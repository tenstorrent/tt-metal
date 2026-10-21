# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeviceSampler on one Galaxy row (TP=8) against the exact full-vocabulary draw, eager and traced.

The decode users (rows 0-3 of a [32, 152576] fp32 logit tile) cycle through sets of temperature / top_k / top_p, each
drawn several times. The reference is vLLM's rule over the whole vocabulary (softmax(logits / T), or over the top_k
logits; keep a token while the mass ranked above it is below top_p) with the same inverse-CDF draw and the uniform
the device's splitmix64 stream gives (replayed on the host from the seeds), so the device token must equal it on
every row whose nucleus and top_k lie in the row's CANDIDATES_PER_CHIP most likely tokens, except where the uniform
lands within rounding of a bucket edge. Greedy rows must take the argmax, the lowest id among tied maxima, and leave
their stream alone. Rows past the users must get token 0. The traced replays draw anew each step and time the
sampling head.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.device_sampler import CANDIDATES_PER_CHIP, DeviceSampler

VOCAB = 152576
B = 32
USERS = 4
DRAWS = 6
REPLAYS = 50
# How close (in cumulative probability) a uniform may land to a bucket edge before a mismatch counts as rounding.
EDGE = 2e-5
# (temperature, top_k, top_p) of users 0-3; top_k -1 or the vocabulary size is none, as the plugin sends it.
PARAM_SETS = [
    ((0.0, 1.0, 0.6, 1.0), (-1, -1, 20, 1), (1.0, 0.95, 0.95, 0.0)),
    ((1.0, 0.8, 1.2, 0.7), (VOCAB, -1, 40, 5), (0.95, 0.9, 0.95, 1.0)),
    ((1.0, 1.0, 0.9, 1.5), (100, -1, 256, -1), (0.9, 0.99, 0.95, 0.5)),
    ((0.6, 1.0, 0.0, 1.0), (-1, 50, -1, 3), (0.95, 1.0, 1.0, 0.8)),
]
SCALES = [(6.0, 4.0, 3.0, 2.5), (5.0, 2.0, 1.0, 0.7), (8.0, 3.0, 1.5, 4.0)]
MASK = (1 << 64) - 1


class Streams:
    """The device's per-row splitmix64 streams (kernels/sample_fixed.hpp), replayed on the host."""

    def __init__(self, gen):
        self.seeds = torch.randint(-(2**62), 2**62, (B,), generator=gen, dtype=torch.int64)
        self.state = [int(x) & MASK for x in self.seeds]

    def draw(self, sampled):
        """The next uniforms of the sampled rows (others 0, their streams unchanged)."""
        out = torch.zeros(B, dtype=torch.float64)
        for r in range(B):
            if not sampled[r]:
                continue
            self.state[r] = (self.state[r] + 0x9E3779B97F4A7C15) & MASK
            z = self.state[r]
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
            z ^= z >> 31
            out[r] = (z >> 40) / float(1 << 24)
        return out


def make_logits(gen, scales):
    logits = torch.randn(B, VOCAB, generator=gen) * torch.tensor(list(scales) + [1.0] * (B - USERS))[:, None]
    # Tied maxima on two chips (the greedy rule takes the lower id) and inside one chip.
    top = logits[0].max() + 1.0
    logits[0, 70000] = logits[0, 3000] = top
    logits[1, 5000] = logits[1, 5001] = logits[1].max() + 0.5
    return logits.float()


def reference(logits, temperature, top_k, top_p, uniform):
    """Per user: (token, nucleus size, whether the uniform is within EDGE of a bucket edge)."""
    out = []
    for r in range(USERS):
        t, k, p, u = float(temperature[r]), int(top_k[r]), float(top_p[r]), float(uniform[r])
        row = logits[r].double()
        if t == 0.0 or k == 1:
            out.append((int(torch.argmax(row)), 1, False))  # torch.argmax: the first maximum
            continue
        vals, ids = torch.sort(row / t, descending=True, stable=True)
        probs = torch.softmax(vals, dim=0)
        if 1 <= k < VOCAB:
            probs = probs[:k] / probs[:k].sum()
            ids = ids[:k]
        above = probs.cumsum(0) - probs
        keep = above < p
        keep[0] = True
        kept = probs[keep]
        cum = kept.cumsum(0)
        threshold = u * cum[-1]
        pick = min(int((cum <= threshold).sum()), kept.numel() - 1)
        near = bool(((cum - threshold).abs() < EDGE).any())
        out.append((int(ids[keep][pick]), int(keep.sum()), near))
    return out


def read_tokens(t):
    chips = [ttnn.to_torch(d).reshape(-1)[:B].to(torch.int64) for d in ttnn.get_device_tensors(t)]
    for c in range(1, len(chips)):
        assert torch.equal(chips[c], chips[0]), f"chip {c} drew other tokens than chip 0"
    return chips[0]


def check(tag, tokens, ref, logits, top_k):
    """Rows 0-3 inside the candidates must match the reference (bar edge rounding), rows past them be 0; returns
    (bad rows, rows compared exactly)."""
    assert not tokens[USERS:].any(), f"{tag}: rows past the users hold {tokens[USERS:].tolist()}"
    bad, exact_rows = [], 0
    for r in range(USERS):
        token, nucleus, near = ref[r]
        k = int(top_k[r])
        exact = nucleus <= CANDIDATES_PER_CHIP and not (CANDIDATES_PER_CHIP < k < VOCAB)
        exact_rows += exact
        if int(tokens[r]) == token:
            continue
        logger.info(
            f"{tag} row {r}: device {int(tokens[r])} ref {token} nucleus {nucleus} near_edge {near} exact {exact} "
            f"logit dev {float(logits[r, int(tokens[r])]):.4f} ref {float(logits[r, token]):.4f}"
        )
        if exact and not near:
            bad.append((tag, r))
    return bad, exact_rows


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_device_sampler(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    vocab_local = VOCAB // plan.tp
    gen = torch.Generator().manual_seed(0)
    sampler = DeviceSampler(plan, vocab_local, USERS, B)
    token_input = ttnn.from_torch(
        torch.zeros(1, B, dtype=torch.int32),
        device=plan.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.replicate(),
    )
    bad, exact_rows, compared = [], 0, 0
    for s, scales in enumerate(SCALES):
        logits = make_logits(gen, scales)
        logits_tt = ttnn.from_torch(
            logits.reshape(1, 1, B, VOCAB),
            device=plan.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.shard(3),
        )
        for i, (temperature, top_k, top_p) in enumerate(PARAM_SETS):
            temperature, top_k, top_p = torch.tensor(temperature), torch.tensor(top_k), torch.tensor(top_p)
            streams = Streams(gen)
            hosts, _ = sampler.host_params(temperature, top_k, top_p, seeds=streams.seeds)
            sampler.set_params(hosts)
            sampled = [r < USERS and float(temperature[r]) != 0 and int(top_k[r]) != 1 for r in range(B)]
            for d in range(DRAWS):
                uniform = streams.draw(sampled)
                tokens = read_tokens(sampler(logits_tt, token_input=token_input))
                assert torch.equal(read_tokens(token_input), tokens), "token_input differs from next_tokens"
                ref = reference(logits, temperature, top_k, top_p, uniform)
                for r in range(USERS):
                    if temperature[r] == 0 or top_k[r] == 1:
                        assert int(tokens[r]) == ref[r][0], f"greedy row {r}: {int(tokens[r])} != {ref[r][0]}"
                b, e = check(f"logits {s} params {i} draw {d}", tokens, ref, logits, top_k)
                bad += b
                exact_rows += e
                compared += USERS
        if s == 0:
            logger.info(f"nucleus sizes (last draw): {[n for _, n, _ in ref]}")

        # Traced: capture once, replay with new uniforms (the last parameter set).
        if s == len(SCALES) - 1:
            sampler(logits_tt, token_input=token_input)
            streams.draw(sampled)  # that step's draw; the capture below runs nothing
            ttnn.synchronize_device(plan.mesh_device)
            trace = ttnn.begin_trace_capture(plan.mesh_device, cq_id=0)
            out = sampler(logits_tt, token_input=token_input)
            ttnn.end_trace_capture(plan.mesh_device, trace, cq_id=0)
            for d in range(DRAWS):
                uniform = streams.draw(sampled)
                ttnn.execute_trace(plan.mesh_device, trace, cq_id=0, blocking=False)
                tokens = read_tokens(out)
                b, e = check(
                    f"replay {d}", tokens, reference(logits, temperature, top_k, top_p, uniform), logits, top_k
                )
                bad += b
                exact_rows += e
                compared += USERS
            ttnn.synchronize_device(plan.mesh_device)
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(plan.mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(plan.mesh_device)
            per = (time.perf_counter() - t0) / REPLAYS
            logger.info(f"DeviceSampler traced: {1e3 * per:.3f} ms per step")
            ttnn.release_trace(plan.mesh_device, trace)
        logits_tt.deallocate(True)
    logger.info(f"{compared} rows compared, {exact_rows} inside the candidates, mismatches there: {bad}")
    assert not bad, f"rows {bad} differ from the reference inside the candidates"
