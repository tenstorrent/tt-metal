# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VerifySampler (kernels/sample_verify.cpp) on one Galaxy row (TP=8), eager and traced. Galaxy only.

The decode slots 0-3 of a [32, 152576] fp32 logit tile hold verify passes (reference.spec_decode.plan_verify_passes):
one request with three drafts, two requests sharing a pass, greedy and sampled requests, top_k and nucleus rows. Two
checks per pass:

- bit for bit: the kernel's output page equals the host mirror (reference.sample_verify.verify_pass) run on the
  device's own candidate pages (DeviceSampler.chip_cands, read back per chip) and parameters: steps 1-2 are the decode
  sampler's, and the accept step is integer arithmetic;
- semantics: a request whose candidates hold its nucleus (not flagged truncated) commits what the float64 reference of
  tests/test_sample_verify_host.py commits on the full logits with the same Q24 uniforms, away from rounding edges.
"""

import random
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import sample_verify as sv
from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS, plan_verify_passes
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes
from models.experimental.mimo_v2_6_flash.tests.test_sample_verify_host import q24, ref_walk
from models.experimental.mimo_v2_6_flash.tt.common import TILE, MeshPlan
from models.experimental.mimo_v2_6_flash.tt.device_sampler import CANDIDATES_PER_CHIP, DeviceSampler
from models.experimental.mimo_v2_6_flash.tt.verify_sampler import VerifySampler

VOCAB = 152576
B = 32
USERS = 4
REPLAYS = 50
# Per case: the pass's requests (drafts, (temperature, top_k, top_p)); temperature 0 is greedy.
CASES = [
    [(3, (1.0, 0, 0.95))],
    [(3, (0.7, 20, 0.9))],
    [(3, (0.0, 0, 1.0))],
    [(1, (1.0, 0, 0.95)), (0, (1.3, 40, 1.0)), (0, (0.0, 0, 1.0))],
    [(2, (1.0, 5, 1.0)), (0, (0.9, 0, 0.5))],
]


def raw_page(t):
    """A [32, 32] uint32 TILE page's raw words (the kernels address the page as bytes): undo the untilize."""
    rows = ttnn.to_torch(t).reshape(TILE, TILE).to(torch.int64) & 0xFFFFFFFF
    raw = [0] * (TILE * TILE)
    for r in range(TILE):
        for c in range(TILE):
            face = (r // 16) * 2 + c // 16
            raw[face * 256 + (r % 16) * 16 + c % 16] = int(rows[r, c])
    return raw


def device_lists(sampler):
    """Per slot, per chip: (sorted (key, local id) pairs, normalizer), from each chip's candidate page."""
    pages = [raw_page(t) for t in ttnn.get_device_tensors(sampler.chip_cands)[: sampler.plan.tp]]
    k = sampler.k
    lists = []
    for u in range(USERS):
        per_chip = []
        for words in pages:
            pairs = [(words[u * 2 * k + 2 * j], words[u * 2 * k + 2 * j + 1]) for j in range(k)]
            norm = words[512 + u * 4] | words[512 + u * 4 + 1] << 32
            per_chip.append((pairs, norm))
        lists.append(per_chip)
    return lists


def make_case(gen, rng, case):
    """(logits [B, VOCAB], argmax [B], pass, spec_sampling) for one case; the drafts favour the slot's argmax."""
    logits = (
        torch.randn(B, VOCAB, generator=gen) * torch.tensor([rng.choice((4.0, 6.0, 9.0)) for _ in range(B)])[:, None]
    )
    rows, sampling = [], {"temperature": [], "top_k": [], "top_p": [], "greedy": [], "uniforms": []}
    slot = 0
    for row, (n, (t, k, p)) in enumerate(case):
        drafts = [int(logits[slot + j].argmax()) if rng.random() < 0.7 else rng.randrange(VOCAB) for j in range(n)]
        tokens = [rng.randrange(VOCAB)] + drafts + [-1] * (MAX_DRAFTS - n)
        positions = list(range(100, 101 + n)) + [-1] * (MAX_DRAFTS - n)
        rows.append((row, tokens, positions, n))
        sampling["temperature"].append(t)
        sampling["top_k"].append(k if k else VOCAB)
        sampling["top_p"].append(p)
        sampling["greedy"].append(t == 0.0)
        sampling["uniforms"].append([rng.random() for _ in range(MAX_DRAFTS + 2)])
        slot += n + 1
    (vp,) = plan_verify_passes(rows)
    spec_sampling = {
        "temperature": torch.tensor(sampling["temperature"], dtype=torch.float32),
        "top_k": torch.tensor(sampling["top_k"], dtype=torch.int32),
        "top_p": torch.tensor(sampling["top_p"], dtype=torch.float32),
        "greedy": torch.tensor(sampling["greedy"]),
        "uniforms": torch.tensor(sampling["uniforms"], dtype=torch.float32),
    }
    top = logits.max(dim=-1, keepdim=True).values
    argmax = torch.where(logits == top, torch.arange(VOCAB), VOCAB).min(dim=-1).values  # lowest id on ties
    return logits.float(), argmax, vp, spec_sampling


def check_semantics(logits, vp, spec_sampling, out):
    """Requests not flagged truncated against the float64 reference; returns (compared, skipped near an edge)."""
    compared, near_edges = 0, 0
    for g, row in enumerate(dict.fromkeys(r for _, r, _ in vp.answers)):
        count, tokens, truncated = out[g]
        if truncated:
            continue
        slots = sorted((c, s) for s, r, c in vp.answers if r == row)
        n = len(slots) - 1
        t, k, p = (float(spec_sampling[x][row]) for x in ("temperature", "top_k", "top_p"))
        # The plugin's [B, K + 2] uniforms: column 0 the bonus, 1..n the request's accept draws, K + 1 the correction
        # (K = MAX_DRAFTS, the block's draft columns, whatever the request's n).
        u = [float(x) for x in spec_sampling["uniforms"][row]]
        drafts = [vp.tokens[s] for _, s in slots[1:]]
        want, want_tokens, near = ref_walk(
            [logits[s].double().tolist() for _, s in slots],
            (t, int(k) if 1 <= int(k) < VOCAB else 0, p),
            drafts,
            bool(spec_sampling["greedy"][row]),
            q24(u[0]),
            [q24(x) for x in u[1 : 1 + n]],
            q24(u[MAX_DRAFTS + 1]),
        )
        if near:
            near_edges += 1
            continue
        assert (count, tokens) == (
            want,
            want_tokens,
        ), f"request {row}: device {(count, tokens)} != {(want, want_tokens)}"
        compared += 1
    return compared, near_edges


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_verify_sampler(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    vocab_local = VOCAB // plan.tp
    gen, rng = torch.Generator().manual_seed(0), random.Random(0)
    sampler = DeviceSampler(plan, vocab_local, USERS, B)
    verify = VerifySampler(sampler, VOCAB)
    argmax_tt = ttnn.from_torch(
        torch.zeros(1, B, dtype=torch.int32),
        device=plan.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.replicate(),
    )
    compared = near = 0
    logits_tt = trace = None
    for i, case in enumerate(CASES * 2):
        logits, argmax, vp, spec_sampling = make_case(gen, rng, case)
        host = ttnn.from_torch(
            logits.reshape(1, 1, B, VOCAB), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=plan.shard(3)
        )
        if logits_tt is None:
            logits_tt = ttnn.to_device(host, plan.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host, logits_tt)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                argmax.reshape(1, B).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=plan.replicate(),
            ),
            argmax_tt,
        )
        slots, groups = sv.pass_inputs(vp, spec_sampling)
        verify.set_inputs(verify.host_inputs(slots, groups))
        if i < len(CASES):
            verify(logits_tt, argmax_tt)  # eager
        else:
            if trace is None:  # the second round replays one capture with each case's inputs
                ttnn.synchronize_device(plan.mesh_device)
                trace = ttnn.begin_trace_capture(plan.mesh_device, cq_id=0)
                verify(logits_tt, argmax_tt)
                ttnn.end_trace_capture(plan.mesh_device, trace, cq_id=0)
            ttnn.execute_trace(plan.mesh_device, trace, cq_id=0, blocking=False)
        words = ttnn.to_torch(ttnn.get_device_tensors(verify.out)[0]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        words = words.tolist()
        out = sv.unpack_out(words)
        # Bit for bit against the mirror on the device's own candidates.
        params = [sv.param_words(*slots.get(s, (0.0, 0, 1.0)), VOCAB) for s in range(USERS)]
        mirror = sv.verify_pass(device_lists(sampler), params, argmax.tolist(), sv.control_words(groups), vocab_local)
        assert words == mirror, f"case {i}: kernel {out} != mirror {sv.unpack_out(mirror)}"
        c, e = check_semantics(logits, vp, spec_sampling, out)
        compared, near = compared + c, near + e
    assert compared >= len(CASES), f"only {compared} requests compared against the reference ({near} near an edge)"
    ttnn.synchronize_device(plan.mesh_device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(plan.mesh_device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(plan.mesh_device)
    logger.info(
        f"VerifySampler: {compared} requests equal the reference ({near} near an edge skipped); traced "
        f"{1e3 * (time.perf_counter() - t0) / REPLAYS:.3f} ms per pass; {CANDIDATES_PER_CHIP} candidates per chip"
    )
    ttnn.release_trace(plan.mesh_device, trace)
