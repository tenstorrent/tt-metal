# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeviceSampler on one Galaxy row (TP=8) against the exact full-vocabulary draw, eager and traced.

The decode users (rows 0-3, or 0-7: the scan's 8 user slots, of a [32, 152576] fp32 logit tile) cycle through sets
of temperature / top_k / top_p (user u takes column u % 4 of a set), each drawn several times. The reference is vLLM's rule over the whole vocabulary (softmax(logits / T), or over the top_k
logits; keep a token while the mass ranked above it is below top_p) with the same inverse-CDF draw and the uniform
the device's splitmix64 stream gives (replayed on the host from the seeds), so the device token must equal it on
every row whose nucleus and top_k lie in the row's CANDIDATES_PER_CHIP most likely tokens, except where the uniform
lands within rounding of a bucket edge. Greedy rows must take the argmax, the lowest id among tied maxima, and leave
their stream alone. Rows past the users must get token 0. The traced replays draw anew each step and time the
sampling head.

The grammar test constrains the users with vLLM bitmasks of shapes aimed at the scan's masking (``grammar_allowed``):
every token of a constrained row must be one its mask allows, and the draw must equal the reference over the allowed
tokens alone under the same rules. The traced replays then take a second mask and the lifted constraint in place.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.device_sampler import (
    CANDIDATES_PER_CHIP,
    DeviceSampler,
    reference_draw,
    splitmix_next,
)

VOCAB = 152576
B = 32
DRAWS = 6
REPLAYS = 50
# (temperature, top_k, top_p) of users 0-3; top_k -1 or the vocabulary size is none, as the plugin sends it.
PARAM_SETS = [
    ((0.0, 1.0, 0.6, 1.0), (-1, -1, 20, 1), (1.0, 0.95, 0.95, 0.0)),
    ((1.0, 0.8, 1.2, 0.7), (VOCAB, -1, 40, 5), (0.95, 0.9, 0.95, 1.0)),
    ((1.0, 1.0, 0.9, 1.5), (100, -1, 256, -1), (0.9, 0.99, 0.95, 0.5)),
    ((0.6, 1.0, 0.0, 1.0), (-1, 50, -1, 3), (0.95, 1.0, 1.0, 0.8)),
]
SCALES = [(6.0, 4.0, 3.0, 2.5), (5.0, 2.0, 1.0, 0.7), (8.0, 3.0, 1.5, 4.0)]


class Streams:
    """The device's per-row splitmix64 streams (kernels/sample_fixed.hpp), replayed on the host."""

    def __init__(self, gen):
        self.seeds = torch.randint(-(2**62), 2**62, (B,), generator=gen, dtype=torch.int64)
        self.state = [int(x) & ((1 << 64) - 1) for x in self.seeds]

    def draw(self, sampled):
        """The next uniforms of the sampled rows (others 0, their streams unchanged)."""
        out = torch.zeros(B, dtype=torch.float64)
        for r in range(B):
            if not sampled[r]:
                continue
            self.state[r], out[r] = splitmix_next(self.state[r])
        return out


def per_user(values, users):
    """A 4-column parameter or scale set for ``users`` rows (user u takes column u % 4)."""
    return [values[u % 4] for u in range(users)]


def make_logits(gen, scales, users):
    logits = torch.randn(B, VOCAB, generator=gen) * torch.tensor(per_user(scales, users) + [1.0] * (B - users))[:, None]
    # Tied maxima on two chips (the greedy rule takes the lower id) and inside one chip.
    top = logits[0].max() + 1.0
    logits[0, 70000] = logits[0, 3000] = top
    logits[1, 5000] = logits[1, 5001] = logits[1].max() + 0.5
    return logits.float()


def reference(logits, temperature, top_k, top_p, uniform):
    """Per user: (token, nucleus size, whether the uniform is within rounding of a bucket edge)."""
    return [
        reference_draw(logits[r], float(temperature[r]), int(top_k[r]), float(top_p[r]), float(uniform[r]))
        for r in range(len(temperature))
    ]


def read_tokens(t):
    chips = [ttnn.to_torch(d).reshape(-1)[:B].to(torch.int64) for d in ttnn.get_device_tensors(t)]
    for c in range(1, len(chips)):
        assert torch.equal(chips[c], chips[0]), f"chip {c} drew other tokens than chip 0"
    return chips[0]


def check(tag, tokens, ref, logits, top_k):
    """The users' rows inside the candidates must match the reference (bar edge rounding), rows past them be 0;
    returns (bad rows, rows compared exactly)."""
    users = len(ref)
    assert not tokens[users:].any(), f"{tag}: rows past the users hold {tokens[users:].tolist()}"
    bad, exact_rows = [], 0
    for r in range(users):
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


def grammar_allowed(gen, logits, users):
    """Per user the tokens its grammar allows (bool [VOCAB]), or None (unconstrained): (0) a sparse random set without
    the row's 16 most likely tokens, (1) 300 tokens of chip 5's shard (every other chip's shard disallowed), (2) the
    least likely token alone, (3) none, (4) short runs across word, tile and chip edges, (5) all but the 64 most likely
    tokens (mostly whole allowed words), (6) every other token, (7) the first and last columns of each tile face."""
    vocab_local = VOCAB // 8
    col = torch.arange(VOCAB) % 32
    out = []
    for u in range(users):
        a = torch.zeros(VOCAB, dtype=torch.bool)
        top = torch.topk(logits[u], 64).indices
        kind = u % 8
        if kind == 0:
            a = torch.rand(VOCAB, generator=gen) < 0.02
            a[top[:16]] = False
        elif kind == 1:
            a[torch.randperm(vocab_local, generator=gen)[:300] + 5 * vocab_local] = True
        elif kind == 2:
            a[int(torch.argmin(logits[u]))] = True
        elif kind == 3:
            a = None
        elif kind == 4:
            a[1000:1100] = True
            a[vocab_local - 3 : vocab_local + 5] = True
            a[70001] = True
        elif kind == 5:
            a[:] = True
            a[top] = False
        elif kind == 6:
            a[::2] = True
        else:
            a = (col == 0) | (col == 15) | (col == 16) | (col == 31)
        out.append(a)
    return out


def pack_mask(allowed):
    """vLLM's grammar bitmask of the users (int32 [users, VOCAB // 32], bit j of word w allows token 32 * w + j; -1 for
    an unconstrained row) and the constrained rows (bool [users])."""
    rows = []
    for a in allowed:
        if a is None:
            rows.append(torch.full((VOCAB // 32,), -1, dtype=torch.int32))
            continue
        w = (a.reshape(-1, 32).to(torch.int64) << torch.arange(32)).sum(1)
        rows.append(torch.where(w >= 1 << 31, w - (1 << 32), w).to(torch.int32))
    return torch.stack(rows), torch.tensor([a is not None for a in allowed])


def reference_masked(logits, allowed, temperature, top_k, top_p, uniform):
    """``reference`` with each constrained row drawn from its allowed tokens alone (vLLM's sampler after the
    bitmask: the disallowed logits -inf)."""
    out = []
    for r in range(len(temperature)):
        args = (float(temperature[r]), int(top_k[r]), float(top_p[r]), float(uniform[r]))
        if allowed[r] is None:
            out.append(reference_draw(logits[r], *args))
            continue
        ids = allowed[r].nonzero().squeeze(1)
        token, nucleus, near = reference_draw(logits[r, ids], *args)
        out.append((int(ids[token]), nucleus, near))
    return out


def check_masked(tag, tokens, ref, logits, temperature, top_k, allowed):
    """``check``, after every constrained row's token is one its mask allows and every greedy row's the reference."""
    for r, a in enumerate(allowed):
        token = int(tokens[r])
        assert a is None or a[token], f"{tag} row {r}: token {token} is disallowed"
        if temperature[r] == 0 or top_k[r] == 1:
            assert token == ref[r][0], f"{tag} greedy row {r}: {token} != {ref[r][0]}"
    return check(tag, tokens, ref, logits, top_k)


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("users", [4, 8])
def test_device_sampler(mesh_device, users):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    vocab_local = VOCAB // plan.tp
    gen = torch.Generator().manual_seed(0)
    sampler = DeviceSampler(plan, vocab_local, users, B)
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
        logits = make_logits(gen, scales, users)
        logits_tt = ttnn.from_torch(
            logits.reshape(1, 1, B, VOCAB),
            device=plan.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.shard(3),
        )
        for i, (temperature, top_k, top_p) in enumerate(PARAM_SETS):
            temperature, top_k, top_p = (torch.tensor(per_user(x, users)) for x in (temperature, top_k, top_p))
            streams = Streams(gen)
            hosts, _ = sampler.host_params(temperature, top_k, top_p, states=streams.seeds)
            sampler.set_params(hosts)
            sampled = [r < users and float(temperature[r]) != 0 and int(top_k[r]) != 1 for r in range(B)]
            for d in range(DRAWS):
                uniform = streams.draw(sampled)
                tokens = read_tokens(sampler(logits_tt, token_input=token_input))
                assert torch.equal(read_tokens(token_input), tokens), "token_input differs from next_tokens"
                ref = reference(logits, temperature, top_k, top_p, uniform)
                for r in range(users):
                    if temperature[r] == 0 or top_k[r] == 1:
                        assert int(tokens[r]) == ref[r][0], f"greedy row {r}: {int(tokens[r])} != {ref[r][0]}"
                b, e = check(f"logits {s} params {i} draw {d}", tokens, ref, logits, top_k)
                bad += b
                exact_rows += e
                compared += users
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
                compared += users
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


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("users", [4, 8])
def test_device_sampler_grammar(mesh_device, users):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    gen = torch.Generator().manual_seed(1)
    sampler = DeviceSampler(plan, VOCAB // plan.tp, users, B)
    token_input = ttnn.from_torch(
        torch.zeros(1, B, dtype=torch.int32),
        device=plan.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.replicate(),
    )
    bad, exact_rows, compared = [], 0, 0

    def compare(tag, tokens, logits, allowed, params, uniform):
        nonlocal bad, exact_rows, compared
        temperature, top_k, top_p = params
        ref = reference_masked(logits, allowed, temperature, top_k, top_p, uniform)
        b, e = check_masked(tag, tokens, ref, logits, temperature, top_k, allowed)
        bad += b
        exact_rows += e
        compared += users

    for s, scales in enumerate(SCALES):
        logits = make_logits(gen, scales, users)
        logits_tt = ttnn.from_torch(
            logits.reshape(1, 1, B, VOCAB),
            device=plan.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.shard(3),
        )
        allowed = grammar_allowed(gen, logits, users)
        mask, rows = pack_mask(allowed)
        for i, param_set in enumerate(PARAM_SETS):
            params = tuple(torch.tensor(per_user(x, users)) for x in param_set)
            temperature, top_k, _ = params
            streams = Streams(gen)
            hosts, _ = sampler.host_params(*params, states=streams.seeds)
            sampler.set_params(hosts)
            sampler.set_grammar(mask, rows)
            sampled = [r < users and float(temperature[r]) != 0 and int(top_k[r]) != 1 for r in range(B)]
            for d in range(DRAWS):
                uniform = streams.draw(sampled)
                tokens = read_tokens(sampler(logits_tt, token_input=token_input))
                compare(f"logits {s} params {i} draw {d}", tokens, logits, allowed, params, uniform)
            # Lifted: the rows draw from the whole vocabulary again.
            sampler.set_grammar()
            uniform = streams.draw(sampled)
            tokens = read_tokens(sampler(logits_tt, token_input=token_input))
            compare(f"logits {s} params {i} lifted", tokens, logits, [None] * users, params, uniform)

        # Traced: captured under the mask; replays under it, under another mask and with the constraint lifted, each
        # uploaded in place between replays as the generator does.
        if s == len(SCALES) - 1:
            sampler.set_grammar(mask, rows)
            sampler(logits_tt, token_input=token_input)
            streams.draw(sampled)
            ttnn.synchronize_device(plan.mesh_device)
            trace = ttnn.begin_trace_capture(plan.mesh_device, cq_id=0)
            out = sampler(logits_tt, token_input=token_input)
            ttnn.end_trace_capture(plan.mesh_device, trace, cq_id=0)
            for tag, current in (("replay", allowed), ("replay other", allowed[::-1]), ("replay lifted", None)):
                if current is None:
                    sampler.set_grammar()
                    current = [None] * users
                else:
                    sampler.set_grammar(*pack_mask(current))
                for d in range(DRAWS):
                    uniform = streams.draw(sampled)
                    ttnn.execute_trace(plan.mesh_device, trace, cq_id=0, blocking=False)
                    compare(f"{tag} {d}", read_tokens(out), logits, current, params, uniform)

            def per_step():
                ttnn.synchronize_device(plan.mesh_device)
                t0 = time.perf_counter()
                for _ in range(REPLAYS):
                    ttnn.execute_trace(plan.mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(plan.mesh_device)
                return (time.perf_counter() - t0) / REPLAYS

            plain = per_step()
            sampler.set_grammar(*pack_mask([torch.arange(VOCAB) % 2 == 0] * users))
            masked = per_step()
            logger.info(
                f"DeviceSampler traced: {1e3 * plain:.3f} ms per step unconstrained, {1e3 * masked:.3f} ms with "
                f"every row allowed every other token"
            )
            ttnn.release_trace(plan.mesh_device, trace)
        logits_tt.deallocate(True)
    logger.info(f"{compared} rows compared, {exact_rows} inside the candidates, mismatches there: {bad}")
    assert not bad, f"rows {bad} differ from the reference inside the candidates"
