# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the verify sampler of Qwen36MTPContractForCausalLM (tt/spec_sample.py VerifySampler) on controlled logits.

The sampler runs as the model runs it: a trace of its own over a persistent [1, 1, B*T, V] bfloat16 logits buffer
(V = 248320, four top-k chunks), warmed eagerly and captured with program cache misses disallowed, replayed with a
fresh ``sampling_plan`` per verify. Each sampled row's draws are checked against vLLM's sampling distribution
(sampling_reference.target_probs, chi-square), on every device of the mesh; then the plugin's id walk over the draws
(accept_greedy_drafts) against the target: the first committed token and the acceptance of the first draft. Greedy,
top-k 1 and padding rows draw nothing; a seeded row draws the same token for the same position.

    pytest models/demos/blackhole/qwen36/tests/test_spec_sample.py    (Blackhole, up to 2 chips)
"""

import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p, target_probs
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt.spec_sample import VerifySampler, sampling_plan

VOCAB = 248320
MESH = _resolve_mesh_shape(2)
P_MIN = 1e-4
# Support tokens of a row: spread over the four top-k chunks of the vocabulary, the ends included.
SUPPORT = (5, 62079, 62080, 130001, 200003, VOCAB - 1)
SUPPORT_LOGITS = (2.0, 1.4, 0.9, 0.5, 0.1, -0.4)
FLOOR = -60.0  # every other token: no mass at any temperature used here


def _parametrize(fn):
    fn = pytest.mark.parametrize("device_params", [{"trace_region_size": 64 << 20}], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


class _Sampler:
    """VerifySampler over a persistent logits buffer, warmed, then captured as the model captures it."""

    def __init__(self, mesh, B, T):
        self.mesh, self.B, self.T = mesh, B, T
        self.logits = ttnn.from_torch(
            torch.zeros(1, 1, B * T, VOCAB),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.sampler = VerifySampler(mesh, B, T, VOCAB)
        self.sampler.body(self.logits)
        ttnn.synchronize_device(mesh)
        mesh.set_program_cache_misses_allowed(False)
        try:
            self.trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                self.sampler.body(self.logits)
            finally:
                ttnn.end_trace_capture(mesh, self.trace, cq_id=0)
        finally:
            mesh.set_program_cache_misses_allowed(True)
        ttnn.synchronize_device(mesh)

    def stage_logits(self, logits):
        h = ttnn.from_torch(
            logits.reshape(1, 1, self.B * self.T, VOCAB),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(h, self.logits)

    def draw(self, plan):
        """[devices, B*T] draws of one replay."""
        self.sampler.stage(plan)
        ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        rows = self.B * self.T
        return torch.stack(
            [ttnn.to_torch(t).reshape(-1)[:rows].to(torch.int64) for t in ttnn.get_device_tensors(self.sampler.draws)]
        )

    def release(self):
        ttnn.release_trace(self.mesh, self.trace)


def _row_logits(r):
    """Row r's logits: the support tokens in a row-dependent order, FLOOR elsewhere."""
    x = torch.full((VOCAB,), FLOOR)
    order = torch.roll(torch.arange(len(SUPPORT)), r % len(SUPPORT))
    for i, tok in zip(order.tolist(), SUPPORT):
        x[tok] = SUPPORT_LOGITS[i]
    return x


# user -> (temperature, top_k, top_p, seed); 3 greedy, 5 top-k 1, 7 a padding row
USERS = {
    0: (1.0, 20, 1.0, None),
    1: (0.7, 4, 1.0, None),
    2: (1.3, 32, 0.75, None),
    3: (0.0, 1, 1.0, None),
    4: (1.0, 10, 1.0, 1234),
    5: (1.0, 1, 1.0, None),
    6: (0.5, 32, 0.9, None),
}


def _params(B):
    rows = [USERS.get(u, (0.0, 1, 1.0, None)) for u in range(B)]
    return SimpleNamespace(
        **{k: [row[i] for row in rows] for i, k in enumerate(("temperature", "top_k", "top_p", "seed"))}
    )


@_parametrize
def test_verify_draws_follow_the_target_distribution(mesh_device):
    from vllm_tt_plugin.spec_decode import accept_greedy_drafts

    B, T, N = 8, 4, 2000
    s = _Sampler(mesh_device, B, T)
    logits = torch.stack([_row_logits(r) for r in range(B * T)])
    s.stage_logits(logits)
    params = _params(B)
    positions = [100 + 7 * u for u in range(B - 1)] + [-1]
    try:
        draws = []
        for i in range(N):
            # a new position per replay, so the seeded row draws from N streams too
            plan = sampling_plan(params, [p + i if p >= 0 else p for p in positions], T)
            d = s.draw(plan)
            assert bool((d == d[0]).all()), "devices of the mesh drew different tokens"
            draws.append(d[0])
        draws = torch.stack(draws)  # [N, B*T]
        sampled = torch.tensor(plan.sampled[: B * T])
        assert sampled.tolist() == [u in (0, 1, 2, 4, 6) for u in range(B) for _ in range(T)]
        rules = {}
        for r in sampled.nonzero().reshape(-1).tolist():
            temp, k, p, _ = USERS[r // T]
            want = target_probs(logits[r], temp, k, p)
            counts = torch.bincount(draws[:, r], minlength=VOCAB)
            pv = chi_square_p(counts, want)
            if p < 1.0:
                rules[r] = {
                    rule: chi_square_p(counts, target_probs(logits[r], temp, k, p, rule))
                    for rule in ("vllm", "at_most")
                }
            else:
                assert pv > P_MIN, f"row {r}: draws do not follow the target (p = {pv})"
        # the walk over the draws: user 0 drafts the target's mode at column 0, user 1 a token of small mass
        for u, draft in ((0, 0), (1, 3)):
            r0 = u * T
            want = target_probs(logits[r0], *USERS[u][:3])
            d0 = int(torch.argsort(want, descending=True)[draft])
            drafts = torch.tensor([[d0, -1, -1]] * N)
            committed, counts = accept_greedy_drafts(draws[:, r0 : r0 + T], drafts, torch.ones(N, dtype=torch.int32))
            assert chi_square_p(torch.bincount(committed[:, 0].to(torch.int64), minlength=VOCAB), want) > P_MIN
            rate = float((counts == 2).float().mean())
            sigma = (float(want[d0]) * (1 - float(want[d0])) / N) ** 0.5
            assert (
                abs(rate - float(want[d0])) < 5 * sigma + 1e-3
            ), f"user {u}: accept rate {rate} vs p {float(want[d0])}"
        # the columns of a user draw independently: with the same logits they agree at the collision rate only
        same = (draws[:, 6 * T] == draws[:, 6 * T + 1]).float().mean()
        want0, want1 = (target_probs(logits[6 * T + j], *USERS[6][:3]) for j in (0, 1))
        assert abs(float(same) - float((want0 * want1).sum())) < 0.05
        # a seeded row draws the same token for the same position, whichever replay
        plan = sampling_plan(params, positions, T)
        again = [s.draw(plan)[0][4 * T : 5 * T] for _ in range(3)]
        assert all(torch.equal(again[0], x) for x in again)
        for r, by_rule in rules.items():
            logger.info(f"row {r} top-p {USERS[r // T][2]}: chi-square p by rule {by_rule}")
            assert max(by_rule.values()) > P_MIN, f"row {r}: draws follow neither top-p rule ({by_rule})"
    finally:
        s.release()


@_parametrize
@pytest.mark.parametrize("K, B", [(1, 16), (7, 4), (11, 2)])
def test_verify_draws_at_every_supported_k(mesh_device, K, B):
    """Fewer than 32 verify rows (K = 11) pad the sampler's rows; every sampled row draws a support token."""
    T = K + 1
    s = _Sampler(mesh_device, B, T)
    logits = torch.stack([_row_logits(r) for r in range(B * T)])
    s.stage_logits(logits)
    params = SimpleNamespace(temperature=[1.0] * B, top_k=[8] * B, top_p=[1.0] * B, seed=[None] * B)
    try:
        for i in range(20):
            d = s.draw(sampling_plan(params, [50 + i] * B, T))
            assert bool((d == d[0]).all())
            assert set(d[0].tolist()) <= set(SUPPORT)
    finally:
        s.release()


@_parametrize
def test_verify_sampler_device_time(mesh_device):
    B, T, N = 8, 4, 200
    s = _Sampler(mesh_device, B, T)
    plan = sampling_plan(_params(B), [100] * (B - 1) + [-1], T)
    try:
        s.sampler.stage(plan)
        ttnn.execute_trace(mesh_device, s.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.execute_trace(mesh_device, s.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        us = (time.perf_counter() - t0) / N * 1e6
        t0 = time.perf_counter()
        for _ in range(N):
            s.sampler.stage(plan)
        stage_us = (time.perf_counter() - t0) / N * 1e6
    finally:
        s.release()
    logger.info(f"verify sampler, B={B} T={T} V={VOCAB}: {us:.1f} us per replay, staging a plan {stage_us:.1f} us")
