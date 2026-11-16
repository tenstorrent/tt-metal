# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the verify sampler of Qwen36MTPContractForCausalLM (tt/spec_sample.py VerifySampler) on controlled logits.

The sampler runs as the model runs it: a trace of its own over a persistent [1, 1, B*T, V] bfloat16 logits buffer
(V = 248320, four top-k chunks), warmed eagerly and captured with program cache misses disallowed, replayed with a
fresh ``sampling_plan`` per verify. Each sampled row's draws are checked against vLLM's sampling distribution
(sampling_reference.target_probs, chi-square), on every device of the mesh; then the plugin's id walk over the draws
(accept_greedy_drafts) against the target: the first committed token and the acceptance of the first draft. Greedy,
top-k 1 and padding rows draw nothing; a seeded row draws the same token for the same position, also on a chained
verify, whose seeds the chain program writes (spec_chain.ChainInputs).

The sampler takes the top-32 of its four vocabulary chunks in one ttnn.topk call over the chunks stacked as rows. Next
to the sampler with one call per chunk (tests/spec_op_graphs.py ChunkedVerifySampler), on logits full of exact ties,
it hands ttnn.sampling the same values and ids in the same order and draws the same tokens, on every device.

    pytest models/demos/blackhole/qwen36/tests/test_spec_sample.py    (Blackhole, up to 2 chips)
"""

import random
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p, target_probs
from models.demos.blackhole.qwen36.tests.spec_op_graphs import ChunkedVerifySampler
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tests.test_spec_sample_topk_host import _logits as _tie_logits
from models.demos.blackhole.qwen36.tt.spec_sample import VerifySampler, chained_sampling_plan, sampling_plan

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
    """VerifySampler over a persistent logits buffer, warmed, then captured as the model captures it. With
    ``capture=False`` the caller captures once every device buffer that the replays run next to exists: a buffer
    allocated after the capture and alive at a replay can sit where the trace writes."""

    def __init__(self, mesh, B, T, capture=True, cls=VerifySampler, logits=None):
        self.mesh, self.B, self.T = mesh, B, T
        self.logits = logits
        if logits is None:
            self.logits = ttnn.from_torch(
                torch.zeros(1, 1, B * T, VOCAB),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        self.sampler = cls(mesh, B, T, VOCAB)
        self.trace = None
        if capture:
            self.capture()

    def capture(self):
        mesh = self.mesh
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
def test_a_chained_seeded_row_draws_what_an_unchained_verify_draws(mesh_device):
    """The chain program (spec_chain.ChainInputs.body) writes a chained verify's sampler seeds, each user's picked by
    the predecessor's count from the host's candidates (spec_sample.chained_sampling_plan). Over the same logits a
    seeded user then draws, token for token, what an unchained verify whose column 0 sits at that count's position
    draws: the chain kernel's pick, its write into the sampler's 1-D seeds and manual_seed's user-id match lose no
    bit. An unseeded user draws support tokens."""
    from models.demos.blackhole.qwen36.tests.test_spec_accept_host import ROWS
    from models.demos.blackhole.qwen36.tests.test_spec_chain_host import _chain_case, _ctrl_words
    from models.demos.blackhole.qwen36.tt.spec_chain import ChainInputs
    from models.demos.blackhole.qwen36.tt.spec_step import VERIFY_STEP_ROWS, drafter_pack_rows, pack_width

    B, K, groups, nv = 8, 3, 3, 6
    T, R = K + 1, B * (K + 1)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def buffer(t, dtype):
        return ttnn.from_torch(
            torch.as_tensor(t).to(torch.int32),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    def copy_in(host, dev):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.as_tensor(host).to(torch.int32), dtype=dev.dtype, mesh_mapper=rep), dev
        )

    s = _Sampler(mesh_device, B, T, capture=False)
    step_dev = buffer(torch.zeros(VERIFY_STEP_ROWS, pack_width(R)), ttnn.uint32)
    pack_dev = buffer(torch.zeros(drafter_pack_rows(K), pack_width(R)), ttnn.uint32)
    ctrl_devs = [buffer(torch.zeros(1, _ctrl_words(B, nv)), ttnn.int32) for _ in range(groups)]
    ids_dev = buffer(torch.zeros(1, 1, 1, 32), ttnn.uint32)
    draft_devs = [buffer(torch.zeros(1, 1, B, 1), ttnn.uint32) for _ in range(K)]
    chain = ChainInputs(mesh_device, B, K, ROWS, step_dev, pack_dev, ctrl_devs, nv, sampler=s.sampler)
    chain.bind(ids_dev, draft_devs)
    # As in the engine, the chain program and every buffer it reads or writes exist before the sampler is captured.
    chain.body()
    s.capture()
    s.stage_logits(torch.stack([_row_logits(r) for r in range(R)]))
    seeded = [u for u in range(B) if u % 2 == 0]
    params = SimpleNamespace(
        temperature=[1.0] * B,
        top_k=[20] * B,
        top_p=[1.0] * B,
        seed=[1000 + u if u in seeded else None for u in range(B)],
    )
    rng = random.Random(11)
    compared = 0
    try:
        for _ in range(10):
            case = _chain_case(rng, B, K, groups, nv, True)
            p0_prev = [p if a else -1 for p, a in zip(case.p0_prev, case.active)]
            plan, seed_cand = chained_sampling_plan(params, p0_prev, T)
            copy_in(case.ids, ids_dev)
            copy_in(case.pack, pack_dev)
            for d, dev in zip(case.drafts, draft_devs):
                copy_in(d.reshape(1, 1, B, 1), dev)
            copy_in(case.draws[..., :R], s.sampler.draws)  # the predecessor's draws
            chain.stage(case.overlay, case.cand, seed_cand)
            chain.body()
            s.sampler.seeds_written()
            chained = s.draw(plan)
            positions = [p + m + 1 if p >= 0 else -1 for p, m in zip(p0_prev, case.m)]
            unchained = s.draw(sampling_plan(params, positions, T))
            for u in range(B):
                if p0_prev[u] < 0:
                    continue
                cols = slice(u * T, (u + 1) * T)
                if u in seeded:
                    assert torch.equal(chained[:, cols], unchained[:, cols]), f"user {u} at {positions[u]}"
                    compared += 1
                else:
                    assert set(chained[0, cols].tolist()) <= set(SUPPORT)
        assert compared > 10
    finally:
        s.release()
        chain.release()
        for t in (step_dev, pack_dev, ids_dev, *ctrl_devs, *draft_devs):
            ttnn.deallocate(t)


@_parametrize
@pytest.mark.parametrize("B", [8, 4, 2, 1])
def test_verify_sampler_device_time(mesh_device, B):
    """Replay time of the sampler (one top-k call over the stacked chunks) and of the per-chunk sampler, at the row
    counts of the verify row buckets (T = 4)."""
    T, N = 4, 200
    times = {}
    for name, cls in (("stacked", VerifySampler), ("per-chunk", ChunkedVerifySampler)):
        s = _Sampler(mesh_device, B, T, cls=cls)
        plan = sampling_plan(_params(B), [100] * (B - 1) + [-1], T)
        try:
            s.sampler.stage(plan)
            ttnn.execute_trace(mesh_device, s.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(N):
                ttnn.execute_trace(mesh_device, s.trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times[name] = (time.perf_counter() - t0) / N * 1e6
            t0 = time.perf_counter()
            for _ in range(N):
                s.sampler.stage(plan)
            stage_us = (time.perf_counter() - t0) / N * 1e6
        finally:
            s.release()
        logger.info(
            f"verify sampler ({name}), B={B} T={T} V={VOCAB}: {times[name]:.1f} us per replay, "
            f"staging a plan {stage_us:.1f} us"
        )
    logger.info(
        f"verify sampler, {B * T} rows: stacked {times['stacked']:.1f} us, per-chunk {times['per-chunk']:.1f} us"
    )


TIE_KINDS = ("coarse", "plateau", "bf16", "zeros")


def _device_rows(t, rows):
    """Per device, the first ``rows`` rows of ``t`` as torch."""
    return [ttnn.to_torch(d).reshape(-1, int(t.shape[-1]))[:rows] for d in ttnn.get_device_tensors(t)]


@_parametrize
@pytest.mark.parametrize("B, T", [(8, 4), (4, 4), (2, 4), (1, 4), (2, 12)])
def test_the_stacked_top_k_hands_the_sampler_the_per_chunk_candidates(mesh_device, B, T):
    """Eager, on logits with thousands of equal maxima per chunk (coarse), 40 equal maxima at chunk ends (plateau),
    random bfloat16 logits (bf16) and +0.0 / -0.0 maxima (zeros): the stacked sampler's values and global ids equal
    the per-chunk sampler's bit for bit, position for position, on every device. At B * T = 32 the stacked call has
    128 rows: on a 110-core worker grid some cores reduce two rows."""
    rows = B * T
    s1 = _Sampler(mesh_device, B, T, capture=False)
    s0 = _Sampler(mesh_device, B, T, capture=False, cls=ChunkedVerifySampler, logits=s1.logits)
    for i, kind in enumerate(TIE_KINDS):
        s1.stage_logits(_tie_logits(kind, rows, VOCAB, seed=100 + i))
        got, want = s1.sampler.candidates(s1.logits), s0.sampler.candidates(s1.logits)
        for a, b, what in zip(got, want, ("values", "global ids")):
            for d, (x, y) in enumerate(zip(_device_rows(a, rows), _device_rows(b, rows))):
                if what == "values":
                    x, y = x.view(torch.int16), y.view(torch.int16)  # bit patterns: -0.0 is not +0.0
                assert torch.equal(x, y), f"{kind}: {what} differ on device {d}"
        for t in (*got, *want):
            ttnn.deallocate(t)


@_parametrize
@pytest.mark.parametrize("B, T", [(8, 4), (1, 4)])
def test_the_stacked_sampler_draws_what_the_per_chunk_sampler_draws(mesh_device, B, T):
    """Both samplers captured as the model captures its sampler, over one logits buffer: for every tie-heavy logits
    kind, 25 plans (seeded rows at advancing positions and unseeded rows with fresh seeds, top-k 32 and top-k 20
    top-p 0.95, a greedy row) staged into both give the same draws on every device."""
    rows = B * T
    s1 = _Sampler(mesh_device, B, T, capture=False)
    s0 = _Sampler(mesh_device, B, T, capture=False, cls=ChunkedVerifySampler, logits=s1.logits)
    s1.sampler.body(s1.logits)  # both compiled and every buffer allocated before either capture
    s0.sampler.body(s1.logits)
    s1.capture()
    s0.capture()
    rng = random.Random(7)
    users = [(1.0, 32, 1.0, 900 + u) if u % 3 == 0 else (1.0, 20, 0.95, None) for u in range(B)]
    if B > 2:
        users[2] = (0.0, 1, 1.0, None)
    params = SimpleNamespace(
        **{k: [row[i] for row in users] for i, k in enumerate(("temperature", "top_k", "top_p", "seed"))}
    )
    compared = 0
    try:
        for i, kind in enumerate(TIE_KINDS):
            s1.stage_logits(_tie_logits(kind, rows, VOCAB, seed=200 + i))
            for step in range(25):
                plan = sampling_plan(
                    params, [40 + 5 * u + step for u in range(B)], T, lambda: rng.randrange(1, 10**6)
                )
                d1, d0 = s1.draw(plan), s0.draw(plan)
                assert torch.equal(d1, d0), f"{kind} step {step}: the draws differ"
                assert bool((d1 == d1[0]).all()), "devices of the mesh drew different tokens"
                compared += sum(plan.sampled[:rows])
    finally:
        s1.release()
        s0.release()
    assert compared >= 4 * 25 * T
