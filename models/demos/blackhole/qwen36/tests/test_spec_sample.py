# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the verify sampler of Qwen36MTPContractForCausalLM (tt/spec_sample.py VerifySampler) on controlled logits.

The sampler runs as the model runs it: a trace of its own over a persistent logits buffer that holds each device's
vocabulary shard [1, 1, rows, V / TP] bfloat16 (V = 248320), warmed eagerly and captured with program cache misses
disallowed, replayed with a fresh ``sampling_plan`` per verify. Each sampled row's draws are checked against vLLM's
sampling distribution (sampling_reference.target_probs, chi-square), on every device of the mesh; then the plugin's id
walk over the draws (accept_greedy_drafts) against the target: the first committed token and the acceptance of the
first draft. Greedy, top-k 1 and padding rows draw nothing; a seeded row draws the same token for the same position,
also on a chained verify, whose seeds the chain program writes (spec_chain.ChainInputs).

The sampler takes, on each device, the stable top-32 of its shard (padded to 32768 columns at TP = 8) and all-gathers
the candidates in shard order. On logits full of exact ties, every device holds each shard's top-32 in value order,
equal values lowest id first (the stable torch sort of test_spec_sample_topk_host.py, bit for bit), and the sampler
draws, for every seed, the token that ttnn.sampling draws from the first candidates of a stable torch sort of the whole
row, at the full width and at the row counts of the verify row buckets (the shard of the bucket's first b*T rows).

    pytest models/demos/blackhole/qwen36/tests/test_spec_sample.py    (Blackhole, up to 2 chips; MESH_DEVICE=P150x8)
"""

import random
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p, target_probs
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tests.test_spec_sample_topk_host import _logits as _tie_logits
from models.demos.blackhole.qwen36.tests.test_spec_sample_topk_host import reference_candidates, reference_order
from models.demos.blackhole.qwen36.tt.spec_sample import PAD_LOGIT, VerifySampler, chained_sampling_plan, sampling_plan

VOCAB = 248320
MESH = _resolve_mesh_shape(2)
P_MIN = 1e-4
# Support tokens of a row: spread over the vocabulary shards, the ends included.
SUPPORT = (5, 62079, 62080, 130001, 200003, VOCAB - 1)
SUPPORT_LOGITS = (2.0, 1.4, 0.9, 0.5, 0.1, -0.4)
FLOOR = -60.0  # every other token: no mass at any temperature used here


def _parametrize(fn):
    # The sampler all-gathers the shards' candidates.
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _shards(mesh):
    return mesh.get_num_devices()


def _mapper(mesh):
    """Each device's vocabulary shard of a [1, 1, rows, V] tensor, as the vocab-sharded LM head lays it out."""
    return ttnn.ShardTensorToMesh(mesh, dim=3) if _shards(mesh) > 1 else ttnn.ReplicateTensorToMesh(mesh)


def _logits_buffer(mesh, rows):
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=_mapper(mesh),
    )


def _stage_logits(mesh, logits, buf):
    """Host -> ``buf`` copy of ``logits`` [rows, V], each device its shard."""
    rows = int(buf.shape[2])
    h = ttnn.from_torch(
        logits[:rows].reshape(1, 1, rows, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_mapper(mesh),
    )
    ttnn.copy_host_to_device_tensor(h, buf)


def _capture(mesh, fn):
    """Trace of ``fn``, captured with program cache misses disallowed (its programs compiled before)."""
    mesh.set_program_cache_misses_allowed(False)
    try:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
    finally:
        mesh.set_program_cache_misses_allowed(True)
    ttnn.synchronize_device(mesh)
    return trace


def _draws(t, rows):
    """[devices, rows] draws of the [1, 1, 1, >= rows] ``t``."""
    return torch.stack([ttnn.to_torch(d).reshape(-1)[:rows].to(torch.int64) for d in ttnn.get_device_tensors(t)])


class _Sampler:
    """VerifySampler over a persistent buffer of each device's logits shard, warmed, then captured as the model captures
    it. With ``capture=False`` the caller captures once every device buffer that the replays run next to exists: a
    buffer allocated after the capture and alive at a replay can sit where the trace writes."""

    def __init__(self, mesh, B, T, capture=True):
        self.mesh, self.B, self.T = mesh, B, T
        self.logits = _logits_buffer(mesh, B * T)
        self.sampler = VerifySampler(mesh, B, T, VOCAB, shards=_shards(mesh))
        self.trace = None
        if capture:
            self.capture()

    def capture(self):
        self.sampler.body(self.logits)
        ttnn.synchronize_device(self.mesh)
        self.trace = _capture(self.mesh, lambda: self.sampler.body(self.logits))

    def stage_logits(self, logits):
        _stage_logits(self.mesh, logits, self.logits)

    def draw(self, plan, trace=None):
        """[devices, B*T] draws of one replay of ``trace`` (default the full-width one)."""
        self.sampler.stage(plan)
        ttnn.execute_trace(self.mesh, self.trace if trace is None else trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        return _draws(self.sampler.draws, self.B * self.T)

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
def test_verify_sampler_device_time(mesh_device):
    """Replay time of the sampler of B = 8 users x T = 4 over the logits of the full width (32 rows) and of the verify
    row buckets' 4, 8 and 16 rows, as the model's sampler traces run."""
    B, T, N = 8, 4, 200
    s = _Sampler(mesh_device, B, T, capture=False)
    buffers = {rows: _logits_buffer(mesh_device, rows) for rows in (4, 8, 16)}
    buffers[B * T] = s.logits
    for buf in buffers.values():
        s.sampler.body(buf)
    ttnn.synchronize_device(mesh_device)
    traces = {rows: _capture(mesh_device, lambda buf=buf: s.sampler.body(buf)) for rows, buf in buffers.items()}
    plan = sampling_plan(_params(B), [100] * (B - 1) + [-1], T)
    try:
        s.sampler.stage(plan)
        for rows, trace in sorted(traces.items()):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(N):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            us = (time.perf_counter() - t0) / N * 1e6
            logger.info(
                f"verify sampler, B={B} T={T} V={VOCAB} TP={_shards(mesh_device)}: {rows} rows {us:.1f} us per replay"
            )
        t0 = time.perf_counter()
        for _ in range(N):
            s.sampler.stage(plan)
        logger.info(f"verify sampler: staging a plan {(time.perf_counter() - t0) / N * 1e6:.1f} us")
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh_device, trace)
        for rows, buf in buffers.items():
            if rows != B * T:
                ttnn.deallocate(buf)


TIE_KINDS = ("coarse", "plateau", "bf16", "zeros", "flat")


def _device_rows(t, rows):
    """Per device, the first ``rows`` rows of ``t`` as torch."""
    return [ttnn.to_torch(d).reshape(-1, int(t.shape[-1]))[:rows] for d in ttnn.get_device_tensors(t)]


def _bits(values):
    return values.to(torch.bfloat16).view(torch.int16)


@_parametrize
@pytest.mark.parametrize("B, T, rows", [(8, 4, 32), (8, 4, 16), (8, 4, 8), (8, 4, 4), (2, 12, 24), (4, 4, 16)])
def test_every_device_holds_each_shards_top_32_lowest_id_first(mesh_device, B, T, rows):
    """Eager, on logits with thousands of equal maxima per shard (coarse), equal values on every shard boundary and
    across the 32nd place (plateau), random bfloat16 logits (bf16), +0.0 / -0.0 maxima (zeros) and one value everywhere
    (flat): on every device the candidates' values (bit patterns, -0.0 returned as +0.0) and global ids equal the
    stable torch sort's top-32 of each shard, position for position, for the logits of ``rows`` rows (fewer than B*T:
    a verify row bucket's); the rows past them hold PAD_LOGIT only (in bfloat16). Logits that every device holds whole
    give the same candidates."""
    shards = _shards(mesh_device)
    R = B * T
    sampler = VerifySampler(mesh_device, B, T, VOCAB, shards=shards)
    buf = _logits_buffer(mesh_device, rows)
    whole = ttnn.from_torch(
        torch.zeros(1, 1, rows, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )  # fmt: skip
    try:
        for i, kind in enumerate(TIE_KINDS):
            logits = _tie_logits(kind, rows, VOCAB, shards, seed=100 + i + rows)
            _stage_logits(mesh_device, logits, buf)
            h = ttnn.from_torch(
                logits.reshape(1, 1, rows, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )  # fmt: skip
            ttnn.copy_host_to_device_tensor(h, whole)
            want_v, want_i = reference_candidates(logits, shards)
            for src, name in ((buf, "shards"), (whole, "whole logits")):
                vals, ids = sampler.candidates(src)
                for d, (v, g) in enumerate(zip(_device_rows(vals, R), _device_rows(ids, R))):
                    where = f"{kind}, {name}, device {d}"
                    assert torch.equal(v[:rows].view(torch.int16), _bits(want_v)), f"{where}: values differ"
                    assert torch.equal(g[:rows].to(torch.int64), want_i), f"{where}: global ids differ"
                    # PAD_LOGIT as the device's pad converts it to bfloat16 (rounded or truncated).
                    pad = v[rows:].float().unique()
                    assert bool(((pad / PAD_LOGIT - 1).abs() < 1e-2).all()), f"{where}: rows past the logits {pad[:4]}"
                ttnn.deallocate(vals)
                ttnn.deallocate(ids)
    finally:
        ttnn.deallocate(buf)
        ttnn.deallocate(whole)


@_parametrize
def test_the_sampler_draws_what_a_stable_sort_of_the_whole_row_draws(mesh_device):
    """The sampler of B = 8 users x T = 4, captured over the full-width logits and over the logits of the verify row
    buckets' 4, 8 and 16 rows, as the model captures its sampler traces. The reference is ttnn.sampling over the first
    256 candidates of a stable torch sort of each whole row (value descending, equal values lowest id first), with the
    same seeds and per-row parameters. For each tie-heavy logits kind and 20 plans per row count (seeded and unseeded
    sampled rows at top-k 32, 20 with top-p 0.95, 5, and a greedy row), every sampled row of every replay draws the
    reference's token, on every device, and the bucket traces draw the full-width trace's tokens."""
    B, T = 8, 4
    R = B * T
    shards = _shards(mesh_device)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    s = _Sampler(mesh_device, B, T, capture=False)
    buffers = {rows: _logits_buffer(mesh_device, rows) for rows in (4, 8, 16)}
    buffers[R] = s.logits
    width = s.sampler.candidates_per_row
    ref_v = ttnn.from_torch(
        torch.zeros(1, 1, R, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    ref_i = ttnn.from_torch(
        torch.zeros(1, 1, R, width, dtype=torch.int32), dtype=ttnn.uint32, device=mesh_device, mesh_mapper=rep
    )
    ref_d = ttnn.from_torch(
        torch.zeros(1, 1, 1, R, dtype=torch.int32), dtype=ttnn.uint32, device=mesh_device, mesh_mapper=rep
    )

    def reference():
        ttnn.manual_seed(seeds=s.sampler.seeds, user_ids=s.sampler.user_ids)
        ttnn.sampling(ref_v, ref_i, k=s.sampler.k, p=s.sampler.p, temp=s.sampler.temp, output_tensor=ref_d)

    for buf in buffers.values():
        s.sampler.body(buf)
    reference()
    ttnn.synchronize_device(mesh_device)
    traces = {rows: _capture(mesh_device, lambda buf=buf: s.sampler.body(buf)) for rows, buf in buffers.items()}
    rng = random.Random(9)
    users = [(1.0, 32, 1.0, 700), (0.8, 20, 0.95, None), (0.0, 1, 1.0, None), (1.2, 5, 1.0, 702)]
    params = SimpleNamespace(
        **{k: [users[u % 4][i] for u in range(B)] for i, k in enumerate(("temperature", "top_k", "top_p", "seed"))}
    )
    compared = 0
    try:
        for i, kind in enumerate(TIE_KINDS):
            logits = _tie_logits(kind, R, VOCAB, shards, seed=300 + i)
            for buf in buffers.values():
                _stage_logits(mesh_device, logits, buf)
            top_v, top_i = reference_order(logits)
            h = ttnn.from_torch(top_v[:, :width].reshape(1, 1, R, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=rep)  # fmt: skip
            ttnn.copy_host_to_device_tensor(h, ref_v)
            h = ttnn.from_torch(top_i[:, :width].reshape(1, 1, R, width).to(torch.int32), dtype=ttnn.uint32,
                                mesh_mapper=rep)  # fmt: skip
            ttnn.copy_host_to_device_tensor(h, ref_i)
            for b in (1, 2, 4, 8):
                rows = b * T
                for step in range(20):
                    positions = [60 + 3 * u + step if u < b else -1 for u in range(B)]
                    plan = sampling_plan(params, positions, T, lambda: rng.randrange(1, 10**6))
                    sampled = torch.tensor(plan.sampled[:rows]).nonzero().reshape(-1)
                    got = s.draw(plan, traces[rows])
                    full = s.draw(plan, traces[R]) if rows < R else got
                    reference()
                    ttnn.synchronize_device(mesh_device)
                    want = _draws(ref_d, R)
                    where = f"{kind}, {rows} rows, step {step}"
                    assert torch.equal(got[:, sampled], want[:, sampled]), f"{where}: draws differ from the reference"
                    assert torch.equal(full[:, sampled], got[:, sampled]), f"{where}: bucket and full width differ"
                    assert bool((got == got[0]).all()), "devices of the mesh drew different tokens"
                    compared += len(sampled)
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh_device, trace)
        for t in (ref_v, ref_i, ref_d, *(buf for rows, buf in buffers.items() if rows != R)):
            ttnn.deallocate(t)
    assert compared == len(TIE_KINDS) * 20 * (1 + 2 + 3 + 6) * T
