# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: one deferred verify whose rows hold every kind of device sampling Qwen36MTPContractForCausalLM serves
(token mask, penalties, logprobs, top-p certificate, plain greedy, plain sampled), on controlled logits.

The bodies run as the model runs a token-masked verify at sample_deferred (Qwen36Model.verify_enqueue_post), each a
trace of its own over a persistent buffer holding each device's vocabulary shard [1, 1, B*T, V / TP] bfloat16
(V = 248320), warmed eagerly and captured with program cache misses disallowed:

  verify stand-in (restore the raw logits, write the verify ids lanes with the fused per-shard argmax)
  -> token mask (spec_mask.VerifyMask) -> penalties (spec_penalties.VerifyPenalties) -> certificate sampler
  (spec_top_p.TopPCertificate, which replaces the sampler trace on a verify with a certificate row)
  -> logprobs (spec_logprobs.VerifyLogprobs).

B = 8 users x T = 4 columns; per user (vLLM's order: the mask, then the penalties, then temperature and top-k/top-p):

  0 plain greedy: the verify ids are the argmax of the raw logits, bit for bit;
  1 plain sampled (top-k 20): chi-square against vLLM's top-k distribution;
  2 masked greedy: it draws with top-k 1, the masked argmax, bit for bit;
  3 masked greedy with a repetition penalty: it does not draw; its ids lanes are the float32 argmax of the masked
    penalized logits, bit for bit, and its allowed logits are the bfloat16 rounding of vLLM's penalties;
  4 sampled (top-k 20) with frequency and presence penalties: chi-square against the bfloat16 penalized target;
  5 top-p certificate (top-k off, top_p 0.9) with repetition and presence penalties: chi-square against vLLM's nucleus
    of the bfloat16 penalized logits;
  6 masked top-p certificate: chi-square against the nucleus of the masked logits;
  7 sampled (top-k 20) with logprobs: chi-square, and its payload (raw logprob, rank and top-20 of each drawn token)
    against float64 log_softmax of the raw logits within 1e-4, the tolerance test_spec_logprobs uses.

Every live row's answered token is among its logprobs candidates (the payload raises otherwise). Every row is checked
before the test fails; a row with chi-square p below 0.01 logs its most likely tokens.

    pytest models/demos/blackhole/qwen36/tests/test_spec_combined.py    (Blackhole; MESH_DEVICE=P150x8)
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, FusedShardArgmax
from models.demos.blackhole.qwen36.tt.spec_logprobs import TOP_LOGPROBS, VerifyLogprobs, payload_from_reads
from models.demos.blackhole.qwen36.tt.spec_mask import MASK_NEG, WORD_BITS, VerifyMask, mask_words, normalise_rows
from models.demos.blackhole.qwen36.tt.spec_penalties import StepPenalties, VerifyPenalties, reference_penalties
from models.demos.blackhole.qwen36.tt.spec_sample import VerifySampler, sampling_plan, with_draws
from models.demos.blackhole.qwen36.tt.spec_step import VERIFY_STEP_ROWS, VERIFY_TOKENS, pack_width
from models.demos.blackhole.qwen36.tt.spec_top_p import (
    CERT_EPS,
    TopPCertificate,
    certificate_rows,
    failed_rows,
    row_distribution,
    sampler_params,
)

VOCAB = 248320
MESH = _resolve_mesh_shape(8)
P_MIN = 1e-4
LOGPROBS_TOL = 1e-4
N = 2000
PAYLOAD_REPLAYS = 64  # replays whose logprobs payload is read and checked
B, T = 8, 4
R = B * T
# user -> (temperature, top_k, top_p, (repetition, frequency, presence) or None, masked); top_k VOCAB is off.
USERS = {
    0: (0.0, 1, 1.0, None, False),
    1: (1.0, 20, 1.0, None, False),
    2: (0.0, 1, 1.0, None, True),
    3: (0.0, 1, 1.0, (1.3, 0.0, 0.0), True),
    4: (1.0, 20, 1.0, (1.0, 0.4, 0.3), False),
    5: (1.0, VOCAB, 0.9, (1.2, 0.0, 0.5), False),
    6: (1.0, VOCAB, 0.9, None, True),
    7: (1.0, 20, 1.0, None, False),
}
LOGPROBS_USER = 7
GREEDY_PENALTY_USERS = [u for u, c in USERS.items() if c[3] is not None and c[0] <= 0.0]


def _parametrize(fn):
    # The fused argmax all-gathers each shard's maximum.
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _shards(mesh):
    return mesh.get_num_devices()


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=3) if _shards(mesh) > 1 else ttnn.ReplicateTensorToMesh(mesh)


def _replicated(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )  # fmt: skip


def _device0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _capture(mesh, fn):
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


def _pack(allowed):
    """Plugin packing int32 [R, ceil(V/32)] of a bool [R, V]."""
    rows, vocab = allowed.shape
    words = mask_words(vocab)
    bits = torch.zeros(rows, words * WORD_BITS, dtype=torch.int64)
    bits[:, :vocab] = allowed.to(torch.int64)
    packed = (bits.reshape(rows, words, WORD_BITS) << torch.arange(WORD_BITS)).sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


class _Case:
    """The verify's controlled inputs: raw logits [R, V] (12 peaked tokens, 30 down to 25, over N(0, 1) per row), the
    block's tokens (drafts among the peaked tokens), each penalty user's prompt and output (holding the tokens its
    penalties must move), each masked row's allowed tokens, and the per-row references."""

    def __init__(self, g):
        self.peaks = [torch.randperm(VOCAB, generator=g)[:12] for _ in range(B)]
        x = torch.randn(R, VOCAB, generator=g)
        for r in range(R):
            x[r, self.peaks[r // T]] += torch.linspace(30.0, 25.0, 12)
        self.raw = x.to(torch.bfloat16).float()
        self.tokens = torch.zeros(B, T, dtype=torch.int64)
        for u in range(B):
            self.tokens[u] = self.peaks[u][[11, 2, 5, 8]]  # column 0 the last commit, columns 1..3 drafts
        p = self.peaks
        self.prompts = {3: p[3][[4, 9]], 4: p[4][[7]], 5: p[5][[0, 1]]}
        self.outputs = {3: torch.cat([p[3][[0, 0, 6]], p[3][[11]]]), 4: p[4][[0, 0, 0, 1, 3, 11]], 5: p[5][[3, 11]]}
        self.allowed = torch.ones(R, VOCAB, dtype=torch.bool)
        for u, keep in ((2, [3, 4, 7, 10]), (3, [0, 1, 2, 6, 8]), (6, [1, 2, 4, 5, 7, 9, 10, 11])):
            for j in range(T):
                row = torch.zeros(VOCAB, dtype=torch.bool)
                row[p[u][keep]] = True
                row[torch.randperm(VOCAB, generator=g)[:200]] = True
                row[p[u][keep[j % len(keep)]]] = j % 2 == 0  # each column a mask of its own
                self.allowed[u * T + j] = row

    def history(self, u, j):
        return torch.cat([self.outputs[u], self.tokens[u, 1 : j + 1]])

    def target(self, r):
        """float32 [V]: vLLM's logits of row r after the mask (-inf) and the penalties; the second value is the bfloat16
        rounding the device samples from."""
        u, j = divmod(r, T)
        x = torch.where(self.allowed[r], self.raw[r], torch.tensor(float("-inf")))
        pen = USERS[u][3]
        if pen is not None:
            x = reference_penalties(x, self.prompts[u], self.history(u, j), *pen)
        return x, x.to(torch.bfloat16).to(torch.float32)

    def mask_rows(self):
        masked = torch.tensor([USERS[r // T][4] for r in range(R)])
        rows = torch.full((R, mask_words(VOCAB)), -1, dtype=torch.int32)
        rows[masked] = _pack(self.allowed[masked])
        return normalise_rows(rows, VOCAB)


def _params():
    return SimpleNamespace(
        temperature=[USERS[u][0] for u in range(B)],
        top_k=[USERS[u][1] for u in range(B)],
        top_p=[USERS[u][2] for u in range(B)],
        seed=[None] * B,
    )


def _penalties():
    rep, freq, pres = torch.ones(B), torch.zeros(B), torch.zeros(B)
    for u, c in USERS.items():
        if c[3] is not None:
            rep[u], freq[u], pres[u] = c[3]
    return StepPenalties(rep, freq, pres, torch.tensor([USERS[u][3] is not None for u in range(B)]))


def _check_draws(r, counts, want, failures):
    """Pearson chi-square p of row ``r``'s draw ``counts`` [V] against ``want`` [V]; below P_MIN the row fails, below
    0.01 its 8 most likely tokens are logged."""
    pv = chi_square_p(counts, want)
    if pv < 0.01:
        n = float(counts.sum())
        top = torch.argsort(want, descending=True, stable=True)[:8].tolist()
        logger.warning(
            f"row {r}: chi-square p {pv:.3g}; "
            + "; ".join(
                f"{t}: p {float(want[t]):.4f} expected {n * float(want[t]):.1f} observed {int(counts[t])}" for t in top
            )
        )
    if pv <= P_MIN:
        failures.append((r, pv))
    return pv


def _dense(x, temperature, top_k, top_p):
    ids, prob = row_distribution(x, temperature, 0 if top_k >= VOCAB else top_k, top_p)
    return torch.zeros(VOCAB, dtype=torch.float64).index_put((ids,), prob)


@_parametrize
def test_one_verify_with_every_row_kind_draws_each_row_from_its_own_target(mesh_device):
    S = _shards(mesh_device)
    case = _Case(torch.Generator().manual_seed(2027))
    raw_buf = ttnn.from_torch(
        case.raw.reshape(1, 1, R, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper=_mapper(mesh_device),
    )  # fmt: skip
    shard = ttnn.from_torch(
        torch.zeros(1, 1, R, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper=_mapper(mesh_device),
    )  # fmt: skip
    ids_buf = _replicated(mesh_device, torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
    step = torch.zeros(VERIFY_STEP_ROWS, pack_width(R), dtype=torch.int32)
    step[VERIFY_TOKENS, :R] = case.tokens.reshape(-1).to(torch.int32)
    step_buf = _replicated(mesh_device, step, ttnn.uint32)
    fused = FusedShardArgmax(mesh_device, VOCAB, S)
    sampler = VerifySampler(mesh_device, B, T, VOCAB, shards=S)
    mask = VerifyMask(mesh_device, R, VOCAB, shards=S)
    pen = VerifyPenalties(mesh_device, B, T, VOCAB, shards=S, argmax=lambda x: fused(x, LANES))
    pen.bind(step=step_buf, ids_out=ids_buf, draws=sampler.draws)
    cert_dev = TopPCertificate(sampler)
    lp = VerifyLogprobs(mesh_device, B, T, VOCAB, shards=S, source=sampler)

    def verify_stand_in():
        ttnn.copy(raw_buf, shard)
        ids = fused(shard, LANES)
        ttnn.copy(ids, ids_buf)
        ttnn.deallocate(ids)

    params, penalties = _params(), _penalties()
    cert = certificate_rows(params, B, VOCAB)
    assert cert.rows == [u in (5, 6) for u in range(B)]
    masked = [USERS[u][4] for u in range(B)]
    draw = [m and u not in GREEDY_PENALTY_USERS for u, m in enumerate(masked)]
    positions = [100 + 9 * u for u in range(B)]

    def plan_at(i):
        return sampling_plan(sampler_params(params, cert), [p + i for p in positions], T, masked=draw)

    plan = plan_at(0)
    assert [plan.sampled[u * T] for u in range(B)] == [False, True, True, False, True, True, True, True]
    pen.warm_eager()
    pen.rebuild({u: (case.prompts[u], case.outputs[u]) for u in case.prompts})
    greedy_lanes = [int(r // T in GREEDY_PENALTY_USERS) for r in range(R)]
    pen.stage(penalties, greedy_lanes)
    mask.stage(case.mask_rows())
    sampler.stage(plan)
    cert_dev.stage(cert, T)
    bodies = [verify_stand_in, lambda: mask.apply_(shard), lambda: pen.apply_body(shard)]
    bodies += [lambda: cert_dev.body(shard), lambda: lp.body(shard)]
    for body in bodies:  # compile
        body()
    ttnn.synchronize_device(mesh_device)
    traces = [_capture(mesh_device, body) for body in bodies]

    def replay(upto=len(traces)):
        for trace in traces[:upto]:
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

    try:
        # The logits the sampler reads: allowed logits of a penalty row are vLLM's penalties rounded once to bfloat16
        # (within one bfloat16 ulp, as test_spec_penalties checks), masked ones far below every allowed one, and the
        # allowed logits of a row without a penalty untouched.
        replay(upto=3)
        edited = ttnn.to_torch(shard, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).reshape(R, VOCAB)
        edited = edited.float()
        worst = 0.0
        for r in range(R):
            u = r // T
            want = case.target(r)[1]
            allowed = case.allowed[r]
            assert bool((edited[r, ~allowed] <= MASK_NEG / 2).all()), f"row {r}: a masked logit is not masked"
            if USERS[u][3] is None:
                assert torch.equal(edited[r, allowed], case.raw[r, allowed]), f"row {r}: an allowed logit changed"
                continue
            got, ref = edited[r, allowed], want[allowed]
            ulp = torch.pow(2.0, torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
            worst = max(worst, float(((got - ref).abs() / ulp).max()))
        logger.info(f"penalized allowed logits: worst {worst:.3f} bf16 ulp from vLLM's penalties")
        assert worst <= 1.0, f"a penalized logit is {worst} bf16 ulp off vLLM's penalties"

        draws, failures = [], []
        for i in range(N):
            plan = plan_at(i)
            sampler.stage(plan)
            replay()
            per_device = [
                ttnn.to_torch(d).reshape(-1)[:R].to(torch.int64) for d in ttnn.get_device_tensors(sampler.draws)
            ]
            assert all(torch.equal(d, per_device[0]) for d in per_device), "devices drew different tokens"
            lanes = _device0(ids_buf).reshape(-1)[:R].to(torch.int64)
            ids = with_draws(lanes.reshape(B, T), per_device[0], plan)
            draws.append(ids.reshape(-1))
            if i < PAYLOAD_REPLAYS:
                q = _device0(cert_dev.q).reshape(-1)[:R].float()
                assert not bool(failed_rows(q, cert, T).any()), f"a certificate failed: q {q.tolist()}"
                assert bool((q[[r for r in range(R) if cert.rows[r // T]]] >= 0.9 + CERT_EPS).all())
                reads = [_device0(t) for t in lp.outputs]
                lp_tok, rank, top_lp, top_ids = payload_from_reads(reads, ids, [True] * B, B, T)
                for j in range(T):
                    r = LOGPROBS_USER * T + j
                    ref = torch.log_softmax(case.raw[r].double(), dim=-1)
                    chosen = int(ids[LOGPROBS_USER, j])
                    assert abs(float(lp_tok[LOGPROBS_USER, j]) - float(ref[chosen])) < LOGPROBS_TOL
                    assert int(rank[LOGPROBS_USER, j]) == int((case.raw[r] >= case.raw[r, chosen]).sum())
                    order = torch.sort(case.raw[r], descending=True, stable=True).indices[:TOP_LOGPROBS]
                    assert top_ids[LOGPROBS_USER, j].tolist() == order.tolist()
                    assert torch.allclose(top_lp[LOGPROBS_USER, j].double(), ref[order], atol=LOGPROBS_TOL)
        draws = torch.stack(draws)

        for r in range(R):
            u = r // T
            temp, k, p, pen_c, _ = USERS[u]
            x32, x16 = case.target(r)
            if temp <= 0.0:
                want = int(x32.argmax())  # float32 argmax, lowest id among ties
                assert bool((draws[:, r] == want).all()), f"row {r}: greedy ids {draws[:3, r].tolist()}, want {want}"
                continue
            counts = torch.bincount(draws[:, r], minlength=VOCAB)
            pv = _check_draws(r, counts, _dense(x16, temp, k, p), failures)
            logger.info(f"user {u} column {r % T}: chi-square p {pv:.3g}")
        assert not failures, f"rows whose draws do not follow their target (row, p): {failures}"
    finally:
        for trace in traces:
            ttnn.release_trace(mesh_device, trace)
        for t in (raw_buf, shard, ids_buf, step_buf):
            ttnn.deallocate(t)
        mask.release()
        pen.release()
        cert_dev.release()
        lp.release()
