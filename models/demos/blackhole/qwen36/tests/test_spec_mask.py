# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the token mask of a deferred verify of Qwen36MTPContractForCausalLM (tt/spec_mask.py VerifyMask) on
controlled logits, ahead of the verify sampler (tt/spec_sample.py VerifySampler).

The mask runs as the model runs it: a trace of its own that edits a persistent logits buffer in place (each device's
vocabulary shard [1, 1, rows, V / TP] bfloat16, V = 248320), replayed before the sampler's trace over the same buffer;
both are warmed eagerly and captured with program cache misses disallowed, then replayed with a fresh mask and
``sampling_plan`` per step. Checked on every device of the mesh:

* the edited logits equal the host ``masked_logits`` (``where(allowed, x, MASK_NEG)`` in bfloat16) bit for bit, at the
  full width and at the row count of a verify row bucket, for dense, sparse, single-token, all-ones and tail masks;
* a greedy masked user draws the masked argmax (lowest id among ties), also where the mask rules out the argmax;
* a sampled masked user draws only allowed tokens, distributed as vLLM's sampler over the masked logits (chi-square).

    pytest models/demos/blackhole/qwen36/tests/test_spec_mask.py    (Blackhole, up to 2 chips; MESH_DEVICE=P150x8)
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p, target_probs
from models.demos.blackhole.qwen36.tests.test_spec_sample import (
    MESH,
    VOCAB,
    _capture,
    _draws,
    _logits_buffer,
    _shards,
    _stage_logits,
)
from models.demos.blackhole.qwen36.tt.spec_mask import MASK_NEG, VerifyMask, allowed_tokens, masked_logits
from models.demos.blackhole.qwen36.tt.spec_sample import VerifySampler, sampling_plan

P_MIN = 1e-4
WORDS = VOCAB // 32


def _parametrize(fn):
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _pack(allowed):
    """Plugin packing int32 [R, V / 32] of a bool [R, V]."""
    R = allowed.shape[0]
    bits = allowed.reshape(R, WORDS, 32).to(torch.int64) << torch.arange(32)
    packed = bits.sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def _case(kind, r, g):
    """bool [V]: the allowed tokens of mask case ``kind`` for row r."""
    a = torch.zeros(VOCAB, dtype=torch.bool)
    if kind == "dense":
        return torch.rand(VOCAB, generator=g) < 0.5
    if kind == "ones":
        return ~a
    if kind == "sparse":
        a[torch.randint(0, VOCAB, (2 + r % 7,), generator=g)] = True
    elif kind == "per_shard":  # one token in every quarter of the vocabulary, the ends included
        a[torch.tensor([0, VOCAB // 4 - 1, VOCAB // 2, VOCAB - 1])] = True
    elif kind == "tail":  # the last ids of the vocabulary only
        a[VOCAB - 1 - torch.randperm(40, generator=g)[: 1 + r % 5]] = True
    return a


def _read_logits(mesh, buf, rows):
    """[rows, V] bfloat16 of the sharded ``buf``: every device's shard side by side."""
    parts = [ttnn.to_torch(t).reshape(-1, int(t.shape[-1]))[:rows] for t in ttnn.get_device_tensors(buf)]
    return torch.cat(parts, dim=1) if _shards(mesh) > 1 else parts[0]


class _MaskedSampler:
    """VerifyMask and VerifySampler over one persistent logits buffer of B*T rows (and one of the first ``bucket``
    users' rows), warmed, then each captured as the model captures them: the mask trace, then the sampler trace."""

    def __init__(self, mesh, B, T, bucket=None):
        self.mesh, self.B, self.T, self.rows = mesh, B, T, B * T
        self.logits = _logits_buffer(mesh, self.rows)
        self.bucket = None if bucket is None else _logits_buffer(mesh, bucket * T)
        self.sampler = VerifySampler(mesh, B, T, VOCAB, shards=_shards(mesh))
        self.mask = VerifyMask(mesh, self.rows, VOCAB, shards=_shards(mesh))
        self.traces = {}
        for name, buf in (("full", self.logits), ("bucket", self.bucket)):
            if buf is None:
                continue
            self.mask.apply_(buf)
            self.sampler.body(buf)
            ttnn.synchronize_device(mesh)
            self.traces[name] = (
                _capture(mesh, lambda b=buf: self.mask.apply_(b)),
                _capture(mesh, lambda b=buf: self.sampler.body(b)),
            )

    def step(self, logits, mask, plan=None, which="full"):
        """Stage ``logits`` [R, V] and ``mask`` (engine rows [<= B*T, V/32]), replay the mask trace (and with ``plan`` the
        sampler trace); returns the edited logits [R, V] and the draws [devices, B*T] (None without ``plan``)."""
        buf = self.logits if which == "full" else self.bucket
        rows = int(buf.shape[2])
        _stage_logits(self.mesh, logits, buf)
        self.mask.stage(mask)
        mask_trace, sample_trace = self.traces[which]
        ttnn.execute_trace(self.mesh, mask_trace, cq_id=0, blocking=False)
        draws = None
        if plan is not None:
            self.sampler.stage(plan)
            ttnn.execute_trace(self.mesh, sample_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        if plan is not None:
            draws = _draws(self.sampler.draws, self.rows)
        return _read_logits(self.mesh, buf, rows), draws

    def draw(self, plan):
        """Replay the sampler trace alone over the logits the last ``step`` masked (the sampler reads them, never
        writes them); returns the draws [devices, B*T]."""
        self.sampler.stage(plan)
        ttnn.execute_trace(self.mesh, self.traces["full"][1], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        return _draws(self.sampler.draws, self.rows)

    def release(self):
        for pair in self.traces.values():
            for trace in pair:
                ttnn.release_trace(self.mesh, trace)
        self.mask.release()


@_parametrize
def test_the_mask_trace_edits_the_logits_bit_for_bit(mesh_device):
    B, T, bucket = 8, 4, 2
    g = torch.Generator().manual_seed(0)
    s = _MaskedSampler(mesh_device, B, T, bucket=bucket)
    try:
        for which, rows in (("full", B * T), ("bucket", bucket * T)):
            for kind in ("dense", "sparse", "per_shard", "tail", "ones"):
                logits = torch.randn(rows, VOCAB, generator=g) * 3
                allowed = torch.stack([_case(kind, r, g) for r in range(rows)])
                mask = _pack(allowed)
                got, _ = s.step(logits, mask, which=which)
                want = masked_logits(logits, mask, VOCAB)
                bad = (got.float() != want.float()).sum().item()
                assert bad == 0, f"{which} {kind}: {bad} logits differ from the host where()"
                assert bool((got.float()[~allowed] == MASK_NEG).all())
    finally:
        s.release()


@_parametrize
def test_a_greedy_masked_user_draws_the_masked_argmax(mesh_device):
    """Every column of a masked greedy user draws with top-k 1: the masked argmax, lowest id among equal logits, also
    where the mask rules out the unmasked argmax; an unmasked greedy user draws nothing."""
    B, T = 8, 4
    g = torch.Generator().manual_seed(1)
    s = _MaskedSampler(mesh_device, B, T)
    params = SimpleNamespace(temperature=[0.0] * B, top_k=[1] * B, top_p=[1.0] * B, seed=[None] * B)
    try:
        for step in range(6):
            logits = torch.round(torch.randn(B * T, VOCAB, generator=g) * 2) / 2  # many exact ties
            kinds = ("dense", "sparse", "per_shard", "tail")
            allowed = torch.stack([_case(kinds[(step + r) % 4], r, g) for r in range(B * T)])
            allowed[torch.arange(B * T), logits.argmax(dim=1)] = False  # the unmasked argmax is never allowed
            allowed[torch.arange(B * T), torch.randint(0, VOCAB, (B * T,), generator=g)] = True
            masked = [u % 2 == 0 for u in range(B)]
            for u in range(B):
                if not masked[u]:
                    allowed[u * T : (u + 1) * T] = True
            plan = sampling_plan(params, [100 + u for u in range(B)], T, masked=masked)
            assert plan.sampled == [m for m in masked for _ in range(T)]
            _, draws = s.step(logits, _pack(allowed), plan)
            assert bool((draws == draws[0]).all()), "devices of the mesh drew different tokens"
            want = masked_logits(logits, _pack(allowed), VOCAB).float().argmax(dim=1)
            rows = torch.tensor(plan.sampled)
            assert torch.equal(draws[0][rows], want[rows]), f"step {step}: draws {draws[0][rows]} vs {want[rows]}"
    finally:
        s.release()


@_parametrize
def test_masked_draws_follow_the_masked_target(mesh_device):
    B, T, N = 8, 4, 1500
    g = torch.Generator().manual_seed(2)
    s = _MaskedSampler(mesh_device, B, T)
    users = {0: (1.0, 20, 1.0), 1: (0.7, 32, 1.0), 2: (1.0, 8, 1.0)}
    params = SimpleNamespace(
        temperature=[users.get(u, (0.0,))[0] for u in range(B)],
        top_k=[users.get(u, (0, 1))[1] for u in range(B)],
        top_p=[1.0] * B,
        seed=[None] * B,
    )
    logits = torch.full((B * T, VOCAB), -60.0)
    support = torch.tensor([5, 62079, 62080, 130001, 200003, VOCAB - 1, 777, 150000])
    logits[:, support] = torch.linspace(2.0, -0.4, len(support))
    allowed = torch.ones(B * T, VOCAB, dtype=torch.bool)
    allowed[:, support[torch.tensor([0, 3, 6])]] = False  # the mode and two more support tokens are ruled out
    allowed[:T, support[1]] = False
    mask = _pack(allowed)
    masked = [True] * 3 + [False] * (B - 3)
    try:
        draws = [s.step(logits, mask, sampling_plan(params, [100] * B, T, masked=masked))[1][0]]
        for i in range(1, N):
            d = s.draw(sampling_plan(params, [100 + i] * B, T, masked=masked))
            assert bool((d == d[0]).all()), "devices of the mesh drew different tokens"
            draws.append(d[0])
        draws = torch.stack(draws)
        for u, (temp, k, p) in users.items():
            for j in range(T):
                r = u * T + j
                assert bool(allowed[r, draws[:, r]].all()), f"row {r} drew a masked token"
                x = torch.where(allowed_tokens(mask[r], VOCAB), logits[r], torch.tensor(float("-inf")))
                pv = chi_square_p(torch.bincount(draws[:, r], minlength=VOCAB), target_probs(x, temp, k, p))
                assert pv > P_MIN, f"row {r}: masked draws do not follow the masked target (p = {pv})"
    finally:
        s.release()
