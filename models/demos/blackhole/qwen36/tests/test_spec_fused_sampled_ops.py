# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE-chip device test of FusedSampledDevice (tt/spec_fused_sampled.py): the fused sampled accept + drafter-input op
sequence equals fused_sampled_reference, eagerly (>= 300 random cases) and in a trace (>= 50 re-staged replays), plus the
replay time. No model, no weights.

Equality rules (device fp32 exp/divide differ from torch by ~1e-7):
  * overflow flags equal; committed / counts equal for non-overflow users, except BOUNDARY users (a draft compare
    |p_d - u| < 1e-5, an inverse-CDF edge or a top-p tail edge within 1e-5 on an examined row) whose result may flip: they
    are counted and printed, any other difference fails;
  * pack / rs_pt bit-equal to fused_drafter_reference computed from the DEVICE's committed / counts.
Run (needs the chip): TT_VISIBLE_DEVICES=0 pytest -o addopts="" --timeout=3600 --import-mode=importlib -sv \
  models/demos/blackhole/qwen36/tests/test_spec_fused_sampled_ops.py"""

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fused_sampled_host import NB, PER_SHARD, TP, K, M, T, _case
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    _ONE_MINUS,
    FusedSampledDevice,
    fused_drafter_reference,
    fused_sampled_reference,
)

pytestmark = pytest.mark.skipif(os.environ.get("TT_VISIBLE_DEVICES") != "0", reason="needs TT_VISIBLE_DEVICES=0")

W = 32
EPS = 1e-5
N_EAGER, N_TRACE = 100, 20  # per B in (1, 4, 8): 300 eager, 60 trace cases


@pytest.fixture(scope="module")
def mesh():
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield m
    ttnn.close_mesh_device(m)


def _make(g, B):
    c = _case(g, B)
    uni = torch.rand(B, K + 2, generator=g).clamp(max=_ONE_MINUS)
    ids = c["mi"].reshape(B, T, M)[:, :, 0].reshape(-1)
    ref = fused_sampled_reference(
        c["uv"], c["ul"], c["tok_buf"], ids, c["aux"], c["pt"], B=B, K=K, per_shard=PER_SHARD, M=M,
        greedy=c["greedy"], temperature=c["temp"], top_k=c["top_k"], top_p=c["top_p"], uniforms=uni,
    )  # fmt: skip
    return c, uni, ids, ref


def _stage(fd, c, uni, ids):
    fd.stage_sampled(
        c["uv"], c["ul"], ids.to(torch.int32), c["tok_buf"].to(torch.int32), c["aux"], c["pt"].repeat_interleave(T, dim=0),
        c["temp"], c["top_k"], c["top_p"], c["greedy"], uni,
    )  # fmt: skip


def _boundary(c, ref, uni, u, m_cols):
    """True if a decision of user u is within EPS of its threshold on a row examined under either walk."""
    nv = int(c["nv"][u])
    prob, cum = ref.prob.double().reshape(-1, T, M)[u], ref.cum.double().reshape(-1, T, M)[u]
    gid = ref.cand_ids.reshape(-1, T, M)[u]
    mx = max(m_cols)
    for j in range(min(mx, nv - 1) + 1) if nv else []:
        if abs(float(ref.p_d[u, j]) - float(uni[u, 1 + j])) < EPS:
            return True
    for m in set(m_cols):
        corr = m < nv
        w = prob[m].clone()
        if corr:
            w[gid[m] == c["drafts"][u, m]] = 0
        cdf = w.cumsum(0) / w.sum()
        uu = float(uni[u, K + 1] if corr else uni[u, 0])
        if float((cdf - uu).abs().min()) < EPS:
            return True
    if float(c["top_p"][u]) < 1.0:
        for m in range(mx + 1):
            x = cum[m][cum[m] > 0]
            if x.numel() and float((x - (1 - float(c["top_p"][u]))).abs().min()) < EPS:
                return True
    return False


def _check(fd, c, uni, ref, tag, stats):
    B, R = fd.B, fd.R
    u = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(fd.mesh, dim=0))[: t.shape[0]]
    res = u(fd.result_out).to(torch.int32)
    committed, counts, ovf = res[0, :R].reshape(B, T), res[1, :B], res[2, :B]
    for us in range(B):
        stats["users"] += 1
        greedy = bool(c["greedy"][us])
        bnd = False
        same = bool(torch.equal(committed[us], ref.committed[us])) and int(counts[us]) == int(ref.counts[us])
        if not greedy and (int(ovf[us]) != int(ref.overflow[us]) or not same):
            bnd = _boundary(c, ref, uni, us, [int(ref.counts[us]) - 1, int(counts[us]) - 1])
            stats["boundary"] += int(bnd)
        if int(ovf[us]) != int(ref.overflow[us]):
            assert bnd and int(counts[us]) != int(
                ref.counts[us]
            ), f"{tag} user {us}: overflow flag {ovf[us]} != {ref.overflow[us]}"
            continue
        stats["ovf"] += int(ref.overflow[us])
        if int(ref.overflow[us]):
            continue
        if not same:
            stats["diff"] += 1
            assert (
                bnd
            ), f"{tag} user {us}: committed {committed[us].tolist()}/{int(counts[us])} != ref {ref.committed[us].tolist()}/{int(ref.counts[us])}"
    # the drafter fields from the DEVICE's committed / counts
    pack, rs_pt = fused_drafter_reference(committed, counts, c["aux"], c["pt"], B=B, K=K)[:2]
    assert torch.equal(u(fd.pack_out).to(torch.int64) & 0xFFFFFFFF, pack.to(torch.int64) & 0xFFFFFFFF), f"{tag}: pack"
    assert torch.equal(u(fd.rs_pt_out).to(torch.int32), rs_pt), f"{tag}: rs_pt"


def test_fused_sampled_ops_eager_and_trace(mesh):
    stats = dict(users=0, boundary=0, diff=0, ovf=0)
    for B in (1, 4, 8):
        _run(mesh, B, stats)
    print(f"RESULT totals {stats}")


def _run(mesh, B, stats):
    g = torch.Generator().manual_seed(20260501 + B)
    fd = FusedSampledDevice.allocate(mesh, B, K, NB, W, TP, M, PER_SHARD)
    for i in range(N_EAGER):  # the first call also compiles every program
        c, uni, ids, ref = _make(g, B)
        _stage(fd, c, uni, ids)
        fd.body()
        _check(fd, c, uni, ref, f"B={B} eager {i}", stats)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fd.body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for i in range(N_TRACE):
        c, uni, ids, ref = _make(g, B)
        _stage(fd, c, uni, ids)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        _check(fd, c, uni, ref, f"B={B} trace {i}", stats)
    times = []
    for _ in range(30):  # timing: the replay alone (inputs already staged)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        times.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    print(f"RESULT B={B} eager {N_EAGER} trace {N_TRACE} replay_median_us={statistics.median(times):.0f} stats={stats}")
