# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE-chip device test of QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED (tt/spec_fused_kernels.SampledAcceptKernel, the stage-S2 sampled
accept as one kernel). No model, no weights. Every case stages the same inputs FusedSampledDevice.stage_sampled stages and
poisons pack_out / rs_pt_out / result_out before the kernel runs; then:
  1. kernel == tt/spec_sampled_kernel_math.sampled_accept_twin, word for word, every page (padding included);
  2. kernel vs fused_sampled_reference (the S2 algorithm) and vs the op graph it replaces (FusedSampledDevice.body with the
     kernel detached): equal except float32 boundary users (an examined accept within 1e-6 of its uniform, an inverse-CDF
     edge, a top-p tail edge), which are counted and printed; overflow flags equal (or a boundary user); the drafter
     tail (pack, reseed table) equal to the S1 tail of each one's own commit;
  3. the all-zero warm-up inputs (the engine's eager compile): kernel == op graph, word for word;
  4. a trace of the kernel replayed with freshly staged inputs, every replay == the model;
  5. trace replay time, op graph vs kernel.
Cases: random union blocks with natural and injected bf16 ties (at the k-th value, through the 64th candidate = overflow,
in the tail), top_k 1..63, top_p 1 / 0.95 / 0.8 / 0.5, temperature 0.3..1.5, greedy / inactive users, nv 0..K, decline /
keep rows; B 1/4/8 at K = 3 (production) and B 1/2/4 at K = 7 (the adaptive-K wide family: 32 rows); nb 6 / 537 (32K
harness width) / 4097 (serving width).
Run on a chip: TT_VISIBLE_DEVICES=0 pytest -o addopts="" --timeout=3600 --import-mode=importlib -sv \
  models/demos/blackhole/qwen36/tests/test_spec_sampled_accept_kernel.py
Short device session: K2S_OPS_SHAPES=prod (B=8 K=3 at nb 537 / 4097 and B=4 K=7 at nb 4097). In ttsim-perf set
TT_METAL_SIMULATOR (the gate accepts it) and K2S_CASES=2 K2S_TRACE_REPLAYS=2."""

import os
import statistics
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests import test_spec_fused_sampled_host as fsh
from models.demos.blackhole.qwen36.tests.test_spec_sampled_kernel_host import _bf16_words, _boundary
from models.demos.blackhole.qwen36.tt import spec_sampled_kernel_math as km
from models.demos.blackhole.qwen36.tt.spec_fused_kernels import SampledAcceptKernel
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    _ONE_MINUS,
    FusedSampledDevice,
    fused_drafter_reference,
    fused_sampled_reference,
    pack_samp_host,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TT_VISIBLE_DEVICES") != "0" and not os.environ.get("TT_METAL_SIMULATOR"),
    reason="needs TT_VISIBLE_DEVICES=0 (one chip) or TT_METAL_SIMULATOR",
)

W = 32
POISON = 0x5A5A5A5A
N_CASES = int(os.environ.get("K2S_CASES", "40"))
N_TRACE = int(os.environ.get("K2S_TRACE_REPLAYS", "100"))
_PROD = os.environ.get("K2S_OPS_SHAPES", "all") == "prod"
SHAPES = [(B, 3, nb) for B in (1, 4, 8) for nb in (6, 537, 4097)] + [(B, 7, nb) for B in (1, 2, 4) for nb in (6, 4097)]
PROD_SHAPES = [(8, 3, 537), (8, 3, 4097), (4, 7, 4097)]


@pytest.fixture(scope="module")
def mesh():
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    yield m
    ttnn.close_mesh_device(m)


def _words(t):
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return x.reshape(tuple(int(d) for d in t.shape)).to(torch.int64) & 0xFFFFFFFF


def _h2d(fd, t, host, dtype):
    h = ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(fd.mesh))
    ttnn.copy_host_to_device_tensor(h, t)


def _poison(fd):
    for n in ("pack_out", "rs_pt_out", "result_out"):
        t = getattr(fd, n)
        _h2d(fd, t, torch.full(tuple(int(d) for d in t.shape), POISON, dtype=torch.int32), t.dtype)


def _read(fd):
    return {n: _words(getattr(fd, n)) for n in ("pack_out", "rs_pt_out", "result_out")}


def _graph_body(fd):
    k = fd._kernel
    fd._kernel = None
    try:
        type(fd).body(fd)
    finally:
        fd._kernel = k


def make_case(g, B, K, nb):
    c = fsh._case(g, B)
    c["pt"] = torch.randint(1, 4000, (B, nb), generator=g, dtype=torch.int32)
    c["uni"] = torch.rand(B, K + 2, generator=g).clamp(max=_ONE_MINUS)
    c["ids"] = c["mi"].reshape(B, K + 1, fsh.M)[:, :, 0].reshape(-1)
    return c


def zero_case(B, K, nb):
    """The engine's eager warm-up inputs: every word zero (every user inactive, rope_rows = 0)."""
    R, TPM = B * (K + 1), fsh.TP * fsh.M
    return dict(
        uv=torch.zeros(R, TPM), ul=torch.zeros(R, TPM, dtype=torch.int64), ids=torch.zeros(R, dtype=torch.int64),
        tok_buf=torch.zeros(R, dtype=torch.int64), aux=torch.zeros(7, W, dtype=torch.int32),
        pt=torch.zeros(B, nb, dtype=torch.int32), temp=torch.zeros(B), top_k=torch.zeros(B), top_p=torch.zeros(B),
        greedy=torch.zeros(B, dtype=torch.bool), uni=torch.zeros(B, K + 2),
    )  # fmt: skip


def stage(fd, c):
    T = fd.T
    fd.stage_sampled(
        c["uv"], c["ul"], c["ids"].to(torch.int32), c["tok_buf"].to(torch.int32), c["aux"],
        c["pt"].repeat_interleave(T, dim=0), c["temp"], c["top_k"], c["top_p"], c["greedy"], c["uni"],
    )  # fmt: skip


def twin(fd, c):
    B, K = fd.B, fd.K
    samp = pack_samp_host(c["temp"], c["top_k"], c["top_p"], c["greedy"], c["uni"], fd.W).numpy()
    aux = torch.zeros(7, fd.W, dtype=torch.int32)
    aux[:, : c["aux"].shape[1]] = c["aux"]
    pk, rs, rr = km.sampled_accept_twin(
        _bf16_words(c["uv"]), c["ul"].numpy(), c["tok_buf"].numpy(), c["ids"].numpy(), aux.numpy(), samp,
        c["pt"].repeat_interleave(K + 1, dim=0).numpy(), B=B, K=K, M=fd.M, TP=fd.TP, per_shard=fd.per_shard, W=fd.W,
        nb=fd.nb,
    )  # fmt: skip
    return {"pack_out": torch.from_numpy(pk), "rs_pt_out": torch.from_numpy(rs), "result_out": torch.from_numpy(rr)}


def assert_same(got, want, tag):
    for k in want:
        g, w = got[k], want[k]
        assert tuple(g.shape) == tuple(w.shape), f"{tag}: {k} shape {tuple(g.shape)} != {tuple(w.shape)}"
        if not torch.equal(g, w):
            bad = (g != w).nonzero()[:6].tolist()
            raise AssertionError(
                f"{tag}: {k} differs at {bad}: got {[hex(int(g[tuple(i)])) for i in bad]} "
                f"want {[hex(int(w[tuple(i)])) for i in bad]}"
            )


def vs_reference(fd, c, out, tag, stats):
    """Equal to fused_sampled_reference except float32 boundary users; the drafter tail is the S1 tail of the commit."""
    B, K, R = fd.B, fd.K, fd.R
    ref = fused_sampled_reference(
        c["uv"], c["ul"], c["tok_buf"], c["ids"], c["aux"], c["pt"], B=B, K=K, per_shard=fd.per_shard, M=fd.M,
        greedy=c["greedy"], temperature=c["temp"], top_k=c["top_k"], top_p=c["top_p"], uniforms=c["uni"],
    )  # fmt: skip
    res = out["result_out"]
    committed = ((res[0, :R] + 2**31) % 2**32 - 2**31).reshape(B, K + 1)
    counts, ovf = res[1, :B], res[2, :B]
    for u in range(B):
        stats["users"] += 1
        same = torch.equal(committed[u], ref.committed[u].to(torch.int64)) and int(counts[u]) == int(ref.counts[u])
        if int(ovf[u]) != int(ref.overflow[u]) or (not int(ref.overflow[u]) and not same):
            bnd = _boundary(c, ref, c["uni"], u, [int(ref.counts[u]) - 1, int(counts[u]) - 1], K, fd.M)
            assert bnd, f"{tag} user {u}: {committed[u].tolist()}/{int(counts[u])} ovf {int(ovf[u])} != ref " \
                f"{ref.committed[u].tolist()}/{int(ref.counts[u])} ovf {int(ref.overflow[u])}"
            stats["boundary"] += 1
        stats["ovf"] += int(ref.overflow[u])
    pack, rs_pt = fused_drafter_reference(committed, counts, c["aux"], c["pt"], B=B, K=K)[:2]
    want_pk = torch.zeros(2 * K + 5, fd.W, dtype=torch.int64)
    want_pk[:, : pack.shape[1]] = pack.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(out["pack_out"], want_pk), f"{tag}: pack != S1 tail of the commit"
    assert torch.equal(out["rs_pt_out"], rs_pt.to(torch.int64) & 0xFFFFFFFF), f"{tag}: rs_pt != S1 tail of the commit"


def _alloc(mesh, monkeypatch, B, K, nb):
    monkeypatch.setenv("QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED", "1")
    fd = FusedSampledDevice.allocate(mesh, B, K, nb, W, fsh.TP, fsh.M, fsh.PER_SHARD)
    assert isinstance(fd._kernel, SampledAcceptKernel), "the kernel did not attach"
    return fd


@pytest.mark.parametrize("B,K,nb", PROD_SHAPES if _PROD else SHAPES)
def test_sampled_kernel_matches(mesh, monkeypatch, B, K, nb):
    monkeypatch.setattr(fsh, "K", K)
    monkeypatch.setattr(fsh, "T", K + 1)
    g = torch.Generator().manual_seed(7000 * B + 100 * K + nb)
    fd = _alloc(mesh, monkeypatch, B, K, nb)
    st_k = dict(users=0, boundary=0, ovf=0)
    st_g = dict(users=0, boundary=0, ovf=0)
    # the all-zero warm-up inputs (every user inactive): kernel == op graph, word for word
    z = zero_case(B, K, nb)
    stage(fd, z)
    _graph_body(fd)
    graph = _read(fd)
    stage(fd, z)
    _poison(fd)
    fd.body()
    assert_same(_read(fd), graph, f"B={B} K={K} nb={nb} zero inputs: kernel vs op graph")
    assert_same(_read(fd), twin(fd, z), f"B={B} K={K} nb={nb} zero inputs: kernel vs model")
    for i in range(N_CASES):
        c = make_case(g, B, K, nb)
        stage(fd, c)
        _poison(fd)
        fd.body()
        got = _read(fd)
        assert_same(got, twin(fd, c), f"B={B} K={K} nb={nb} case {i}: kernel vs model")
        vs_reference(fd, c, got, f"B={B} K={K} nb={nb} case {i} kernel", st_k)
        stage(fd, c)
        _graph_body(fd)
        vs_reference(fd, c, _read(fd), f"B={B} K={K} nb={nb} case {i} op graph", st_g)
    print(f"RESULT B={B} K={K} nb={nb} cases={N_CASES}: kernel == model; vs reference kernel {st_k} op graph {st_g}")


@pytest.mark.parametrize("B,K,nb", [(8, 3, 4097), (4, 7, 4097)])
def test_sampled_kernel_trace(mesh, monkeypatch, B, K, nb):
    """A trace of the kernel replayed with re-staged inputs (every replay == the model), then replay time vs the op graph."""
    monkeypatch.setattr(fsh, "K", K)
    monkeypatch.setattr(fsh, "T", K + 1)
    g = torch.Generator().manual_seed(99 + B + K)
    fd = _alloc(mesh, monkeypatch, B, K, nb)
    c = make_case(g, B, K, nb)
    stage(fd, c)
    fd.body()  # compiled before the capture
    _graph_body(fd)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fd.body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for i in range(N_TRACE):
        c = make_case(g, B, K, nb)
        stage(fd, c)
        _poison(fd)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        assert_same(_read(fd), twin(fd, c), f"B={B} K={K} nb={nb} replay {i}: kernel vs model")
    times = {}
    for name, fn in (("kernel", fd.body), ("op graph", lambda: _graph_body(fd))):
        if name == "op graph":
            ttnn.release_trace(mesh, tid)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times[name] = 1e3 * (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, tid)
    print(
        f"RESULT trace B={B} K={K} nb={nb}: {N_TRACE} replays == model; replay ms kernel {times['kernel']:.4f} "
        f"op graph {times['op graph']:.4f}"
    )
