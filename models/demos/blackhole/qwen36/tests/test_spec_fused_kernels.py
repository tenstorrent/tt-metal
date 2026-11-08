# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE-chip device test of QWEN36_SPEC_CHAIN_KERNEL / QWEN36_SPEC_ACCEPT_KERNEL (tt/spec_fused_kernels.py): the one-kernel
chain and greedy-accept bodies write the same words as the op graphs they replace (spec_chain.ChainDevice.body,
spec_fused.FusedGreedyDevice.body) and as the exact host references (chain_reference, fused_greedy_reference), word for word,
padding included (every output is poisoned before the kernel runs). Random batches with joins / finishes (host-owned rows),
declined / skip / inactive rows, per-row M-RoPE deltas that hit both rope-table clamps, capacity-edge fits, B 1/2/4/8, K 1..7,
nb 4 / 64 / 537 / 4097 (the serving width); the all-zero warm-up inputs; 1000 trace replays per kernel with freshly staged
inputs. Also times trace replays of op graph vs kernel. No model, no weights.
Run: TT_VISIBLE_DEVICES=0 pytest -o addopts="" --timeout=3600 --import-mode=importlib -sv \
  models/demos/blackhole/qwen36/tests/test_spec_fused_kernels.py"""

import os
import statistics
import time

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words
from models.demos.blackhole.qwen36.tt.spec_chain import (
    ChainDevice,
    chain_reference,
    fused_fit_mode_range,
    pack_ovr_host,
    res_c_reference,
)
from models.demos.blackhole.qwen36.tt.spec_fused import (
    FUSED_AUX_ROWS,
    FusedGreedyDevice,
    fused_accept_aux_host,
    fused_fit_mode,
    fused_greedy_reference,
)
from models.demos.blackhole.qwen36.tt.spec_fused_kernels import AcceptKernel, ChainKernel

SCRATCH, BS, VOCAB = 5000, 64, 248320
POISON = 0x5A5A5A5A
ROPE = [(0, 262144), (5, 150), (-7, 60)]  # (rope delta, rope table rows): the last two hit both clamps
N_TRACE = int(os.environ.get("K2_TRACE_REPLAYS", "1000"))


# --------------------------------------------------------------------------------------------------------------------- #
# case generators (the distributions of test_spec_fused_host / test_spec_chain_host, without the vLLM plugin import)
# --------------------------------------------------------------------------------------------------------------------- #
def accept_case(g, B, K, nb, edge=False, mixed=False, zero=False):
    """dict(ids [32], tok [R], aux [7, W], pt_rows [R, nb], ref) or None (a 'mixed' fit row: the host falls back)."""
    T, R, W = K + 1, B * (K + 1), 32
    ri = lambda lo, hi, *s: torch.randint(lo, hi, s, generator=g)
    if zero:  # the eager warm-up of FusedGreedyDevice.body: every input zero
        z = lambda *s: torch.zeros(*s, dtype=torch.int32)
        aux, pt = z(FUSED_AUX_ROWS, W), z(B, nb)
        ids, tok = z(32), z(R)  # rope_rows = 0: outside rope_row_index's contract, so kernel vs op graph only
        return dict(ids=ids, tok=tok, aux=aux, pt_rows=pt.repeat_interleave(T, dim=0), ref=None)
    cap = nb * BS
    pad = torch.rand(B, generator=g) < 0.25
    skip = (torch.rand(B, generator=g) < 0.2) & ~pad
    nv = ri(0, K + 1, B)
    nv[pad] = 0
    start0 = ri(1, 60, B)
    if edge:  # near the capacity edge: fits_all / fits_none / mixed all occur
        start0 = cap - 2 * K - 2 + ri(0, 2 * K + 4, B)
    ids = ri(1, VOCAB, B, T)
    tok = torch.cat([ri(1, VOCAB, B, 1), ri(1, VOCAB, B, K)], 1)
    for u in range(B):  # forced correct-draft prefix so m covers 0..K
        mm = int(ri(0, K + 1, 1))
        tok[u, 1 : 1 + mm] = ids[u, :mm]
        if mm < K:
            tok[u, 1 + mm] = (ids[u, mm] + 1) % VOCAB
    active = ~pad
    for u in range(B):  # sanitize_verify_block: pad tail = column 0, inactive row = 0, start0 = max(start_pos[u, 0], 0)
        if not active[u]:
            tok[u] = 0
            start0[u] = 0
        else:
            tok[u, int(nv[u]) + 1 :] = tok[u, 0]
    delta, rows = ROPE[int(ri(0, len(ROPE), 1))]
    if mixed:
        delta = [delta + int(x) for x in ri(-30, 30, B)]
    try:
        aux = fused_accept_aux_host(nv, start0, active, skip, cap, K, SCRATCH, delta, rows)
    except ValueError:
        return None
    pt = torch.randint(1, 4000, (B, nb), generator=g, dtype=torch.int32)
    ids32 = torch.zeros(32, dtype=torch.int32)
    ids32[:R] = ids.reshape(-1).to(torch.int32)
    tokw = tok.reshape(-1).to(torch.int32)
    ref = fused_greedy_reference(ids32[:R], tokw, aux, pt, B=B, K=K)
    return dict(ids=ids32, tok=tokw, aux=aux, pt_rows=pt.repeat_interleave(T, dim=0), ref=ref)


def accept_cases(g, n, B, K, nb, mixed=False):
    out = []
    while len(out) < n:
        c = accept_case(g, B, K, nb, edge=len(out) % 3 == 2, mixed=mixed)
        if c is not None:
            out.append(c)
    return out


def chain_case(g, B, K, Nv, parity, mixed=False, zero=False, W_res=32):
    """dict(res [rows, W_res], prev_aux [7, 32], drafts [B, K], ovr [11, W_o], ref (tok, vaux, next aux))."""
    T = K + 1
    ri = lambda lo, hi, *s: torch.randint(lo, hi, s, generator=g)
    W_o = -(-max(B * T, B, 5) // 32) * 32
    if zero:  # the eager warm-up of ChainDevice.body: every input zero (rope_rows = 0: kernel vs op graph only)
        z = lambda *s: torch.zeros(*s, dtype=torch.int32)
        return dict(res=z(3, W_res), prev_aux=z(FUSED_AUX_ROWS, 32), drafts=z(B, K), ovr=z(11, W_o), ref=None)
    # same distribution as test_spec_chain_host._case
    cap = int(ri(40, 200, 1)) if torch.rand(1, generator=g) < 0.7 else 100000
    rope_rows = int(ri(30, 90, 1))  # small: exercises the clamp
    delta = int(ri(-40, 40, 1))
    if mixed:
        delta = [int(x) for x in ri(-40, 40, B)]
        delta = [delta[0] if u % 2 else delta[u] for u in range(B)]
    counts = ri(1, T + 1, B)
    committed = ri(0, VOCAB, B, T)
    committed[torch.arange(T)[None, :] >= counts[:, None]] = -1
    p0_prev = ri(1, 70, B)
    drafts = ri(0, VOCAB, B, K)
    mask = (torch.rand(B, generator=g) < 0.3).long()
    act = (torch.rand(B, generator=g) > 0.2).long()
    skip = (torch.rand(B, generator=g) < 0.25).bool()
    nv = torch.where(act.bool(), ri(0, K + 1, B), torch.zeros(B, dtype=torch.long))
    p0_host = torch.where(act.bool(), ri(1, 70, B), torch.zeros(B, dtype=torch.long))
    mi_host = torch.where(act.bool(), ri(0, T, B), torch.zeros(B, dtype=torch.long))
    host_tok = torch.where(act.bool()[:, None], ri(0, VOCAB, B, T), torch.zeros(B, T, dtype=torch.long))
    for u in range(B):
        host_tok[u, int(nv[u]) + 1 :] = host_tok[u, 0] if act[u] else 0
    rmode = fused_fit_mode_range(p0_prev, cap, act, K)
    hmode = fused_fit_mode(p0_host, cap, act, K)
    skip = skip & act.bool()
    dec, keep = [], []
    for u in range(B):
        mode = hmode[u] if mask[u] else rmode[u]
        if mode == "mixed" and not skip[u]:
            skip[u] = bool(act[u])
        dec.append(int((not act[u]) or skip[u] or mode == "fits_none"))
        keep.append(int(skip[u] and act[u]))
    prev_aux = torch.randint(0, 1000, (FUSED_AUX_ROWS, 32), generator=g, dtype=torch.int32)  # junk but the p0 row
    prev_aux[4, :B] = p0_prev.to(torch.int32)
    res = res_c_reference(committed, counts, p0_prev, torch.randint(0, 2, (B,), generator=g))
    res = torch.cat([res[:3], torch.zeros(3, max(0, W_res - res.shape[1]), dtype=torch.int32)], 1)[:, :W_res]
    ovr = pack_ovr_host(
        nv, act, dec, keep, p0_host, mask, host_tok, mi_host, parity,
        K=K, scratch_block=SCRATCH, rope_delta=delta, rope_rows=rope_rows,
    )  # fmt: skip
    drafts = drafts.to(torch.int32)
    ref = chain_reference(
        res, drafts, prev_aux, ovr, B=B, K=K, Nv=Nv, rope_rows=rope_rows, ctrl_words=spec_ctrl_words(B, Nv)
    )
    return dict(res=res, prev_aux=prev_aux, drafts=drafts, ovr=ovr, ref=ref)


# --------------------------------------------------------------------------------------------------------------------- #
# staging / reading
# --------------------------------------------------------------------------------------------------------------------- #
def _h2d(mesh, dst, t, dt):
    host = ttnn.from_torch(
        torch.as_tensor(t).to(torch.int32).reshape(tuple(int(d) for d in dst.shape)),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.copy_host_to_device_tensor(host, dst)


def _words(mesh, t):
    """The device words of ``t`` (device 0 of the mesh) as int64 bit patterns."""
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return x.reshape(tuple(int(d) for d in t.shape)).to(torch.int64) & 0xFFFFFFFF


def _poison(mesh, t):
    _h2d(mesh, t, torch.full(tuple(int(d) for d in t.shape), POISON, dtype=torch.int32), t.dtype)


def _w(t):
    return torch.as_tensor(t).to(torch.int64) & 0xFFFFFFFF


# ---- accept
def stage_accept(fd, c):
    fd.stage(c["ids"][: fd.R], c["tok"], c["aux"], c["pt_rows"])


def read_accept(fd):
    return {n: _words(fd.mesh, getattr(fd, n)) for n in ("pack_out", "rs_pt_out", "result_out")}


def want_accept(fd, ref):
    B, R, W = fd.B, fd.R, fd.W
    res = torch.zeros(2, W, dtype=torch.int64)
    res[0, :R] = ref.committed.reshape(-1).to(torch.int64)
    res[1, :B] = ref.counts.to(torch.int64)
    return {"pack_out": _w(ref.pack), "rs_pt_out": _w(ref.rs_pt), "result_out": _w(res)}


def poison_accept(fd):
    for n in ("pack_out", "rs_pt_out", "result_out"):
        _poison(fd.mesh, getattr(fd, n))


def assert_same(got, want, tag):
    for k in want:
        g, w = got[k], want[k]
        assert tuple(g.shape) == tuple(w.shape), f"{tag}: {k} shape {tuple(g.shape)} != {tuple(w.shape)}"
        if not torch.equal(g, w):
            bad = (g != w).nonzero()[:6].tolist()
            raise AssertionError(f"{tag}: {k} differs at {bad}: got {[int(g[tuple(i)]) for i in bad]} want {[int(w[tuple(i)]) for i in bad]}")


def accept_pair(fd, kern, c, tag):
    """Op graph then kernel on the same staged inputs: both equal the host reference (kernel outputs poisoned first)."""
    stage_accept(fd, c)
    _graph_body(fd)
    graph = read_accept(fd)
    stage_accept(fd, c)
    poison_accept(fd)
    kern.run()
    got = read_accept(fd)
    if c["ref"] is not None:
        assert_same(graph, want_accept(fd, c["ref"]), f"{tag} op graph vs reference")
    assert_same(got, graph, f"{tag} kernel vs op graph")


# ---- chain
def stage_chain(cd, c):
    rows = int(cd.result_in.shape[0])
    _h2d(cd.mesh, cd.result_in, c["res"][:rows], ttnn.int32)
    _h2d(cd.mesh, cd.aux_prev_in, c["prev_aux"], ttnn.int32)
    for k, d in enumerate(cd.draft_ins):
        _h2d(cd.mesh, d, c["drafts"][:, k], ttnn.uint32)
    cd.stage_ovr(c["ovr"])


def read_chain(cd):
    return {n: _words(cd.mesh, getattr(cd, n)) for n in ("tok_out", "vaux_out", "aux_prev_in")}


def want_chain(ref):
    tok, vaux, aux = ref
    return {"tok_out": _w(tok).reshape(1, -1), "vaux_out": _w(vaux), "aux_prev_in": _w(aux)}


def chain_pair(cd, kern, c, tag, check_ref=True):
    stage_chain(cd, c)
    _graph_body(cd)
    graph = read_chain(cd)
    stage_chain(cd, c)  # aux_prev_in is also an output: re-stage the input words
    _poison(cd.mesh, cd.tok_out)
    _poison(cd.mesh, cd.vaux_out)
    kern.run()
    got = read_chain(cd)
    if check_ref and c["ref"] is not None:
        assert_same(graph, want_chain(c["ref"]), f"{tag} op graph vs reference")
    assert_same(got, graph, f"{tag} kernel vs op graph")


def _graph_body(obj):
    """The op-graph body of a device object whatever its flag (the kernel attribute is per instance; a class without it
    is the pre-K2 one, whose body is the op graph)."""
    k = getattr(obj, "_kernel", None)
    obj._kernel = None
    try:
        type(obj).body(obj)
    finally:
        obj._kernel = k


def trace_ms(mesh, fn, reps=200):
    """Per-replay wall ms of a trace of ``fn`` (non-blocking replays, one synchronize): ~device time of the trace."""
    fn()  # every program compiled before the capture
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(5):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = 1e3 * (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, tid)
    return ms


# --------------------------------------------------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mesh():
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    yield m
    ttnn.close_mesh_device(m)


ACCEPT_SHAPES = [(B, K, nb) for B in (1, 2, 4, 8) for K in (3,) for nb in (4, 64, 537, 4097)] + [
    (8, 1, 537), (4, 7, 537), (2, 5, 64), (8, 2, 4097), (1, 7, 4097),
]  # fmt: skip
# K2_OPS_SHAPES=prod: only the production bucket (B=8 K=3; nb 537 = the 32K harness width, 4097 = the serving width), for a
# short device session; default: every shape (each new shape compiles the op graph's programs once)
_PROD = os.environ.get("K2_OPS_SHAPES", "all") == "prod"


@pytest.mark.parametrize("mixed", [False, True], ids=["common_delta", "row_deltas"])
def test_accept_kernel_matches(mesh, mixed):
    """>= 20 random cases per shape (and the zero warm-up inputs): kernel == op graph == fused_greedy_reference."""
    n_cases = 0
    for B, K, nb in [x for x in ACCEPT_SHAPES if not _PROD or (x[0], x[1]) == (8, 3) and x[2] in (64, 537, 4097)]:
        g = torch.Generator().manual_seed(1000 * B + 10 * K + nb + int(mixed))
        fd = FusedGreedyDevice.allocate(mesh, B, K, nb, 32)
        for cores in (32, 7):
            kern = AcceptKernel(fd, cores)
            for i, c in enumerate([accept_case(g, B, K, nb, zero=True)] + accept_cases(g, 10, B, K, nb, mixed)):
                accept_pair(fd, kern, c, f"accept B={B} K={K} nb={nb} cores={kern.n_cores} case {i}")
                n_cases += 1
        print(f"RESULT accept B={B} K={K} nb={nb} mixed={mixed} chunk={kern.chunk} ok")
    print(f"RESULT accept cases={n_cases} all identical")


CHAIN_SHAPES = [(B, 3, Nv) for B in (1, 2, 4, 8) for Nv in (2, 6)] + [(8, 1, 6), (4, 7, 6), (2, 5, 2), (8, 2, 6)]


@pytest.mark.parametrize("mixed", [False, True], ids=["common_delta", "row_deltas"])
def test_chain_kernel_matches(mesh, mixed):
    """>= 20 random cases per shape, S1 [2, W] and S2 [3, W] result buffers, both parities (and the zero warm-up inputs):
    kernel == op graph == chain_reference."""
    n_cases = 0
    for B, K, Nv in [x for x in CHAIN_SHAPES if not _PROD or (x[0], x[1]) == (8, 3)]:
        g = torch.Generator().manual_seed(7000 + 100 * B + 10 * K + Nv + int(mixed))
        for rows in (2, 3):
            cd = ChainDevice.allocate(mesh, B, K, Nv, result_rows=rows)
            kern = ChainKernel(cd)
            cases = [chain_case(g, B, K, Nv, 0, zero=True, W_res=cd.W_res)]
            cases += [chain_case(g, B, K, Nv, j % 2, mixed=mixed, W_res=cd.W_res) for j in range(10)]
            for j, c in enumerate(cases):
                chain_pair(cd, kern, c, f"chain B={B} K={K} Nv={Nv} rows={rows} case {j}", check_ref=j > 0)
                n_cases += 1
        print(f"RESULT chain B={B} K={K} Nv={Nv} mixed={mixed} ok")
    print(f"RESULT chain cases={n_cases} all identical")


@pytest.mark.parametrize("nb", [537, 4097])
def test_accept_kernel_trace(mesh, nb):
    """B=8 K=3: the kernel in a trace, N_TRACE replays, each on freshly staged inputs, each equal to the reference (outputs
    poisoned every 50 replays); then trace replay time of the op graph vs the kernel."""
    B, K = 8, 3
    g = torch.Generator().manual_seed(42 + nb)
    fd = FusedGreedyDevice.allocate(mesh, B, K, nb, 32)
    kern = AcceptKernel(fd, 32)
    c0 = accept_cases(g, 1, B, K, nb)[0]
    stage_accept(fd, c0)
    kern.run()  # compile before capture
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    kern.run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    pool = accept_cases(g, 64, B, K, nb) + accept_cases(g, 32, B, K, nb, mixed=True)
    for i in range(N_TRACE):
        c = pool[i % len(pool)] if i >= len(pool) else pool[i]
        stage_accept(fd, c)
        if i % 50 == 0:
            poison_accept(fd)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        assert_same(read_accept(fd), want_accept(fd, c["ref"]), f"accept trace replay {i} nb={nb}")
    ttnn.release_trace(mesh, tid)
    stage_accept(fd, c0)
    t_graph = trace_ms(mesh, lambda: _graph_body(fd))
    t_kern = trace_ms(mesh, kern.run)
    print(
        f"RESULT accept trace nb={nb} replays={N_TRACE} identical; trace ms/replay op_graph={t_graph:.4f} "
        f"kernel={t_kern:.4f} saved={t_graph - t_kern:.4f} (cores={kern.n_cores} chunk={kern.chunk})"
    )


def test_chain_kernel_trace(mesh):
    """B=8 K=3 Nv=6: the kernel in a trace, N_TRACE replays on freshly staged inputs, each equal to the reference; then trace
    replay time of the op graph vs the kernel."""
    B, K, Nv = 8, 3, 6
    g = torch.Generator().manual_seed(4242)
    cd = ChainDevice.allocate(mesh, B, K, Nv, result_rows=2)
    kern = ChainKernel(cd)
    pool = [chain_case(g, B, K, Nv, i % 2, mixed=i % 3 == 0, W_res=cd.W_res) for i in range(96)]
    stage_chain(cd, pool[0])
    kern.run()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    kern.run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for i in range(N_TRACE):
        c = pool[i % len(pool)]
        stage_chain(cd, c)
        if i % 50 == 0:
            _poison(cd.mesh, cd.tok_out)
            _poison(cd.mesh, cd.vaux_out)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        assert_same(read_chain(cd), want_chain(c["ref"]), f"chain trace replay {i}")
    ttnn.release_trace(mesh, tid)
    stage_chain(cd, pool[0])
    t_graph = trace_ms(mesh, lambda: _graph_body(cd))
    stage_chain(cd, pool[0])
    t_kern = trace_ms(mesh, kern.run)
    print(
        f"RESULT chain trace replays={N_TRACE} identical; trace ms/replay op_graph={t_graph:.4f} kernel={t_kern:.4f} "
        f"saved={t_graph - t_kern:.4f}"
    )


def test_flag_attach(mesh, monkeypatch):
    """The flags attach the kernels at allocate (and only to the greedy class); flag off leaves the op graph."""
    from models.demos.blackhole.qwen36.tt.spec_fused_sampled import FusedSampledDevice  # noqa: F401 (subclass exists)

    monkeypatch.delenv("QWEN36_SPEC_CHAIN_KERNEL", raising=False)
    monkeypatch.delenv("QWEN36_SPEC_ACCEPT_KERNEL", raising=False)
    assert ChainDevice.allocate(mesh, 8, 3, 6)._kernel is None
    assert FusedGreedyDevice.allocate(mesh, 8, 3, 64, 32)._kernel is None
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_KERNEL", "1")
    monkeypatch.setenv("QWEN36_SPEC_ACCEPT_KERNEL", "1")
    monkeypatch.setenv("QWEN36_SPEC_ACCEPT_KERNEL_CORES", "16")
    cd = ChainDevice.allocate(mesh, 8, 3, 6)
    fd = FusedGreedyDevice.allocate(mesh, 8, 3, 4097, 32)
    assert isinstance(cd._kernel, ChainKernel) and isinstance(fd._kernel, AcceptKernel)
    assert fd._kernel.n_cores == 16
    g = torch.Generator().manual_seed(5)
    c = chain_case(g, 8, 3, 6, 1, W_res=cd.W_res)
    stage_chain(cd, c)
    cd.body()  # the flag routes body() to the kernel
    assert_same(read_chain(cd), want_chain(c["ref"]), "flagged chain body")
    a = accept_cases(g, 1, 8, 3, 4097)[0]
    stage_accept(fd, a)
    poison_accept(fd)
    fd.body()
    assert_same(read_accept(fd), want_accept(fd, a["ref"]), "flagged accept body")
    l1_ids = ttnn.from_torch(
        torch.zeros(1, 1, 1, 32, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    with pytest.raises(AssertionError, match="QWEN36_SPEC_ACCEPT_KERNEL"):  # other accessor: refused (fused accept off)
        fd.bind_ids(l1_ids)
