# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoEBlock (C++ ops) on a padded chunk, LoudBox 2x4 (FABRIC_2D): GLM-5.3-Flash at chunk 1280, real random weights,
combine_snap 2^-14 (the combine's sum is order-independent, so a token's rows are bit-comparable across routings).

A padded chunk comes as main's chunked-prefill metadata scalars (actual_start, actual_end) and is laid out as main's
KV-pad-aware rotated layout over the SP rows (moe_padding_config's writer): main_row_real below is a host port of its
closed form, checked against the layout's positions. Each SP row's real rows are a prefix; TP chip p of the row owns
its rows p S ... Padding rows are one repeated row (decode via prefill / the last chunk: every padding row is the same
token's embedding, so all of them pick the same experts). With (actual_start, actual_end), chunk aligned or not:
  * the real tokens' outputs (the combine's y_own and the block's output) are bit-identical to a run of the same real
    rows with the padding replaced by distinct real-looking rows (no span: every row real);
  * the padding tokens' combined rows are zero - exactly the host model's padding rows, so every chip's real count is
    the model's - and the routing planner counted no row for them (each expert's count = the real tokens' picks);
  * traced: one capture replayed with other (start, end) (device words) matches the eager forward.

MOE_PAD_CHUNK: the chunk (1280; 2560 fits too, 5120 not on the LoudBox's grid).
test_padding_perf (TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_MID_RUN_DUMP=1, TT_METAL_PROFILER_CPP_POST_PROCESS=1):
the traced forward's device time with 40% real + identical padding, with / without the span, and fully real."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

R, C = 2, 4
CHUNK = int(os.environ.get("MOE_PAD_CHUNK", "1280"))
ACC = 9  # (the combine plans 9 accumulators on the LoudBox's 11x10 grid)
E, K, H, I, Is = 288, 8, 4096, 2048, 2048  # GLM-5.3-Flash
SNAP = 2.0**-14
REPO = Path(__file__).resolve().parents[4]
N_DISTINCT = 4  # (distinct routed experts' weights, cycled over each chip's experts: the host packs them once)


def _fw():
    path = REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py"
    spec = importlib.util.spec_from_file_location("flat_routed_expert_weights", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Capture:
    """block.mb with fused_tail's routed input (the combine's y_own) and the head's outputs kept."""

    def __init__(self, inner):
        self._inner = inner
        self.y_own = None

    def __getattr__(self, k):
        f = getattr(self._inner, k)
        if k != "fused_tail":
            return f

        def tail(y_own, *a, **kw):
            self.y_own = y_own
            return f(y_own, *a, **kw)

        return tail


@pytest.fixture(scope="module")
def mesh():
    if ttnn.get_num_devices() < R * C:
        pytest.skip(f"needs a {R}x{C} LoudBox")
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = max(7232, (2 * H + 32 + 63) // 64 * 64)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D, ttnn.FabricReliabilityMode.RELAXED_INIT, router_config=rc)
    m = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=5888, trace_region_size=64 << 20)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def setup(mesh):
    mb = ttnn.experimental.moe_block
    fw = _fw()
    act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=10.0)
    dims = mb.MoEModelDims(
        num_experts=E, top_k=K, emb_dim=H, routed_expert_hidden_dim=I, routed_expert_activation=act,
        router=mb.RouterSpec(score_func=mb.RouterScoreFunc.Sigmoid, route_scale=2.5),
        shared_expert_hidden_dim=Is, shared_expert_activation=act,
    )  # fmt: skip
    N = R * C
    dtypes = mb.MoEDtypes()
    cfg = mb.MoEBlockConfig(chunk_tokens=CHUNK, num_links=2, persist_l1_small=True)
    cfg.combine_snap = SNAP
    res = mb.MoEBlockResources(mesh, dims, cfg, dtypes)
    block = MoEBlock(mesh, res, dtypes=dtypes, accumulators=ACC, activation=fw.ACTS["silu"])
    g = torch.Generator().manual_seed(3)
    r = lambda *s, sc=1.0: ttnn.from_torch(torch.randn(*s, generator=g) * sc, dtype=ttnn.float32)
    head_w = mb.prepare_fused_head_weights(
        r(Is, H, sc=H**-0.5), r(Is, H, sc=H**-0.5), r(H, Is, sc=Is**-0.5), None, r(E, H, sc=H**-0.5), r(E, sc=0.01),
        resources=res, dtypes=dtypes, accumulators=ACC,
    )  # fmt: skip
    EPC = E // N
    distinct = [
        (
            torch.randn(H, I, generator=g) * H**-0.5,
            torch.randn(H, I, generator=g) * H**-0.5,
            torch.randn(I, H, generator=g) * I**-0.5,
        )
        for _ in range(N_DISTINCT)
    ]
    local = [distinct[i % N_DISTINCT] for i in range(EPC)]
    gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
    experts = fw.FlatRoutedExpert(
        mesh, [local] * N, m=block.expert_kw["max_tokens_per_expert"], H=dims.routed_width, I=I, gids=gids,
        n_global=E, act="silu", pin=1, layout=block.expert_kw["layout"],
    )  # fmt: skip
    weights = MoEBlockWeights(head=head_w, expert_ids=experts.gidx, gate_up=experts.w_gu, down=experts.w_d)
    base = torch.randn(CHUNK, H, generator=g).to(torch.bfloat16)  # (distinct rows everywhere)
    pad_row = torch.randn(1, H, generator=g).to(torch.bfloat16)
    cap = _Capture(block.mb)
    block.mb = cap
    yield dict(
        mesh=mesh, mb=mb, block=block, weights=weights, base=base, pad_row=pad_row, cap=cap, EPC=EPC, N=N,
        distinct=distinct,
    )  # fmt: skip
    experts.close()
    res.release()


def _x(mesh, rows, host=False):
    return ttnn.from_torch(
        rows.reshape(1, 1, CHUNK, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None if host else mesh,
        memory_config=None if host else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
    )  # fmt: skip


def main_row_real(start, end, sp, tpc, my):
    """moe_padding_config's writer (origin/main): SP row my's real rows under the KV-pad-aware rotated layout"""
    pc = lambda e, b, n: 0 if e <= b else min(e - b, n)  # noqa: E731
    chunk = sp * tpc
    slab, bchip, boff = start // chunk, (start // tpc) % sp, start % tpc
    if my != bchip:
        return pc(end, (slab + 1 if my < bchip else slab) * chunk + my * tpc, tpc)
    return pc(end, slab * chunk + my * tpc + boff, tpc - boff) + pc(end, (slab + 1) * chunk + my * tpc, boff)


def rotated_positions(start, sp, tpc):
    """[sp][tpc] the absolute position of each SP row's rows (update_padded_kv_cache's rotation)"""
    chunk = sp * tpc
    slab, bchip, boff = start // chunk, (start // tpc) % sp, start % tpc
    out = []
    for my in range(sp):
        if my != bchip:
            base = (slab + 1 if my < bchip else slab) * chunk + my * tpc
            out.append([base + j for j in range(tpc)])
        else:
            lo = [slab * chunk + my * tpc + boff + j for j in range(tpc - boff)]
            out.append(lo + [(slab + 1) * chunk + my * tpc + j for j in range(boff)])
    return out


def real_mask(start, end):
    """[CHUNK] in the block's row order (SP row r's rows r CHUNK/R ..): real rows of the span"""
    tpc = CHUNK // R
    m = torch.zeros(CHUNK, dtype=torch.bool)
    for r in range(R):
        m[r * tpc : r * tpc + main_row_real(start, end, R, tpc, r)] = True
    return m


def chip_counts(start, end):
    """each chip's real tokens (row-major chips): its SP row's real rows less p S, in [0, S]"""
    tpc, S = CHUNK // R, CHUNK // (R * C)
    return [min(max(main_row_real(start, end, R, tpc, r) - p * S, 0), S) for r in range(R) for p in range(C)]


ALIGNED = [(0, CHUNK * 2 // 5), (0, 0), (0, 1), (0, 37), (0, 160), (0, 161), (0, 497), (0, CHUNK - 1), (0, CHUNK)]
ALIGNED += [(3 * CHUNK, 3 * CHUNK + 512)]
ROTATED = [  # (actual_start not chunk aligned: a multi-turn resume)
    (100, 612),
    (700, 1700),
    (1613, 1613 + CHUNK),
    (1613, 1614),
    (650, 650 + 640),
    (CHUNK - 1, CHUNK - 1 + 700),
    (2 * CHUNK + CHUNK // 2, 2 * CHUNK + CHUNK // 2 + 300),
    (5000, 5999),
]


def test_host_model():
    """main_row_real against the rotated layout's positions: every SP row's positions increase (its real rows are a
    prefix), the chunk's positions are [start, start + CHUNK), and the closed form counts the positions below end"""
    tpc = CHUNK // R
    for start in list(range(0, 3 * CHUNK, 7)) + [x[0] for x in ROTATED]:
        pos = rotated_positions(start, R, tpc)
        assert sorted(p for row in pos for p in row) == list(range(start, start + CHUNK))
        for end in (start, start + 1, start + 333, start + tpc, start + tpc + 1, start + CHUNK - 1, start + CHUNK):
            for r in range(R):
                assert all(a < b for a, b in zip(pos[r], pos[r][1:]))
                assert main_row_real(start, end, R, tpc, r) == sum(p < end for p in pos[r]), (start, end, r)
            if start % CHUNK == 0:  # (chunk aligned: the natural order)
                assert torch.equal(real_mask(start, end), torch.arange(CHUNK) < end - start)


def _padded(s, mask):
    rows = s["base"].clone()
    rows[~mask] = s["pad_row"]
    return rows


def _owner_rows(t):
    """per-chip [1, 1, S, W] (the owners' rows, row-major chips) -> [CHUNK, W] in the chunk's token order"""
    return torch.cat([ttnn.to_torch(d).reshape(-1, d.shape[-1]) for d in ttnn.get_device_tensors(t)], dim=0)


def _residual(mesh, t):
    """the block's output [1, 1, S/SP, H/TP] a chip -> [CHUNK, H]"""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(R, C), dims=(2, 3))).reshape(
        CHUNK, H
    )


def _counts(s):
    """each chip's experts' counts from the planner (counts [.., NG] a chip: its EPC experts) -> [E]"""
    EPC, out = s["EPC"], []
    for i, d in enumerate(ttnn.get_device_tensors(s["block"].last_head.counts)):
        out.append(ttnn.to_torch(d).reshape(-1)[i * EPC : (i + 1) * EPC].to(torch.int64))
    return torch.cat(out)


def _picks(s):
    """the routing rows' expert ids [CHUNK, K] (chunk order) - every token's, padding included"""
    out = []
    for d in ttnn.get_device_tensors(s["block"].last_head.rows):
        r = ttnn.to_torch(d).reshape(-1, 2 * K).to(torch.int64) & 0xFFFF  # (a token: K ids, then K bf16 weights)
        out.append(r[: CHUNK // (R * C), :K])
    return torch.cat(out)


def _clobber(s):
    """junk into the persistent tensors a forward must rewrite (the dispatch buffer, the combine's output, the planner's
    counts): a dispatch / combine / planner that skipped a row would otherwise pass on the last call's data"""
    mesh, block = s["mesh"], s["block"]
    head = getattr(block, "last_head", None)
    if head is None:
        return
    ts = [(head.xbuf, 1000.0), (head.counts, 0xDEAD)]
    if s["cap"].y_own is not None:
        ts.append((s["cap"].y_own, 7.0))
    for t, v in ts:
        d = ttnn.get_device_tensors(t)[0]
        host = ttnn.from_torch(
            torch.full(tuple(d.shape), v, dtype=torch.int32 if t.dtype == ttnn.uint32 else torch.float32),
            dtype=t.dtype, layout=t.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )  # fmt: skip
        ttnn.copy_host_to_device_tensor(host, t)


def _span_kw(block, span):
    if span is None:
        return {}
    start, end = span
    if isinstance(start, int):
        start, end = block.chunk_scalars(start, end)
    return dict(actual_start=start, actual_end=end)


def _forward(s, rows, span=None):
    mesh, block = s["mesh"], s["block"]
    _clobber(s)
    x = _x(mesh, rows)
    y = block(x, s["weights"], **_span_kw(block, span))
    ttnn.synchronize_device(mesh)
    out = dict(y_own=_owner_rows(s["cap"].y_own), out=_residual(mesh, y), counts=_counts(s), picks=_picks(s))
    ttnn.deallocate(y)
    ttnn.deallocate(x)
    return out


def _bits(t):
    return t.contiguous().view(torch.int16)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


def test_unpadded_reference(setup):
    """The unpadded forward: counts = every token's picks, deterministic."""
    s = setup
    o = _forward(s, s["base"])
    hist = torch.bincount(o["picks"].reshape(-1), minlength=E)
    assert torch.equal(o["counts"], hist), "the planner's counts vs the routing rows' picks"
    assert int(o["counts"].sum()) == CHUNK * K
    o2 = _forward(s, s["base"])
    assert _same(o["y_own"], o2["y_own"]) and _same(o["out"], o2["out"]), "not deterministic"


def _routes(s):
    """the routing rows [CHUNK, K] expert ids and [CHUNK, K] gate weights (chunk order)"""
    ids, ws = [], []
    for d in ttnn.get_device_tensors(s["block"].last_head.rows):
        r = ttnn.to_torch(d).reshape(-1, 2 * K).to(torch.int64)[: CHUNK // (R * C)]
        ids.append(r[:, :K] & 0xFFFF)
        ws.append(((r[:, K:] & 0xFFFF) << 16).to(torch.int32).view(torch.float32))
    return torch.cat(ids), torch.cat(ws)


def _torch_y_own(s, rows, tokens):
    """the routed experts' combined output of `tokens` in torch (fp32, the device's routes and gate weights)"""
    ids, ws = _routes(s)
    out = []
    for t in tokens:
        x = rows[t].float()
        y = torch.zeros(H)
        for k in range(K):
            wg, wu, wd = s["distinct"][(int(ids[t, k]) % s["EPC"]) % N_DISTINCT]
            y += float(ws[t, k]) * ((torch.nn.functional.silu(x @ wg) * (x @ wu)) @ wd)
        out.append(y)
    return torch.stack(out)


def _pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.reshape(-1).float(), b.reshape(-1).float()]))[0, 1])


def _check_padded(s, ref, o, span):
    m = real_mask(*span)
    assert _same(o["y_own"][m], ref["y_own"][m]), f"{span}: the real tokens' y_own differs"
    assert _same(o["out"][m], ref["out"][m]), f"{span}: the real tokens' block output differs"
    pad = o["y_own"][~m]
    assert int(pad.abs().sum()) == 0 and not torch.isnan(pad).any(), f"{span}: padding rows' routed output"
    hist = torch.bincount(o["picks"][m].reshape(-1), minlength=E)
    assert torch.equal(o["counts"], hist), f"{span}: the counts are not the real tokens' picks"
    assert int(o["counts"].sum()) == int(m.sum()) * K


def test_padded(setup):
    s = setup
    ref = _forward(s, s["base"])  # (the same real rows, the padding distinct real-looking rows)
    span = ALIGNED[0]
    o = _forward(s, _padded(s, real_mask(*span)), span)
    _check_padded(s, ref, o, span)
    # (the real rows against torch: a sample of real tokens' routed output, the device's routes and gate weights)
    real = torch.nonzero(real_mask(*span)).reshape(-1)
    sample = real[torch.randperm(len(real), generator=torch.Generator().manual_seed(1))[:32]].tolist()
    pcc = _pcc(o["y_own"][sample].float(), _torch_y_own(s, _padded(s, real_mask(*span)), sample))
    assert pcc > 0.97, f"{span}: real tokens' routed output vs torch, PCC {pcc:.5f}"
    hot = torch.bincount(o["picks"][~real_mask(*span)].reshape(-1), minlength=E)
    logger.info(f"MOE_PAD (a) {span}: 32 real tokens' routed output vs torch PCC {pcc:.5f}")
    logger.info(
        f"MOE_PAD (a, b) {span}: real rows bit-identical, padding rows zero; counts sum {int(o['counts'].sum())}; "
        f"the padding's picks (unplanned): {int((hot > 0).sum())} distinct experts"
    )
    for span in ALIGNED[1:]:
        _check_padded(s, ref, _forward(s, _padded(s, real_mask(*span)), span), span)
    logger.info(f"MOE_PAD (a, b) chunk aligned {ALIGNED}: ok")
    for span in ROTATED:
        _check_padded(s, ref, _forward(s, _padded(s, real_mask(*span)), span), span)
        logger.info(f"MOE_PAD (a, b) rotated {span}: chips' real tokens {chip_counts(*span)} ok")
    # main's metadata tuple (slot_id, actual_start, actual_end) as it is
    span = ROTATED[1]
    st, en = s["block"].chunk_scalars(*span)
    slot = ttnn.from_torch(torch.zeros(1, 1, 1, 1, dtype=torch.int64), dtype=ttnn.uint32, device=s["mesh"],
                           mesh_mapper=ttnn.ReplicateTensorToMesh(s["mesh"]))  # fmt: skip
    _clobber(s)
    x = _x(s["mesh"], _padded(s, real_mask(*span)))
    y = s["block"](x, s["weights"], metadata=(slot, st, en))
    ttnn.synchronize_device(s["mesh"])
    o = dict(y_own=_owner_rows(s["cap"].y_own), out=_residual(s["mesh"], y), counts=_counts(s), picks=_picks(s))
    _check_padded(s, ref, o, span)
    logger.info("MOE_PAD metadata=(slot_id, actual_start, actual_end): ok")


def test_padded_traced(setup):
    s = setup
    mesh, block = s["mesh"], s["block"]
    ref = _forward(s, s["base"])
    x = _x(mesh, _padded(s, real_mask(0, 0)))  # (the trace's x: each span's rows written before its replay)
    st, en = block.chunk_scalars(*ALIGNED[0])
    ttnn.deallocate(block(x, s["weights"], actual_start=st, actual_end=en))  # (warm: every program built)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    y = block(x, s["weights"], actual_start=st, actual_end=en)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    y_own = s["cap"].y_own
    spans = [ALIGNED[0], ROTATED[1], (0, CHUNK * 3 // 4), ROTATED[0], (0, 37), ROTATED[2], (0, CHUNK), ALIGNED[0]]
    try:
        for span in spans:
            ttnn.copy_host_to_device_tensor(_x(mesh, _padded(s, real_mask(*span)), host=True), x)
            block.chunk_scalars(*span)  # (the same two words the capture reads)
            _clobber(s)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            o = dict(y_own=_owner_rows(y_own), out=_residual(mesh, y), counts=_counts(s), picks=_picks(s))
            _check_padded(s, ref, o, span)
            e = _forward(s, _padded(s, real_mask(*span)), span)
            assert _same(o["y_own"], e["y_own"]) and _same(o["out"], e["out"]), f"{span}: traced vs eager"
        logger.info(f"MOE_PAD (d) traced: one capture replayed with (start, end) {spans} = eager, ok")
    finally:
        ttnn.release_trace(mesh, tid)


KERNEL_DURATION = "DEVICE KERNEL DURATION [ns]"


def _device_us(s, rows, span, calls=15):
    """Traced (no host gaps): the forward's device time a chip - its tail's end less its head's start, the same clock -
    max over chips, and each program's kernel time (max over chips); medians of calls replays. us."""
    mesh, block = s["mesh"], s["block"]
    x = _x(mesh, rows)
    kw = _span_kw(block, span)  # (the words written before the capture: no host write in it)
    ttnn.deallocate(block(x, s["weights"], **kw))
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    y = block(x, s["weights"], **kw)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.ReadDeviceProfiler(mesh)
    spans, per = [], []
    for _ in range(calls):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(mesh)
        progs, span, shape = {}, 0.0, set()
        for dev, ps in (ttnn.get_latest_programs_perf_data() or {}).items():
            ps = sorted(ps, key=lambda p: p.program_execution_uid.runtime_id)
            shape.add(len(ps))
            rs = [p.program_analyses_results[KERNEL_DURATION] for p in ps]
            span = max(span, (max(r.end_timestamp for r in rs) - min(r.start_timestamp for r in rs)) / 1000.0)
            for i, r in enumerate(rs):
                progs[i] = max(progs.get(i, 0.0), r.duration / 1000.0)
        assert shape == {4}, f"programs a chip: {shape}"
        spans.append(span)
        per.append([progs[i] for i in sorted(progs)])
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(y)
    ttnn.deallocate(x)
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    return med(spans), [med([p[i] for p in per]) for i in range(4)]


def test_padding_perf(setup):
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        pytest.skip("needs TT_METAL_DEVICE_PROFILER=1 (+ MID_RUN_DUMP, CPP_POST_PROCESS)")
    s = setup
    names = ["head", "dispatch", "experts+combine", "tail"]

    def run(what, rows, sp):
        span, us = _device_us(s, rows, sp)
        parts = ", ".join(f"{n} {u:.1f}" for n, u in zip(names, us))
        logger.info(f"MOE_PAD (e) chunk {CHUNK}, {what}: forward {span:.1f} us a chip (traced); programs: {parts}")

    V = CHUNK * 2 // 5
    run("100% real, no span", s["base"], None)
    run(f"{V} real + identical padding, no span", _padded(s, real_mask(0, V)), None)
    run(f"{V} real + identical padding, span (0, {V})", _padded(s, real_mask(0, V)), (0, V))
    run(f"{V} real + identical padding, rotated span {ROTATED[0]}", _padded(s, real_mask(*ROTATED[0])), ROTATED[0])
    run(f"100% real, span (0, {CHUNK})", s["base"], (0, CHUNK))
