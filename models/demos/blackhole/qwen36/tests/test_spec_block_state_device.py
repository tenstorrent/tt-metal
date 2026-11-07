# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the spec class's GDN state pages (tt/spec_block_state.py) with gdn_spec_step, on one chip.

The pools are flat float32 alias views of real BF8 KV cache buffers ([N, 1, block, 256], DRAM interleaved), addressed
by SpecBlockLayout through SpecBlockBook's ring_page_idx and ctrl pages, as the model runs them. Each row's start state
is written with SpecGdnBlockState's own prefill write path. Checks against the op's default ring layout (the test
harness of tests/ttnn/.../kda/test_gdn_spec_step.py): output, conv window and every token's state bit-identical,
read back with SpecGdnBlockState's restore path; held rows write nothing; blocks outside the windows stay untouched.
Also the conv history write / read round trip, and traced time per call, ring vs pages.

    pytest models/demos/blackhole/qwen36/tests/test_spec_block_state_device.py
"""

import importlib.util
import pathlib
import time
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import spec_block_state as sbs

_OP_TEST = (
    pathlib.Path(__file__).resolve().parents[5]
    / "tests/ttnn/unit_tests/operations/experimental/kda/test_gdn_spec_step.py"
)
_spec = importlib.util.spec_from_file_location("gdn_spec_step_case", _OP_TEST)
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)

T = 4


def _kv(device, num_blocks, block):
    return ttnn.from_torch(
        torch.zeros(num_blocks, 1, block, 256),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _state(device, g, block, num_blocks):
    """A SpecGdnBlockState over one GDN layer of geometry g, bound to fresh BF8 K and V buffers."""
    k, v = _kv(device, num_blocks, block), _kv(device, num_blocks, block)
    nb = int(device.dram_grid_size().x)
    S = sbs.kv_block_bytes_per_bank(k.shape, "bfloat8_b", nb)
    lay = sbs.SpecBlockLayout(S, nb, num_blocks, g.Nv, sbs.conv_carry_pages(4, g.C))
    half = [lay.pool1_pages // sbs.HALF_PAGES, 1, 64, 128]
    views = sbs._LayerViews(
        rec1=ttnn.experimental.alias_view(k, half, ttnn.float32, check_slices=False),
        rec2=ttnn.experimental.alias_view(v, half, ttnn.float32, check_slices=False),
        conv=ttnn.experimental.alias_view(v, [lay.conv_pages_total, 1, 32, 32], ttnn.bfloat16, check_slices=False),
        conv_k=ttnn.experimental.alias_view(k, [lay.conv_pages_total, 1, 32, 32], ttnn.bfloat16, check_slices=False),
        group=0,
        two_pool=lay.heads_in_k < g.Nv,
        k=k,
        v=v,
    )
    dn = SimpleNamespace(Nv=g.Nv, Nk=g.Nk, Dk=128, Dv=128, K=4, qkv_dim_tp=g.C, _spec_paged=views)
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.model, st.mesh, st.T, st.G, st.layout = None, device, T, 1, lay
    st.layers, st.layer_groups = [(0, dn, 0)], {0: 0}
    st.kv_block_size, st._zeros = block, None
    return st, dn, k, v


def _windows(B, num_blocks, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(num_blocks - 1, generator=g)[: B * T] + 1  # block 0 is vLLM's null block
    return ids.reshape(B, T).to(torch.int64)


def _dev_u32(device, t):
    return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# (geometry, block): TP=4 at 2048 (the QuietBox runs, heads in K and V), TP=8 at 1024 (Galaxy primary, heads in K and
# V), TP=8 at 2048 (every head in K)
CASES = [(op.GEOS[1], 2048), (op.GEOS[0], 1024), (op.GEOS[0], 2048)]


# the deployment's gdn_spec_step options: QWEN36_SPEC_STEP_VALUE_BLOCKS=4, QWEN36_SPEC_STEP_OPT=250 and
# QWEN36_SPEC_STEP_NARROW_GR=1 (kernel_opts bit 1024)
STEP_OPTS = {
    "default": {},
    "vb4_opt250": dict(value_blocks=4, kernel_opts=250),
    "vb4_opt1274": dict(value_blocks=4, kernel_opts=1274),
}


@pytest.mark.parametrize("step_opts", list(STEP_OPTS))
@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
@pytest.mark.parametrize("B,hold", [(1, ()), (4, (2,)), (8, ())])
def test_pages_match_ring(device, geo_block, B, hold, step_opts):
    g, block = geo_block
    so = STEP_OPTS[step_opts]
    num_blocks = 48
    c = op.Case(device, g, B, T, seed=7 + B, hold=hold, par=B & 1, R=op.round_up32(B * T), mode=op.MODES[2])
    out_ref, ring_ref, _, win_ref = c.snapshot(c.run(**so))
    c.reseed()
    st, dn, k, v = _state(device, g, block, num_blocks)
    win = _windows(B, num_blocks, seed=B)
    book = sbs.SpecBlockBook(st.layout, 1, B, T, "none")
    st.book = book
    # each row's start state into its start column with the prefill write path
    for u in range(B):
        book.set_prefill_blocks([torch.tensor([int(win[u, c.mi[u]])])])
        rec = ttnn.from_torch(c.state0[u : u + 1].float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        with sbs._PageTables(st) as pts:
            st._write_rec(dn, int(win[u, c.mi[u]]), rec, pts)
    live = [u for u in range(B) if u not in c.hold]
    book.set_step([win], torch.tensor(c.mi), live)
    words = op.ctrl_page(g, B, T, c.mi, c.par, c.hold).shape[-1]
    c.ring = dn._spec_paged.rec1
    c.ctrl = _dev_u32(device, book.ctrl_rows(c.mi, c.par, words, hold=c.hold))
    idx = _dev_u32(device, book.idx_host[0])
    kw = {"ring_page_idx": idx, **so}
    if dn._spec_paged.two_pool:
        kw["ring_pool2"] = dn._spec_paged.rec2
    o = c.run(**kw)
    out = ttnn.to_torch(o).reshape(-1, g.VD).float()
    rows = [r for r in range(out.shape[0]) if r // T not in c.hold]
    assert torch.equal(out[rows], out_ref[rows]), "output differs from the default ring layout"
    assert torch.equal(ttnn.to_torch([c.win_a, c.win_b][1 - c.par]).float(), win_ref), "window differs"
    for u in range(B):
        for t in range(T):
            got = ttnn.to_torch(st._read_rec(dn, int(win[u, t]))).reshape(g.Nv, 128, 128).float()
            if u in c.hold:
                want = c.state0[u] if t == c.mi[u] else torch.zeros_like(got)
            else:
                want = ring_ref.reshape(T, B, g.Nv, 128, 128)[t, u]
            assert torch.equal(got, want), f"row {u} column {t} block {int(win[u, t])}"
    outside = sorted(set(range(1, num_blocks)) - set(win.reshape(-1).tolist()))[:4]
    for b in outside:
        assert not ttnn.to_torch(st._read_rec(dn, b)).float().any(), f"block {b} outside every window was written"


@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_conv_history_round_trip(device, geo_block):
    """_write_carry / _read_carry, and materialize from a conv window: exact bf16 rows."""
    g, block = geo_block
    st, dn, k, v = _state(device, g, block, 32)
    torch.manual_seed(3)
    carry = torch.randn(1, 3, g.C).bfloat16()
    rm = ttnn.from_torch(carry, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with sbs._PageTables(st) as pts:
        for b in (5, 6, 17):
            st._write_carry(dn, b, rm, pts)
    for b in (5, 6, 17):
        assert torch.equal(ttnn.to_torch(st._read_carry(dn, b)).reshape(1, 3, g.C), carry), f"block {b}"
    B = 4
    window = torch.randn(B, 3 + T, g.C).bfloat16()
    w_dev = ttnn.from_torch(window, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dn.verify_win_cur = lambda: w_dev
    for u, t, dst in ((0, 0, 9), (2, 1, 10), (3, 3, 11)):
        with sbs._PageTables(st) as pts:
            st.materialize(u, t, 0, dst, pts)
        got = ttnn.to_torch(st._read_carry(dn, dst)).reshape(3, g.C)
        assert torch.equal(got, window[u, t + 1 : t + 4]), f"row {u} column {t}"
    assert torch.equal(ttnn.to_torch(st._read_carry(dn, 5)).reshape(1, 3, g.C), carry), "a neighbor block changed"


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize("geo_block", [CASES[0], CASES[1]], ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_traced_time_ring_vs_pages(device, geo_block):
    """Traced time per call at B=8, T=4, the model's arithmetic: the default ring vs the KV-buffer pages."""
    g, block = geo_block
    N, REPS, B = 48, 10, 8
    if B * g.Nv > op.num_cores(device):
        pytest.skip("over the grid")
    c = op.Case(device, g, B, T, seed=23, par=0, R=op.round_up32(B * T), mode=op.MODES[2])
    ring, ctrl = c.ring, c.ctrl
    st, dn, k, v = _state(device, g, block, 64)
    win = _windows(B, 64, seed=5)
    book = sbs.SpecBlockBook(st.layout, 1, B, T, "none")
    book.set_step([win], torch.tensor(c.mi), range(B))
    words = op.ctrl_page(g, B, T, c.mi, c.par).shape[-1]
    pctrl = _dev_u32(device, book.ctrl_rows(c.mi, c.par, words))
    kw = {"ring_page_idx": _dev_u32(device, book.idx_host[0])}
    if dn._spec_paged.two_pool:
        kw["ring_pool2"] = dn._spec_paged.rec2

    def per_call_us(r, ct, extra):
        c.ring, c.ctrl = r, ct
        ttnn.deallocate(c.run(**extra))
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(N):
            ttnn.deallocate(c.run(**extra))
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            times.append((time.perf_counter() - t0) / N * 1e6)
        ttnn.release_trace(device, tid)
        return min(times), sorted(times)[len(times) // 2]

    r_min, r_med = per_call_us(ring, ctrl, {})
    p_min, p_med = per_call_us(dn._spec_paged.rec1, pctrl, kw)
    print(
        f"\n[{g.id} block {block}, {st.layout.heads_in_k}/{g.Nv} heads in K] gdn_spec_step B={B} T={T}: "
        f"ring {r_min:.1f} us (median {r_med:.1f}), KV-buffer pages {p_min:.1f} us (median {p_med:.1f}), "
        f"{100 * (p_min - r_min) / r_min:+.2f}%"
    )


@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_block_copies_keep_the_state(device, geo_block):
    """SpecGdnBlockState.copy_blocks: a whole-block copy between blocks of the same phase and a page-by-page copy
    between blocks of different phase (BF8 1024: odd and even blocks) both leave the source's state at the
    destination's layout pages."""
    g, block = geo_block
    st, dn, k, v = _state(device, g, block, 32)
    st.model = SimpleNamespace(_paged_kv_caches=[(k, v)], mesh_device=device)
    st._slot_layers = {0: dn}
    st.book = sbs.SpecBlockBook(st.layout, 1, 8, T, "none")
    torch.manual_seed(11)
    rec = torch.randn(1, g.Nv, 128, 128)
    carry = torch.randn(1, 3, g.C).bfloat16()
    rec_dev = ttnn.from_torch(rec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with sbs._PageTables(st) as pts:
        st._write_rec(dn, 7, rec_dev, pts)
        rm = ttnn.from_torch(carry, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        st._write_carry(dn, 7, rm, pts)
    # the source is the bound prefill scratch in the model: the write must leave it allocated (every head in K)
    assert rec_dev.is_allocated() and torch.equal(ttnn.to_torch(rec_dev).reshape(rec.shape).float(), rec)
    for dst in (9, 10, 13, 20):
        st.copy_blocks([(7, dst)])
        got = ttnn.to_torch(st._read_rec(dn, dst)).reshape(1, g.Nv, 128, 128).float()
        assert torch.equal(got, rec), f"7 -> {dst} (phases {st.layout.phase(7)}, {st.layout.phase(dst)}): state"
        assert torch.equal(ttnn.to_torch(st._read_carry(dn, dst)).reshape(1, 3, g.C), carry), f"7 -> {dst}: conv"


@pytest.mark.parametrize("block", [2048, 1024])
def test_zero_blocks_zeroes_only_the_listed_blocks(device, block):
    """zero_blocks: every listed block is zeroed, in fills of several blocks (11 blocks: more than one fill), and every
    other block keeps its bytes."""
    num_blocks = 40
    k = ttnn.from_torch(
        torch.ones(num_blocks, 1, block, 256),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v = ttnn.clone(k)
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.model = SimpleNamespace(_paged_kv_caches=[(k, v)], mesh_device=device)
    st.mesh, st.kv_block_size, st._zeros = device, block, None
    st.book = SimpleNamespace(forget_block=lambda b: None)
    listed = [3, 4, 5, 11, 12, 13, 20, 21, 22, 30, 31]
    st.zero_blocks(listed)
    for name, t in (("K", k), ("V", v)):
        host = ttnn.to_torch(t).float()
        for b in range(num_blocks):
            if b in listed:
                assert not host[b].any(), f"{name} block {b} was not zeroed"
            else:
                assert torch.equal(host[b], torch.ones_like(host[b])), f"{name} block {b} changed"


@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_chained_copies_select_dst_by_the_device_count(device, geo_block):
    """run_chain_copies with the counts on the device (the chain copies of a chained step's mamba_state): a row whose
    count selects its copy gets the source's state and the pending verify's conv rows of that column at dst; a row
    whose count does not keeps dst as it was (the copy lands in the null block 0)."""
    from models.demos.blackhole.qwen36.tt.spec_chain import RES_COUNTS

    g, block = geo_block
    st, dn, k, v = _state(device, g, block, 40)
    st.model = SimpleNamespace(_paged_kv_caches=[(k, v)], mesh_device=device)
    st._slot_layers, st.moves_counts, st.chain_copies = {0: dn}, None, []
    B = 8
    st.book = sbs.SpecBlockBook(st.layout, 1, B, T, "align", block)
    w = torch.zeros(B, T, dtype=torch.int64)
    w[2], w[5] = torch.tensor([20, 21, 22, 23]), torch.tensor([30, 31, 32, 33])
    st.book.set_step([w], torch.zeros(B, dtype=torch.int64), [2, 5])
    torch.manual_seed(5)
    recs = {b: torch.randn(1, g.Nv, 128, 128) for b in (21, 32, 9, 12)}
    carries = {b: torch.randn(1, 3, g.C).bfloat16() for b in (9, 12)}
    with sbs._PageTables(st) as pts:
        for b, r in recs.items():
            st._write_rec(dn, b, ttnn.from_torch(r, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device), pts)
        for b, c in carries.items():
            rm = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            st._write_carry(dn, b, rm, pts)
    window = torch.randn(B, 3 + T, g.C).bfloat16()
    w_dev = ttnn.from_torch(window, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dn.verify_win_cur = lambda: w_dev
    # row 2: counts 2..4 copy window column 1 (block 21) into 9; row 5: counts 3..4 copy column 2 (block 32) into 12
    cp = torch.zeros(B, T, 2, dtype=torch.int64)
    cp[2, 1:] = torch.tensor([21, 9])
    cp[5, 2:] = torch.tensor([32, 12])
    st.capture_chain_copies([cp], [2, 5])
    assert st.chain_copies == [(2, 0, 21, 9, 1, (2, 3, 4)), (5, 0, 32, 12, 2, (3, 4))]
    res = torch.zeros(3, 32, dtype=torch.int32)
    res[RES_COUNTS, 2], res[RES_COUNTS, 5] = 3, 2  # row 2 selects its copy, row 5 does not
    st.run_chain_copies(result=ttnn.from_torch(res, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device))
    rec_at = lambda b: ttnn.to_torch(st._read_rec(dn, b)).reshape(1, g.Nv, 128, 128).float()
    conv_at = lambda b: ttnn.to_torch(st._read_carry(dn, b)).reshape(1, 3, g.C)
    assert torch.equal(rec_at(9), recs[21]), "row 2: selected copy, state"
    assert torch.equal(conv_at(9), window[2:3, 2:5]), "row 2: selected copy, conv rows after token 1"
    assert torch.equal(rec_at(12), recs[12]) and torch.equal(conv_at(12), carries[12]), "row 5: dst changed"
    assert torch.equal(rec_at(32), recs[32]) and torch.equal(rec_at(21), recs[21]), "a source changed"


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_copy_traces_match_the_eager_block_ops(device, geo_block):
    """_CopyTraces: the replayed whole-block copy, page-by-page copy (blocks of either phase), conv write
    from either window half and the chained copy with device-selected pages leave what the eager ops leave."""
    from models.demos.blackhole.qwen36.tt.spec_chain import RES_COUNTS

    g, block = geo_block
    st, dn, k, v = _state(device, g, block, 40)
    st.model = SimpleNamespace(_paged_kv_caches=[(k, v)], mesh_device=device)
    st._slot_layers, st.moves_counts, st.chain_copies = {0: dn}, None, []
    B = 8
    st.book = sbs.SpecBlockBook(st.layout, 1, B, T, "align", block)
    torch.manual_seed(7)
    recs = {b: torch.randn(1, g.Nv, 128, 128) for b in (7, 21, 12)}
    carries = {b: torch.randn(1, 3, g.C).bfloat16() for b in (7, 21, 12)}
    with sbs._PageTables(st) as pts:
        for b in recs:
            st._write_rec(
                dn, b, ttnn.from_torch(recs[b], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device), pts
            )
            rm = ttnn.from_torch(carries[b], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            st._write_carry(dn, b, rm, pts)
    windows = [torch.randn(B, 3 + T, g.C).bfloat16() for _ in range(2)]
    dn._verify_win_pair = [
        ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for w in windows
    ]
    dn._spec_win_par = 0
    dn.verify_win_cur = lambda: dn._verify_win_pair[dn._spec_win_par]
    zeros = lambda shape, dt: ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    dn.rec_state, dn.conv_carry = zeros((1, g.Nv, 128, 128), ttnn.float32), zeros((1, 3, g.C), ttnn.bfloat16)
    # a fused step's result buffer (row RES_COUNTS = accepted counts by engine row), as the engine hands it over
    res_dev = ttnn.from_torch(
        torch.zeros(3, 32, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    st.chain_results = [res_dev]
    ct = sbs._CopyTraces(st)
    ct.capture()
    ct.capture_restore()
    st._copy_traces = ct
    rec_at = lambda b: ttnn.to_torch(st._read_rec(dn, b)).reshape(1, g.Nv, 128, 128).float()
    conv_at = lambda b: ttnn.to_torch(st._read_carry(dn, b)).reshape(1, 3, g.C)
    # restore of a block into the scratch
    for b in (7, 12):
        st.book.set_prefill_blocks([torch.tensor([b])])
        st.restore(0)
        assert torch.equal(ttnn.to_torch(dn.rec_state).reshape(recs[b].shape).float(), recs[b]), f"restore {b}: state"
        assert torch.equal(ttnn.to_torch(dn.conv_carry).reshape(1, 3, g.C), carries[b]), f"restore {b}: conv"
    raw = lambda t, b: ttnn.to_torch(ttnn.slice(t, (b, 0, 0, 0), [b + 1] + [int(x) for x in list(t.shape)[1:]])).float()
    # bit-equal blocks (state bytes read as BF8 include nan, which torch.equal never matches)
    same = lambda a, b: torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    # whole-block copy: every byte of K and V
    st._copy_block(7, 9)
    assert same(raw(k, 9), raw(k, 7)) and same(raw(v, 9), raw(v, 7)), "whole-block copy"
    # page-by-page copies into blocks of both phases
    for dst in (10, 13):
        with sbs._PageTables(st) as pts:
            st._relayout(21, dst, pts)
        assert torch.equal(rec_at(dst), recs[21]) and torch.equal(conv_at(dst), carries[21]), f"21 -> {dst}"
    # conv rows of a window column, both window halves
    for par, (u, t, dst) in ((0, (2, 1, 15)), (1, (5, 3, 16))):
        dn._spec_win_par = par
        with sbs._PageTables(st) as pts:
            st.materialize(u, t, 0, dst, pts)
        assert torch.equal(conv_at(dst), windows[par][u : u + 1, t + 1 : t + 4]), f"window half {par}"
    dn._spec_win_par = 0
    # chained copies with the counts on the device (traced selection): row 2 selects its copy (column 1, block 21 -> 18),
    # row 5 does not (counts 3..4 copy column 2, block 32 -> 12; its count is 2)
    recs[32] = torch.randn(1, g.Nv, 128, 128)
    with sbs._PageTables(st) as pts:
        st._write_rec(
            dn, 32, ttnn.from_torch(recs[32], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device), pts
        )
    w = torch.zeros(B, T, dtype=torch.int64)
    w[2], w[5] = torch.tensor([20, 21, 22, 23]), torch.tensor([30, 31, 32, 33])
    st.book.set_step([w], torch.zeros(B, dtype=torch.int64), [2, 5])
    cp = torch.zeros(B, T, 2, dtype=torch.int64)
    cp[2, 1:] = torch.tensor([21, 18])
    cp[5, 2:] = torch.tensor([32, 12])
    st.capture_chain_copies([cp], [2, 5])
    res = torch.zeros(3, 32, dtype=torch.int32)
    res[RES_COUNTS, 2], res[RES_COUNTS, 5] = 3, 2
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(res, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), res_dev)
    assert ct.result_index(res_dev) == 0, "the result buffer's select trace"
    st.run_chain_copies(result=res_dev)
    assert torch.equal(rec_at(18), recs[21]), "chained copy: state"
    assert torch.equal(conv_at(18), windows[0][2:3, 2:5]), "chained copy: conv rows"
    assert torch.equal(rec_at(12), recs[12]) and torch.equal(
        conv_at(12), carries[12]
    ), "a block outside the copies changed"


def _layers(device, g, block, num_blocks, n):
    """A SpecGdnBlockState over ``n`` GDN layers of geometry g, layer i in group i with its own K and V buffers, each
    with a bound B=1 scratch (rec_state, conv_carry) and in-replay snapshot buffers of 3 slots."""
    sts = [_state(device, g, block, num_blocks) for _ in range(n)]
    st = sts[0][0]
    st.layers = [(i, s[1], i) for i, s in enumerate(sts)]
    st.layer_groups = {i: i for i in range(n)}
    st.G = n
    st.book = sbs.SpecBlockBook(st.layout, n, 8, T, "none")
    st.model = SimpleNamespace(_paged_kv_caches=[(s[2], s[3]) for s in sts], mesh_device=device)
    tile = lambda x, dt: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    for _, dn, _ in st.layers:
        dn.rec_state = tile(torch.zeros(1, g.Nv, 128, 128), ttnn.float32)
        dn.conv_carry = tile(torch.zeros(1, 3, g.C), ttnn.bfloat16)
        dn._snap_rec = tile(torch.randn(3, g.Nv, 128, 128), ttnn.float32)
        dn._snap_conv_rm = ttnn.from_torch(
            torch.randn(1, 9, g.C).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
    return st, sts


@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_block_io_kernel_matches_the_op_chains(device, geo_block):
    """GdnBlockIO (kernels/gdn_block_io.cpp): the scratch write, the snapshot write and the restore of two GDN layers in
    two groups move the same pages and bytes as _write_rec / _write_carry / _read_rec / _read_carry, into blocks of
    both phases, and leave every other block as it was."""
    from models.demos.blackhole.qwen36.tt.gdn_block_io import GdnBlockIO

    g, block = geo_block
    st, sts = _layers(device, g, block, 24, 2)
    io = GdnBlockIO(st, st.model._paged_kv_caches)
    assert io.reason() is None and io.zero_reason() is None
    torch.manual_seed(13)
    recs = [torch.randn(1, g.Nv, 128, 128) for _ in st.layers]
    carries = [torch.randn(1, 3, g.C).bfloat16() for _ in st.layers]
    for (_, dn, _), r, c in zip(st.layers, recs, carries):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(r, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), dn.rec_state)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dn.conv_carry)
    raw = lambda t, b: ttnn.to_torch(ttnn.slice(t, (b, 0, 0, 0), [b + 1] + [int(x) for x in list(t.shape)[1:]])).float()
    same = lambda a, b: torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    before = {(i, b): (raw(s[2], b), raw(s[3], b)) for i, s in enumerate(sts) for b in range(24)}
    # kernel into (6, 7) and (11, 10); the op chains into (8, 9) and (13, 12): same phase pairwise
    io.write([6, 7])
    io.write([11, 10])
    with sbs._PageTables(st) as pts:
        for (_, dn, gg), blocks in ((st.layers[0], (8, 13)), (st.layers[1], (9, 12))):
            for b in blocks:
                st._write_rec(dn, b, dn.rec_state, pts)
                rm = ttnn.to_layout(dn.conv_carry, ttnn.ROW_MAJOR_LAYOUT)
                st._write_carry(dn, b, rm, pts)
    pairs = {0: ((6, 8), (11, 13)), 1: ((7, 9), (10, 12))}
    for i, (_, dn, _) in enumerate(st.layers):
        k, v = sts[i][2], sts[i][3]
        for kb, eb in pairs[i]:
            assert torch.equal(ttnn.to_torch(st._read_rec(dn, kb)).reshape(recs[i].shape).float(), recs[i]), f"{i} {kb}"
            assert torch.equal(ttnn.to_torch(st._read_carry(dn, kb)).reshape(1, 3, g.C), carries[i]), f"{i} {kb}"
            assert same(raw(k, kb), raw(k, eb)) and same(raw(v, kb), raw(v, eb)), f"layer {i}: block {kb} vs {eb}"
    written = {0: {6, 8, 11, 13}, 1: {7, 9, 10, 12}}
    for (i, b), (bk, bv) in before.items():
        if b not in written[i]:
            assert same(raw(sts[i][2], b), bk) and same(raw(sts[i][3], b), bv), f"layer {i} block {b} changed"
    # snapshot slot 1 into (15, 14)
    io.write_snapshot([15, 14], 1)
    for (_, dn, _), b in zip(st.layers, (15, 14)):
        snap = ttnn.to_torch(dn._snap_rec).float()[1:2]
        rows = ttnn.to_torch(dn._snap_conv_rm)[:, 3:6]
        assert torch.equal(ttnn.to_torch(st._read_rec(dn, b)).reshape(snap.shape).float(), snap), f"snapshot {b}"
        assert torch.equal(ttnn.to_torch(st._read_carry(dn, b)).reshape(1, 3, g.C), rows), f"snapshot conv {b}"
    # restore (6, 7) into cleared scratch
    for _, dn, _ in st.layers:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.zeros(1, g.Nv, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), dn.rec_state
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.ones(1, 3, g.C).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            dn.conv_carry,
        )
    io.restore([6, 7])
    for (_, dn, _), r, c in zip(st.layers, recs, carries):
        assert torch.equal(ttnn.to_torch(dn.rec_state).reshape(r.shape).float(), r), "restore: state"
        assert torch.equal(ttnn.to_torch(dn.conv_carry).reshape(1, 3, g.C), c), "restore: conv"


@pytest.mark.parametrize("block", [2048, 1024])
def test_block_io_zero_matches_zero_blocks(device, block):
    """GdnBlockIO.zero: every listed block of every K and V buffer is zeroed (more ids than one program takes), every
    other block keeps its bytes."""
    from models.demos.blackhole.qwen36.tt.gdn_block_io import ZERO_MAX_BLOCKS, GdnBlockIO

    num_blocks = 48
    ones = lambda: ttnn.from_torch(
        torch.ones(num_blocks, 1, block, 256),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kv = [(ones(), ones()), (ones(), ones())]
    st = SimpleNamespace(mesh=device, layers=[], layout=None, G=0)
    io = GdnBlockIO(st, kv)
    assert io.zero_reason() is None
    listed = [1, 2, 3, 5, 8, 13, 21, 34, 40, 41, 42, 43, 44, 45, 46, 47] + list(range(22, 22 + ZERO_MAX_BLOCKS - 10))
    io.zero(listed)
    for n, (k, v) in enumerate(kv):
        for name, t in (("K", k), ("V", v)):
            host = ttnn.to_torch(t).float()
            for b in range(num_blocks):
                if b in listed:
                    assert not host[b].any(), f"{name}{n} block {b} was not zeroed"
                else:
                    assert torch.equal(host[b], torch.ones_like(host[b])), f"{name}{n} block {b} changed"


@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_block_io_window_matches_materialize(device, geo_block):
    """GdnBlockIO.window (the boundary conv writes): the conv rows after token t of row u's verify window land in block
    b's conv history of every layer of the entry's group, as materialize writes them; the other group's layer and
    the recurrent pages stay as they were."""
    from models.demos.blackhole.qwen36.tt.gdn_block_io import GdnBlockIO

    g, block = geo_block
    st, sts = _layers(device, g, block, 24, 2)
    io = GdnBlockIO(st, st.model._paged_kv_caches)
    B = 8
    torch.manual_seed(17)
    windows = [torch.randn(B, 3 + T, g.C).bfloat16() for _ in st.layers]
    for (_, dn, _), w in zip(st.layers, windows):
        dev = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dn.verify_win_cur = lambda dev=dev: dev
    assert io.window_reason() is None
    rec = torch.randn(1, g.Nv, 128, 128)
    rec_dev = ttnn.from_torch(rec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with sbs._PageTables(st) as pts:
        for _, dn, _ in st.layers:
            for b in (5, 6, 9):
                st._write_rec(dn, b, rec_dev, pts)
    conv_at = lambda dn, b: ttnn.to_torch(st._read_carry(dn, b)).reshape(3, g.C)
    before = {(i, b): conv_at(dn, b) for i, (_, dn, _) in enumerate(st.layers) for b in (5, 6, 9)}
    # (u, t, group, block): blocks of both phases, every window column, both groups
    entries = [(0, 0, 0, 5), (7, 3, 0, 6), (3, 1, 1, 9), (2, 2, 1, 6)]
    io.window(entries)
    for i, (_, dn, gg) in enumerate(st.layers):
        mine = {b: (u, t) for u, t, eg, b in entries if eg == gg}
        for b in (5, 6, 9):
            if b in mine:
                u, t = mine[b]
                assert torch.equal(conv_at(dn, b), windows[i][u, t + 1 : t + 4]), f"layer {i} block {b}"
            else:
                assert torch.equal(conv_at(dn, b), before[(i, b)]), f"layer {i} block {b} changed"
            got = ttnn.to_torch(st._read_rec(dn, b)).reshape(rec.shape).float()
            assert torch.equal(got, rec), f"layer {i} block {b}: recurrent pages changed"


@pytest.mark.parametrize("geo_block", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}")
def test_block_io_chain_selects_by_the_device_count(device, geo_block):
    """GdnBlockIO.chain (the chained copies): with the result buffer, an entry copies src's state and conv history (or
    the window rows after token t) into dst in every layer of its group only when row u's count is in its counts;
    without it, every entry runs. Layers of the other group and unselected destinations keep their bytes."""
    from models.demos.blackhole.qwen36.tt.gdn_block_io import GdnBlockIO
    from models.demos.blackhole.qwen36.tt.spec_chain import RES_COUNTS

    g, block = geo_block
    st, sts = _layers(device, g, block, 24, 2)
    io = GdnBlockIO(st, st.model._paged_kv_caches)
    B = 8
    torch.manual_seed(19)
    windows = [torch.randn(B, 3 + T, g.C).bfloat16() for _ in st.layers]
    for (_, dn, _), w in zip(st.layers, windows):
        dev = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dn.verify_win_cur = lambda dev=dev: dev
    blocks = (9, 10, 12, 13, 15, 16, 21)
    recs = {(i, b): torch.randn(1, g.Nv, 128, 128) for i in range(2) for b in blocks}
    carries = {(i, b): torch.randn(1, 3, g.C).bfloat16() for i in range(2) for b in blocks}
    with sbs._PageTables(st) as pts:
        for i, (_, dn, _) in enumerate(st.layers):
            for b in blocks:
                r = ttnn.from_torch(recs[(i, b)], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
                st._write_rec(dn, b, r, pts)
                c = ttnn.from_torch(carries[(i, b)], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                st._write_carry(dn, b, c, pts)
    rec_at = lambda dn, b: ttnn.to_torch(st._read_rec(dn, b)).reshape(1, g.Nv, 128, 128).float()
    conv_at = lambda dn, b: ttnn.to_torch(st._read_carry(dn, b)).reshape(1, 3, g.C)
    res = torch.zeros(3, 32, dtype=torch.int32)
    res[RES_COUNTS, 2], res[RES_COUNTS, 5], res[RES_COUNTS, 4] = 3, 2, 1
    res_dev = ttnn.from_torch(res, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # (u, group, src, dst, t, counts): selected from the window, not selected, selected from src's conv history
    entries = [(2, 0, 21, 9, 1, (2, 3, 4)), (5, 1, 13, 12, None, (3, 4)), (4, 1, 15, 10, None, (1,))]
    io.chain(entries, res_dev)
    (_, dn0, _), (_, dn1, _) = st.layers
    assert torch.equal(rec_at(dn0, 9), recs[(0, 21)]) and torch.equal(conv_at(dn0, 9), windows[0][2:3, 2:5])
    assert torch.equal(rec_at(dn1, 9), recs[(1, 9)]) and torch.equal(conv_at(dn1, 9), carries[(1, 9)]), "other group"
    assert torch.equal(rec_at(dn1, 12), recs[(1, 12)]) and torch.equal(conv_at(dn1, 12), carries[(1, 12)]), "unselected"
    assert torch.equal(rec_at(dn1, 10), recs[(1, 15)]) and torch.equal(conv_at(dn1, 10), carries[(1, 15)])
    assert torch.equal(rec_at(dn0, 10), recs[(0, 10)]), "other group"
    # host-selected: every entry runs
    io.chain([(5, 1, 13, 16, None, (3, 4)), (3, 0, 15, 13, 3, (4,))])
    assert torch.equal(rec_at(dn1, 16), recs[(1, 13)]) and torch.equal(conv_at(dn1, 16), carries[(1, 13)])
    assert torch.equal(rec_at(dn0, 13), recs[(0, 15)]) and torch.equal(conv_at(dn0, 13), windows[0][3:4, 4:7])
