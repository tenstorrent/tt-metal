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


def _state(device, g, block, num_blocks, layout):
    """A SpecGdnBlockState over one GDN layer of geometry g, bound to fresh BF8 K and V buffers."""
    k, v = _kv(device, num_blocks, block), _kv(device, num_blocks, block)
    nb = int(device.dram_grid_size().x)
    S = sbs.kv_block_bytes_per_bank(k.shape, "bfloat8_b", nb)
    lay = sbs.SpecBlockLayout(S, nb, num_blocks, g.Nv, sbs.conv_carry_pages(4, g.C), layout)
    half = [lay.pool1_pages // sbs.HALF_PAGES, 1, 64, 128]
    views = sbs._LayerViews(
        rec1=ttnn.experimental.alias_view(k, half, ttnn.float32, check_slices=False),
        rec2=ttnn.experimental.alias_view(v, half, ttnn.float32, check_slices=False),
        conv=ttnn.experimental.alias_view(v, [lay.conv_pages_total, 1, 32, 32], ttnn.bfloat16, check_slices=False),
        conv_k=ttnn.experimental.alias_view(k, [lay.conv_pages_total, 1, 32, 32], ttnn.bfloat16, check_slices=False),
        group=0,
        two_pool=lay.heads_in_k < g.Nv,
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


# (geometry, block, layout): TP=4 two pools at 2048 (the QuietBox runs), TP=8 two pools at 1024 (Galaxy primary),
# TP=8 K only at 2048 (option 1)
CASES = [(op.GEOS[1], 2048, "two_pool"), (op.GEOS[0], 1024, "two_pool"), (op.GEOS[0], 2048, "k_only")]


# the deployment's gdn_spec_step options: QWEN36_SPEC_STEP_VALUE_BLOCKS=4, QWEN36_SPEC_STEP_OPT=250 and
# QWEN36_SPEC_STEP_NARROW_GR=1 (kernel_opts bit 1024)
STEP_OPTS = {
    "default": {},
    "vb4_opt250": dict(value_blocks=4, kernel_opts=250),
    "vb4_opt1274": dict(value_blocks=4, kernel_opts=1274),
}


@pytest.mark.parametrize("step_opts", list(STEP_OPTS))
@pytest.mark.parametrize("geo_block_layout", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}_{c[2]}")
@pytest.mark.parametrize("B,hold", [(1, ()), (4, (2,)), (8, ())])
def test_pages_match_ring(device, geo_block_layout, B, hold, step_opts):
    g, block, layout = geo_block_layout
    so = STEP_OPTS[step_opts]
    num_blocks = 48
    c = op.Case(device, g, B, T, seed=7 + B, hold=hold, par=B & 1, R=op.round_up32(B * T), mode=op.MODES[2])
    out_ref, ring_ref, _, win_ref = c.snapshot(c.run(**so))
    c.reseed()
    st, dn, k, v = _state(device, g, block, num_blocks, layout)
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


@pytest.mark.parametrize("geo_block_layout", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}_{c[2]}")
def test_conv_history_round_trip(device, geo_block_layout):
    """_write_carry / _read_carry, and materialize from a conv window: exact bf16 rows."""
    g, block, layout = geo_block_layout
    st, dn, k, v = _state(device, g, block, 32, layout)
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
@pytest.mark.parametrize("geo_block_layout", [CASES[0], CASES[1]], ids=lambda c: f"{c[0].id}_b{c[1]}_{c[2]}")
def test_traced_time_ring_vs_pages(device, geo_block_layout):
    """Traced time per call at B=8, T=4, the model's arithmetic: the default ring vs the KV-buffer pages."""
    g, block, layout = geo_block_layout
    N, REPS, B = 48, 10, 8
    if B * g.Nv > op.num_cores(device):
        pytest.skip("over the grid")
    c = op.Case(device, g, B, T, seed=23, par=0, R=op.round_up32(B * T), mode=op.MODES[2])
    ring, ctrl = c.ring, c.ctrl
    st, dn, k, v = _state(device, g, block, 64, layout)
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
        f"\n[{g.id} block {block} {layout}] gdn_spec_step B={B} T={T}: ring {r_min:.1f} us (median {r_med:.1f}), "
        f"KV-buffer pages {p_min:.1f} us (median {p_med:.1f}), {100 * (p_min - r_min) / r_min:+.2f}%"
    )


@pytest.mark.parametrize("geo_block_layout", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}_{c[2]}")
def test_block_copies_keep_the_state(device, geo_block_layout):
    """SpecGdnBlockState.copy_blocks: a whole-block copy between blocks of the same phase and a page-by-page copy
    between blocks of different phase (BF8 1024: odd and even blocks) both leave the source's state at the
    destination's layout pages."""
    g, block, layout = geo_block_layout
    st, dn, k, v = _state(device, g, block, 32, layout)
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
    # the source is the bound prefill scratch in the model: the write must leave it allocated (k_only: one pool)
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


@pytest.mark.parametrize("geo_block_layout", CASES, ids=lambda c: f"{c[0].id}_b{c[1]}_{c[2]}")
def test_chained_copies_select_dst_by_the_device_count(device, geo_block_layout):
    """run_chain_copies with the counts on the device (a chained step's state_chain_copies_per_layer): a row whose
    count selects its copy gets the source's state and the pending verify's conv rows of that column at dst; a row
    whose count does not keeps dst as it was (the copy lands in the null block 0)."""
    from models.demos.blackhole.qwen36.tt.spec_chain import RES_COUNTS

    g, block, layout = geo_block_layout
    st, dn, k, v = _state(device, g, block, 40, layout)
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
