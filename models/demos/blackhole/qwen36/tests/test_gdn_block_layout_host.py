# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the KV cache specs of Qwen36ForCausalLM and the place of the GDN state in a KV cache block
(tt/kv_cache_spec.py), on 8 DRAM banks (Blackhole P150, P300) and on 7 (P100a, Bin6 Blackhole Galaxy).

The spec tests need vLLM and skip without it; the geometry, packing and page index tests need only torch.
Qwen3.6-27B / Qwen3.8-27B text config: 64 layers, 16 full attention (4 KV heads x 256) and 48 GDN (16 key heads, 48
value heads, 128 x 128, conv kernel 4).
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tt.kv_cache_spec import (
    BF16_PAGE_BYTES,
    DRAM_BANKS_ENV,
    FP32_PAGE_BYTES,
    HIST_SLOTS,
    GdnDims,
    GdnLayoutError,
    choose_block_size,
    mesh_tp_degree,
    pack_conv_taps,
    pack_history_rows,
    plan_gdn_block_layout,
    planned_dram_banks,
    unpack_history_rows,
)

QWEN_27B = {
    "num_hidden_layers": 64,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "hidden_size": 5120,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}
# Qwen3.6-35B-A3B: 40 layers (10 attention), 2 KV heads, 32 GDN value heads.
QWEN_35B = {
    **QWEN_27B,
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "linear_num_value_heads": 32,
}

# (checkpoint, TP, bfloat8_b K / V) -> attention block size that holds one GDN state.
SUPPORTED = [
    (QWEN_27B, 8, False, 512),
    (QWEN_27B, 8, True, 1024),
    (QWEN_27B, 4, False, 1024),
    (QWEN_27B, 4, True, 2048),
    (QWEN_35B, 4, False, 1024),
    (QWEN_35B, 4, True, 2048),
]
IDS = ["27b-tp8-bf16", "27b-tp8-bf8", "27b-tp4-bf16", "27b-tp4-bf8", "35b-tp4-bf16", "35b-tp4-bf8"]
# The same on 7 DRAM banks, where a power-of-two block's tiles do not spread evenly over the banks.
SUPPORTED_7 = [
    (QWEN_27B, 8, False, 1024),
    (QWEN_27B, 8, True, 2048),
    (QWEN_27B, 4, False, 1024),
    (QWEN_27B, 4, True, 2048),
    (QWEN_35B, 4, False, 1024),
    (QWEN_35B, 4, True, 2048),
]
# (checkpoint, TP, bfloat8_b K / V, block size, DRAM banks) of every supported layout.
LAYOUTS = [(*c, 8) for c in SUPPORTED] + [(*c, 7) for c in SUPPORTED_7]
LAYOUT_IDS = [f"{i}-8banks" for i in IDS] + [f"{i}-7banks" for i in IDS]


@pytest.fixture(autouse=True)
def _planned_for_8_banks(monkeypatch):
    """The specs plan for QWEN36_DRAM_BANKS; tests that want 7 banks set it themselves."""
    monkeypatch.delenv(DRAM_BANKS_ENV, raising=False)


def _layout(tc, tp, bf8, banks=8, block_size=None):
    dims = GdnDims.from_text_config(tc, tp)
    kv_heads = max(1, tc["num_key_value_heads"] // tp)
    if block_size is None:
        return choose_block_size(kv_heads=kv_heads, head_dim=tc["head_dim"], kv_bf8=bf8, dims=dims, banks=banks)
    return block_size, plan_gdn_block_layout(
        block_size=block_size, kv_heads=kv_heads, head_dim=tc["head_dim"], kv_bf8=bf8, dims=dims, banks=banks
    )


@pytest.mark.parametrize("tc, tp, bf8, block, banks", LAYOUTS, ids=LAYOUT_IDS)
def test_block_size_is_the_smallest_that_holds_the_state(tc, tp, bf8, block, banks):
    got, layout = _layout(tc, tp, bf8, banks=banks)
    assert got == block and layout.banks == banks
    smaller = [b for b in (64, 128, 256, 512, 1024) if b < block]
    for b in smaller:
        with pytest.raises(GdnLayoutError):  # allow-pytest.raises: host test without the repo conftest
            _layout(tc, tp, bf8, banks=banks, block_size=b)


def test_tp8_bf8_1024_matches_the_operator_test_layout():
    """The layout test_gdn_decode_step_paged.py's KvBlocks checks on device: 4 heads in K, 2 heads in V, the history in
    the last three bfloat16 page rows of V; a block is 8.5 float32 pages per bank, so odd blocks start one page row
    later."""
    _, lay = _layout(QWEN_27B, 8, True, block_size=1024)
    assert (lay.heads_k, lay.heads_v, lay.bank_bytes, lay.hist_tiles, lay.hist_first) == (4, 2, 34816, 136, 112)
    assert not lay.fp32_slices_are_blocks
    assert [lay.first_fp32_page(b) for b in range(4)] == [0, 72, 136, 208]


def _used_ranges(lay, block):
    """[(buffer, bank, first byte, end byte)]: the pages one request's state uses in ``block`` (buffer 0 = K, 1 = V),
    each page at bank p % banks, row p // banks of its view."""
    used = []
    nb, hp = lay.banks, lay.dims.head_pages
    first = lay.first_fp32_page(block)
    for h in range(lay.dims.nv):
        buf, page0 = (0, first + hp * h) if h < lay.heads_k else (1, first + hp * (h - lay.heads_k))
        used += [
            (buf, p % nb, p // nb * FP32_PAGE_BYTES, (p // nb + 1) * FP32_PAGE_BYTES) for p in range(page0, page0 + hp)
        ]
    for p in range(lay.hist_page(block, 0), lay.hist_page(block, 0) + HIST_SLOTS * lay.dims.nv):
        used.append((1, p % nb, p // nb * BF16_PAGE_BYTES, (p // nb + 1) * BF16_PAGE_BYTES))
    return used


def _tile_blocks(lay, bank, lo, hi):
    """The blocks whose K / V tiles hold bytes [lo, hi) of ``bank``: tile row r of a bank is tile r * banks + bank of
    the buffer, and block b holds tiles [b * tiles, (b + 1) * tiles)."""
    rows = range(lo // lay.kv_tile_bytes, (hi - 1) // lay.kv_tile_bytes + 1)
    return {(r * lay.banks + bank) // lay.tiles for r in rows}


@pytest.mark.parametrize("tc, tp, bf8, block, banks", LAYOUTS, ids=LAYOUT_IDS)
def test_state_pages_stay_inside_their_block(tc, tp, bf8, block, banks):
    """Every state and history page of a block lies in tiles of that block on its bank, and no two pages of a block
    share a byte, for every block offset pattern: so blocks never share a byte. On 7 banks a block's tiles do not
    spread evenly, and the tile rows a block holds on only some banks stay unused."""
    _, lay = _layout(tc, tp, bf8, banks=banks)
    for b in range(2 * banks * FP32_PAGE_BYTES // BF16_PAGE_BYTES + 3):
        used = sorted(_used_ranges(lay, b))
        for buf, bank, lo, hi in used:
            assert _tile_blocks(lay, bank, lo, hi) == {b}, f"block {b}: bytes [{lo}, {hi}) of bank {bank} buffer {buf}"
        for (buf, bank, lo, hi), (buf2, bank2, lo2, _) in zip(used, used[1:]):
            assert (buf, bank) != (buf2, bank2) or hi <= lo2, f"block {b}: two pages share bank {bank} of buffer {buf}"


@pytest.mark.parametrize("tc, tp, bf8, block, banks", LAYOUTS, ids=LAYOUT_IDS)
def test_views_fit_the_buffers(tc, tp, bf8, block, banks):
    """The float32 and bfloat16 views need no more bytes per bank than the K / V buffer has: whole page rows up to the
    end of the last block's tiles on every bank."""
    _, lay = _layout(tc, tp, bf8, banks=banks)
    for n in (1, 5, 33):
        allocated = -(-n * lay.tiles // banks) * lay.kv_tile_bytes
        assert lay.fp32_pages(n) // banks * FP32_PAGE_BYTES <= allocated
        assert lay.fp32_pages(n) % banks == 0 and lay.fp32_pages(n) % lay.slice_pages == 0
        assert lay.hist_view_shape(n)[0] // banks * BF16_PAGE_BYTES <= allocated


@pytest.mark.parametrize("tc, tp, bf8, block, banks", LAYOUTS, ids=LAYOUT_IDS)
def test_index_follows_the_rows_blocks(tc, tp, bf8, block, banks):
    """state_page_idx column r names the pages of row r's block; padding rows name block 0; moving a request to
    another row only moves its column (no state moves)."""
    _, lay = _layout(tc, tp, bf8, banks=banks)
    num_blocks, nv, hp = 40, lay.dims.nv, lay.dims.head_pages
    idx = lay.index_host([7, 0, 12, 0], num_blocks)
    assert idx.shape == (2 * nv, 4) and idx.dtype == torch.int32
    p_k = lay.fp32_pages(num_blocks)
    for col, b in ((0, 7), (2, 12), (1, 0), (3, 0)):
        first = lay.first_fp32_page(b)
        want = [first + hp * h if h < lay.heads_k else p_k + first + hp * (h - lay.heads_k) for h in range(nv)]
        want += [lay.hist_page(b, h) for h in range(nv)]
        assert idx[:, col].tolist() == want
    moved = lay.index_host([12, 7, 0, 0], num_blocks)
    assert torch.equal(moved[:, 0], idx[:, 2]) and torch.equal(moved[:, 1], idx[:, 0])


@pytest.mark.parametrize("tc, tp, bf8, block, banks", LAYOUTS, ids=LAYOUT_IDS)
def test_write_tables_name_the_pages_the_index_reads(tc, tp, bf8, block, banks):
    """The paged_fill_cache tables that write a block's state address the same pages that gdn_decode_step's index
    reads: slices of the K (then V) view (half heads of 8 float32 pages on 8 banks, single pages on 7), one history
    page per table entry."""
    _, lay = _layout(tc, tp, bf8, banks=banks)
    num_blocks, nv, hp = 40, lay.dims.nv, lay.dims.head_pages
    sp = lay.slice_pages
    assert sp == (8 if banks == 8 else 1) and hp == 16
    for b in (1, 2, 7):
        idx = lay.index_host([b], num_blocks)[:, 0].tolist()
        tables = lay.write_tables(b)
        k_pages = [sp * s for s in tables[0]]
        assert k_pages == [idx[h] + sp * j for h in range(lay.heads_k) for j in range(hp // sp)]
        if lay.heads_v:
            p_k = lay.fp32_pages(num_blocks)
            v_pages = [sp * s + p_k for s in tables[1]]
            assert v_pages == [idx[h] + sp * j for h in range(lay.heads_k, nv) for j in range(hp // sp)]
        assert tables[-1] == [idx[nv + h] + s for h in range(nv) for s in range(HIST_SLOTS)]
        assert len(tables) == (3 if lay.heads_v else 2)


@pytest.mark.parametrize("banks", [8, 7])
def test_page_index_values_fit_the_views(banks):
    """The largest index value addresses the last pages of the float32 views (K view then V view, gdn_decode_step's
    two pools) and of the history view."""
    _, lay = _layout(QWEN_27B, 8, True, banks=banks)
    n = 33
    idx = lay.index_host([n - 1], n)
    nv, hp = lay.dims.nv, lay.dims.head_pages
    assert int(idx[:nv].max()) + hp <= 2 * lay.fp32_pages(n)
    assert int(idx[nv:].max()) + HIST_SLOTS <= lay.hist_view_shape(n)[0]
    assert lay.fp32_view_shape(n)[0] * lay.slice_pages == lay.fp32_pages(n)
    if banks == 8:
        assert lay.hist_view_shape(n)[0] == n * lay.hist_tiles


def test_unsupported_geometries_name_the_reason():
    dims = GdnDims.from_text_config(QWEN_27B, 8)
    # On 7 banks a 1024-token bfloat8_b block holds 8 or 9 whole float32 rows (56 or 63 pages) per buffer: K takes
    # no head (3 heads would leave the V heads' workspace copy off bank 0), and V cannot take all 6.
    with pytest.raises(GdnLayoutError, match="K holds 0 of 6 GDN value heads"):  # allow-pytest.raises: no conftest
        plan_gdn_block_layout(block_size=1024, kv_heads=1, head_dim=256, kv_bf8=True, dims=dims, banks=7)
    with pytest.raises(GdnLayoutError, match="not whole bfloat16 history pages"):  # allow-pytest.raises: see above
        plan_gdn_block_layout(block_size=512, kv_heads=1, head_dim=256, kv_bf8=True, dims=dims)
    with pytest.raises(GdnLayoutError, match="above the largest"):  # allow-pytest.raises: see above
        choose_block_size(kv_heads=1, head_dim=256, kv_bf8=False, dims=dims, min_block_size=4096)
    assert GdnDims.from_text_config(QWEN_27B, 2).decode_step_problems() == [
        "24 value heads per device > 16 (a and b share one 32-wide tile)"
    ]


def _pack_reference(rows, dims):
    """gdn_decode_step's packed head tiles from 4 rows [C] (``pack_head_tiles`` of the operator tests): tile (h, s)
    holds rows[s]'s channels [q_hk | k_hk | v_h] as 32-wide chunks, chunk c in tile rows 2c and 2c + 1."""
    rf, kd = dims.nv // dims.nk, dims.nk * dims.dk
    out = torch.zeros(dims.nv, 4, 32, 32)
    for h in range(dims.nv):
        hk = h // rf
        for s, r in enumerate(rows):
            chunks = torch.cat(
                [
                    r[hk * dims.dk : (hk + 1) * dims.dk],
                    r[kd + hk * dims.dk : kd + (hk + 1) * dims.dk],
                    r[2 * kd + h * dims.dv : 2 * kd + (h + 1) * dims.dv],
                ]
            ).reshape(-1, 32)
            out[h, s, 0 : 2 * len(chunks) : 2] = chunks
            out[h, s, 1 : 2 * len(chunks) : 2] = chunks
    return out


@pytest.mark.parametrize("tp", [4, 8])
def test_history_and_tap_packing(tp):
    dims = GdnDims.from_text_config(QWEN_27B, tp)
    gen = torch.Generator().manual_seed(tp)
    carry = torch.randn(dims.kconv - 1, dims.conv_dim, generator=gen)
    table = pack_history_rows(dims)
    packed = carry.reshape(-1, 32)[table].reshape(dims.nv, HIST_SLOTS, 32, 32)
    want = _pack_reference([torch.zeros(dims.conv_dim)] + list(carry), dims)
    used = 2 * (2 * dims.dk + dims.dv) // 32
    # Slot 0 and the rows past the head row are never read; the slots the operator reads must match.
    assert torch.equal(packed[:, 1:, :used], want[:, 1:, :used])
    taps = [torch.randn(dims.conv_dim, generator=gen) for _ in range(4)]
    assert torch.equal(pack_conv_taps(taps, dims), _pack_reference(taps, dims))


@pytest.mark.parametrize("tc, tp", [(QWEN_27B, 4), (QWEN_27B, 8), (QWEN_35B, 4)], ids=["27b-tp4", "27b-tp8", "35b-tp4"])
def test_history_unpack_reads_back_the_packed_carry(tc, tp):
    """A prefill that resumes from a block rebuilds its convolution carry from the packed history: unpacking the
    packed rows gives the carry back, also when the slot and rows the operator never reads hold other values."""
    dims = GdnDims.from_text_config(tc, tp)
    gen = torch.Generator().manual_seed(tp)
    carry = torch.randn(dims.kconv - 1, dims.conv_dim, generator=gen)
    packed = carry.reshape(-1, 32)[pack_history_rows(dims)].reshape(dims.nv, HIST_SLOTS, 32, 32)
    used = 2 * (2 * dims.dk + dims.dv) // 32
    packed[:, 0] = torch.randn(dims.nv, 32, 32, generator=gen)
    packed[:, :, used:] = torch.randn(dims.nv, HIST_SLOTS, 32 - used, 32, generator=gen)
    rows = packed.reshape(-1, 32)[unpack_history_rows(dims)]
    assert torch.equal(rows.reshape(dims.kconv - 1, dims.conv_dim), carry)


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_float32_pages_start_at_one_offset_except_bf8_tp8(tc, tp, bf8, block):
    """The first whole float32 page of a block starts at the block's first byte in every layout except bfloat8_b
    1024-token blocks at TP=8, where a block holds 8.5 float32 pages per bank: even blocks start on a page, odd blocks
    half a page later. A copy between two blocks must therefore address each block's own first page."""
    _, lay = _layout(tc, tp, bf8)
    leads = [lay.first_fp32_page(b) * FP32_PAGE_BYTES // lay.banks - b * lay.bank_bytes for b in range(8)]
    assert leads == ([0, 2048] * 4 if (tp, bf8) == (8, True) else [0] * 8)


@pytest.mark.parametrize("tc, tp, bf8, block, banks", LAYOUTS, ids=LAYOUT_IDS)
def test_block_views_start_on_bank_0(tc, tp, bf8, block, banks):
    """GdnBlockState reads and copies a block through ttnn.narrow, which needs the first page of the narrowed range on
    bank 0: the first state page of every block, the first page of its packed history, and the page of the batch-1
    workspace where restore puts the V heads. The state starts on a whole slice of the float32 views."""
    _, lay = _layout(tc, tp, bf8, banks=banks)
    for b in range(2 * lay.banks + 3):
        assert lay.first_fp32_page(b) % lay.banks == 0
        assert lay.first_fp32_page(b) % lay.slice_pages == 0
        assert lay.hist_page(b, 0) % lay.banks == 0
    if lay.heads_v:
        assert lay.state_slices(lay.heads_k) * lay.slice_pages % lay.banks == 0


def test_7_bank_layouts():
    """On 7 banks at TP=8, K holds every GDN value head and V only the history; at TP=4 K holds 7 heads, so the V
    heads start at workspace page 112, on bank 0."""
    want = {
        (8, False): (1024, 6),
        (8, True): (2048, 6),
        (4, False): (1024, 7),
        (4, True): (2048, 7),
    }
    for (tp, bf8), (block, heads_k) in want.items():
        got, lay = _layout(QWEN_27B, tp, bf8, banks=7)
        assert (got, lay.heads_k, lay.slice_pages, lay.slice_shape) == (block, heads_k, 1, [32, 32])
        assert not lay.even_spread and not lay.fp32_slices_are_blocks and not lay.hist_slices_are_blocks
        assert not lay.workspace_spreads_evenly
    _, lay = _layout(QWEN_27B, 8, True, banks=7)
    assert lay.fp32_view_shape(10) == [lay.fp32_pages(10), 1, 32, 32]
    assert lay.workspace_view_shape() == [96, 1, 32, 32] and lay.workspace_rows_shape() == [1, 1, 96 * 32, 32]


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_8_bank_layouts_keep_the_even_spread_offsets(tc, tp, bf8, block):
    """With an even spread block b owns bytes [b * s, (b + 1) * s) of every bank, s = tiles / banks * tile bytes: the
    state starts at the first whole float32 row from b * s, the history at hist_first bfloat16 pages after the
    block's first bfloat16 page, and the views hold the half-head slices of 8 pages."""
    _, lay = _layout(tc, tp, bf8)
    s, n = lay.bank_bytes, lay.banks
    assert lay.even_spread and lay.slice_pages == 8 and lay.slice_shape == [64, 128]
    for b in range(20):
        assert lay.owned_bytes(b) == (b * s, (b + 1) * s)
        assert lay.first_fp32_page(b) == n * -(-b * s // FP32_PAGE_BYTES)
        assert lay.hist_page(b, 1) == b * lay.hist_tiles + lay.hist_first + HIST_SLOTS
    assert lay.fp32_view_shape(9) == [n * (9 * s // FP32_PAGE_BYTES) // 8, 1, 64, 128]
    assert lay.hist_view_shape(9) == [9 * lay.hist_tiles, 1, 32, 32]
    assert lay.workspace_view_shape() == [2 * lay.dims.nv, 1, 64, 128]
    assert lay.workspace_rows_shape() == [1, 1, lay.dims.nv * 128, 128]
    assert lay.state_writes(3) == [(0, lay.first_fp32_page(3) // 8, 0, 2 * lay.heads_k)] + (
        [(1, lay.first_fp32_page(3) // 8, 2 * lay.heads_k, 2 * lay.dims.nv)] if lay.heads_v else []
    )


def test_planned_dram_banks(monkeypatch):
    assert planned_dram_banks() == 8
    monkeypatch.setenv(DRAM_BANKS_ENV, "7")
    assert planned_dram_banks() == 7
    for bad in ("0", "seven", "-7"):
        monkeypatch.setenv(DRAM_BANKS_ENV, bad)
        with pytest.raises(ValueError, match="not a positive DRAM bank count"):  # allow-pytest.raises: no conftest
            planned_dram_banks()


def _devices(n):
    return ",".join(str(i) for i in range(n))


@pytest.mark.parametrize(
    "mesh, visible, tp",
    [
        ("P150x8", None, 8),
        ("P150x4", None, 4),
        ("(1, 8)", None, 8),
        ("(2, 4)", None, 4),
        ("P150x8", "0,1,2,3", 4),
        ("BH-Galaxy", None, 4),  # one engine opens the (8, 4) grid
        ("BH-Galaxy", _devices(8), 8),  # DP=4 rank: (1, 8)
        ("BH-Galaxy", _devices(16), 8),  # DP=2 rank: (2, 8)
        ("BH-Galaxy", _devices(4), 4),  # DP=8 rank: (1, 4)
        ("(4, 8)", _devices(8), 8),
    ],
)
def test_mesh_tp_degree(mesh, visible, tp):
    assert mesh_tp_degree(mesh, visible or "") == tp


@pytest.mark.parametrize(
    "mesh, parent, devices", [("BH-Galaxy", (4, 8), 32), ("(4, 8)", (4, 8), 32), ("P150x8", (1, 8), 8)]
)
def test_mesh_tp_degree_matches_the_plugins_standard_dp_ranks(mesh, parent, devices):
    """A standard DP rank sees its submesh's devices in TT_VISIBLE_DEVICES; its TP degree is the submesh's columns."""
    dp_submeshes = pytest.importorskip("vllm_tt_plugin.utils.dp_submeshes")
    for dp in (2, 4, 8):
        _, rank_mesh = dp_submeshes.dp_submesh_shapes(parent, dp)
        assert mesh_tp_degree(mesh, _devices(devices // dp)) == rank_mesh[1], f"{mesh} DP={dp}"


@pytest.mark.parametrize(
    "mesh, visible, tp", [("P150x8", "", 8), ("BH-Galaxy", _devices(8), 8), ("(4, 8)", _devices(8), 8), ("", "", 4)]
)
def test_spec_plan_tp_is_mesh_tp_degree(monkeypatch, mesh, visible, tp):
    """The MTP class's spec_plan sizes its state at mesh_tp_degree, and at TP=4 without a MESH_DEVICE."""
    qwen36_vllm = pytest.importorskip("models.demos.blackhole.qwen36.tt.qwen36_vllm")
    monkeypatch.setenv("MESH_DEVICE", mesh)
    monkeypatch.setenv("TT_VISIBLE_DEVICES", visible)
    assert qwen36_vllm._mesh_tp() == tp


def test_mesh_tp_degree_needs_mesh_device():
    with pytest.raises(ValueError, match="set MESH_DEVICE"):  # allow-pytest.raises: host test without the repo conftest
        mesh_tp_degree("", "")


def test_plugin_layout_shares_slots_like_vllm():
    """Layer slot j holds attention layer j and the GDN layer at position j of every GDN group."""
    attention = [i % 4 == 3 for i in range(16)]
    dims = GdnDims.from_text_config(QWEN_27B, 8)
    lay = PluginLayout(attention, 1, 256, dims, False, num_blocks=10)
    assert lay.gdn_groups == [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]]
    slots = {}
    for i, entry in enumerate(lay.per_layer_specs):
        slots.setdefault(entry[2], []).append(i)
    assert sorted(len(v) for v in slots.values()) == [4] * 4


# --------------------------------------------------------------------------------------------- vLLM specs and groups


@pytest.fixture(scope="module")
def vllm_kv():
    return (
        pytest.importorskip("vllm.v1.kv_cache_interface"),
        pytest.importorskip("vllm.v1.core.kv_cache_utils"),
    )


@pytest.mark.parametrize("tc, tp, bf8, block, banks", LAYOUTS, ids=LAYOUT_IDS)
def test_specs(vllm_kv, monkeypatch, tc, tp, bf8, block, banks):
    """The specs plan for QWEN36_DRAM_BANKS (8 when unset)."""
    from models.demos.blackhole.qwen36.tt.kv_cache_spec import qwen36_kv_cache_spec

    kv_iface, _ = vllm_kv
    if banks != 8:
        monkeypatch.setenv(DRAM_BANKS_ENV, str(banks))
    specs = qwen36_kv_cache_spec(tc, tp, kv_bf8=bf8, mamba_block_size=65000, min_block_size=64)
    attn = {n: s for n, s in specs.items() if isinstance(s, kv_iface.FullAttentionSpec)}
    gdn = {n: s for n, s in specs.items() if isinstance(s, kv_iface.MambaSpec)}
    n_layers = tc["num_hidden_layers"]
    assert len(attn) == n_layers // 4 and len(gdn) == n_layers - n_layers // 4
    assert [int(n.split(".")[2]) for n in specs] == list(range(n_layers))
    a, g = next(iter(attn.values())), next(iter(gdn.values()))
    dims = GdnDims.from_text_config(tc, tp)
    assert a.block_size == block and a.num_heads == 1 and a.head_size == 256
    assert a.page_size_bytes == 2 * (block * 256 // 1024) * (1088 if bf8 else 2048)
    assert g.shapes == ((dims.nv, 4, 32, 32), (dims.nv, 128, 128)) and g.dtypes == (torch.bfloat16, torch.float32)
    assert g.real_page_size_bytes == dims.nv * (4 * 2048 + 128 * 128 * 4) <= a.page_size_bytes == g.page_size_bytes
    assert g.mamba_cache_mode == "none" and g.num_speculative_blocks == 0
    # A multiple of the attention block, at least the requested state block size.
    assert g.block_size % block == 0 and 65000 <= g.block_size < 65000 + block


def test_specs_reject_unsupported_meshes(vllm_kv):
    from models.demos.blackhole.qwen36.tt.kv_cache_spec import qwen36_kv_cache_spec

    with pytest.raises(ValueError, match="48 value heads per device > 16"):  # allow-pytest.raises: no repo conftest
        qwen36_kv_cache_spec(QWEN_27B, 1, kv_bf8=False, mamba_block_size=4096)
    with pytest.raises(ValueError, match="24 value heads per device > 16"):  # allow-pytest.raises: see above
        qwen36_kv_cache_spec(QWEN_27B, 2, kv_bf8=False, mamba_block_size=4096)
    # Qwen3.6-35B-A3B at TP=2 fits the operator (16 value heads per device); it is untested, not refused.
    qwen36_kv_cache_spec(QWEN_35B, 2, kv_bf8=False, mamba_block_size=4096)
    tiny = {**QWEN_35B, "linear_num_key_heads": 2, "linear_num_value_heads": 4}
    with pytest.raises(ValueError, match="single-device GDN layer"):  # allow-pytest.raises: see above
        qwen36_kv_cache_spec(tiny, 1, kv_bf8=False, mamba_block_size=4096)


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED[:4], ids=IDS[:4])
def test_vllm_groups_share_layer_slots(vllm_kv, tc, tp, bf8, block):
    """vLLM's own grouping makes one attention group and three GDN groups of 16 layers with one page size, in the
    groups that PluginLayout (the device tests' stand-in for the plugin) forms."""
    from models.demos.blackhole.qwen36.tt.kv_cache_spec import qwen36_kv_cache_spec

    kv_iface, kv_utils = vllm_kv
    specs = qwen36_kv_cache_spec(tc, tp, kv_bf8=bf8, mamba_block_size=32768)
    groups = kv_utils._get_kv_cache_groups_uniform_page_size(specs)
    attn_groups = [g for g in groups if isinstance(g.kv_cache_spec, kv_iface.FullAttentionSpec)]
    gdn_groups = [g for g in groups if isinstance(g.kv_cache_spec, kv_iface.MambaSpec)]
    assert len(attn_groups) == 1 and len(gdn_groups) == 3
    assert [len(g.layer_names) for g in groups] == [16] * 4
    assert len({g.kv_cache_spec.page_size_bytes for g in groups}) == 1
    dims = GdnDims.from_text_config(tc, tp)
    stand_in = PluginLayout([i % 4 == 3 for i in range(64)], 1, 256, dims, bf8, num_blocks=8)
    assert [tuple(g.layer_names) for g in gdn_groups] == stand_in.group_names


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_align_specs(vllm_kv, tc, tp, bf8, block):
    """Mode "align": the state blocks have the attention block size (the plugin requires it), the MambaSpec asks for
    one prefill checkpoint block on block boundaries, and vLLM's grouping keeps one page size."""
    from models.demos.blackhole.qwen36.tt.kv_cache_spec import qwen36_kv_cache_spec

    kv_iface, kv_utils = vllm_kv
    specs = qwen36_kv_cache_spec(tc, tp, kv_bf8=bf8, mamba_block_size=None, mamba_cache_mode="align")
    gdn = [s for s in specs.values() if isinstance(s, kv_iface.MambaSpec)]
    attn = [s for s in specs.values() if isinstance(s, kv_iface.FullAttentionSpec)]
    assert {s.block_size for s in attn} == {s.block_size for s in gdn} == {block}
    g = gdn[0]
    assert (g.mamba_cache_mode, g.num_prefill_checkpoint_blocks, g.prefill_checkpoint_alignment) == ("align", 1, block)
    groups = kv_utils._get_kv_cache_groups_uniform_page_size(specs)
    assert len({grp.kv_cache_spec.page_size_bytes for grp in groups}) == 1
