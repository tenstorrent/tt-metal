# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the KV cache specs of Qwen36ForCausalLM and the place of the GDN state in a KV cache block
(tt/kv_cache_spec.py).

The spec tests need vLLM and skip without it; the geometry, packing and page index tests need only torch.
Qwen3.6-27B / Qwen3.8-27B text config: 64 layers, 16 full attention (4 KV heads x 256) and 48 GDN (16 key heads, 48
value heads, 128 x 128, conv kernel 4).
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tt.kv_cache_spec import (
    BF16_PAGE_BYTES,
    FP32_PAGE_BYTES,
    HIST_SLOTS,
    GdnDims,
    GdnLayoutError,
    choose_block_size,
    mesh_tp_degree,
    pack_conv_taps,
    pack_history_rows,
    plan_gdn_block_layout,
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


def _layout(tc, tp, bf8, banks=8, block_size=None):
    dims = GdnDims.from_text_config(tc, tp)
    kv_heads = max(1, tc["num_key_value_heads"] // tp)
    if block_size is None:
        return choose_block_size(kv_heads=kv_heads, head_dim=tc["head_dim"], kv_bf8=bf8, dims=dims, banks=banks)
    return block_size, plan_gdn_block_layout(
        block_size=block_size, kv_heads=kv_heads, head_dim=tc["head_dim"], kv_bf8=bf8, dims=dims, banks=banks
    )


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_block_size_is_the_smallest_that_holds_the_state(tc, tp, bf8, block):
    got, layout = _layout(tc, tp, bf8)
    assert got == block
    smaller = [b for b in (64, 128, 256, 512, 1024) if b < block]
    for b in smaller:
        with pytest.raises(GdnLayoutError):  # allow-pytest.raises: host test without the repo conftest
            _layout(tc, tp, bf8, block_size=b)


def test_tp8_bf8_1024_matches_the_operator_test_layout():
    """The layout test_gdn_decode_step_paged.py's KvBlocks checks on device: 4 heads in K, 2 heads in V, the history in
    the last three bfloat16 page rows of V; a block is 8.5 float32 pages per bank, so odd blocks start one page row
    later."""
    _, lay = _layout(QWEN_27B, 8, True, block_size=1024)
    assert (lay.heads_k, lay.heads_v, lay.bank_bytes, lay.hist_tiles, lay.hist_first) == (4, 2, 34816, 136, 112)
    assert not lay.fp32_slices_are_blocks
    assert [lay.first_fp32_page(b) for b in range(4)] == [0, 72, 136, 208]


def _owned_bytes(lay, block):
    """{(buffer, bank): set of byte offsets} one request's state uses in ``block`` (buffer 0 = K, 1 = V)."""
    used = {}
    nb, hp = lay.banks, lay.dims.head_pages
    first = lay.first_fp32_page(block)
    for h in range(lay.dims.nv):
        buf, page0 = (0, first + hp * h) if h < lay.heads_k else (1, first + hp * (h - lay.heads_k))
        for p in range(page0, page0 + hp):
            row = p // nb
            used.setdefault((buf, p % nb), set()).update(range(row * FP32_PAGE_BYTES, (row + 1) * FP32_PAGE_BYTES))
    for h in range(lay.dims.nv):
        for s in range(HIST_SLOTS):
            p = lay.hist_page(block, h) + s
            row = p // nb
            bytes_ = set(range(row * BF16_PAGE_BYTES, (row + 1) * BF16_PAGE_BYTES))
            assert not bytes_ & used.get((1, p % nb), set()), f"block {block}: history page {p} overlaps a state page"
            used.setdefault((1, p % nb), set()).update(bytes_)
    return used


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_state_pages_stay_inside_their_block(tc, tp, bf8, block):
    """Every state and history page of a block lies inside the bytes the block owns on its bank, and blocks never
    share a byte, for every block offset pattern."""
    _, lay = _layout(tc, tp, bf8)
    blocks = range(2 * FP32_PAGE_BYTES // BF16_PAGE_BYTES + 3)
    seen = {}
    for b in blocks:
        for key, offsets in _owned_bytes(lay, b).items():
            lo, hi = b * lay.bank_bytes, (b + 1) * lay.bank_bytes
            assert min(offsets) >= lo and max(offsets) < hi, f"block {b} uses bytes outside [{lo}, {hi}) of {key}"
            assert not offsets & seen.get(key, set()), f"block {b} shares bytes with another block on {key}"
            seen.setdefault(key, set()).update(offsets)


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_index_follows_the_rows_blocks(tc, tp, bf8, block):
    """state_page_idx column r names the pages of row r's block; padding rows name block 0; moving a request to
    another row only moves its column (no state moves)."""
    _, lay = _layout(tc, tp, bf8)
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


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_write_tables_name_the_pages_the_index_reads(tc, tp, bf8, block):
    """The paged_fill_cache tables that write a block's state address the same pages that gdn_decode_step's index
    reads: half-head slices of 8 float32 pages of the K (then V) view, one history page per table entry."""
    _, lay = _layout(tc, tp, bf8)
    num_blocks, nv, hp = 40, lay.dims.nv, lay.dims.head_pages
    for b in (1, 2, 7):
        idx = lay.index_host([b], num_blocks)[:, 0].tolist()
        tables = lay.write_tables(b)
        k_pages = [8 * s for s in tables[0]]
        assert k_pages == [idx[h] + 8 * j for h in range(lay.heads_k) for j in range(2)]
        if lay.heads_v:
            p_k = lay.fp32_pages(num_blocks)
            v_pages = [8 * s + p_k for s in tables[1]]
            assert v_pages == [idx[h] + 8 * j for h in range(lay.heads_k, nv) for j in range(2)]
        assert tables[-1] == [idx[nv + h] + s for h in range(nv) for s in range(HIST_SLOTS)]
        assert len(tables) == (3 if lay.heads_v else 2) and hp == 16


def test_page_index_values_fit_the_views():
    """The largest index value addresses the last pages of the float32 views (K view then V view, gdn_decode_step's
    two pools) and of the history view."""
    _, lay = _layout(QWEN_27B, 8, True)
    n = 33
    idx = lay.index_host([n - 1], n)
    nv, hp = lay.dims.nv, lay.dims.head_pages
    assert int(idx[:nv].max()) + hp <= 2 * lay.fp32_pages(n)
    assert int(idx[nv:].max()) + HIST_SLOTS <= n * lay.hist_tiles
    assert lay.fp32_view_shape(n)[0] * 8 == lay.fp32_pages(n)


def test_unsupported_geometries_name_the_reason():
    dims = GdnDims.from_text_config(QWEN_27B, 8)
    with pytest.raises(GdnLayoutError, match="do not spread evenly over 7"):  # allow-pytest.raises: no repo conftest
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


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_block_views_start_on_bank_0(tc, tp, bf8, block):
    """GdnBlockState reads and copies a block through ttnn.narrow, which needs the first page of the narrowed range on
    bank 0: the first state page of every block and the first page of its packed history."""
    _, lay = _layout(tc, tp, bf8)
    for b in range(lay.banks + 3):
        assert lay.first_fp32_page(b) % lay.banks == 0
        assert lay.hist_page(b, 0) % lay.banks == 0


@pytest.mark.parametrize(
    "mesh, visible, tp",
    [("P150x8", None, 8), ("P150x4", None, 4), ("(1, 8)", None, 8), ("(2, 4)", None, 4), ("P150x8", "0,1,2,3", 4)],
)
def test_mesh_tp_degree(mesh, visible, tp):
    assert mesh_tp_degree(mesh, visible or "") == tp


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


@pytest.mark.parametrize("tc, tp, bf8, block", SUPPORTED, ids=IDS)
def test_specs(vllm_kv, tc, tp, bf8, block):
    from models.demos.blackhole.qwen36.tt.kv_cache_spec import qwen36_kv_cache_spec

    kv_iface, _ = vllm_kv
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
