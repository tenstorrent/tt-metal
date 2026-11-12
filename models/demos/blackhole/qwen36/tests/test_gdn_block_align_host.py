# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): prefix restore, prefill checkpoints and decode block crossings of the GDN state in vLLM KV
cache blocks (mamba_cache_mode "align", tt/gdn_block_state.py and Qwen36ForCausalLM.prefill_forward).

The block-state tests run GdnBlockState's own read and copy code on an emulation of the interleaved DRAM buffers:
``ttnn.experimental.alias_view``, ``ttnn.narrow`` and ``ttnn.copy`` are replaced by functions that address pages as
the device does (page q of a buffer on bank q % 8, at offset (q // 8) * page size), so a block's float32 pages start
half a page later on odd bfloat8_b 1024-token blocks, as on the device. The packed history is read back through a
stand-in for to_layout / reshape / embedding on host tensors (row-major tiles).
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
from models.demos.blackhole.qwen36.tt.kv_cache_spec import (
    HIST_SLOTS,
    GdnDims,
    pack_history_rows,
    plan_gdn_block_layout,
    unpack_history_rows,
)

QWEN_27B_TP8 = GdnDims(nk=2, nv=6, dk=128, dv=128, kconv=4)
ATTENTION = [i % 4 == 3 for i in range(16)]  # 4 attention layers (4 layer slots), 12 GDN layers in 3 groups
NUM_BLOCKS = 12
BMAX = 4
BANKS = 8


# ---------------------------------------------------------------------------------------- interleaved DRAM emulation


@dataclass
class Buf:
    """One interleaved DRAM buffer: the same byte range on each bank."""

    banks: list


@dataclass
class View:
    """A device tensor on Buf: page q at bank q % BANKS, byte offset base + (q // BANKS) * page."""

    buf: Buf
    base: int
    page: int
    shape: tuple

    @property
    def pages(self):
        return math.prod(self.shape[:-2]) * (self.shape[-2] // 32) * (self.shape[-1] // 32)

    def _at(self, q):
        return self.buf.banks[q % BANKS], self.base + (q // BANKS) * self.page

    def read(self, q):
        bank, off = self._at(q)
        assert off + self.page <= len(bank), "read past the buffer"
        return bytes(bank[off : off + self.page])

    def write(self, q, data):
        bank, off = self._at(q)
        assert len(data) == self.page and off + self.page <= len(bank), "write past the buffer"
        bank[off : off + self.page] = data


@dataclass
class Host:
    """A host-side tensor standing in for a row-major device tensor or the conv carry workspace."""

    t: torch.Tensor


def _page_bytes():
    return {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}


def _alias_view(t, shape, dtype, check_slices=True):
    view = View(t.buf, t.base, _page_bytes()[dtype], tuple(int(d) for d in shape))
    assert view.pages * view.page <= t.pages * t.page, "alias_view larger than its input"
    return view


def _narrow(t, dim, start, length):
    assert all(t.shape[i] == 1 for i in range(dim)), "narrow: dims before dim must be 1"
    assert dim < len(t.shape) - 2, "the emulation narrows outer dims only"
    per = t.pages // t.shape[dim]
    first = start * per
    assert first % BANKS == 0, "narrow: the first page must be on bank 0"
    assert start + length <= t.shape[dim]
    shape = list(t.shape)
    shape[dim] = length
    return View(t.buf, t.base + first // BANKS * t.page, t.page, tuple(shape))


def _copy(src, dst):
    if isinstance(src, Host):
        dst.t[...] = src.t.reshape(dst.t.shape)
        return
    assert src.shape == dst.shape and src.page == dst.page
    for q in range(src.pages):
        dst.write(q, src.read(q))


def _tiles_of(view):
    """[pages, 32, 32] bfloat16 of a bfloat16 View (the emulation stores a tile row-major)."""
    raw = b"".join(view.read(q) for q in range(view.pages))
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(-1, 32, 32)


def _to_layout(t, layout, memory_config=None):
    if isinstance(t, View):
        return Host(_tiles_of(t).reshape(t.shape))
    return t


def _reshape(t, shape):
    return Host(t.t.reshape(shape))


def _embedding(table, rows, layout=None, memory_config=None):
    return Host(rows.t[table.t.reshape(-1)].reshape(1, -1, rows.t.shape[-1]))


@pytest.fixture
def dram(monkeypatch):
    """Route the ttnn calls of GdnBlockState's read and copy paths to the emulation."""
    monkeypatch.setattr(ttnn.experimental, "alias_view", _alias_view, raising=False)
    monkeypatch.setattr(ttnn, "narrow", _narrow)
    monkeypatch.setattr(ttnn, "copy", _copy)
    monkeypatch.setattr(ttnn, "to_layout", _to_layout)
    monkeypatch.setattr(ttnn, "reshape", _reshape)
    monkeypatch.setattr(ttnn, "embedding", _embedding)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(gbs, "_free_copy", lambda view, source: None)
    index_copies = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: SimpleNamespace(host=t.clone()))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: index_copies.append((dev, host.host)))
    return index_copies


def _buffer(layout, num_blocks, gen):
    """A K or V buffer of ``num_blocks`` blocks full of random bytes (stale content of earlier owners)."""
    size = num_blocks * layout.bank_bytes
    banks = [
        bytearray(torch.randint(0, 256, (size,), dtype=torch.uint8, generator=gen).numpy().tobytes())
        for _ in range(BANKS)
    ]
    tile = layout.kv_tile_bytes
    return View(Buf(banks), 0, tile, (num_blocks, 1, layout.block_size, layout.tiles * 1024 // layout.block_size))


def _state(kv_bf8=True, block_size=1024, seed=0):
    """A GdnBlockState over 4 layer slots (12 GDN layers in 3 groups) on emulated K and V buffers."""
    gen = torch.Generator().manual_seed(seed)
    lay = plan_gdn_block_layout(block_size=block_size, kv_heads=1, head_dim=256, kv_bf8=kv_bf8, dims=QWEN_27B_TP8)
    st = object.__new__(gbs.GdnBlockState)
    st.mesh, st.bmax, st.num_blocks, st.dims, st.layout = None, BMAX, NUM_BLOCKS, QWEN_27B_TP8, lay
    st.kv_caches = [(_buffer(lay, NUM_BLOCKS, gen), _buffer(lay, NUM_BLOCKS, gen)) for _ in range(4)]
    slot_views = []
    for k, v in st.kv_caches:
        rec_k = _alias_view(k, lay.fp32_view_shape(NUM_BLOCKS), ttnn.float32)
        rec_v = _alias_view(v, lay.fp32_view_shape(NUM_BLOCKS), ttnn.float32) if lay.heads_v else None
        slot_views.append((rec_k, rec_v, _alias_view(v, lay.hist_view_shape(NUM_BLOCKS), ttnn.bfloat16)))
    plugin = PluginLayout(ATTENTION, 1, 256, QWEN_27B_TP8, kv_bf8, num_blocks=NUM_BLOCKS, block_size=block_size)
    st.layers = {}
    for group in plugin.gdn_groups:
        for slot, i in enumerate(group):
            rec_k, rec_v, hist = slot_views[slot]
            ws = View(Buf([bytearray(QWEN_27B_TP8.nv * 2 * 4096) for _ in range(BANKS)]), 0, 4096, (1, 6, 128, 128))
            st.layers[i] = SimpleNamespace(
                name=i,
                state_blocks=SimpleNamespace(rec_k=rec_k, rec_v=rec_v, hist=hist, idx=f"idx{i}"),
                rec_state=ws,
                conv_carry=Host(torch.zeros(1, 3, QWEN_27B_TP8.conv_dim, dtype=torch.bfloat16)),
            )
    st._unpack_table = Host(unpack_history_rows(QWEN_27B_TP8).reshape(1, -1))
    st._group_names = st._group_layers = None
    st._decode_blocks, st._index_refs = {}, {}
    return st, plugin


def _slot_views(st, slot):
    """(rec_k, rec_v, hist) views of layer slot ``slot``: those of the slot's layer in the first GDN group."""
    sb = st.layers[[i for i, a in enumerate(ATTENTION) if not a][0::3][slot]].state_blocks
    return sb.rec_k, sb.rec_v, sb.hist


def _slot_state(st, slot, block):
    """(16 float32 pages per head, packed history tiles [Nv, 4, 32, 32]) of ``block`` in layer slot ``slot``, read
    from the buffers by the layout's definition, independently of GdnBlockState."""
    lay = st.layout
    rec_k, rec_v, hist = _slot_views(st, slot)
    first = lay.first_fp32_page(block)
    heads = []
    for h in range(lay.dims.nv):
        view, p0 = (rec_k, first + 16 * h) if h < lay.heads_k else (rec_v, first + 16 * (h - lay.heads_k))
        heads.append(b"".join(view.read(p0 + j) for j in range(16)))
    p0 = lay.hist_page(block, 0)
    raw = b"".join(hist.read(p0 + j) for j in range(lay.dims.nv * HIST_SLOTS))
    tiles = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(lay.dims.nv, HIST_SLOTS, 32, 32)
    return heads, tiles.clone()


def _write_slot_state(st, slot, block, gen):
    """Write a random full state (recurrent pages and the packed history of a random carry) into ``block`` of layer
    slot ``slot`` by the layout's definition. Returns (heads, carry)."""
    lay, dims = st.layout, st.dims
    rec_k, rec_v, hist = _slot_views(st, slot)
    first = lay.first_fp32_page(block)
    heads = []
    for h in range(dims.nv):
        view, p0 = (rec_k, first + 16 * h) if h < lay.heads_k else (rec_v, first + 16 * (h - lay.heads_k))
        pages = torch.randn(16, 1024, generator=gen).numpy().tobytes()
        for j in range(16):
            view.write(p0 + j, pages[j * 4096 : (j + 1) * 4096])
        heads.append(pages)
    carry = torch.randn(dims.kconv - 1, dims.conv_dim, generator=gen).to(torch.bfloat16)
    packed = carry.reshape(-1, 32)[pack_history_rows(dims)].reshape(-1, 32, 32)
    p0 = lay.hist_page(block, 0)
    for j in range(packed.shape[0]):
        hist.write(p0 + j, packed[j].contiguous().view(torch.uint8).numpy().tobytes())
    return heads, carry


def _block_bytes(st, block):
    """Every byte of ``block`` in every K and V buffer."""
    out = []
    for k, v in st.kv_caches:
        for buf in (k, v):
            lo, hi = block * st.layout.bank_bytes, (block + 1) * st.layout.bank_bytes
            out.append(b"".join(bytes(bank[lo:hi]) for bank in buf.buf.banks))
    return out


def _write_full_state(st, block, gen):
    return [_write_slot_state(st, slot, block, gen) for slot in range(len(st.kv_caches))]


def _assert_state(st, block, want, label):
    # Compare to a bool first: pytest's report of two unequal multi-kilobyte byte strings takes minutes.
    for slot, (heads, carry) in enumerate(want):
        got_heads, tiles = _slot_state(st, slot, block)
        same = got_heads == heads
        assert same, f"{label}: recurrent state of slot {slot}"
        packed = carry.reshape(-1, 32)[pack_history_rows(st.dims)].reshape(st.dims.nv, HIST_SLOTS, 32, 32)
        used = 2 * (2 * st.dims.dk + st.dims.dv) // 32
        assert torch.equal(tiles[:, 1:, :used], packed[:, 1:, :used]), f"{label}: history of slot {slot}"


# --------------------------------------------------------------------------------------------- prefill planning


def test_prefill_segments():
    """Where a prefill stops: at the checkpoint, on KV block boundaries only."""
    seg = gbs.prefill_segments
    assert seg(0, 2500, 2048, 1024) == [(0, 2048), (2048, 2500)]  # one 2500-token chunk, checkpoint after 2048
    assert seg(0, 2048, 0, 1024) == [(0, 2048)]  # chunked equivalent: chunk 1 ...
    assert seg(2048, 2500, 0, 1024) == [(2048, 2500)]  # ... chunk 2 resumes from chunk 1's end state
    assert seg(1024, 3600, 3072, 1024) == [(1024, 3072), (3072, 3600)]  # partial hit at 1024, own checkpoint
    for args, msg in [
        ((300, 700, 0, 512), "inside a 512-token KV block"),
        ((0, 2500, 2000, 1024), "not on a 1024-token block boundary"),
        ((1024, 2500, 1024, 1024), "not on a 1024-token block boundary"),
        ((0, 2048, 2048, 1024), "not on a 1024-token block boundary"),
        ((5, 5, 0, 1), "is empty"),
    ]:
        with pytest.raises(ValueError, match=msg):  # allow-pytest.raises: host test without the repo conftest
            seg(*args)


def test_row_blocks():
    plugin = PluginLayout(ATTENTION, 1, 256, QWEN_27B_TP8, True, num_blocks=NUM_BLOCKS, block_size=1024)
    state = plugin.step_state(
        [(4, 5, 6), (7, 8, 9)], BMAX, start_blocks=[(1, 2, 3), None], checkpoints=[None, (2048, (10, 11, 12))]
    )
    assert gbs.row_blocks(state, 0) == gbs.RowBlocks(start=(1, 2, 3), dest=(4, 5, 6), checkpoint=None, checkpoint_pos=0)
    assert gbs.row_blocks(state, 1) == gbs.RowBlocks(
        start=(0, 0, 0), dest=(7, 8, 9), checkpoint=(10, 11, 12), checkpoint_pos=2048
    )
    old = SimpleNamespace(groups=tuple(SimpleNamespace(windows=g.windows) for g in state.groups))
    with pytest.raises(ValueError, match="has no start_blocks"):  # allow-pytest.raises: no repo conftest
        gbs.row_blocks(old, 0)


# --------------------------------------------------------------------------------- reads and decode crossings


@pytest.mark.parametrize("block", [5, 6], ids=["odd", "even"])
def test_restore_loads_the_start_block_and_leaves_it_unchanged(dram, block):
    """restore puts the start block's recurrent state into the batch-1 workspace page for page and rebuilds the
    convolution carry from the packed history, for even and odd bfloat8_b 1024-token blocks. No block changes."""
    st, plugin = _state()
    gen = torch.Generator().manual_seed(block)
    want = _write_full_state(st, block, gen)
    before = [_block_bytes(st, b) for b in range(NUM_BLOCKS)]
    state = plugin.step_state([(9, 10, 11)], BMAX, start_blocks=[(block, block, block)])
    st.restore(state, (block, block, block))
    for g, layers in enumerate(plugin.gdn_groups):
        for slot, i in enumerate(layers):
            dn = st.layers[i]
            heads, carry = want[slot]
            same = b"".join(dn.rec_state.read(q) for q in range(dn.rec_state.pages)) == b"".join(heads)
            assert same, f"layer {i}: workspace recurrent state"
            assert torch.equal(dn.conv_carry.t[0], carry), f"layer {i}: conv carry"
    same = [_block_bytes(st, b) for b in range(NUM_BLOCKS)] == before
    assert same, "restore changed a block"


def test_decode_crossing_copies_the_start_state_into_the_destination(dram):
    """A decode row whose start block differs from its destination (its token crosses into a new block-table column)
    gets the start block's full state copied into the destination before the index names the destination, between
    odd and even bfloat8_b 1024-token blocks in both directions (groups 0 and 2: 1 -> 4 and 3 -> 6, group 1: 2 -> 5);
    the start block is unchanged, and a row whose start block is its destination copies nothing."""
    st, plugin = _state()
    gen = torch.Generator().manual_seed(7)
    starts, dests, stay = (1, 2, 3), (4, 5, 6), (7, 8, 9)
    want = {b: _write_full_state(st, b, gen) for b in starts + stay}
    before = {b: _block_bytes(st, b) for b in starts + stay}
    state = plugin.step_state([dests, stay, None], BMAX, start_blocks=[starts, "same", None])
    st.set_decode_blocks(state)
    for g in range(3):
        _assert_state(st, dests[g], want[starts[g]], f"group {g} destination")
    for b in starts + stay:
        same = _block_bytes(st, b) == before[b]
        assert same, f"block {b} changed"
    index = dict(dram)
    for g, layers in enumerate(plugin.gdn_groups):
        want_idx = st.layout.index_host([dests[g], stay[g], 0, 0], NUM_BLOCKS)
        for i in layers:
            assert torch.equal(index[f"idx{i}"], want_idx)


# ------------------------------------------------------------------------------------------ adapter orchestration


@pytest.fixture
def adapter(monkeypatch):
    """Qwen36ForCausalLM.prefill_forward over a stand-in model that records prefill calls and state block traffic."""
    from models.common.tests.vllm_stubs import stub_vllm_if_missing

    stub_vllm_if_missing(monkeypatch)
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    events = []
    vocab = 8

    def prefill(tokens, pt, actual_len, vision_tokens=None, start_pos=0, rope_token_ids=None):
        assert tokens.shape[1] == actual_len
        if rope_token_ids is not None:
            # The RoPE table of a multimodal row comes from the whole row.
            events.append(("prefill", start_pos, actual_len, vision_tokens, rope_token_ids.shape[1]))
        else:
            events.append(("prefill", start_pos, actual_len, vision_tokens))
        return torch.full((1, 1, vocab), float(actual_len))

    blocks = SimpleNamespace(
        layout=SimpleNamespace(block_size=1024),
        restore=lambda state, b: events.append(("restore", tuple(b))),
        write_state=lambda state, b: events.append(("write", tuple(b))),
    )
    model = SimpleNamespace(
        args=SimpleNamespace(max_batch_size=4, vocab_size=vocab),
        mesh_device=None,
        gdn_blocks=blocks,
        prefill_traced_chunked=prefill,
        vision_rows_host=lambda rows: events.append(("vision rows to host", rows)) or f"host {rows}",
    )
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "ConcatMeshToTensor", lambda *a, **kw: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    wrapper = Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)
    wrapper.model = [model]
    wrapper.data_parallel = 1
    plugin = PluginLayout(ATTENTION, 1, 256, QWEN_27B_TP8, True, num_blocks=64, block_size=1024)

    def run(start, end, dest, start_blocks=None, checkpoint=None, max_batch_size=4, **kw):
        events.clear()
        model.args.max_batch_size = max_batch_size
        state = plugin.step_state([dest], BMAX, start_blocks=[start_blocks], checkpoints=[checkpoint])
        tokens, page_table = torch.zeros(1, end, dtype=torch.int64), torch.zeros(1, 4, dtype=torch.int32)
        logits, _ = wrapper.prefill_forward(tokens, page_table, None, [end], start_pos=[start], mamba_state=state, **kw)
        assert float(logits[0, 0, 0]) == end
        return list(events)

    return run


def test_prefill_writes_the_checkpoint_between_two_parts(adapter):
    """An unchunked 2500-token prompt with a checkpoint after 2048: the prefill stops after token 2048, writes the
    checkpoint blocks, continues to 2500 and writes the destination blocks."""
    events = adapter(0, 2500, (4, 5, 6), checkpoint=(2048, (1, 2, 3)))
    assert events == [
        ("prefill", 0, 2048, None),
        ("write", (1, 2, 3)),
        ("prefill", 2048, 2500, None),
        ("write", (4, 5, 6)),
    ]


def test_prefill_resumes_from_the_start_blocks_and_only_reads_them(adapter):
    """A prefix hit or a later chunk restores the start blocks and computes only its own tokens; the start blocks are
    read, never written."""
    events = adapter(2048, 2500, (7, 8, 9), start_blocks=(1, 2, 3))
    assert events == [("restore", (1, 2, 3)), ("prefill", 2048, 2500, None), ("write", (7, 8, 9))]
    events = adapter(1024, 3600, (7, 8, 9), start_blocks=(1, 2, 3), checkpoint=(3072, (4, 5, 6)))
    assert events == [
        ("restore", (1, 2, 3)),
        ("prefill", 1024, 3072, None),
        ("write", (4, 5, 6)),
        ("prefill", 3072, 3600, None),
        ("write", (7, 8, 9)),
    ]


def test_prefill_refuses_a_start_inside_a_kv_block(adapter, expect_error):
    """A text row that starts inside a KV block is refused in both cache modes (start block equal to the destination,
    or another block): the attention fill writes from the first row of a block, and vllm-tt-plugin ends prefill chunks
    on block boundaries."""
    for start_blocks in ("same", (1, 2, 3)):
        with expect_error(ValueError, "prefill resumes at 1920, inside a 1024-token KV block"):
            adapter(1920, 2500, (4, 5, 6), start_blocks=start_blocks)


def test_prefill_checks_the_start_blocks(adapter, expect_error):
    with expect_error(ValueError, "has the start blocks"):
        adapter(2048, 2500, (4, 5, 6))
    with expect_error(ValueError, "has the start blocks"):
        adapter(0, 2500, (4, 5, 6), start_blocks=(1, 2, 3))


@pytest.fixture
def multimodal(monkeypatch):
    """Rows carry pixels, and the vision tower answers "vision"."""
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    monkeypatch.setattr(Qwen36ForCausalLM, "_has_visual", staticmethod(lambda kwargs, key: key == "pixel_values"))
    monkeypatch.setattr(Qwen36ForCausalLM, "_compute_vision_tokens", lambda self, model, kwargs: "vision")


def test_multimodal_prefill_writes_its_checkpoint_with_the_whole_row_rope(adapter, multimodal):
    """A whole multimodal prompt with a checkpoint after 2048 tokens runs as two segments, both with the vision rows
    and the RoPE table of all 2500 tokens, so an image across token 2048 keeps its M-RoPE positions in both."""
    events = adapter(0, 2500, (4, 5, 6), checkpoint=(2048, (1, 2, 3)), max_batch_size=1, pixel_values=["x"])
    # The device rows are read once, before the first segment's trace replays can overwrite them.
    assert events == [
        ("vision rows to host", "vision"),
        ("prefill", 0, 2048, "host vision", 2500),
        ("write", (1, 2, 3)),
        ("prefill", 2048, 2500, "host vision", 2500),
        ("write", (4, 5, 6)),
    ]


def test_multimodal_prefill_continues_from_its_start_blocks(adapter, multimodal, expect_error):
    """A multimodal request replayed after preemption that outgrows one step continues after its images from the
    state its first chunk wrote; like a text row, it must start on a KV block boundary."""
    events = adapter(2048, 2500, (4, 5, 6), start_blocks=(1, 2, 3), max_batch_size=1, pixel_values=["x"])
    assert events == [
        ("vision rows to host", "vision"),
        ("restore", (1, 2, 3)),
        ("prefill", 2048, 2500, "host vision", 2500),
        ("write", (4, 5, 6)),
    ]
    with expect_error(ValueError, "prefill resumes at 1920, inside a 1024-token KV block"):
        adapter(1920, 2500, (4, 5, 6), start_blocks=(1, 2, 3), max_batch_size=1, pixel_values=["x"])
