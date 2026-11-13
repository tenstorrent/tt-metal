# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): prefix restore, prefill checkpoints and decode block crossings of the GDN state in vLLM KV
cache blocks (mamba_cache_mode "align", tt/gdn_block_state.py, Qwen36ForCausalLM.prefill_forward, and the checkpoint
routing of Qwen36Model.prefill_traced_chunked and gdn.tp.PrefillCheckpoint).

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


CHUNK = 2048  # Qwen36Model.prefill_chunk_size


def test_prefill_segments():
    """How a prefill gets vLLM's checkpoint: from the forward of its last chunk (one range that carries the
    checkpoint), or by stopping at the checkpoint (two ranges), on KV block boundaries only."""

    def seg(start, end, checkpoint, block=1024):
        return gbs.prefill_segments(start, end, checkpoint, block, CHUNK)

    assert seg(0, 2000, 1024) == [(0, 2000, 1024)]  # 2000-token prompt: one masked forward computes both states
    assert seg(2048, 3764, 3072) == [(2048, 3764, 3072)]  # follow-up turn: hit at 2048, checkpoint in the tail
    assert seg(0, 3500, 3072) == [(0, 3500, 3072)]  # one replayed chunk, checkpoint in the eager tail
    assert seg(0, 2048, 1024) == [(0, 2048, 1024)]  # a whole last chunk
    assert seg(0, 1800, 1536, block=512) == [(0, 1800, 1536)]
    assert seg(0, 2500, 2048) == [(0, 2048, 0), (2048, 2500, 0)]  # on a chunk boundary: replayed chunk, then tail
    assert seg(1024, 3600, 3072) == [(1024, 3072, 0), (3072, 3600, 0)]  # partial hit at 1024, own checkpoint
    assert seg(0, 5000, 1024) == [(0, 1024, 0), (1024, 5000, 0)]  # inside an earlier chunk
    assert seg(0, 2048, 0) == [(0, 2048, 0)]  # chunked equivalent: chunk 1 ...
    assert seg(2048, 2500, 0) == [(2048, 2500, 0)]  # ... chunk 2 resumes from chunk 1's end state
    for args, msg in [
        ((300, 700, 0, 512), "inside a 512-token KV block"),
        ((0, 2500, 2000), "not on a 1024-token block boundary"),
        ((1024, 2500, 1024), "not on a 1024-token block boundary"),
        ((0, 2048, 2048), "not on a 1024-token block boundary"),
        ((5, 5, 0, 1), "is empty"),
    ]:
        with pytest.raises(ValueError, match=msg):  # allow-pytest.raises: host test without the repo conftest
            seg(*args)


@pytest.mark.parametrize("block", [512, 1024, 2048])
def test_every_vllm_checkpoint_costs_no_extra_forward(block):
    """vLLM's rule (vllm.v1.kv_cache_interface.get_mamba_prefill_checkpoint_position and
    is_mamba_prefill_checkpoint_valid, prefix_match_unit = block size) puts a prefill's checkpoint on the last block
    boundary before its end, at least one block after its start. prefill_segments then either computes it in the last
    chunk's forward or stops on a chunk boundary, never inside an earlier chunk."""
    for start in range(0, 4 * CHUNK + 1, block):
        for end in range(start + 1, start + 3 * CHUNK + 2 * block, 97):
            p = (end - 1) // block * block
            if not start + block <= p < end:
                continue
            ranges = gbs.prefill_segments(start, end, p, block, CHUNK)
            if len(ranges) == 2:
                assert (p - start) % CHUNK == 0, (start, end, p, ranges)
            else:
                assert ranges == [(start, end, p)], (start, end, p, ranges)


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

    def prefill(tokens, pt, actual_len, vision_tokens=None, start_pos=0, rope_token_ids=None, checkpoint_pos=0):
        assert tokens.shape[1] == actual_len
        if rope_token_ids is not None:
            # The RoPE table of a multimodal row comes from the whole row.
            events.append(("prefill", start_pos, actual_len, vision_tokens, checkpoint_pos, rope_token_ids.shape[1]))
        else:
            events.append(("prefill", start_pos, actual_len, vision_tokens, checkpoint_pos))
        return torch.full((1, 1, vocab), float(actual_len))

    blocks = SimpleNamespace(
        layout=SimpleNamespace(block_size=1024),
        restore=lambda state, b: events.append(("restore", tuple(b))),
        write_state=lambda state, b: events.append(("write", tuple(b))),
        write_checkpoint=lambda state, b: events.append(("checkpoint", tuple(b))),
    )
    model = SimpleNamespace(
        args=SimpleNamespace(max_batch_size=4, vocab_size=vocab),
        mesh_device=None,
        gdn_blocks=blocks,
        prefill_traced_chunked=prefill,
        vision_rows_host=lambda rows: events.append(("vision rows to host", rows)) or f"host {rows}",
        prefill_chunk_size=CHUNK,
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
    """An unchunked 2500-token prompt with a checkpoint after 2048, a chunk boundary: the prefill stops after token
    2048, writes the workspace into the checkpoint blocks, continues to 2500 and writes the destination blocks."""
    events = adapter(0, 2500, (4, 5, 6), checkpoint=(2048, (1, 2, 3)))
    assert events == [
        ("prefill", 0, 2048, None, 0),
        ("write", (1, 2, 3)),
        ("prefill", 2048, 2500, None, 0),
        ("write", (4, 5, 6)),
    ]


def test_prefill_computes_a_checkpoint_in_its_last_chunk_in_one_forward(adapter):
    """A 2000-token prompt with a checkpoint after 1024: one prefill computes both states; the checkpoint blocks get
    the state the forward computed at 1024 (write_checkpoint), the destination blocks the workspace. The same for a
    follow-up turn whose checkpoint lies in its recomputed tail."""
    events = adapter(0, 2000, (4, 5, 6), checkpoint=(1024, (1, 2, 3)))
    assert events == [("prefill", 0, 2000, None, 1024), ("checkpoint", (1, 2, 3)), ("write", (4, 5, 6))]
    events = adapter(2048, 3764, (7, 8, 9), start_blocks=(1, 2, 3), checkpoint=(3072, (4, 5, 6)))
    assert events == [
        ("restore", (1, 2, 3)),
        ("prefill", 2048, 3764, None, 3072),
        ("checkpoint", (4, 5, 6)),
        ("write", (7, 8, 9)),
    ]


def test_prefill_resumes_from_the_start_blocks_and_only_reads_them(adapter):
    """A prefix hit or a later chunk restores the start blocks and computes only its own tokens; the start blocks are
    read, never written."""
    events = adapter(2048, 2500, (7, 8, 9), start_blocks=(1, 2, 3))
    assert events == [("restore", (1, 2, 3)), ("prefill", 2048, 2500, None, 0), ("write", (7, 8, 9))]
    events = adapter(1024, 3600, (7, 8, 9), start_blocks=(1, 2, 3), checkpoint=(3072, (4, 5, 6)))
    assert events == [
        ("restore", (1, 2, 3)),
        ("prefill", 1024, 3072, None, 0),
        ("write", (4, 5, 6)),
        ("prefill", 3072, 3600, None, 0),
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
    """A whole multimodal prompt with a checkpoint after 2048 tokens, a chunk boundary, runs as two segments, both
    with the vision rows and the RoPE table of all 2500 tokens, so an image across token 2048 keeps its M-RoPE
    positions in both."""
    events = adapter(0, 2500, (4, 5, 6), checkpoint=(2048, (1, 2, 3)), max_batch_size=1, pixel_values=["x"])
    # The device rows are read once, before the first segment's trace replays can overwrite them.
    assert events == [
        ("vision rows to host", "vision"),
        ("prefill", 0, 2048, "host vision", 0, 2500),
        ("write", (1, 2, 3)),
        ("prefill", 2048, 2500, "host vision", 0, 2500),
        ("write", (4, 5, 6)),
    ]


def test_multimodal_prefill_continues_from_its_start_blocks(adapter, multimodal, expect_error):
    """A multimodal request replayed after preemption that outgrows one step continues after its images from the
    state its first chunk wrote; like a text row, it must start on a KV block boundary."""
    events = adapter(2048, 2500, (4, 5, 6), start_blocks=(1, 2, 3), max_batch_size=1, pixel_values=["x"])
    assert events == [
        ("vision rows to host", "vision"),
        ("restore", (1, 2, 3)),
        ("prefill", 2048, 2500, "host vision", 0, 2500),
        ("write", (4, 5, 6)),
    ]
    with expect_error(ValueError, "prefill resumes at 1920, inside a 1024-token KV block"):
        adapter(1920, 2500, (4, 5, 6), start_blocks=(1, 2, 3), max_batch_size=1, pixel_values=["x"])


def test_multimodal_prefill_computes_a_checkpoint_in_its_last_chunk_in_one_forward(adapter, multimodal):
    """A 2000-token multimodal prompt with a checkpoint after 1024 runs one prefill, with the vision rows and the
    whole-row RoPE table, that also computes the checkpoint (an image may cross token 1024)."""
    events = adapter(0, 2000, (4, 5, 6), checkpoint=(1024, (1, 2, 3)), max_batch_size=1, pixel_values=["x"])
    assert events == [
        ("vision rows to host", "vision"),
        ("prefill", 0, 2000, "host vision", 1024, 2000),
        ("checkpoint", (1, 2, 3)),
        ("write", (4, 5, 6)),
    ]


# ------------------------------------------------------------------------- checkpoint inside a prefill forward


@pytest.fixture
def chunked(monkeypatch):
    """Qwen36Model.prefill_traced_chunked and _prefill_traced_chunked_tp at TP=8 with 1024-token KV blocks, without a
    device: the chunk-trace replays are counted, the eager masked-bucket forwards recorded."""
    from models.demos.blackhole.qwen36.tt import model as model_module

    calls = []
    model = object.__new__(model_module.Qwen36Model)
    model.num_devices = 8
    model.device = "mesh"
    model._chunked_chunk_size = CHUNK
    model._chunked_trace_id = 1
    model._chunked_trace_output = "chunk_out"
    model._chunk_full_page_table_buf = SimpleNamespace(shape=[1, 8])
    for name in ("token", "start_idx", "page_table", "cos", "sin"):
        setattr(model, f"_chunk_{name}_buf" if name != "start_idx" else "_chunk_start_idx_tensor", name)
    model._paged_kv_caches = None
    model._build_request_rope = lambda tokens, vision_tokens: None
    model.vision_rows_host = lambda rows: rows
    model._reset_gdn_state_for_new_sequence = lambda: None
    model._rope_tp_cos_sin_torch = lambda start, length: (None, None)
    model._set_vision_merge = lambda *a: None
    model._vis_row_offset_for = lambda tokens, start: 0
    model._masked_bucket_logits_tp = lambda hidden, actual_len, bucket: ("trace_logits", hidden)

    def masked(tokens, pt, actual_len, chunk_start=0, checkpoint_len=None, vision_tokens=None, **kw):
        assert tokens.shape[1] == actual_len
        calls.append(
            ("eager", chunk_start, actual_len, checkpoint_len) + (() if vision_tokens is None else ("vision",))
        )
        return "logits"

    model.prefill_masked_bucket = masked
    monkeypatch.setattr(model_module, "get_block_size", lambda kv: 1024)
    monkeypatch.setattr(ttnn, "execute_trace", lambda *a, **kw: calls.append(("replay",)))

    def run(start, end, checkpoint=0, **kw):
        calls.clear()
        tokens = torch.zeros(1, end, dtype=torch.int64)
        model.prefill_traced_chunked(
            tokens,
            torch.zeros(1, 8, dtype=torch.int32),
            actual_len=end,
            start_pos=start,
            checkpoint_pos=checkpoint,
            **kw,
        )
        return list(calls)

    return run, model


def test_checkpoint_rides_in_the_eager_forward_of_the_last_chunk(chunked):
    """The chunk trace replays the same full chunks as without a checkpoint, and the eager forward of the last chunk
    gets the checkpoint's offset in it. A whole last chunk runs eagerly instead of replaying."""
    run, _ = chunked
    assert run(0, 2000, 1024) == [("eager", 0, 2000, 1024)]
    assert run(0, 2000) == [("eager", 0, 2000, None)]
    assert run(0, 2048, 1024) == [("eager", 0, 2048, 1024)]
    assert run(0, 2048) == [("replay",)]
    assert run(0, 3500, 3072) == [("replay",), ("eager", 2048, 1452, 1024)]
    assert run(0, 3500) == [("replay",), ("eager", 2048, 1452, None)]
    assert run(0, 4096, 3072) == [("replay",), ("eager", 2048, 2048, 1024)]
    assert run(2048, 3764, 3072) == [("eager", 2048, 1716, 1024)]
    assert run(2048, 7000) == [("replay",), ("replay",), ("eager", 6144, 856, None)]
    # A multimodal prompt: the same forwards, each splicing the vision rows.
    assert run(0, 2000, 1024, vision_tokens="rows") == [("eager", 0, 2000, 1024, "vision")]
    assert run(0, 4096, 3072, vision_tokens="rows") == [("replay",), ("eager", 2048, 2048, 1024, "vision")]


def test_checkpoint_outside_the_last_chunk_is_refused(chunked, expect_error):
    run, model = chunked
    with expect_error(ValueError, r"not inside the last chunk \[4096, 5000\)"):
        run(0, 5000, 1024)
    with expect_error(ValueError, r"not inside the last chunk \[2048, 4096\)"):
        run(0, 4096, 2048)
    model.num_devices = 1
    with expect_error(ValueError, "needs a TP mesh"):
        run(0, 2000, 1024)


def test_prefill_checkpoint_constants(monkeypatch, expect_error):
    """PrefillCheckpoint's gate mask zeroes beta and g from the checkpoint on, and its one-hot rows pick the K - 1
    inputs before it, the conv history after ``length`` tokens."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import PrefillCheckpoint

    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    T, L, kconv = 256, 128, 4
    ck = PrefillCheckpoint(None, T, L, kconv)
    assert ck.length == L and ck.gate_mask.shape == (1, T, 1)
    assert ck.gate_mask[0, :L].eq(1).all() and ck.gate_mask[0, L:].eq(0).all()
    x = torch.randn(1, T, 64).to(torch.bfloat16)
    assert torch.equal((ck.conv_select.float() @ x.float()).to(torch.bfloat16), x[:, L - (kconv - 1) : L])
    with expect_error(ValueError, "needs 3 <= length < 256"):
        PrefillCheckpoint(None, T, T, kconv)
    with expect_error(ValueError, "needs 3 <= length < 256"):
        PrefillCheckpoint(None, T, 2, kconv)


def _delta_rule(k, v, beta, g, state):
    """Gated delta rule state recurrence, one token at a time: S = exp(g_t) S; S += beta_t k_t (v_t - S^T k_t)^T."""
    for t in range(k.shape[0]):
        state = torch.exp(g[t]) * state
        state = state + beta[t] * torch.outer(k[t], v[t] - state.T @ k[t])
    return state


def test_gate_mask_gives_the_state_at_the_checkpoint(monkeypatch):
    """Beta and g times PrefillCheckpoint.gate_mask turn every token from the checkpoint on into an identity update,
    so the recurrence over all T tokens ends at the state after the checkpoint's ``length`` tokens."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import PrefillCheckpoint

    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    gen = torch.Generator().manual_seed(0)
    T, L, D = 96, 64, 16
    k = torch.nn.functional.normalize(torch.randn(T, D, generator=gen, dtype=torch.float64), dim=-1)
    v = torch.randn(T, D, generator=gen, dtype=torch.float64)
    beta = torch.rand(T, generator=gen, dtype=torch.float64)
    g = -torch.rand(T, generator=gen, dtype=torch.float64)
    s0 = torch.randn(D, D, generator=gen, dtype=torch.float64)
    mask = PrefillCheckpoint(None, T, L, 4).gate_mask.reshape(T).double()
    at_checkpoint = _delta_rule(k[:L], v[:L], beta[:L], g[:L], s0)
    assert torch.equal(_delta_rule(k, v, beta * mask, g * mask, s0), at_checkpoint)
    assert not torch.allclose(_delta_rule(k, v, beta, g, s0), at_checkpoint)


def _gdn_layer():
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

    dn = object.__new__(TPGatedDeltaNet)
    dn.tw, dn.Nk, dn.Nv, dn.Dk, dn.Dv = {}, 2, 6, 128, 128
    dn.prefill_checkpoint = None
    dn.prefill_checkpoint_buffers = None
    return dn


def test_gdn_prefill_checkpoint_needs_a_carried_single_sequence_prefill(expect_error):
    dn = _gdn_layer()
    x = SimpleNamespace(shape=(1, 2048, 640))
    for length, kw in [
        (1024, {"return_state": True}),
        (1024, {"carry_state": False}),
        (1024, {"sp": object()}),
        (2000, {"valid_len": 2000}),
        (2048, {}),
    ]:
        with expect_error(ValueError, "needs a carried single-sequence prefill"):
            dn.forward_prefill(x, checkpoint=SimpleNamespace(length=length), **kw)


def test_gdn_prefill_checkpoint_is_taken_once(monkeypatch, expect_error):
    """A layer keeps the last forward's checkpoint (freeing an untaken older one) until take_prefill_checkpoint."""
    freed = []
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    dn = _gdn_layer()
    with expect_error(ValueError, "holds no prefill checkpoint"):
        dn.take_prefill_checkpoint()
    dn._keep_prefill_checkpoint("rec0", "conv0")
    dn._keep_prefill_checkpoint("rec1", "conv1")
    assert freed == ["rec0", "conv0"]
    assert dn.take_prefill_checkpoint() == ("rec1", "conv1")
    with expect_error(ValueError, "holds no prefill checkpoint"):
        dn.take_prefill_checkpoint()
