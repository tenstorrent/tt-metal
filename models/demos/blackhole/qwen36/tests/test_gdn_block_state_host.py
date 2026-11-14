# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: GdnBlockState (QWEN36_MAMBA_SPEC) page math and index updates, with ttnn stubbed out."""

import math
from types import SimpleNamespace

import pytest
import torch

import models.demos.blackhole.qwen36.tt.gdn_block_state as gbs

NV, DK, DV = 12, 128, 128  # TP=4 per-device GDN shapes of Qwen3.6-27B
NUM_BLOCKS, HD = 10, 256
BANKS = 8  # DRAM banks of a P150


class _Tensor(SimpleNamespace):
    pass


@pytest.fixture
def fake(monkeypatch):
    calls = {"views": [], "h2d": []}

    def alias_view(t, shape, dtype, check_slices=True):
        calls["views"].append((t.name, list(shape), dtype, check_slices))
        return _Tensor(name=f"{t.name}:view", shape=list(shape), dtype=dtype)

    def from_torch(x, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        return _Tensor(name="idx" if device is not None else "host", data=x.clone(), shape=list(x.shape), dtype=dtype)

    def copy_host_to_device_tensor(src, dst):
        calls["h2d"].append(dst)
        dst.data = src.data.clone()

    stub = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        bfloat8_b="bf8",
        uint32="u32",
        int32="i32",
        ROW_MAJOR_LAYOUT="rm",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        ReplicateTensorToMesh=lambda mesh: None,
        from_torch=from_torch,
        copy_host_to_device_tensor=copy_host_to_device_tensor,
        experimental=SimpleNamespace(alias_view=alias_view),
    )
    stub_map = {"bf16": 2048, "fp32": 4096, "bf8": 1088}
    monkeypatch.setattr(gbs, "ttnn", stub)
    monkeypatch.setattr(gbs, "_tile_bytes", lambda dtype: stub_map[dtype])
    monkeypatch.setattr(gbs, "_dram_banks", lambda mesh: BANKS)
    return calls


def _model(n_layers=8, bmax=8, nv=NV):
    attn_idx = [i for i in range(n_layers) if i % 4 == 3]
    layers = []
    for i in range(n_layers):
        dn = SimpleNamespace(Nv=nv, Nk=nv // 3, Dk=DK, Dv=DV, K=4, _decode_op=True, use_fused_recurrent_decode=False)
        layers.append(SimpleNamespace(is_full_attention=i in attn_idx, attention=dn))
    return SimpleNamespace(
        mesh_device="mesh", args=SimpleNamespace(max_batch_size=bmax), _attention_layer_indices=attn_idx, layers=layers
    )


def _group_of(model):
    """{GDN layer index: its KV cache group}: group i % 4, so layer slot j holds GDN layers 4j, 4j + 1, 4j + 2 of groups
    0, 1, 2 (as _specs places them)."""
    return {i: i % 4 for i, layer in enumerate(model.layers) if not layer.is_full_attention}


def _specs(model, kv_dtype="bf8", block=4096):
    """per_layer_specs as the plugin builds them: attention j -> tensor_idx j, GDN layer -> the slot of its group."""
    specs, j = [], 0
    for i, layer in enumerate(model.layers):
        if layer.is_full_attention:
            specs.append(((NUM_BLOCKS, 1, block, HD), kv_dtype, j))
            j += 1
        else:
            specs.append((((3, 2560), (NV, DK, DV)), ("bf16", "fp32"), (i // 4) % 2))
    return specs


def _kv(n, dtype="bf8", block=4096):
    shape = [NUM_BLOCKS, 1, block, HD]
    return [
        [_Tensor(name=f"k{j}", shape=shape, dtype=dtype), _Tensor(name=f"v{j}", shape=shape, dtype=dtype)]
        for j in range(n)
    ]


def test_is_mamba_entry():
    assert gbs.is_mamba_entry((((3, 2560), (12, 128, 128)), ("a", "b"), 0))
    assert not gbs.is_mamba_entry(((10, 1, 2048, 256), "bf16", 0))


@pytest.mark.parametrize(
    "dtype, block, nv, rec_pages, hist_tiles, heads_k, hist_first",
    [
        ("bf8", 2048, 6, 136, 272, 6, 0),  # TP=8 (Galaxy shapes): 512 tiles x 1088 B = 557,056 B = 136 fp32 pages
        ("bf8", 4096, 12, 272, 544, 12, 0),  # TP=4: 1,114,112 B per block
        ("bf16", 2048, 12, 256, 512, 12, 0),  # 1 MiB per block
        # TP=8 at BF8 1024: 8.5 fp32 pages per bank, 64 whole pages per block -> 4 heads in K, 2 in V, the history in
        # the last 3 bf16 page rows of the V block (pages 112..135)
        ("bf8", 1024, 6, 68, 136, 4, 112),
        ("bf8", 2048, 12, 136, 272, 8, 224),  # TP=4 at BF8 2048: 8 + 4 heads
    ],
)
def test_views_and_page_counts(fake, dtype, block, nv, rec_pages, hist_tiles, heads_k, hist_first):
    model = _model(nv=nv)
    st = gbs.GdnBlockState(model, _specs(model, dtype, block), _kv(2, dtype, block))
    assert len(st.layers) == 6
    i, dn = st.layers[0]
    ps = dn._paged_state
    assert ps.hist_pages == hist_tiles and ps.heads_k == heads_k and ps.hist_first == hist_first
    pages = BANKS * (NUM_BLOCKS * rec_pages // BANKS)  # whole fp32 page rows of the buffer
    assert ps.rec_pages_k == pages
    assert ps.rec_view.shape == [pages // 8, 1, DK // 2, DV] and ps.rec_view.dtype == "fp32"
    assert ps.hist_view.shape == [NUM_BLOCKS, hist_tiles, 32, 32] and ps.hist_pool is ps.hist_view
    assert ps.hist_view.name == "v0:view"
    split = heads_k < nv
    assert (ps.rec_view2 is not None) == split and (ps.rec_pool2 is ps.rec_view2)
    whole = rec_pages % BANKS == 0  # a block holds whole fp32 pages on every bank
    layer0 = fake["views"][: 3 if split else 2]
    assert [c for name, _, dt, c in layer0 if dt == "fp32"] == [whole] * (2 if split else 1)


@pytest.mark.parametrize("dtype, block, nv", [("bf8", 1024, 12), ("bf8", 512, 6)])
def test_blocks_without_room_for_the_state_are_refused(fake, dtype, block, nv):
    # BF8 1024 at TP=4: 4 heads in K, V cannot hold 8; BF8 512 at TP=8: 3 whole fp32 rows in some blocks.
    model = _model(nv=nv)
    with pytest.raises(ValueError):
        gbs.GdnBlockState(model, _specs(model, dtype, block), _kv(2, dtype, block))


@pytest.mark.parametrize(
    "dtype, block, nv",
    [("bf8", 1024, 6), ("bf8", 2048, 6), ("bf8", 2048, 12), ("bf8", 4096, 12), ("bf16", 1024, 12)],
)
def test_state_pages_lie_inside_their_block_on_every_bank(fake, dtype, block, nv):
    """Interleaved page q of a view with page size p is bytes [(q // 8) * p, +p) of bank q % 8; block b owns bytes
    [b * s, (b + 1) * s) of every bank. Every page the decode index names lies wholly inside its block, and no two
    pages of a block overlap."""
    model = _model(nv=nv)
    st = gbs.GdnBlockState(model, _specs(model, dtype, block), _kv(2, dtype, block))
    i, dn = st.layers[0]
    ps = dn._paged_state
    blocks = list(range(1, NUM_BLOCKS))
    for lo in range(0, len(blocks), 8):
        row = blocks[lo : lo + 8]
        ids = [torch.tensor(row + [0] * (8 - len(row)), dtype=torch.int32)] * 3
        st._host_idx.clear()
        st.set_decode_blocks(ids, _group_of(model))
        idx = ps.idx.data
        for d, b in enumerate(row):
            used = set()  # (bank, byte) granules of 2048 B owned by this block's pages
            for h in range(2 * nv):
                p = int(idx[h, d])
                if h < nv:
                    pool, q0, n, size, s = (
                        ("k", p, 16, 4096, ps.bank_bytes_k)
                        if p < ps.rec_pages_k
                        else ("v", p - ps.rec_pages_k, 16, 4096, ps.bank_bytes_v)
                    )
                else:
                    pool, q0, n, size, s = "v", p, 4, 2048, ps.bank_bytes_v
                for q in range(q0, q0 + n):
                    bank, start = q % BANKS, (q // BANKS) * size
                    assert b * s <= start and start + size <= (b + 1) * s, (b, h, q)
                    for g in range(start // 2048, (start + size) // 2048):
                        assert (pool, bank, g) not in used, (b, h, q)
                        used.add((pool, bank, g))


def test_decode_index_follows_rows_and_skips_unchanged(fake):
    model = _model()
    st = gbs.GdnBlockState(model, _specs(model), _kv(2))
    ids = [torch.tensor([5, 7, 0, 0, 0, 0, 0, 0], dtype=torch.int32)] * 3
    row_of = [3, 0, 1, 2, 4, 5, 6, 7]  # runner slot 0 -> device row 3, slot 1 -> row 0
    st.set_decode_blocks(ids, _group_of(model), row_of=row_of)
    i, dn = st.layers[0]
    idx = dn._paged_state.idx.data
    assert list(idx.shape) == [2 * NV, 8]
    for h in range(NV):
        assert idx[h].tolist() == [7 * 272 + 16 * h, 16 * h, 16 * h, 5 * 272 + 16 * h] + [16 * h] * 4
        assert idx[NV + h].tolist() == [7 * 544 + 4 * h, 4 * h, 4 * h, 5 * 544 + 4 * h] + [4 * h] * 4
    n = len(fake["h2d"])
    assert n == len(st.layers)
    st.set_decode_blocks(ids, _group_of(model), row_of=row_of)
    assert len(fake["h2d"]) == n  # unchanged indices: no copies


def test_decode_index_of_each_layer_follows_its_group(fake):
    """Each GDN layer points at the block its group holds for the row; a later step that moves one group's block
    copies only that group's layers."""
    model = _model()
    st = gbs.GdnBlockState(model, _specs(model), _kv(2))
    group_of = _group_of(model)
    ids = [torch.tensor([3 + 2 * g] + [0] * 7, dtype=torch.int32) for g in range(3)]
    st.set_decode_blocks(ids, group_of)
    for i, dn in st.layers:
        assert int(dn._paged_state.idx.data[0, 0]) == (3 + 2 * group_of[i]) * 272, i
    n = len(fake["h2d"])
    ids[1] = torch.tensor([9] + [0] * 7, dtype=torch.int32)
    st.set_decode_blocks(ids, group_of)
    assert len(fake["h2d"]) - n == sum(1 for g in group_of.values() if g == 1)
    for i, dn in st.layers:
        assert int(dn._paged_state.idx.data[0, 0]) == (9 if group_of[i] == 1 else 3 + 2 * group_of[i]) * 272, i


def test_mirror_table_duplicates_the_row_parity():
    st = gbs.GdnBlockState.__new__(gbs.GdnBlockState)
    st.mesh = None
    calls = {}
    real = gbs.ttnn
    try:
        gbs.ttnn = SimpleNamespace(
            uint32="u32", ROW_MAJOR_LAYOUT="rm", ReplicateTensorToMesh=lambda m: None, from_torch=lambda x, **k: x
        )
        for parity in (0, 1):
            t = st._mirror_table(2, parity).reshape(2, 32)
            assert t[1, 0].item() == 32 + parity and t[1, 1].item() == 32 + parity  # rows 0 and 1 <- row parity
            assert t[0, 22].item() == 22 + parity and t[0, 23].item() == 22 + parity  # chunk 11: rows 22, 23
    finally:
        gbs.ttnn = real


# --- Device ops on torch: prefill write, block copy, restore ------------------------------------------------------

NK, KCONV = 4, 4  # TP=4 per-device key heads, conv kernel
CONV_DIM = 2 * NK * DK + NV * DV  # 2560
POOL_BLOCKS = 24
_TORCH_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class _Paged:
    """A paged cache buffer [N, 1, block, hd] of `dtype` tiles as raw bytes, one row per block."""

    def __init__(self, n, block, dtype="bf8"):
        self.shape = [n, 1, block, HD]
        self.dtype = dtype
        self.bytes = torch.zeros(n, block * HD // 1024 * {"bf8": 1088, "bf16": 2048}[dtype], dtype=torch.uint8)


class _TorchTTNN:
    """The ttnn surface of gdn_block_state on torch tensors. alias_view is a real torch view of a _Paged buffer's
    bytes, so writes through a view, block copies and later reads see one memory (layouts are no-ops)."""

    bfloat16, float32, bfloat8_b, uint32, int32 = "bf16", "fp32", "bf8", "u32", "i32"
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = "rm", "tile", "dram"

    def __init__(self):
        self.fills = []
        self.experimental = SimpleNamespace(
            alias_view=self.alias_view, slice_write=self.slice_write, paged_fill_cache=self.paged_fill_cache
        )

    @staticmethod
    def ReplicateTensorToMesh(mesh):
        return None

    @staticmethod
    def from_torch(x, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        return x.clone()

    @staticmethod
    def alias_view(t, shape, dtype, check_slices=True):
        flat = t.bytes.view(_TORCH_DTYPES[dtype]).reshape(-1)
        return flat[: math.prod(shape)].reshape(shape)

    @staticmethod
    def concat(ts, dim=0):
        return torch.cat(ts, dim=dim)

    @staticmethod
    def reshape(t, shape):
        return t.reshape(tuple(shape))

    @staticmethod
    def slice(t, start, end, memory_config=None):
        if isinstance(t, _Paged):
            out = _Paged.__new__(_Paged)
            out.shape, out.dtype = [1] + t.shape[1:], t.dtype
            out.bytes = t.bytes[start[0] : start[0] + 1].clone()
            return out
        return t[tuple(slice(s, e) for s, e in zip(start, end))].clone()

    @staticmethod
    def slice_write(src, dst, start, end, step):
        dst[tuple(slice(s, e) for s, e in zip(start, end))] = src

    def paged_fill_cache(self, cache, blk, pt, batch_idx=0):
        self.fills.append((cache, int(pt[batch_idx][0])))
        cache.bytes[int(pt[batch_idx][0])] = blk.bytes[0]

    @staticmethod
    def to_layout(t, layout, memory_config=None):
        return t.clone()

    @staticmethod
    def embedding(idx, weight, layout=None, memory_config=None):
        return weight[idx.long()]

    @staticmethod
    def copy(src, dst):
        dst.copy_(src)

    @staticmethod
    def copy_host_to_device_tensor(src, dst):
        dst.copy_(src)

    @staticmethod
    def deallocate(t):
        pass


@pytest.fixture
def torch_ttnn(monkeypatch):
    t = _TorchTTNN()
    monkeypatch.setattr(gbs, "ttnn", t)
    monkeypatch.setattr(gbs, "_tile_bytes", lambda dtype: {"bf16": 2048, "fp32": 4096, "bf8": 1088}[dtype])
    monkeypatch.setattr(gbs, "_dram_banks", lambda mesh: BANKS)
    monkeypatch.setattr(gbs.GdnBlockState, "_halves_shard_memcfg", lambda self, dn, halves: None)
    return t


def _pack_tables(monkeypatch, bmax, nv=NV, nk=NK):
    """The production RM <-> packed history tables (TPGatedDeltaNet._pack_tables) on torch."""
    from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp

    monkeypatch.setattr(gdn_tp, "ttnn", _TorchTTNN())
    me = SimpleNamespace(
        _decode_B=bmax, Nv=nv, Nk=nk, Dk=DK, Dv=DV, K=KCONV, mesh=None, _shared_const=lambda key, make: make()
    )
    return gdn_tp.TPGatedDeltaNet._pack_tables(me)


def _staged_model(monkeypatch, bmax=8, n_layers=8, block=4096, nv=NV, nk=NK):
    """A model whose GDN decode buffers hold random state: rec_state, and the packed history of a random conv carry
    per row, packed for the row's parity by the production table."""
    tab = _pack_tables(monkeypatch, bmax, nv, nk)
    cpr = tab["cpr"]
    conv_dim = 2 * nk * DK + nv * DV
    model = _model(n_layers=n_layers, bmax=bmax, nv=nv)
    g = torch.Generator().manual_seed(7)
    carries = torch.randn(bmax, KCONV - 1, conv_dim, generator=g).to(torch.bfloat16)
    for layer in model.layers:
        dn = layer.attention
        dn.Nk, dn.K, dn.qkv_dim_tp, dn._decode_B = nk, KCONV, conv_dim, bmax
        dn.rec_state = torch.randn(bmax, nv, DK, DV, generator=g)
        packed = []
        for b in range(bmax):
            w = torch.cat([carries[b].reshape((KCONV - 1) * cpr, 32), torch.zeros(1, 32, dtype=torch.bfloat16)])
            packed.append(w[tab["slot_to_packed"][b & 1].long()].reshape(nv, 4, 32, 32))
        dn._conv_hist_packed = torch.stack(packed)
        dn._rec_row_shard_memcfg = dn._packed_row_shard_memcfg = lambda: None
    model.prepare_gdn_decode_width = lambda b: None
    n_attn = len(model._attention_layer_indices)
    kv = [[_Paged(POOL_BLOCKS, block), _Paged(POOL_BLOCKS, block)] for _ in range(n_attn)]
    model._paged_kv_caches = kv
    return model, kv, carries


def _ids(model, bases):
    """(blocks, group_of) of GdnBlockState: request u holds block bases[u] + 8 * g in GDN group g = layer % 4 (one
    group owns a block id, so the GDN layers that share a layer slot never share a block)."""
    blocks = [torch.tensor([b + 8 * g for b in bases] + [0] * (8 - len(bases)), dtype=torch.int32) for g in range(3)]
    return blocks, _group_of(model)


def _group_copies(src, dst):
    return [(src + 8 * g, dst + 8 * g) for g in range(3)]


@pytest.mark.parametrize(
    "block, nv, nk, src, dst",
    [(4096, NV, NK, 5, 7), (1024, 6, 2, 5, 7), (1024, 6, 2, 4, 7), (1024, 6, 2, 5, 6)],
    ids=["tp4_b4096", "tp8_b1024_split", "tp8_b1024_split_even_to_odd", "tp8_b1024_split_odd_to_even"],
)
def test_prefill_rows_survive_block_copy_and_restore(torch_ttnn, monkeypatch, block, nv, nk, src, dst):
    # BF8 1024 holds 8.5 fp32 pages per bank: the whole pages of an even block start at its first byte, those of an
    # odd block half a page later, so a copy between parities must move the head pages, not only the block's bytes.
    model, kv, carries = _staged_model(monkeypatch, block=block, nv=nv, nk=nk)
    st = gbs.GdnBlockState(model, _specs(model, block=block), kv)
    rec0 = {i: dn.rec_state.clone() for i, dn in st.layers}
    st.write_prefill_rows(*_ids(model, [src, 2]), rows=[3, 0])  # request 0 on device row 3, request 1 on row 0
    st.copy_blocks(_group_copies(src, dst))
    for i, dn in st.layers:  # the B=1 prefill scratch, bound
        dn.rec_state = torch.zeros(1, nv, DK, DV)
        dn.conv_carry = torch.zeros(1, KCONV - 1, dn.qkv_dim_tp, dtype=torch.bfloat16)
    st.set_prefill_blocks(*_ids(model, [dst, 2]))
    for u, row in ((0, 3), (1, 0)):
        st.restore(u)
        for i, dn in st.layers:
            assert torch.equal(dn.rec_state[0], rec0[i][row]), (u, i)
            assert torch.equal(dn.conv_carry[0], carries[row]), (u, i)


def test_pooled_history_holds_both_parities(torch_ttnn, monkeypatch):
    model, kv, _ = _staged_model(monkeypatch)
    st = gbs.GdnBlockState(model, _specs(model), kv)
    st.write_prefill_rows(*_ids(model, [4, 6]), rows=[1, 2])
    nck = 2 * (DK // 32) + DV // 32
    for i, dn in st.layers:
        for base, row in ((4, 1), (6, 2)):
            blk = base + 8 * (i % 4)
            pooled = dn._paged_state.hist_view[blk, : NV * 4].reshape(NV, 4, 32, 32)
            src = dn._conv_hist_packed[row]
            for c in range(nck):
                assert torch.equal(pooled[:, :, 2 * c], src[:, :, 2 * c + (row & 1)])
                assert torch.equal(pooled[:, :, 2 * c + 1], src[:, :, 2 * c + (row & 1)])


def test_block_copies_run_in_order_over_every_buffer(torch_ttnn, monkeypatch):
    model, kv, _ = _staged_model(monkeypatch)
    for j, (k, v) in enumerate(kv):
        for blk in range(POOL_BLOCKS):
            k.bytes[blk] = 2 * blk + 1
            v.bytes[blk] = 2 * blk + 2
    gbs.copy_kv_blocks(model, [(3, 4), (4, 5), (6, 6)])  # 3 -> 4, then that copy -> 5; 6 -> 6 is skipped
    assert len(torch_ttnn.fills) == 2 * 2 * len(kv)
    for k, v in kv:
        assert int(k.bytes[4, 0]) == int(k.bytes[5, 0]) == 7 and int(v.bytes[5, 0]) == 8
        assert int(k.bytes[6, 0]) == 13 and int(k.bytes[3, 0]) == 7
    with pytest.raises(ValueError):
        gbs.copy_kv_blocks(model, [(0, 4)])


def test_unpack_rows_inverts_the_production_packing(monkeypatch):
    tab = _pack_tables(monkeypatch, 2)
    rows = gbs.unpack_rows(NV, NK, DK, DV, KCONV).reshape(-1)
    to_packed = tab["slot_to_packed"][0].reshape(-1)  # parity 0: packed row <- RM carry row
    for rm_row, packed_row in enumerate(rows.tolist()):
        assert int(to_packed[packed_row]) == rm_row
