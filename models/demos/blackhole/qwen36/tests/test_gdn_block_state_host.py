# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: GdnBlockState (QWEN36_MAMBA_SPEC) page math and index updates, with ttnn stubbed out."""

from types import SimpleNamespace

import pytest
import torch

import models.demos.blackhole.qwen36.tt.gdn_block_state as gbs

NV, DK, DV = 12, 128, 128  # TP=4 per-device GDN shapes of Qwen3.6-27B
NUM_BLOCKS, HD = 10, 256


class _Tensor(SimpleNamespace):
    pass


@pytest.fixture
def fake(monkeypatch):
    calls = {"views": [], "h2d": []}

    def alias_view(t, shape, dtype):
        calls["views"].append((t.name, list(shape), dtype))
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
    return calls


def _model(n_layers=8, bmax=8, nv=NV):
    attn_idx = [i for i in range(n_layers) if i % 4 == 3]
    layers = []
    for i in range(n_layers):
        dn = SimpleNamespace(Nv=nv, Dk=DK, Dv=DV, _decode_op=True, use_fused_recurrent_decode=False)
        layers.append(SimpleNamespace(is_full_attention=i in attn_idx, attention=dn))
    return SimpleNamespace(
        mesh_device="mesh", args=SimpleNamespace(max_batch_size=bmax), _attention_layer_indices=attn_idx, layers=layers
    )


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
    "dtype, block, nv, rec_pages, hist_tiles",
    [
        ("bf8", 2048, 6, 136, 272),  # TP=8 (Galaxy shapes): 512 tiles x 1088 B = 557,056 B = 136 fp32 pages
        ("bf8", 4096, 12, 272, 544),  # TP=4: 1,114,112 B per block
        ("bf16", 2048, 12, 256, 512),  # 1 MiB per block
    ],
)
def test_views_and_page_counts(fake, dtype, block, nv, rec_pages, hist_tiles):
    model = _model(nv=nv)
    st = gbs.GdnBlockState(model, _specs(model, dtype, block), _kv(2, dtype, block))
    assert len(st.layers) == 6
    i, dn = st.layers[0]
    ps = dn._paged_state
    assert ps.rec_pages == rec_pages and ps.hist_pages == hist_tiles
    assert ps.rec_view.shape == [NUM_BLOCKS * rec_pages // 8, 1, DK // 2, DV] and ps.rec_view.dtype == "fp32"
    assert ps.hist_view.shape == [NUM_BLOCKS, hist_tiles, 32, 32] and ps.hist_pool is ps.hist_view
    assert ps.hist_view.name == "v0:view"


@pytest.mark.parametrize("dtype, block, nv", [("bf8", 2048, 12), ("bf8", 1024, 6), ("bf16", 1024, 12)])
def test_blocks_without_room_for_the_state_are_refused(fake, dtype, block, nv):
    # BF8 2048 at TP=4: 136 pages < 192; BF8 1024: 8.5 fp32 pages per bank; bf16 1024 at TP=4: 128 < 192.
    model = _model(nv=nv)
    with pytest.raises(ValueError):
        gbs.GdnBlockState(model, _specs(model, dtype, block), _kv(2, dtype, block))


def test_decode_index_follows_rows_and_skips_unchanged(fake):
    model = _model()
    st = gbs.GdnBlockState(model, _specs(model), _kv(2))
    ids = [torch.tensor([[5], [7], [0], [0], [0], [0], [0], [0]], dtype=torch.int32) for _ in model.layers]
    row_of = [3, 0, 1, 2, 4, 5, 6, 7]  # runner slot 0 -> device row 3, slot 1 -> row 0
    st.set_decode_blocks(ids, row_of=row_of)
    i, dn = st.layers[0]
    idx = dn._paged_state.idx.data
    assert idx[0].tolist() == [7 * 272, 0, 0, 5 * 272, 0, 0, 0, 0]
    assert idx[1].tolist() == [7 * 544, 0, 0, 5 * 544, 0, 0, 0, 0]
    n = len(fake["h2d"])
    assert n == len(st.layers)
    st.set_decode_blocks(ids, row_of=row_of)
    assert len(fake["h2d"]) == n  # unchanged indices: no copies
