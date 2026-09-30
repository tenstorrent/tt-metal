# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 prefill cache layout (bead F6.3): geometry goldens, device writes over chunks, and the device-side
zero initialization of large caches (bead F10)."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.v41.cache import (
    WINDOW_SLOT,
    ZERO_BLOCK_BYTES,
    V41CacheGeometry,
    V41PrefillState,
    replicated_zeros,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat


def test_geometry_goldens(expect_error):
    g = V41CacheGeometry(C, max_seq_len=20480, chunk=5120, sp=2)
    assert g.window_rows == 128 + 5120
    assert g.kv_rows(2) == 5248 + 10240 and g.kv_rows(1) == 5248 + 20480 and g.kv_rows(0) == 5248
    assert g.new_compressed_rows(5120, 5120, 2) == (2560, 2560)
    assert g.new_compressed_rows(10240, 1001, 2) == (5120, 500)  # odd valid length: last group incomplete
    assert g.new_compressed_rows(0, 1001, 1) == (0, 1001)
    assert g.kv_row_of_compressed(7) == 5248 + 7
    assert g.kv_row_of_window(5120 - 127, 5120) == 1 and g.kv_row_of_window(5120 + 3, 5120) == 131
    with expect_error(AssertionError, "multiple of 2\\*32\\*sp"):
        V41CacheGeometry(C, max_seq_len=20480, chunk=5120 + 64, sp=2)


MESH_2X4 = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]


@pytest.mark.parametrize("mesh_device, device_params", MESH_2X4, indirect=True)
def test_state_writes_over_chunks(mesh_device, device_params):
    """Two full chunks and a padded tail through SP-sharded rows; host model of the layout as expected."""
    chunk, max_seq = 1024, 3072
    layers = [0, 2, 3, 20]  # ratio 0, ratio-2 source, its consumer, ratio-1 source
    state = V41PrefillState(mesh_device, C, max_seq, chunk, layers)
    shape = tuple(mesh_device.shape)
    gen = torch.Generator().manual_seed(3)
    d, idim = C.HEAD_DIM, C.INDEX_HEAD_DIM
    expect_kv = {l: torch.zeros(state.geometry.kv_rows(C.compress_ratio(l)), d) for l in (2, 20)}
    expect_idx = {l: torch.zeros(state.geometry.compressed_rows(C.compress_ratio(l)), idim) for l in (2, 20)}
    expect_carry = {l: torch.zeros(WINDOW_SLOT, d) for l in layers}

    def sp_sharded(t):
        return ttnn.from_torch(
            t[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, None)),
        )

    def host(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0, 0]

    for length in (chunk, chunk, 700):
        start = state.start
        for source in (2, 20):
            r = C.compress_ratio(source)
            rows_kv = torch.randn(chunk // r, d, generator=gen).to(torch.bfloat16)
            rows_ik = torch.randn(chunk // r, idim, generator=gen).to(torch.bfloat16)
            state.write_compressed(source, sp_sharded(rows_kv), sp_sharded(rows_ik), length)
            first, count = state.geometry.new_compressed_rows(start, length, r)
            w = state.geometry.window_rows
            expect_kv[source][w + first : w + first + count] = rows_kv[:count].float()
            expect_idx[source][first : first + count] = rows_ik[:count].float()
        for layer in layers:
            window = torch.randn(chunk, d, generator=gen).to(torch.bfloat16)
            dst = state.write_window(layer, sp_sharded(window))
            region = torch.cat([expect_carry[layer], window.float()])
            assert torch.equal(host(dst)[: state.geometry.window_rows], region), f"layer {layer} window region"
            state.update_window_carry(layer, length)
            expect_carry[layer] = region[length : length + WINDOW_SLOT]
            assert torch.equal(host(state.window_carry[layer]), expect_carry[layer]), f"layer {layer} carry"
        state.advance(length)

    for source in (2, 20):
        w = state.geometry.window_rows
        assert torch.equal(host(state.kv[source])[w:], expect_kv[source][w:]), f"compressed KV of {source}"
        assert torch.equal(host(state.index_k[source]), expect_idx[source]), f"index-K of {source}"
    assert state.kv_tensor(3) is state.kv[2] and state.kv_tensor(0) is state.swa_scratch


def _all_chips(mesh_device, t) -> torch.Tensor:
    """Raw stored values of every chip's replica, [chips, rows, width] (FP8 leaves through a mesh composer)."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:, 0].float()


@pytest.mark.parametrize("mesh_device, device_params", MESH_2X4, indirect=True)
def test_large_cache_zeroed_on_device(mesh_device, device_params):
    """Caches above ZERO_BLOCK_BYTES are zeroed by device copies of one uploaded block, on every chip, over memory
    that held non-zero data, for both KV storage formats; a tensor within one block is a plain upload."""
    width = 512
    step = ZERO_BLOCK_BYTES // (width * 2)
    rows = 2 * step + 96  # two whole blocks and a partial one
    # dirty the DRAM the zero tensors are likely to reuse
    junk = ttnn.from_torch(
        torch.full((1, 1, 4 * rows, width), 7.0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.deallocate(junk)
    for dtype in (ttnn.bfloat16, MlaKvCacheFormat.SCALED_FP8.storage_dtype):
        for n in (rows, 64):
            t = replicated_zeros(mesh_device, n, width, dtype)
            assert list(t.shape) == [1, 1, n, width] and t.dtype == dtype and t.layout == ttnn.ROW_MAJOR_LAYOUT
            values = _all_chips(mesh_device, t)
            assert values.shape[0] == mesh_device.get_num_devices() and bool((values == 0).all()), (dtype, n)
            ttnn.deallocate(t)
    # the state's ratio-1 KV tensor (window region + one row per token) spans several zero blocks
    chunk, max_seq = 1024, 8192
    for fmt in V41PrefillState.FORMATS:
        state = V41PrefillState(mesh_device, C, max_seq, chunk, [0, 2, 3, 20], kv_format=fmt)
        kv = state.kv[20]
        assert kv.shape[2] == state.geometry.kv_rows(1) and kv.shape[2] * kv.shape[3] > ZERO_BLOCK_BYTES
        for t in (kv, state.kv[2], state.index_k[20], state.swa_scratch, state.window_carry[20]):
            assert bool((_all_chips(mesh_device, t) == 0).all()), fmt
