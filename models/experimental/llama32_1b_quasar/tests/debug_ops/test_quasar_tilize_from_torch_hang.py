# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar from_torch(TILE) tilize hang seen during llama32_1b weight upload.

The llama model builds every weight with ``ttnn.from_torch(fp32_tensor, dtype=bfloat16, layout=TILE, ...)``,
which on device runs: upload row-major fp32 -> TilizeDeviceOperation (fp32 rowmajor->tile) ->
TypecastDeviceOperation (fp32->bf16). During bring-up on the Quasar simulator the run wedged with a
*wide* tilize (reader_unary_stick_layout_split_rows_multicore + tilize_metal2) frozen on core (0,0) at
NTW (reader NOC-transaction wait) / WFW (compute wait-front) — a producer/consumer credit stall. A tiny
32x32 tilize passed, so width is the suspected trigger (the wide-row path in the tilize picker).

This reproduces just that op, parametrized by width, so the deadlock can be debugged without the whole
model build. Run on the sim exactly as the model does, e.g.:

    TT_METAL_WATCHER=12 TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" \
        TT_METAL_SIMULATOR=~/sim/libttsim.so MESH_DEVICE=N150 \
        pytest tests/ttnn/unit_tests/operations/test_quasar_tilize_from_torch_hang.py

Bisect: the first ``shape`` id that hangs (never logs "readback complete") is the smallest width that
trips it. 32x32 is the known-good baseline; 32x2048 crosses the 32-tile wide-row threshold.
"""

import pytest
import torch
from loguru import logger

import ttnn


def _readback(tt, mesh_device):
    try:
        num = mesh_device.get_num_devices()
    except Exception:
        num = 1
    if num > 1:
        return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return ttnn.to_torch(tt)


# (H, W) logical shapes; width in tiles = W/32. The model's weights are 2D [K, N].
_SHAPES = [
    (32, 32),  # 1x1 tiles — known-good baseline (passed in the model log)
    (32, 512),  # 1 x 16 tiles
    (32, 1024),  # 1 x 32 tiles (at the wide-row threshold)
    (32, 2048),  # 1 x 64 tiles (over the threshold — the split_rows_multicore/wide-row path)
    (256, 2048),  # 8 x 64 tiles
    (2048, 2048),  # 64 x 64 tiles (square, large)
]


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_quasar_tilize_from_torch(mesh_device, shape):
    """from_torch(fp32 -> bf16, TILE) — the exact weight-upload op path that deadlocked."""
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.float32)

    logger.info(f"[tilize-repro] from_torch(TILE) begin shape=(1,1,{h},{w}) fp32->bf16")
    tt = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info(f"[tilize-repro] from_torch returned shape=(1,1,{h},{w}); reading back")
    out = _readback(tt, mesh_device).float()
    logger.info(f"[tilize-repro] readback complete shape=(1,1,{h},{w})")

    assert tuple(out.shape)[-2:] == (h, w), f"shape mismatch: {tuple(out.shape)} vs (1,1,{h},{w})"
    assert torch.isfinite(out).all(), "non-finite values after from_torch(TILE)"
    # bf16 round-trip of the fp32 input; loose tolerance covers the dtype cast.
    ref = t.to(torch.bfloat16).float()
    assert torch.allclose(out.reshape(ref.shape), ref, atol=0.05, rtol=0.05), "value mismatch after tilize+typecast"


@pytest.mark.parametrize("shape", _SHAPES, ids=[f"{h}x{w}" for (h, w) in _SHAPES])
def test_quasar_experimental_tilize(mesh_device, shape):
    """Same tilize, but via the Gen2-native ttnn.experimental.quasar.tilize op instead of the mainline
    TilizeDeviceOperation that from_torch(TILE) uses.

    Uploads a ROW_MAJOR tensor first (plain to_device, no device tilize), then calls the quasar tilize op
    directly on it. If this passes where test_quasar_tilize_from_torch hangs, the experimental/quasar
    tilize is the working path (and the model's from_torch(TILE) should route through it, or upload
    row-major + ttnn.experimental.quasar.tilize). Input is bf16 so there is no typecast — this isolates
    the tilize itself.
    """
    h, w = shape
    torch.manual_seed(0)
    t = torch.randn(1, 1, h, w, dtype=torch.bfloat16)

    logger.info(f"[qsr-tilize] upload row-major (1,1,{h},{w})")
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # no device tilize on upload
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info(f"[qsr-tilize] ttnn.experimental.quasar.tilize begin (1,1,{h},{w})")
    tt = ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    logger.info(f"[qsr-tilize] tilize done (1,1,{h},{w}); reading back")
    out = _readback(tt, mesh_device).float()
    logger.info(f"[qsr-tilize] readback complete (1,1,{h},{w})")

    assert tuple(out.shape)[-2:] == (h, w), f"shape mismatch: {tuple(out.shape)} vs (1,1,{h},{w})"
    assert torch.isfinite(out).all(), "non-finite values after quasar.tilize"
    assert torch.allclose(out.reshape(t.float().shape), t.float(), atol=0.05, rtol=0.05), "value mismatch after tilize"
