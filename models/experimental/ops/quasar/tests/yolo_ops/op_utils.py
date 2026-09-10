# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the YOLOv8 per-op test suite (``tests/yolo_ops/``).

Goal
----
Demonstrate that every ttnn op the YOLOv8 model runs — *with that model's shapes* —
works on Quasar, without landing any YOLOv8 model code in main (licensing). These
tests import NO YOLOv8 code; they call the raw ttnn ops directly with the shapes the
model uses (extracted from the ``sdawle/yolov8_bh`` branch:
``models/demos/yolov8{l,s}/tt/*``). Shapes cover the yolov8l and yolov8s variants at
640x640 input, batch 1.

Conventions (mirror tests/ops/):
  * import dims/helpers from here; ground each shape in a real model call site
    (cite the branch file:line in a comment),
  * use the ``ttnn_mesh_device`` fixture (from tests/conftest.py), parametrized
    ``indirect`` — default ``[(1, 1)]``,
  * compare against a torch reference with ``assert_pcc`` where a reference exists;
    otherwise ``assert_shape_dtype`` (shape/dtype/finite).

Emulator subset: the ``emulator`` marker (added by tests/yolo_ops/conftest.py) selects
cases small enough for the 2-node emulator. For YOLO most activations are large
(320/160/80x80 feature maps), so only the smallest (≈ resolution ≤ 40) fit; the rest
run on real Blackhole. Parametrize spatially-sized ops with an int ``hw`` (input
H == W) and/or a ``shape`` tuple so the conftest can classify them.
"""

from __future__ import annotations

import pytest  # noqa: F401  (re-exported convenience for op files)
import torch

import ttnn

# Reuse the generic tensor/assert helpers from the llama op suite (arch-agnostic).
from models.experimental.llama32_1b_quasar.tests.ops.op_utils import (  # noqa: F401
    assert_pcc,
    assert_shape_dtype,
    comp_pcc,
    from_tt,
    to_tt,
    torch_rand,
    with_default_mesh,
)

# =============================================================================
# YOLOv8 (640x640, batch 1) constants
# =============================================================================

BATCH = 1
INPUT_RES = 640
IMG_CH = 3
IMG_CH_PADDED = 16  # model pads the 3-channel input to 16 (test_yolov8l.py: F.pad ... 0,13)
TILE = 32

# Feature-map resolutions the 640x640 backbone/neck operate at (stride 2 halving):
#   640 -> 320 -> 160 -> 80 -> 40 -> 20.  Detect head runs at {80, 40, 20}.
RESOLUTIONS = [320, 160, 80, 40, 20]

# Emulator classification thresholds (heuristic; tune on the emulator).
# A case fits the 2-node emulator when its spatial size / element count is small.
EMU_MAX_HW = 40  # input H==W (feature-map side) that still fits 2 cores
EMU_MAX_ELEMS = 40 * 40 * 512  # ~0.8M elements — fallback for shape-tuple params

# conv2d / max_pool2d allocate from the device L1-small region; the model opens the
# device with this size (yolov8l common.py: _YOLOV8L_L1_SMALL_BASE_640). The default
# test fixture opens with l1_small_size=0, so those ops OOM ("L1_SMALL ... bank size is
# 0 B") unless the device is opened with a nonzero l1_small_size. 32768 gives headroom
# over the model's 24576 base for a single isolated op.
YOLOV8_L1_SMALL_SIZE = 32768


def with_mesh_l1small(l1_small_size=YOLOV8_L1_SMALL_SIZE, mesh=(1, 1)):
    """Parametrize ``ttnn_mesh_device`` to open the device with a nonzero l1_small_size.

    Needed by conv2d / max_pool2d (they allocate from the L1-small region). The fixture
    accepts a dict param ``{"mesh_shape", ...device kwargs}``; conftest._fits_emulator
    reads ``mesh_shape`` out of it for emulator classification.
    """
    return pytest.mark.parametrize(
        "ttnn_mesh_device",
        [pytest.param({"mesh_shape": mesh, "l1_small_size": l1_small_size}, id="mesh")],
        indirect=True,
    )


def to_tile_l1(torch_tensor, mesh_device, *, dtype=ttnn.bfloat16):
    """TILE tensor in L1 — generated via DRAM to dodge the tilize-into-L1 NOC bug.

    ``from_torch(..., layout=TILE, memory_config=L1)`` tilizes on device; for
    non-tile-aligned shapes (e.g. dist2bbox (1,2,8400) — the 2/4 row dim padded to 32)
    the device tilize_with_val_padding into L1 overflows NOC ("NOC target address
    overflow"). Tilizing into DRAM works (the existing DRAM tests pass), so we tilize
    into DRAM and then relocate DRAM->L1 (a plain copy, no re-tilize). The model never
    tilizes these from host anyway — they arrive in L1 via graph ops — so this produces
    the same TILE/L1 state the op under test consumes.
    """
    x = to_tt(torch_tensor, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)


def _corerangeset(mesh_device, num_cores):
    """CoreRangeSet of ``num_cores`` (row-major) on this device; skip if it can't fit."""
    grid = mesh_device.compute_with_storage_grid_size()
    if num_cores > grid.x * grid.y:
        pytest.skip(f"model-faithful shard needs {num_cores} cores; device has {grid.x * grid.y}")
    return ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)


def height_sharded_memcfg(mesh_device, num_cores, shape):
    """HEIGHT_SHARDED memory config over ``num_cores`` — the model's neck/SPPF sharding.

    ``shape`` is the full logical shape; it's HEIGHT-sharded over ``num_cores`` — each
    core holds ``(prod(shape[:-1]) / num_cores, shape[-1])``. ttnn requires the explicit
    per-core shard shape (use_height_and_width_as_shard_shape=True) when the grid is a
    CoreRangeSet, which ``num_cores_to_corerangeset`` produces for non-rectangular core
    counts (e.g. 20/40). Skips when the device lacks ``num_cores`` or the rows don't
    divide evenly.
    """
    total_rows = 1
    for d in shape[:-1]:
        total_rows *= d
    width = shape[-1]
    if total_rows % num_cores != 0:
        pytest.skip(f"height {total_rows} not divisible by {num_cores} cores")
    shard_shape = (total_rows // num_cores, width)
    return ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=_corerangeset(mesh_device, num_cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def height_sharded_tile_memcfg(mesh_device, shape, *, max_cores=None):
    """HEIGHT_SHARDED memcfg for a **TILE** tensor — picks the largest core count whose
    per-core shard height is a multiple of TILE (32), which TILE sharding requires
    ("Physical shard shape must be tile sized").

    Use this for TILE conv-output tensors (e.g. the bottleneck residual add); use
    ``height_sharded_memcfg`` for ROW_MAJOR feature maps where the shard height is free.
    The exact model grid isn't recoverable statically, so we pick a valid tile-aligned
    grid — faithful in layout/strategy/buffer, approximate only in core count.
    """
    total_rows = 1
    for d in shape[:-1]:
        total_rows *= d
    width = shape[-1]
    grid = mesh_device.compute_with_storage_grid_size()
    avail = grid.x * grid.y
    if max_cores is not None:
        avail = min(avail, max_cores)
    chosen = next(
        (nc for nc in range(avail, 0, -1) if total_rows % nc == 0 and (total_rows // nc) % TILE == 0),
        None,
    )
    if chosen is None:
        pytest.skip(f"no tile-aligned height shard for {total_rows} rows on <= {avail} cores")
    return ttnn.create_sharded_memory_config(
        shape=(total_rows // chosen, width),
        core_grid=ttnn.num_cores_to_corerangeset(chosen, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def block_sharded_memcfg(mesh_device, grid_y, grid_x, shape):
    """BLOCK_SHARDED memory config on a (grid_y, grid_x) core grid — the model's
    ``block_shard=True`` conv outputs. Skips if the grid doesn't fit the device."""
    grid = mesh_device.compute_with_storage_grid_size()
    if grid_y > grid.y or grid_x > grid.x:
        pytest.skip(f"model-faithful block shard needs {grid_y}x{grid_x}; device has {grid.y}x{grid.x}")
    return ttnn.create_sharded_memory_config(
        shape=tuple(shape),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def assert_lossless(torch_ref, tt_out, *, mesh_device=None):
    """Exact-equality check for **value-preserving** ops (clone, reshape, permute, slice,
    split, pad, concat, transpose, untilize, layout/sharding moves, reallocate...).

    Those ops must not alter data, so a PCC check is too weak — on the large tensors here
    it can pass even when values are corrupted. We require the logical data to be bit-exact
    (bf16 in == bf16 out). Compare in float32; align on the reference's element count when
    the device output is tile-padded (logical data comes first, row-major).
    """
    got = from_tt(tt_out, mesh_device)
    ref = torch_ref.float()
    if got.shape != ref.shape and got.numel() >= ref.numel():
        got = got.reshape(-1)[: ref.numel()].reshape(ref.shape)
    if not torch.equal(got, ref):
        diff = (got - ref).abs()
        raise AssertionError(
            f"value-preserving op changed data: {(got != ref).sum().item()}/{ref.numel()} "
            f"elements differ, max|diff|={diff.max().item():.6g}"
        )


def nhwc_to_tt(torch_nchw: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """torch NCHW -> ttnn NHWC row-major (the layout ttnn.conv2d / max_pool2d consume).

    YOLOv8 activations enter conv/pool as [N, H, W, C] row-major on device. We permute
    NCHW->NHWC on the host and upload row-major.
    """
    nhwc = torch_nchw.permute(0, 2, 3, 1).contiguous()
    return ttnn.from_torch(
        nhwc,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
