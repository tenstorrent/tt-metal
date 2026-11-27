# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the OFT per-op test suite (``tests/oft_ops/``).

Goal
----
Demonstrate that the ttnn ops the OFT (Orthographic Feature Transform) 3D-detection model runs
— *with that model's shapes* — work on Quasar. These tests import NO model code: each op's
shapes / dtypes / memory configs were reverse-engineered from a **tt-metal kernel cache** of
one Blackhole end-to-end run of the model (``tt-metal-cache-oft-e2e``).

How the shapes were recovered
-----------------------------
The cache directory is keyed by main commit ``1088b10b014`` (2026-02-02); the program
factories at that commit map each kernel's compile-time args (read from the ELF DWARF,
``kernel_compile_time_args``) back to per-core blocking, page sizes, shard specs, dtypes and
fused activations. Runtime args are NOT cached. Docstrings mark each fact:

  * ``[H]`` read directly from compile-time args / defines / CB data formats,
  * ``[M]`` op / dtype / memory config read directly; shape inferred from the architecture,
  * ``[L]`` representative value inside the recovered bounds.

The OFT model lived in ``models/experimental/oft`` until it was removed (``ae4ef6b7c16``), but
the cached run does **not** match that code: it was built from a modified out-of-tree
checkout. The cache is ground truth. What the run actually did:

  * ResNet-18 frontend on a 1x3x384x1280 image; only the stem, maxpool, layer1 and layer2 ran
    on device (layer2 in float32). Every frontend / topdown conv has a bias, conv1 of each block
    has a fused ReLU, weights are bf16. GroupNorm ran on host (no group_norm kernels at all).
  * Only the 1/8 scale (lat8, [1, 48, 160, 256]) went through the device OFT: fp32 integral
    image, *nearest* grid_sample with a precomputed grid of 7 height points per stick, 44 BEV
    slices of 576 rows (159*159 + 63 pad = 25344), fp32 corner arithmetic, and a LoFi linear
    1792 -> 128.
  * Topdown (8 BasicBlocks) and the head ran at **128** channels on the 159x159 BEV grid.
  * Encoder: 5x5 NMS conv + 3x3 max_pool2d with indices, dims decode (exp) and ``atan2``.
  * The run used a **5x4 = 20-core worker grid**: every sharded tensor, work split and mcast
    in the cache has 20 cores.

Conventions (mirror ``tests/vad_variant_ops/``):
  * use the ``ttnn_mesh_device`` fixture, parametrized ``indirect``,
  * compare against a torch reference with ``assert_pcc``; ``assert_lossless`` for
    value-preserving data movement and exact gathers,
  * parametrize with a ``shape`` / ``in_shape`` tuple or an ``hw`` int so conftest can tag
    emulator-sized cases; mark cases pinned to the 20-core grid ``blackhole_scale``.
"""

from __future__ import annotations

import math

import pytest  # noqa: F401  (re-exported convenience for op files)
import torch  # noqa: F401

import ttnn

# Generic helpers shared with the other Quasar op suites. The VAD-variant model also ran on a
# 5x4 grid, so its 20-core memory-config helpers apply unchanged.
from models.experimental.ops.quasar.tests.vad_variant_ops.op_utils import (  # noqa: F401
    DTYPES,
    MODEL_GRID_X,
    MODEL_GRID_Y,
    MODEL_NUM_CORES,
    assert_lossless,
    assert_pcc,
    assert_shape_dtype,
    comp_pcc,
    from_tt,
    hs20,
    is_quasar,
    model_grid,
    resolve_memcfg,
    run_binary,
    seen,
    to_tt,
    to_tt_mem,
    torch_rand,
    with_default_mesh,
    with_mesh_l1small,
    xfail_on_quasar,
)

# =============================================================================
# Model constants (recovered from the kernel cache)
# =============================================================================

TILE = 32

# Image / ResNet-18 frontend (only stem .. layer2 ran on device).
IMG_H, IMG_W = 384, 1280
STEM_H, STEM_W = IMG_H // 2, IMG_W // 2  # 192 x 640 conv1 output
L1_H, L1_W = IMG_H // 4, IMG_W // 4  # 96 x 320 layer1
L2_H, L2_W = IMG_H // 8, IMG_W // 8  # 48 x 160 layer2 / lat8

# OFT (1/8 scale only)
OFT_C = 256  # lateral channels
GRID_POINTS = 7  # y-heights per BEV cell (grid_height 4 / cell 0.5 -> 8 corners -> 7 cells)
BEV = 159  # BEV cells per side
BEV_NHW = BEV * BEV  # 25281
PAD_AMOUNT = 63
BEV_ROWS = BEV_NHW + PAD_AMOUNT  # 25344
NUM_SLICES = 44
SLICE_ROWS = BEV_ROWS // NUM_SLICES  # 576
VOX_C = GRID_POINTS * OFT_C  # 1792 = OFT linear in-features
TD_C = 128  # OFT linear out-features = topdown / head channels
HEAD_OUT = 9  # 1 score + 3 pos + 3 dim + 2 ang

# The NMS heatmap and decode tensors are [1, 159, 159] (one class).
HEATMAP = (1, BEV, BEV)

# Emulator classification thresholds (heuristic; tune on the emulator).
EMU_MAX_HW = 48
EMU_MAX_ELEMS = 48 * 48 * 256

# conv2d / max_pool2d allocate from the device L1-small region.
OFT_L1_SMALL_SIZE = 32768


# =============================================================================
# Backend selection
# =============================================================================


def op(name: str, *, experimental: bool = False):
    """The ttnn op ``name``, using its Quasar port on Quasar.

    Mainline factories that still build legacy ``DataMovementKernel`` / ``ComputeKernel``
    objects are rejected on Quasar (``tt_metal/impl/kernels/kernel.hpp``); the ports live in
    ``ttnn.experimental.quasar``. Elsewhere the mainline op (``ttnn.<name>``, or
    ``ttnn.experimental.<name>`` for ``experimental=True``) is used. Ops whose mainline
    factory is already Metal 2.0 (unary, permute, intimg, interleaved<->sharded) and ops with
    no port yet are called directly instead.
    """
    if is_quasar():
        return getattr(ttnn.experimental.quasar, name)
    return getattr(ttnn.experimental if experimental else ttnn, name)


def xfail_no_quasar_port(op_name: str) -> None:
    """xfail on Quasar for an op whose mainline factory builds legacy kernels and has no port yet."""
    xfail_on_quasar(f"{op_name} has no Quasar port: its mainline factory builds legacy DataMovementKernels")


def hs_rows(rows: int, num_cores: int = MODEL_NUM_CORES) -> int:
    """Shard height the model used for ``rows`` height-sharded over ``num_cores`` (tile-rounded)."""
    return math.ceil(rows / num_cores / TILE) * TILE


def model_core_range_set(mesh_device, num_cores: int = MODEL_NUM_CORES) -> ttnn.CoreRangeSet:
    """The first ``num_cores`` cores of the model's 5x4 grid, row-wise (skips on smaller grids)."""
    model_grid(mesh_device)
    return ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(MODEL_GRID_X, MODEL_GRID_Y), row_wise=True)


def rand_nonneg(shape, dtype=torch.float32):
    """Post-ReLU style activations (the OFT features feed an integral image)."""
    return torch.rand(*shape).to(dtype)
