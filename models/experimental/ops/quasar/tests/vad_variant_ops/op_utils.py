# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the VAD-variant per-op test suite (``tests/vad_variant_ops/``).

Goal
----
Demonstrate that the ttnn ops a VAD/SSR-style autonomous-driving model runs — *with that
model's shapes* — work on Quasar. The model itself can't be run or landed here; it lives
on a fork. These tests import NO model code: each op's shapes / dtypes /
memory configs were reverse-engineered from a **tt-metal kernel cache** of one Blackhole run
of the model (``tt-metal-cache/3204133098312896700``).

How the shapes were recovered
-----------------------------
Each ``kernels/<name>/<hash>/`` cache entry holds the generated defines, CB data formats,
math fidelity, and the compiled ELF. The compile-time args survive in the ELF's DWARF
(``kernel_compile_time_args`` has a ``DW_AT_const_value``). The program factories
map those args back to per-core block sizes, page sizes, shard specs, and so on.
Runtime args are NOT cached, so:

  * shapes are **cache-determined** where a compile-time arg pins them (sharded tensors,
    conv/matmul blocking, page sizes, reduction extents) — marked ``[H]`` in docstrings;
  * shapes are **inferred** from the model architecture where only the op / dtype /
    memory config is pinned (interleaved eltwise ops) — marked ``[M]`` / ``[L]``.

Each test docstring cites the cache kernel(s) (``<kernel>/<hash prefix>``) its cases come from.

Model facts recovered from the cache
------------------------------------
ResNet-50 backbone on 6 camera images of 384x640 (run one image at a time), so the
stride-32 feature map is 12x20 = 240 keys at 1 level. The BEVFormer-style encoder has a
100x100 = 10000-query BEV grid, embed 256, 8 heads x 32 head_dim. Temporal self-attention
uses 4 sampling points over a 2-frame queue; spatial cross-attention uses 8 points.
The run used a **20-core (5x4) worker grid** — every work split and sharded tensor
accessor in the cache has 20 banks — so the "model-faithful" sharded cases shard over
``CoreRange((0,0),(4,3))``.

Fork vs main
------------
The fork predates some main changes (it still has ``eltwise/unary_ng``), so on main an op may
route to a different program factory than in the cache. These tests pin the *op call* (shapes,
dtypes, memory configs, compute config) rather than the kernel that main dispatches to.
The fork's custom ops (deformable attention, nonzero, getitem, setitem, reshard)
are not in main; they are covered with their closest stock-ttnn equivalents at the
recovered shapes.

Conventions (mirror tests/yolo_ops/):
  * use the ``ttnn_mesh_device`` fixture (from tests/conftest.py), parametrized ``indirect``,
  * compare against a torch reference with ``assert_pcc``; ``assert_lossless`` for
    value-preserving data movement,
  * parametrize with a ``shape`` / ``in_shape`` tuple or an ``hw`` int so conftest can tag
    emulator-sized cases; mark 20-core sharded cases ``blackhole_scale``.
"""

from __future__ import annotations

import pytest  # noqa: F401  (re-exported convenience for op files)
import torch  # noqa: F401

import ttnn

# Generic tensor/assert helpers (arch-agnostic), shared with the other Quasar op suites.
from models.experimental.llama32_1b_quasar.tests.ops.op_utils import (  # noqa: F401
    assert_pcc,
    assert_shape_dtype,
    comp_pcc,
    from_tt,
    to_tt,
    torch_rand,
    with_default_mesh,
)
from models.experimental.ops.quasar.tests.yolo_ops.op_utils import (  # noqa: F401
    assert_lossless,
    nhwc_to_tt,
    with_mesh_l1small,
)

# =============================================================================
# Model constants (recovered from the kernel cache)
# =============================================================================

TILE = 32

# Camera images / backbone
NUM_CAMS = 6
IMG_H, IMG_W = 384, 640
FEAT_H, FEAT_W = IMG_H // 32, IMG_W // 32  # 12 x 20 stride-32 map
NUM_KEYS = FEAT_H * FEAT_W  # 240 keys per camera (1 FPN level)

# BEV encoder
BEV_H = BEV_W = 100
NUM_QUERY = BEV_H * BEV_W  # 10000
EMBED = 256
FFN = 512
NUM_HEADS = 8
HEAD_DIM = EMBED // NUM_HEADS  # 32
TSA_POINTS = 4  # temporal self-attention
SCA_POINTS = 8  # spatial cross-attention
BEV_QUEUE = 2  # temporal self-attention batch (prev + cur BEV)

# Per-camera-group padded hit counts the spatial cross-attention ran with
# (nonzero output padded to a multiple of 32): batch 1/1/2/2 -> 6 cameras.
SCA_QUERY_GROUPS = [(1, 3680), (1, 3072), (2, 1376), (2, 992)]

# Worker grid the model ran on (5 x 4 = 20 cores).
MODEL_GRID_X, MODEL_GRID_Y = 5, 4
MODEL_NUM_CORES = MODEL_GRID_X * MODEL_GRID_Y

# Emulator classification thresholds (heuristic; tune on the emulator).
EMU_MAX_HW = 48  # conv/pool input side that still fits the 2-node emulator
EMU_MAX_ELEMS = 48 * 48 * 256  # ~0.6M elements — fallback for shape-tuple params

# conv2d / max_pool2d allocate from the device L1-small region.
VAD_L1_SMALL_SIZE = 32768


def is_quasar() -> bool:
    return "quasar" in ttnn.get_arch_name()


def xfail_on_quasar(reason: str) -> None:
    """Mark the running case xfail on Quasar for a known, tracked Quasar limitation."""
    if is_quasar():
        pytest.xfail(reason)


# =============================================================================
# Memory-config helpers for the model's 20-core grid
# =============================================================================


def model_grid(mesh_device, grid_x=MODEL_GRID_X, grid_y=MODEL_GRID_Y) -> ttnn.CoreRangeSet:
    """A ``grid_x`` x ``grid_y`` sub-grid of the model's 5x4 worker grid; skip when the device can't provide it."""
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < grid_x or grid.y < grid_y:
        pytest.skip(f"model-faithful shard needs a {grid_x}x{grid_y} grid; device has {grid.x}x{grid.y}")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})


def hs20(mesh_device, shard_shape) -> ttnn.MemoryConfig:
    """HEIGHT_SHARDED, ROW_MAJOR orientation over the model's 20 cores with an explicit shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(model_grid(mesh_device), list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )


def bs_colmajor(mesh_device, shard_shape, grid_x=MODEL_GRID_X) -> ttnn.MemoryConfig:
    """BLOCK_SHARDED, COL_MAJOR orientation (height split over x, width over the 4 y-cores).

    ResNet layer3 tensors use the full 5x4 grid; layer4 tensors use 3x4 (240 rows / 96-row shards).
    """
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(model_grid(mesh_device, grid_x), list(shard_shape), ttnn.ShardOrientation.COL_MAJOR),
    )


def resolve_memcfg(spec, mesh_device):
    """Resolve a compact memory-config spec used in case tables.

    ``"dram"`` / ``"l1"`` -> interleaved; ``("hs", (h, w))`` -> ``hs20``;
    ``("bs", (h, w)[, grid_x])`` -> ``bs_colmajor``.
    Specs (not MemoryConfig objects) keep the parametrize tables device-free.
    """
    if spec == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if spec == "l1":
        return ttnn.L1_MEMORY_CONFIG
    kind, shard, *rest = spec
    if kind == "hs":
        return hs20(mesh_device, shard)
    if kind == "bs":
        return bs_colmajor(mesh_device, shard, *rest)
    raise ValueError(f"unknown memory-config spec {spec!r}")


def is_sharded_spec(spec) -> bool:
    return isinstance(spec, tuple)


def to_tt_mem(torch_tensor, mesh_device, spec, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Upload to DRAM, then move to ``spec`` (interleaved or sharded).

    Going via DRAM avoids device tilize straight into L1/sharded buffers, and is how these
    tensors reach the op in the model anyway (produced by a previous op, not uploaded).
    """
    x = to_tt(torch_tensor, mesh_device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if spec == "dram":
        return x
    return ttnn.to_memory_config(x, resolve_memcfg(spec, mesh_device))


DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}


# =============================================================================
# Binary-op runner shared by the eltwise test files
# =============================================================================


def run_binary(
    mesh,
    tt_op,
    torch_op,
    a_shape,
    b_shape,
    a_dt,
    b_dt,
    a_mem,
    b_mem,
    out_mem,
    *,
    relu=False,
    layout=ttnn.TILE_LAYOUT,
    b_transform=None,
    pcc=0.99,
):
    """Run ``tt_op(a, b)`` with the cached dtypes / memory configs and compare to ``torch_op``.

    ``a_mem`` / ``b_mem`` / ``out_mem`` are ``resolve_memcfg`` specs. ``relu`` fuses ReLU via
    ``activations`` (the ResNet residual adds ran with ``PACK_RELU``). ``b_transform`` maps the
    random torch ``b`` into the op's valid domain (e.g. nonzero divisors).
    """
    a_torch = torch_rand(a_shape)
    b_torch = torch_rand(b_shape)
    if b_transform is not None:
        b_torch = b_transform(b_torch)
    a = to_tt_mem(a_torch, mesh, a_mem, dtype=DTYPES[a_dt], layout=layout)
    b = to_tt_mem(b_torch, mesh, b_mem, dtype=DTYPES[b_dt], layout=layout)

    kwargs = {"memory_config": resolve_memcfg(out_mem, mesh)}
    if relu:
        kwargs["activations"] = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
    out = tt_op(a, b, **kwargs)

    # Reference on the values the device actually saw (bfp8 quantization included).
    a_seen = from_tt(a, mesh).reshape(-1)[: a_torch.numel()].reshape(a_shape)
    b_seen = from_tt(b, mesh).reshape(-1)[: b_torch.numel()].reshape(b_shape)
    ref = torch_op(a_seen, b_seen)
    if relu:
        ref = torch.relu(ref)
    assert_pcc(ref, out, pcc=pcc, mesh_device=mesh)


def seen(tt_tensor, like: torch.Tensor, mesh_device=None) -> torch.Tensor:
    """Host copy of ``tt_tensor``'s logical data, shaped like ``like``.

    Use as the reference for value-preserving ops on bfloat8_b inputs: the op must preserve
    the *quantized* values the device holds, not the original float values.
    """
    host = from_tt(tt_tensor, mesh_device)
    return host.reshape(-1)[: like.numel()].reshape(like.shape)
