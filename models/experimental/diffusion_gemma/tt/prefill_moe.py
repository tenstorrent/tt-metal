# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local dense prefill-MoE geometry tuning.

The shared Gemma4 prefill computes every routed expert in 32-token chunks. That
math is retained exactly; only the sparse-matmul grid and K blocking are changed
for the DiffusionGemma 26B TP=4 shape on Blackhole. The tuned layer-0 MoE output
is bit-identical to the shared geometry while reducing a 256-token call from
~135.5 ms to ~21.2 ms on QB2.

The shared Gemma4 source remains untouched. A context-local selector activates
the tuned geometry only for the current DiffusionGemma prefill. Other threads
and async tasks continue to use the shared Gemma4 builder.
"""

from __future__ import annotations

from contextvars import ContextVar
import os
from contextlib import contextmanager
from threading import Lock

import ttnn
import models.demos.gemma4.tt.experts.prefill as gemma4_prefill

FLAG = "DG_PREFILL_MOE_TUNED"

_HIDDEN_SIZE = 2816
_INTERMEDIATE_PER_DEVICE = 192
_MOE_INTERMEDIATE_SIZE = 704
_NUM_EXPERTS = 128
_TOP_K = 8
_MESH_SHAPE = (1, 4)
_COMPUTE_GRID = (13, 10)

_tuned_geometry_active: ContextVar[bool] = ContextVar("diffusion_gemma_tuned_prefill_moe", default=False)
_builder_install_lock = Lock()
_original_builder = gemma4_prefill._build_sparse_matmul_config


def tuned_prefill_moe_enabled() -> bool:
    """Whether the exact dense prefill-MoE geometry is enabled (default on)."""

    return os.environ.get(FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def _find_supported_experts(model):
    """Return all experts only when every measured QB2 invariant matches."""

    mesh_device = getattr(model, "mesh_device", None)
    mesh_config = getattr(model, "mesh_config", None)
    if mesh_device is None or mesh_config is None:
        return None

    try:
        grid = mesh_device.compute_with_storage_grid_size()
        prefill_config = mesh_config.prefill
        supported_mesh = (
            mesh_device.arch() == ttnn.device.Arch.BLACKHOLE
            and tuple(mesh_device.shape) == _MESH_SHAPE
            and mesh_device.get_num_devices() == 4
            and tuple(mesh_config.mesh_shape) == _MESH_SHAPE
            and mesh_config.tp_axis == 1
            and (prefill_config.tp, prefill_config.ep, prefill_config.sp) == (4, 1, 1)
            and (int(grid.x), int(grid.y)) == _COMPUTE_GRID
        )
    except (AttributeError, TypeError):
        return None
    if not supported_mesh or gemma4_prefill.PREFILL_CHUNK_SIZE != ttnn.TILE_SIZE:
        return None

    layers = tuple(getattr(model, "layers", ()))
    experts_per_layer = tuple(getattr(getattr(layer, "moe", None), "experts", None) for layer in layers)
    if not experts_per_layer or any(experts is None for experts in experts_per_layer):
        return None

    for experts in experts_per_layer:
        weights = experts.weights
        config = experts.config
        expert_weights = (weights.gate_proj, weights.up_proj, weights.down_proj)
        if (
            config.hidden_size != _HIDDEN_SIZE
            or config.moe_intermediate_size != _MOE_INTERMEDIATE_SIZE
            or config.num_experts != _NUM_EXPERTS
            or config.top_k != _TOP_K
            or weights.intermediate_size_per_device != _INTERMEDIATE_PER_DEVICE
            or any(weight.get_dtype() != ttnn.bfloat16 for weight in expert_weights)
        ):
            return None
    return experts_per_layer


def _contextual_config_builder(m, n, in0_block_w=1):
    """Select the measured QB2 geometry only in the active call context."""

    if not _tuned_geometry_active.get() or m != ttnn.TILE_SIZE:
        return _original_builder(m, n, in0_block_w)

    if n == _INTERMEDIATE_PER_DEVICE:
        # gate/up: M=32, K=2816 (88 tiles), N=192 (6 tiles).
        grid_x, grid_y = 6, 1
        block_w = 44
        per_core_n = 1
    elif n == _HIDDEN_SIZE:
        # down: M=32, K=192 (6 tiles), N=2816 (88 tiles).
        grid_x, grid_y = 11, 4
        block_w = 3
        per_core_n = 2
    else:
        return _original_builder(m, n, in0_block_w)

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _install_contextual_builder() -> None:
    """Install one stable dispatcher; call-local state controls its behavior."""

    if gemma4_prefill._build_sparse_matmul_config is _contextual_config_builder:
        return
    with _builder_install_lock:
        if gemma4_prefill._build_sparse_matmul_config is _contextual_config_builder:
            return
        gemma4_prefill._build_sparse_matmul_config = _contextual_config_builder


@contextmanager
def use_tuned_prefill_moe(model):
    """Apply the exact QB2 dense-MoE geometry in the current call context."""

    if not tuned_prefill_moe_enabled() or _find_supported_experts(model) is None:
        yield
        return

    _install_contextual_builder()
    token = _tuned_geometry_active.set(True)
    try:
        yield
    finally:
        _tuned_geometry_active.reset(token)
