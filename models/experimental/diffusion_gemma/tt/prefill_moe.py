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
_MIN_GRID = (11, 4)

_tuned_geometry_active: ContextVar[bool] = ContextVar("diffusion_gemma_tuned_prefill_moe", default=False)
_builder_install_lock = Lock()
_original_builder = gemma4_prefill._build_sparse_matmul_config


def tuned_prefill_moe_enabled() -> bool:
    """Whether the exact dense prefill-MoE geometry is enabled (default on)."""

    return os.environ.get(FLAG, "1").strip().lower() not in ("0", "false", "no", "off")


def _find_supported_experts(model):
    for layer in getattr(model, "layers", ()):
        experts = getattr(getattr(layer, "moe", None), "experts", None)
        if experts is None:
            continue
        weights = experts.weights
        config = experts.config
        grid = model.mesh_device.compute_with_storage_grid_size()
        if (
            config.hidden_size == _HIDDEN_SIZE
            and weights.intermediate_size_per_device == _INTERMEDIATE_PER_DEVICE
            and int(grid.x) >= _MIN_GRID[0]
            and int(grid.y) >= _MIN_GRID[1]
        ):
            return experts
        return None
    return None


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
