# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-I2V package init.

Applies two cosmos3-local power-mitigation patches to shared tt_dit infra at
import time. Both are runtime mutations confined to this package — no edits to
files outside experimental/cosmos3_i2v/.

1. Trunk matmul grid clamp (10x10 on multi-chip BH). Cosmos3-Super's FF
   matmuls are ~3.7x larger than Wan2.2 (intermediate=25600 vs 13824, plus
   gate/up combined), so the 11x10 cap that's safe for Wan still trips power
   on Cosmos3. Drop the cap to 10x10 specifically when Cosmos3 is in use.

2. VAE conv3d grid clamp (11x10 on multi-chip BH). The Wan VAE used by the
   Cosmos3 adapter calls `mesh_device.compute_with_storage_grid_size()`
   directly and never goes through `get_matmul_core_grid`. Cosmos3's VAE
   activations are wider than Wan's, so the unclamped 12x10 conv3d is a real
   power risk. Intercept `get_conv3d_config` and clamp its `grid_size` arg.

The patches are gated so they only fire on multi-chip BH; single-chip and WH
runs are unaffected.
"""

import ttnn as _ttnn
from models.tt_dit.models.vae import vae_wan2_1 as _wan_vae_mod
from models.tt_dit.utils import conv3d as _tt_dit_conv3d
from models.tt_dit.utils import matmul as _tt_dit_matmul

# --- (1) Trunk matmul grid: lower threshold to 1 (any multi-chip BH) and cap to 10x10 ---
_tt_dit_matmul._BH_GALAXY_MIN_DEVICES = 1
_tt_dit_matmul._BH_GALAXY_MAX_CORE_GRID = (10, 10)


# --- (2) VAE conv3d grid: wrap get_conv3d_config to clamp the passed grid_size ---
_VAE_CONV3D_CAP = (11, 10)
_orig_get_conv3d_config = _tt_dit_conv3d.get_conv3d_config


def _clamped_get_conv3d_config(in_channels, out_channels, kernel_size, weights_dtype, grid_size, **kwargs):
    """Pass-through to tt_dit's get_conv3d_config with grid_size clamped to 11x10 on multi-chip BH."""
    if (
        grid_size is not None
        and _ttnn.device.is_blackhole()
        and (grid_size.x > _VAE_CONV3D_CAP[0] or grid_size.y > _VAE_CONV3D_CAP[1])
    ):
        grid_size = _ttnn.CoreCoord(min(grid_size.x, _VAE_CONV3D_CAP[0]), min(grid_size.y, _VAE_CONV3D_CAP[1]))
    return _orig_get_conv3d_config(in_channels, out_channels, kernel_size, weights_dtype, grid_size, **kwargs)


# Patch both the source module AND the namespace of vae_wan2_1, since it did
# `from ...utils.conv3d import get_conv3d_config` at module-load time (the
# imported name is a separate reference; patching just the source module
# wouldn't affect callers that already bound the name).
_tt_dit_conv3d.get_conv3d_config = _clamped_get_conv3d_config
_wan_vae_mod.get_conv3d_config = _clamped_get_conv3d_config
