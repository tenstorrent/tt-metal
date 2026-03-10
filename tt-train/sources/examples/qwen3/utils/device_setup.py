# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Tenstorrent device setup for single-device and distributed modes."""

import os

import ttml
from ttml.common.utils import get_tt_metal_home


def setup_device(dp_size: int, tp_size: int, seed: int = 42):
    """Open a Tenstorrent device (single or mesh) and return ``(ctx, device)``.

    Handles:
      - ``TT_METAL_RUNTIME_ROOT`` env-var fallback
      - Fabric enablement and parallelism-context initialisation
    """
    distributed = dp_size > 1 or tp_size > 1
    total_devices = dp_size * tp_size

    if "TT_METAL_RUNTIME_ROOT" not in os.environ:
        tt_metal_home = get_tt_metal_home()
        if tt_metal_home and os.path.exists(tt_metal_home):
            os.environ["TT_METAL_RUNTIME_ROOT"] = tt_metal_home

    if distributed:
        print(
            f"\nEnabling distributed mode: DP={dp_size}, TP={tp_size} "
            f"({total_devices} devices, mesh [{dp_size}, {tp_size}])"
        )

        ttml.core.distributed.enable_fabric(total_devices)

    ctx = ttml.autograd.AutoContext.get_instance()
    if distributed:
        ctx.open_device([dp_size, tp_size])
        ctx.initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=True)
        )
    else:
        ctx.open_device()
    ctx.set_seed(seed)
    return ctx, ctx.get_device()
