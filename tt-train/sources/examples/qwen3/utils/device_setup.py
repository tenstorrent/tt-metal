# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Tenstorrent device setup for single-device and distributed modes."""

import os
import sys

import ttml


_MGD_WARNING = """
================================================================================
  WARNING: TT_MESH_GRAPH_DESC_PATH is NOT set!

  Distributed mode requires a Mesh Graph Descriptor (MGD) file.
  enable_fabric() will attempt automatic selection for 8 or 32 devices,
  but this may not match your hardware topology.

  For reliable operation, set the environment variable explicitly:

      export TT_MESH_GRAPH_DESC_PATH="/path/to/your/mesh_graph_descriptor.textproto"

  Common MGD files (relative to $TT_METAL_HOME):
      tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto
      tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto

  See: https://github.com/tenstorrent/tt-metal/blob/main/tt-train/docs/DISTRIBUTED_TRAINING.md#setting-mgd-files-via-environment-variable
================================================================================
"""


def setup_device(dp_size: int, tp_size: int, seed: int = 42):
    """Open a Tenstorrent device (single or mesh) and return ``(ctx, device)``.

    Handles:
      - Fabric enablement via ``ttml.core.distributed.enable_fabric``
      - Parallelism-context initialisation
    """
    distributed = dp_size > 1 or tp_size > 1
    total_devices = dp_size * tp_size

    if distributed:
        if "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
            print(_MGD_WARNING, file=sys.stderr)

        print(
            f"\nEnabling distributed mode: DP={dp_size}, TP={tp_size} "
            f"({total_devices} devices, mesh [{dp_size}, {tp_size}])"
        )
        ttml.core.distributed.enable_fabric(total_devices)

    ctx = ttml.autograd.AutoContext.get_instance()
    if distributed:
        ctx.open_device([dp_size, tp_size])
        ctx.initialize_parallelism_context(
            ttml.autograd.DistributedConfig(
                enable_ddp=dp_size > 1, enable_tp=tp_size > 1
            )
        )
    else:
        ctx.open_device()
    ctx.set_seed(seed)
    return ctx, ctx.get_device()
