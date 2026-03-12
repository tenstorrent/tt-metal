# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Tenstorrent device setup for single-device and distributed modes."""

import os
from pathlib import Path

import ttnn
import ttml
from ttml.common.utils import get_tt_metal_home

# Mesh-shape → (descriptor filename, FabricConfig).
# The default T3K descriptor assumes [2,4]; a [1,8] mesh needs its own
# descriptor with RING,RING dim_types so the control plane wires up the
# correct topology.  Shapes that aren't listed use auto-discovery.
_MGD_TABLE: dict[tuple[int, int], tuple[str, "ttnn.FabricConfig"]] = {
    (2, 4): ("t3k_mesh_graph_descriptor.textproto", ttnn.FabricConfig.FABRIC_2D),
    (1, 8): (
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
        "t3k_1x8_mesh_graph_descriptor.textproto",
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ),
}


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

        mesh_shape = (dp_size, tp_size)
        user_set_mgd = "TT_MESH_GRAPH_DESC_PATH" in os.environ

        if not user_set_mgd and mesh_shape in _MGD_TABLE:
            mgd_rel, fabric_cfg = _MGD_TABLE[mesh_shape]
            runtime_root = os.environ.get(
                "TT_METAL_RUNTIME_ROOT", get_tt_metal_home() or ""
            )
            mgd_path = str(Path(runtime_root) / mgd_rel)
            if Path(mgd_path).exists():
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd_path
                print(f"  MGD: {mgd_path}")
            else:
                print(f"  WARNING: MGD not found ({mgd_path}), using auto-discovery")
            ttnn.set_fabric_config(fabric_cfg)
        else:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

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
