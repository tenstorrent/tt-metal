# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os

from loguru import logger

import ttnn
from conftest import bh_2d_mesh_device_context
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config

TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS_DEFAULT = "30000"

# Ethernet port used to initialize fabric router coordination.
FABRIC_ROUTER_ETH_PORT = 15232

# Default worker L1 size (bytes) for most ranks in pod pipeline demo runs.
DEFAULT_WORKER_L1_SIZE = 1431568

# Larger worker L1 size (bytes) for LM-head/tail ranks that need extra L1.
LM_HEAD_WORKER_L1_SIZE = 1499000

# Mesh rank using LM-head/tail stage in 64-process topology.
LM_HEAD_RANK_64_PROCS = 62

# Mesh rank using LM-head/tail stage in 16-process topology.
LM_HEAD_RANK_16_PROCS = 14


def _fabric_config_for_num_procs(num_procs: int):
    """Infer fabric config from process count: 4 → FABRIC_2D, 16/64 → FABRIC_2D_TORUS_Y."""
    if num_procs == 4:
        return ttnn.FabricConfig.FABRIC_2D
    if num_procs in (16, 64):
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError(f"Unsupported num_procs for fabric config: {num_procs} (expected 4, 16, or 64)")


def _worker_l1_size_for_rank(num_procs: int, my_rank: int) -> int:
    """Select worker L1 size for rank-specific LM-head memory requirements."""
    if num_procs == 64 and my_rank == LM_HEAD_RANK_64_PROCS:
        return LM_HEAD_WORKER_L1_SIZE
    if num_procs == 16 and my_rank == LM_HEAD_RANK_16_PROCS:
        return LM_HEAD_WORKER_L1_SIZE
    return DEFAULT_WORKER_L1_SIZE


@contextlib.contextmanager
def open_mesh_device():
    """Open mesh device for pod demos with shared fabric and worker L1 settings."""
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS_DEFAULT

    my_rank = int(ttnn.distributed_context_get_rank())
    num_procs = int(ttnn.distributed_context_get_size())
    worker_l1_size = _worker_l1_size_for_rank(num_procs=num_procs, my_rank=my_rank)
    device_params = {
        "fabric_config": _fabric_config_for_num_procs(num_procs),
        "fabric_router_config": create_fabric_router_config(FABRIC_ROUTER_ETH_PORT),
        "worker_l1_size": worker_l1_size,
    }

    logger.info("Opening mesh device...")
    with bh_2d_mesh_device_context(device_params) as mesh_device:
        logger.info(
            "Mesh device opened (id={}, shape={})",
            mesh_device.get_system_mesh_id(),
            mesh_device.shape,
        )
        yield mesh_device
