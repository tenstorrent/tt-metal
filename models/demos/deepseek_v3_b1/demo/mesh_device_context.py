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
FABRIC_PACKET_SIZE_BYTES = 15232

# TODO: Store these values inside the stages and fetch based on pipeline config
DEFAULT_WORKER_L1_SIZE = 1431568
LM_HEAD_WORKER_L1_SIZE = 1456820


def _base_lm_head_ranks(num_procs: int, num_mtp_levels: int) -> list[int]:
    """Mesh ids that run a Base LM-head stage for this topology."""
    if num_procs == 16:
        first = 16 - 2 * num_mtp_levels
        return [first + 2 * k for k in range(num_mtp_levels)]
    if num_procs == 64:
        return [62]
    if num_procs == 80:
        return [62 + 2 * k for k in range(num_mtp_levels)]
    return []


def _fabric_config_for_num_procs(num_procs: int):
    """Infer fabric config from process count: 4 → FABRIC_2D, 16/64/80 → FABRIC_2D_TORUS_Y."""
    if num_procs == 4:
        return ttnn.FabricConfig.FABRIC_2D
    if num_procs in (16, 64, 80):
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError(f"Unsupported num_procs for fabric config: {num_procs} (expected 4, 16, 64, or 80)")


def _needs_extended_worker_l1(num_procs: int, num_mtp_levels: int) -> bool:
    """Return true when this worker must use the larger L1 budget.

    TT_MESH_ID must be provided by launch tooling for 16/64-proc runs.
    """
    if num_mtp_levels == 0:
        return False

    target_ranks = _base_lm_head_ranks(num_procs, num_mtp_levels)
    if not target_ranks:
        return False

    mesh_id = os.environ.get("TT_MESH_ID")
    if mesh_id is None:
        raise RuntimeError("TT_MESH_ID must be set for 16/64-process runs to select worker_l1_size")

    try:
        return int(mesh_id) in target_ranks
    except ValueError as exc:
        raise RuntimeError(f"Invalid TT_MESH_ID={mesh_id!r}; expected an integer") from exc


def _worker_l1_size_for_rank(num_procs: int, num_mtp_levels: int) -> int:
    """Select worker L1 size for rank-specific LM-head memory requirements."""
    if _needs_extended_worker_l1(num_procs=num_procs, num_mtp_levels=num_mtp_levels):
        return LM_HEAD_WORKER_L1_SIZE
    return DEFAULT_WORKER_L1_SIZE


@contextlib.contextmanager
def open_mesh_device(*, num_mtp_levels: int = 1):
    """Open mesh device for pod demos with shared fabric and worker L1 settings."""
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS_DEFAULT

    num_procs = int(ttnn.distributed_context_get_size())
    worker_l1_size = _worker_l1_size_for_rank(num_procs=num_procs, num_mtp_levels=num_mtp_levels)
    device_params = {
        "fabric_config": _fabric_config_for_num_procs(num_procs),
        "fabric_router_config": create_fabric_router_config(FABRIC_PACKET_SIZE_BYTES),
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
