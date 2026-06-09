# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os

from loguru import logger

import ttnn
from conftest import bh_2d_mesh_device_context, get_updated_device_params, reset_fabric, set_fabric
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config

TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS_DEFAULT = "30000"
FABRIC_PACKET_SIZE_BYTES = 15232

# TODO: Store these values inside the stages and fetch based on pipeline config
DEFAULT_WORKER_L1_SIZE = 1431568
LM_HEAD_WORKER_L1_SIZE = 1453716
LM_HEAD_RANK_64_PROCS = 62
LM_HEAD_RANK_16_PROCS = 14


def _fabric_config_for_num_procs(num_procs: int):
    """Infer fabric config from process count: 4 → FABRIC_2D, 16/64 → FABRIC_2D_TORUS_Y."""
    if num_procs == 4:
        return ttnn.FabricConfig.FABRIC_2D
    if num_procs in (16, 64):
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError(f"Unsupported num_procs for fabric config: {num_procs} (expected 4, 16, or 64)")


def _needs_extended_worker_l1(num_procs: int, *, enable_speculative_decode: bool = True) -> bool:
    """Return true when this worker must use the larger L1 budget.

    TT_MESH_ID must be provided by launch tooling for 16/64-proc runs.
    """
    if not enable_speculative_decode:
        return False

    mesh_id = os.environ.get("TT_MESH_ID")
    target_rank = None
    if num_procs == 64:
        target_rank = LM_HEAD_RANK_64_PROCS
    elif num_procs == 16:
        target_rank = LM_HEAD_RANK_16_PROCS
    else:
        return False

    if mesh_id is None:
        raise RuntimeError("TT_MESH_ID must be set for 16/64-process runs to select worker_l1_size")

    try:
        return int(mesh_id) == target_rank
    except ValueError as exc:
        raise RuntimeError(f"Invalid TT_MESH_ID={mesh_id!r}; expected an integer") from exc


def _worker_l1_size_for_rank(num_procs: int, *, enable_speculative_decode: bool = True) -> int:
    """Select worker L1 size for rank-specific LM-head memory requirements."""
    if _needs_extended_worker_l1(num_procs=num_procs, enable_speculative_decode=enable_speculative_decode):
        return LM_HEAD_WORKER_L1_SIZE
    return DEFAULT_WORKER_L1_SIZE


@contextlib.contextmanager
def _galaxy_4rank_mesh_device_context(device_params):
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(8, 4),
        **updated_device_params,
    )
    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        reset_fabric(fabric_config)
        del mesh_device


@contextlib.contextmanager
def open_mesh_device(*, enable_speculative_decode: bool = True):
    """Open mesh device for pod demos with shared fabric and worker L1 settings."""
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS_DEFAULT

    num_procs = int(ttnn.distributed_context_get_size())
    worker_l1_size = _worker_l1_size_for_rank(
        num_procs=num_procs,
        enable_speculative_decode=enable_speculative_decode,
    )
    device_params = {
        "fabric_config": _fabric_config_for_num_procs(num_procs),
        "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        "fabric_router_config": create_fabric_router_config(FABRIC_PACKET_SIZE_BYTES),
        "worker_l1_size": worker_l1_size,
    }

    logger.info("Opening mesh device...")
    mesh_context = _galaxy_4rank_mesh_device_context if num_procs == 4 else bh_2d_mesh_device_context
    with mesh_context(device_params) as mesh_device:
        logger.info(
            "Mesh device opened (id={}, shape={})",
            mesh_device.get_system_mesh_id(),
            mesh_device.shape,
        )
        yield mesh_device
