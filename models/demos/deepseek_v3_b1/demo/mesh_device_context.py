# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os

from loguru import logger

import ttnn
from conftest import reset_fabric, set_fabric
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config
from models.demos.deepseek_v3_b1.demo.stage_family import (
    fabric_config_for_stage_family,
    query_global_stage_mesh_shape,
    stage_family_from_shape,
)

TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS_DEFAULT = "30000"
FABRIC_PACKET_SIZE_BYTES = 15232

# TODO: Store these values inside the stages and fetch based on pipeline config
DEFAULT_WORKER_L1_SIZE = 1431568
LM_HEAD_WORKER_L1_SIZE = 1453716
LM_HEAD_RANK_64_PROCS = 62
LM_HEAD_RANK_16_PROCS = 14


def _fabric_config_for_topology(mesh_shape: ttnn.MeshShape, num_procs: int):
    """Infer fabric config from MGD-derived stage shape plus deployment layout."""

    stage_family = stage_family_from_shape(mesh_shape)
    num_stages_hint = 4 if stage_family.value == "4x2" and num_procs == 4 else None
    return fabric_config_for_stage_family(stage_family, num_stages=num_stages_hint)


def _fabric_config_for_num_procs(num_procs: int):
    """Legacy helper kept for tests that validate current worker setup assumptions."""

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
def open_mesh_device(*, enable_speculative_decode: bool = True):
    """Open mesh device for pod demos with shared fabric and worker L1 settings."""
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS_DEFAULT

    num_procs = int(ttnn.distributed_context_get_size())
    mesh_shape = query_global_stage_mesh_shape()
    logger.info(f"Queried global stage mesh shape from MGD: {mesh_shape}")
    fabric_config = _fabric_config_for_topology(mesh_shape, num_procs)
    worker_l1_size = _worker_l1_size_for_rank(
        num_procs=num_procs,
        enable_speculative_decode=enable_speculative_decode,
    )
    device_params = {
        "fabric_router_config": create_fabric_router_config(FABRIC_PACKET_SIZE_BYTES),
        "worker_l1_size": worker_l1_size,
    }

    logger.info("Opening mesh device with MGD-derived shape {} and fabric {}", mesh_shape, fabric_config)
    set_fabric(fabric_config, fabric_router_config=device_params.pop("fabric_router_config"))
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    try:
        logger.info(
            "Mesh device opened (id={}, shape={})",
            mesh_device.get_system_mesh_id(),
            mesh_device.shape,
        )
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        reset_fabric(fabric_config)
