# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Topology-aware mesh device open/close for tt_dit pipelines.

Device kwargs are sourced from each model's performance test tables.
Does not attach mpi4py — that belongs to the serving runner.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

import ttnn
from models.tt_dit.utils.test import line_params, line_params_8k, ring_params, ring_params_8k


def _get_updated_device_params(device_params: dict[str, Any]) -> dict[str, Any]:
    """Mirror tests.scripts.common.get_updated_device_params for open_mesh_device kwargs."""
    new_device_params = device_params.copy()
    dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
    dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
    fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

    if ttnn.device.is_blackhole():
        fabric_config = new_device_params.get("fabric_config", None)
        if not (fabric_config and fabric_tensix_config):
            if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                logger.warning(
                    "ROW dispatch requires both fabric and tensix config, using DispatchCoreAxis.COL instead."
                )
                dispatch_core_axis = ttnn.DispatchCoreAxis.COL

    new_device_params["dispatch_core_config"] = ttnn.DispatchCoreConfig(
        dispatch_core_type, dispatch_core_axis, fabric_tensix_config
    )
    return new_device_params


def _linear_fabric_params(**extra: Any) -> dict[str, Any]:
    base = line_params_8k if ttnn.device.is_blackhole() else line_params
    return {**base, **extra}


def _ring_fabric_params(**extra: Any) -> dict[str, Any]:
    base = ring_params_8k if ttnn.device.is_blackhole() else ring_params
    return {**base, **extra}


# (model_id, topology) → open_mesh_device kwargs. Missing keys default to {}.
_DEVICE_PARAMS: dict[tuple[str, str], dict[str, Any]] = {
    ("external", "linear"): _linear_fabric_params(),
    ("external", "ring"): _ring_fabric_params(),
    ("wan2.2", "linear"): _linear_fabric_params(trace_region_size=150_000_000),
    ("wan2.2", "ring"): _ring_fabric_params(trace_region_size=150_000_000),
    ("wan2.2-i2v", "linear"): _linear_fabric_params(trace_region_size=150_000_000),
    ("wan2.2-i2v", "ring"): _ring_fabric_params(trace_region_size=150_000_000),
    ("mochi-1", "linear"): _linear_fabric_params(),
    ("mochi-1", "ring"): _ring_fabric_params(),
    ("sd3.5", "linear"): _linear_fabric_params(l1_small_size=32768, trace_region_size=50_000_000),
    ("sd3.5", "ring"): _ring_fabric_params(l1_small_size=32768, trace_region_size=50_000_000),
    ("flux.1-dev", "linear"): _linear_fabric_params(l1_small_size=32768, trace_region_size=51_000_000),
    ("flux.1-dev", "ring"): _ring_fabric_params(l1_small_size=32768, trace_region_size=51_000_000),
    ("flux.1-schnell", "linear"): _linear_fabric_params(l1_small_size=32768, trace_region_size=51_000_000),
    ("flux.1-schnell", "ring"): _ring_fabric_params(l1_small_size=32768, trace_region_size=51_000_000),
    ("motif-image-6b-preview", "linear"): _linear_fabric_params(l1_small_size=32768, trace_region_size=50_000_000),
    ("motif-image-6b-preview", "ring"): _ring_fabric_params(l1_small_size=32768, trace_region_size=50_000_000),
    ("qwen-image", "linear"): _linear_fabric_params(trace_region_size=47_000_000),
    ("qwen-image", "ring"): _ring_fabric_params(trace_region_size=47_000_000),
    ("qwen-image-2512", "linear"): _linear_fabric_params(trace_region_size=47_000_000),
    ("qwen-image-2512", "ring"): _ring_fabric_params(trace_region_size=47_000_000),
}


def _resolve_device_params(model_id: str, topology: str) -> dict[str, Any]:
    if model_id.startswith("external:"):
        model_id = "external"
    return dict(_DEVICE_PARAMS.get((model_id, topology), {}))


def _set_fabric(device_params: dict[str, Any]) -> Any:
    """Pop fabric keys from device_params and call set_fabric_config. Returns fabric_config."""
    fabric_config = device_params.pop("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    fabric_tensix_config = device_params.pop("fabric_tensix_config", ttnn.FabricTensixConfig.DISABLED)
    reliability_mode = device_params.pop("reliability_mode", ttnn.FabricReliabilityMode.STRICT_INIT)
    fabric_manager = device_params.pop("fabric_manager", ttnn.FabricManagerMode.DEFAULT)
    fabric_router_config = device_params.pop("fabric_router_config", None)

    if fabric_router_config is not None:
        ttnn.set_fabric_config(
            fabric_config,
            reliability_mode,
            None,
            fabric_tensix_config,
            ttnn.FabricUDMMode.DISABLED,
            fabric_manager,
            fabric_router_config,
        )
    else:
        ttnn.set_fabric_config(
            fabric_config,
            reliability_mode,
            None,
            fabric_tensix_config,
            ttnn.FabricUDMMode.DISABLED,
            fabric_manager,
        )
    return fabric_config


def get_device(
    mesh_shape: tuple[int, int] | list[int],
    model_id: str,
    topology: str = "ring",
    **device_params: Any,
) -> ttnn.MeshDevice:
    """Open a MeshDevice configured for ``(model_id, mesh_shape, topology)``.

    Args:
        mesh_shape: ``(rows, cols)``, e.g. ``(4, 32)``.
        model_id: Registered id, or ``external:<module.name>``.
        topology: ``linear`` or ``ring`` (default ``ring``).
        **device_params: Optional overrides merged onto the model defaults
            (e.g. ``trace_region_size=...`` for external models).
    """
    shape = tuple(int(x) for x in mesh_shape)
    if len(shape) != 2:
        raise ValueError(f"mesh_shape must be (rows, cols), got {mesh_shape!r}")

    topo = topology.lower()
    raw_params = _resolve_device_params(model_id, topo)
    raw_params.update(device_params)

    updated = _get_updated_device_params(raw_params)
    updated.pop("require_exact_physical_num_devices", None)
    fabric_config = _set_fabric(updated)

    mesh = ttnn.MeshShape(*shape)
    try:
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh, **updated)
    except Exception:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as reset_err:
            logger.warning(f"Failed to reset fabric after open failure: {reset_err}")
        raise

    logger.info(
        f"Opened mesh_device shape={shape} model_id={model_id} topology={topo} "
        f"fabric={fabric_config} devices={mesh_device.get_num_devices()}"
    )
    # Stash fabric so close_device can reset it.
    mesh_device._tt_dit_fabric_config = fabric_config  # type: ignore[attr-defined]
    return mesh_device


def close_device(mesh_device: ttnn.MeshDevice | None) -> None:
    """Close mesh device and disable fabric if it was configured by ``get_device``."""
    if mesh_device is None:
        return
    fabric_config = getattr(mesh_device, "_tt_dit_fabric_config", None)
    try:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
    finally:
        if fabric_config is not None:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception as e:
                logger.warning(f"Failed to disable fabric on close: {e}")
