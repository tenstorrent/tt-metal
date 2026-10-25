# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent sub-device manager for the two-chunk prefill pipeline (pipeline track E1).

``TT_GLM_PERSISTENT_SD=1`` keeps ONE sub-device manager ("P") loaded for the whole forward:

  SD0 "main" = (0, 0)..(gx - 1, gy - 1), gx x gy = the MeshDevice op-grid override (11x10 on a 12x10 BH Galaxy chip)
  SD1 "comm" = the device columns past the override (logical x = 11: 10 cores)

The op-grid override (``MeshDevice.set_compute_grid_override`` or ``TT_METAL_MESH_COMPUTE_GRID_OVERRIDE=11x10`` at
open) makes every op that sizes its work from ``compute_with_storage_grid_size()`` stay inside (0, 0)..(gx - 1,
gy - 1), and the CCLs that default to the first sub-device take SD0's cores, so the unchanged forward runs on SD0
while SD1 stays free for a second stream's communication (dispatch, CCLs called with ``subdevice_id=SD1``).
tt-metal internals keep the device grid, so SD1 is a legal sub-device.

The overlap regions that need their own manager (MoE dispatch || shared expert, full-layer top-k || KVPE gather)
swap P -> region manager -> P (``restore_base_manager``) instead of default -> region -> default. Each swap is a
trace-segment boundary (a full barrier), exactly as the load/clear pair is today, so the segment count does not
change. ``TT_GLM_MOE_DISPATCH_SD=comm`` instead runs the MoE dispatch on SD1 and the shared expert on SD0 with no
swap (pipeline track E1b).

Everything here is inert unless ``TT_GLM_PERSISTENT_SD=1``.
"""

import os
from dataclasses import dataclass
from typing import Optional

from loguru import logger

import ttnn

ENV_PERSISTENT_SD = "TT_GLM_PERSISTENT_SD"
ENV_MOE_DISPATCH_SD = "TT_GLM_MOE_DISPATCH_SD"  # "region" (default: swap to the MoE manager) | "comm" (SD1, no swap)


def persistent_sd_enabled() -> bool:
    return os.getenv(ENV_PERSISTENT_SD, "0").lower() in ("1", "true", "on", "yes")


def moe_dispatch_on_comm() -> bool:
    """MoE dispatch on SD1 (comm column) || shared expert on SD0, with no manager swap. Needs the persistent manager."""
    mode = os.getenv(ENV_MOE_DISPATCH_SD, "region").lower()
    if mode not in ("region", "comm"):
        raise ValueError(f"{ENV_MOE_DISPATCH_SD}={mode!r}: expected 'region' or 'comm'")
    if mode == "comm" and not persistent_sd_enabled():
        raise ValueError(f"{ENV_MOE_DISPATCH_SD}=comm needs {ENV_PERSISTENT_SD}=1")
    return mode == "comm"


@dataclass(frozen=True)
class PersistentSubDevices:
    manager_id: object
    # The same layout under a second id. Loading it while P is loaded (or P while it is) drains and resets every
    # sub-device: the only device-side join of SD0 and SD1 until an in-trace wait op exists (see barrier_target()).
    twin_manager_id: object
    main_sd_id: ttnn.SubDeviceId
    comm_sd_id: ttnn.SubDeviceId
    main_cores: ttnn.CoreRangeSet
    comm_cores: ttnn.CoreRangeSet
    op_grid: tuple  # (x, y) of SD0 = the op-grid override
    device_grid: tuple  # (x, y) of the whole Tensix grid

    # INTERFACES.md names (PipelineManager): sd0 = main, sd1 = comm.
    @property
    def sd0_id(self):
        return self.main_sd_id

    @property
    def sd1_id(self):
        return self.comm_sd_id

    @property
    def sd0_cores(self):
        return self.main_cores

    @property
    def sd1_cores(self):
        return self.comm_cores


PipelineManager = PersistentSubDevices

_cache: dict = {}


def _ensure_override(mesh_device) -> tuple:
    """Return (op_grid, device_grid); set the override to (device_x - 1, device_y) if the process did not."""
    dev = mesh_device.device_compute_with_storage_grid_size()
    override = mesh_device.get_compute_grid_override()
    if override is None:
        n_cached = mesh_device.num_program_cache_entries()
        if n_cached > 0:
            logger.warning(
                f"[persistent_sd] setting the op-grid override after {n_cached} programs were cached; set "
                "TT_METAL_MESH_COMPUTE_GRID_OVERRIDE=11x10 at open instead"
            )
        override = ttnn.CoreCoord(dev.x - 1, dev.y)
        mesh_device.set_compute_grid_override(override)
    if override.y != dev.y or override.x >= dev.x:
        raise ValueError(
            f"[persistent_sd] the op-grid override {override.x}x{override.y} must keep every row and leave at least "
            f"one column of the {dev.x}x{dev.y} device grid for the comm sub-device"
        )
    return (override.x, override.y), (dev.x, dev.y)


def get_persistent_sub_devices(mesh_device) -> Optional[PersistentSubDevices]:
    """The persistent manager of `mesh_device`: created on first use when TT_GLM_PERSISTENT_SD=1, or the one a caller
    already created with get_pipeline_manager(); None otherwise (everything below is then the default behavior)."""
    psd = _cache.get(mesh_device.id())
    if psd is None and persistent_sd_enabled():
        psd = get_pipeline_manager(mesh_device)
    return psd


def get_pipeline_manager(mesh_device) -> PersistentSubDevices:
    """The persistent {SD0 main, SD1 comm} manager of `mesh_device`, created on first use whatever the env (sets the
    op-grid override to (device_x - 1, device_y) if the process did not). Does not load it."""
    key = mesh_device.id()
    psd = _cache.get(key)
    if psd is not None:
        return psd
    (gx, gy), (dx, dy) = _ensure_override(mesh_device)
    main_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    comm_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(gx, 0), ttnn.CoreCoord(dx - 1, dy - 1))])
    layout = [ttnn.SubDevice([main_cores]), ttnn.SubDevice([comm_cores])]
    manager_id = mesh_device.create_sub_device_manager(layout, 0)
    twin_manager_id = mesh_device.create_sub_device_manager(layout, 0)
    psd = PersistentSubDevices(
        manager_id=manager_id,
        twin_manager_id=twin_manager_id,
        main_sd_id=ttnn.SubDeviceId(0),
        comm_sd_id=ttnn.SubDeviceId(1),
        main_cores=main_cores,
        comm_cores=comm_cores,
        op_grid=(gx, gy),
        device_grid=(dx, dy),
    )
    _cache[key] = psd
    logger.info(
        f"[persistent_sd] manager: SD0 main = (0,0)-({gx - 1},{gy - 1}) {main_cores.num_cores()} cores, "
        f"SD1 comm = ({gx},0)-({dx - 1},{dy - 1}) {comm_cores.num_cores()} cores"
    )
    return psd


def load_base_manager(mesh_device) -> None:
    """Load the persistent manager (no-op when disabled or already loaded)."""
    psd = get_persistent_sub_devices(mesh_device)
    if psd is not None:
        mesh_device.load_sub_device_manager(psd.manager_id)


def restore_base_manager(mesh_device) -> None:
    """Leave an overlap region: back to the persistent manager when enabled, else to the default manager."""
    psd = get_persistent_sub_devices(mesh_device)
    if psd is not None:
        mesh_device.load_sub_device_manager(psd.manager_id)
    else:
        mesh_device.clear_loaded_sub_device_manager()


def barrier_target(mesh_device):
    """The manager to load for a full SD0/SD1 join: whichever of P and its twin is NOT loaded (loading the loaded
    manager is a no-op, not a drain). The caller loads it (through the trace controller when one is set, so a
    capture splits there). Same cost as one load/clear (~10-14 us on device + a trace-segment boundary)."""
    psd = get_persistent_sub_devices(mesh_device)
    assert psd is not None, "barrier_target() needs TT_GLM_PERSISTENT_SD=1"
    if mesh_device.is_sub_device_manager_active(psd.manager_id):
        return psd.twin_manager_id
    return psd.manager_id


def release_persistent_sub_devices(mesh_device) -> None:
    """Clear (back to the default manager) and remove the persistent manager. Idempotent."""
    psd = _cache.pop(mesh_device.id(), None)
    if psd is None:
        return
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(psd.manager_id)
    mesh_device.remove_sub_device_manager(psd.twin_manager_id)
