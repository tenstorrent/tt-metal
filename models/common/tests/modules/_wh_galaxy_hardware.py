# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared real-hardware plumbing for common WH Galaxy module tests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import prod
from typing import Any

import pytest

import ttnn
from models.common.models.galaxy import GalaxyModePlan, GalaxyResourcesConfig
from models.common.modules.prefetcher import Prefetcher2D, Prefetcher2DConfig, Prefetcher2DModeConfig

_FACTORY_MODULE = "models.common.models.galaxy"
_FACTORY_NAMES = ("create_galaxy_resources", "create_galaxy_ccl", "get_galaxy_ccl")


@dataclass
class GalaxyHardwareResources:
    """Small adapter around the changing production Galaxy resource API."""

    owner: Any
    ccl: Any
    prefetcher: Any = None

    def context(self, mode: str) -> Any:
        return self.ccl.context(mode)

    def prefetch_context(self, mode: str) -> Any:
        if self.prefetcher is None:
            return None
        context = getattr(self.prefetcher, "context", None)
        return context(mode) if callable(context) else None

    def activate(self, mode: str) -> None:
        resource = self.owner or self.ccl or self.prefetcher
        activate = getattr(resource, "activate", None)
        if callable(activate):
            activate(mode)

    def synchronize(self, mode: str) -> None:
        synchronize = getattr(self.owner, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError("Galaxy production resource owner must expose synchronize(mode)")
        synchronize(mode)

    def cleanup(self) -> None:
        seen: set[int] = set()
        for resource in (self.owner, self.ccl, self.prefetcher):
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            cleanup = getattr(resource, "cleanup", None) or getattr(resource, "close", None)
            if callable(cleanup):
                cleanup()


@dataclass(frozen=True)
class _CCLOnlyContext:
    mode: str
    mesh_device: Any
    sub_device_manager_id: Any
    worker_sub_device_id: Any
    stall_group: tuple[Any, ...]


class _CCLOnlySubdeviceOwner:
    """Test owner for CCL fixtures that must not launch a prefetch producer."""

    def __init__(self, mesh_device: ttnn.MeshDevice, config: GalaxyResourcesConfig):
        self.mesh_device = mesh_device
        self._plans = {plan.mode: plan for plan in (config.prefill, config.decode)}
        self._managers = {
            mode: mesh_device.create_sub_device_manager(list(plan.sub_devices), plan.local_l1_size)
            for mode, plan in self._plans.items()
        }
        self._contexts = {
            mode: _CCLOnlyContext(
                mode=mode,
                mesh_device=mesh_device,
                sub_device_manager_id=self._managers[mode],
                worker_sub_device_id=plan.worker_sub_device_id,
                stall_group=plan.stall_group,
            )
            for mode, plan in self._plans.items()
        }
        self._loaded = False
        self._cleaned = False

    def context(self, mode: str) -> _CCLOnlyContext:
        return self._contexts[mode]

    def borrow_context(
        self,
        mode: str,
        *,
        sub_devices: tuple[Any, ...],
        worker_sub_device_id: Any,
        stall_group: tuple[Any, ...],
        local_l1_size: int,
    ) -> _CCLOnlyContext:
        plan = self._plans[mode]
        requested = (tuple(sub_devices), worker_sub_device_id, tuple(stall_group), local_l1_size)
        expected = (plan.sub_devices, plan.worker_sub_device_id, plan.stall_group, plan.local_l1_size)
        if requested != expected:
            raise ValueError(f"{mode} CCL-only subdevice policy mismatch")
        return self.context(mode)

    def activate(self, mode: str) -> _CCLOnlyContext:
        context = self.context(mode)
        self.mesh_device.load_sub_device_manager(context.sub_device_manager_id)
        self.mesh_device.set_sub_device_stall_group(list(context.stall_group))
        self._loaded = True
        return context

    def cleanup(self) -> None:
        if self._cleaned:
            return
        if self._loaded:
            self.mesh_device.reset_sub_device_stall_group()
            self.mesh_device.clear_loaded_sub_device_manager()
        for mode in ("decode", "prefill"):
            self.mesh_device.remove_sub_device_manager(self._managers[mode])
        self._cleaned = True


def require_galaxy_hardware_resources(
    mesh_device: ttnn.MeshDevice,
    *,
    config: GalaxyResourcesConfig,
    prefetch_weights: tuple[tuple[str, Any], ...],
) -> GalaxyHardwareResources:
    """Create production resources, or fail with the missing API boundary."""

    module = import_module(_FACTORY_MODULE)
    factory = next(
        (getattr(module, name, None) for name in _FACTORY_NAMES if callable(getattr(module, name, None))), None
    )
    if factory is None:
        pytest.fail(
            f"WH Galaxy hardware tests require a production resource factory in {_FACTORY_MODULE}; "
            f"expected one of {', '.join(_FACTORY_NAMES)}. Do not substitute legacy model CCL/prefetch setup."
        )

    if not prefetch_weights:
        pytest.fail("Galaxy production hardware resources require explicit prefetched device weights")
    prefetcher = _create_hardware_prefetcher(mesh_device, config, prefetch_weights)
    try:
        owner = factory(mesh_device=mesh_device, config=config, prefetcher=prefetcher)
    except Exception:
        prefetcher.cleanup()
        raise

    ccl = owner.get("ccl") if isinstance(owner, dict) else getattr(owner, "ccl", owner)
    prefetcher = owner.get("prefetcher") if isinstance(owner, dict) else getattr(owner, "prefetcher", None)
    if not callable(getattr(ccl, "context", None)):
        pytest.fail("Galaxy production resource factory must return a CCL collaborator exposing context(mode)")
    if getattr(ccl, "mesh_device", mesh_device) is not mesh_device:
        pytest.fail("Galaxy production resource factory returned resources for a different mesh")
    return GalaxyHardwareResources(owner=owner, ccl=ccl, prefetcher=prefetcher)


def require_galaxy_ccl_hardware_resources(
    mesh_device: ttnn.MeshDevice,
    *,
    config: GalaxyResourcesConfig,
) -> GalaxyHardwareResources:
    """Create production Galaxy CCL resources without launching a prefetch producer."""

    module = import_module(_FACTORY_MODULE)
    factory = getattr(module, "create_galaxy_resources", None)
    if not callable(factory):
        pytest.fail(f"WH Galaxy hardware tests require create_galaxy_resources in {_FACTORY_MODULE}")
    subdevices = _CCLOnlySubdeviceOwner(mesh_device, config)
    try:
        owner = factory(mesh_device=mesh_device, config=config, prefetcher=subdevices)
    except Exception:
        subdevices.cleanup()
        raise
    return GalaxyHardwareResources(owner=owner, ccl=owner.ccl, prefetcher=subdevices)


def _create_hardware_prefetcher(
    mesh_device: ttnn.MeshDevice,
    resources: GalaxyResourcesConfig,
    weights: tuple[tuple[str, Any], ...],
) -> Prefetcher2D:
    sender_coords = tuple(
        ttnn.CoreCoord(x, y)
        for x, y in ((0, 9), (0, 0), (0, 4), (0, 5), (4, 0), (4, 9), (4, 1), (4, 7), (4, 6), (4, 2), (4, 4), (4, 5))
    )
    receiver_pairs = tuple(((1, y), (2, y)) for y in (9, 0, 4, 5)) + tuple(
        ((5, y), (6, y)) for y in (0, 9, 1, 7, 6, 2, 4, 5)
    )
    receiver_sets = tuple(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end))})
        for start, end in receiver_pairs
    )
    dummy_sender_coords = tuple(
        ttnn.CoreCoord(x, y) for x, y in ((0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (4, 3), (4, 8))
    )

    def ranges(*coordinates: tuple[int, int, int, int]) -> Any:
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in coordinates]
        )

    dummy_receiver_sets = (
        ranges((3, 0, 3, 0), (1, 1, 3, 1)),
        ranges((1, 2, 3, 2)),
        ranges((1, 3, 3, 3), (3, 4, 3, 4)),
        ranges((3, 5, 3, 5), (1, 6, 3, 6)),
        ranges((1, 7, 3, 7)),
        ranges((1, 8, 3, 8), (3, 9, 3, 9)),
        ranges((5, 3, 6, 3)),
        ranges((5, 8, 6, 8)),
    )
    all_sender_coords = sender_coords + dummy_sender_coords
    all_receiver_sets = receiver_sets + dummy_receiver_sets
    sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in sender_coords])
    address_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_cores, [1, len(weights)], ttnn.ShardOrientation.ROW_MAJOR),
    )

    def mode_config(plan: GalaxyModePlan) -> Prefetcher2DModeConfig:
        return Prefetcher2DModeConfig(
            mode=plan.mode,
            sub_devices=plan.sub_devices,
            worker_sub_device_id=plan.worker_sub_device_id,
            stall_group=plan.stall_group,
            local_l1_size=plan.local_l1_size,
        )

    prefetcher = Prefetcher2D(
        Prefetcher2DConfig(
            mesh_device=mesh_device,
            architecture=resources.architecture,
            prefill=mode_config(resources.prefill),
            decode=mode_config(resources.decode),
            sender_receiver_mapping=tuple(zip(all_sender_coords, all_receiver_sets)),
            global_cb_size=728 * 1088,
            expected_weight_count=len(weights),
            address_repeat_count=len(sender_coords),
            address_memory_config=address_memory_config,
            address_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    )
    try:
        prefetcher.initialize()
        for name, tensor in weights:
            prefetcher.register_weight(name, tensor)
        prefetcher.seal()
    except Exception:
        prefetcher.cleanup()
        raise
    return prefetcher


def exact_tensor_resource(context: Any, operation: str, cluster_axis: int, tensor: Any, stage_key: Any = None) -> Any:
    """Select the resource keyed by the tensor geometry seen by the TTNN operation."""

    shape = tuple(int(value) for value in (tensor if isinstance(tensor, tuple) else tensor.shape))
    sequence_key = prod(shape[:-1]) if stage_key is None else (prod(shape[:-1]), stage_key)
    return context.resources(operation, cluster_axis, shape, sequence_key)


def galaxy_mode_plan(
    mode: str,
    collectives: tuple[Any, ...],
    mesh_device: ttnn.MeshDevice,
    *,
    semaphore_cores: Any | None = None,
) -> GalaxyModePlan:
    """Build the common no-prefetch worker envelope used by focused module tests."""

    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_id = ttnn.SubDeviceId(0)
    return GalaxyModePlan(
        mode=mode,
        sub_devices=(ttnn.SubDevice([cores]),),
        worker_sub_device_id=worker_id,
        stall_group=(worker_id,),
        semaphore_cores=semaphore_cores or cores,
        collectives=collectives,
    )


def galaxy_prefetch_decode_mode_plan(collectives: tuple[Any, ...]) -> GalaxyModePlan:
    """Build the canonical WH decode sender/worker subdevice partition."""

    sender_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(core, core)
            for core in (
                ttnn.CoreCoord(0, 9),
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, 4),
                ttnn.CoreCoord(0, 5),
                ttnn.CoreCoord(4, 0),
                ttnn.CoreCoord(4, 9),
                ttnn.CoreCoord(4, 1),
                ttnn.CoreCoord(4, 7),
                ttnn.CoreCoord(4, 6),
                ttnn.CoreCoord(4, 2),
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 5),
            )
        ]
    )
    worker_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    worker_id = ttnn.SubDeviceId(1)
    return GalaxyModePlan(
        mode="decode",
        sub_devices=(ttnn.SubDevice([sender_cores]), ttnn.SubDevice([worker_cores])),
        worker_sub_device_id=worker_id,
        stall_group=(worker_id,),
        semaphore_cores=worker_cores,
        collectives=collectives,
    )


def compose_2d_sharded_tensor(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> Any:
    """Compose a tensor sharded over columns and replicated over rows."""

    result = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=(8, 4)),
    )
    return result[:, :1]


def deallocate_tensor(tensor: Any) -> None:
    if tensor is None:
        return
    deallocate = getattr(tensor, "deallocate", None)
    if callable(deallocate):
        deallocate(True)


def deallocate_module_weights(module: Any, *names: str) -> None:
    seen: set[int] = set()
    for name in names:
        tensor = getattr(module, name, None)
        if tensor is not None and id(tensor) not in seen:
            deallocate_tensor(tensor)
            seen.add(id(tensor))
