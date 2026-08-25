# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared, host-testable ownership for Wormhole Galaxy CCL resources."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Hashable, Literal, Protocol, runtime_checkable

import ttnn

GalaxyMode = Literal["prefill", "decode"]
CollectiveName = Literal[
    "reduce_scatter",
    "all_gather",
    "all_reduce",
    "all_gather_concat",
    "all_reduce_create_qkv_heads",
]
_COLLECTIVES: tuple[CollectiveName, ...] = (
    "reduce_scatter",
    "all_gather",
    "all_reduce",
    "all_gather_concat",
    "all_reduce_create_qkv_heads",
)
_CANONICAL_AXES: dict[CollectiveName, frozenset[int]] = {
    "reduce_scatter": frozenset((0, 1)),
    "all_gather": frozenset((0, 1)),
    "all_reduce": frozenset((0, 1)),
    "all_gather_concat": frozenset((0,)),
    "all_reduce_create_qkv_heads": frozenset((1,)),
}


def _validate_wh_galaxy(mesh_device: Any, mesh_shape: tuple[int, int], architecture: Any) -> None:
    if architecture != ttnn.device.Arch.WORMHOLE_B0:
        raise ValueError(f"Galaxy CCL requires Wormhole B0, got {architecture}")
    if mesh_shape != (8, 4):
        raise ValueError(f"Galaxy CCL requires logical mesh shape (8, 4), got {mesh_shape}")
    if tuple(mesh_device.shape) != mesh_shape:
        raise ValueError(f"mesh device shape {tuple(mesh_device.shape)} does not match resolved shape {mesh_shape}")
    if mesh_device.get_num_devices() != 32:
        raise ValueError(f"Galaxy CCL requires exactly 32 devices, got {mesh_device.get_num_devices()}")
    if mesh_device.arch() != architecture:
        raise ValueError("mesh device architecture does not match the resolved architecture")


@dataclass(frozen=True)
class GalaxyResourceKey:
    """Identity for resources that cannot safely alias across operation shapes."""

    operation: CollectiveName
    cluster_axis: int
    geometry: Hashable
    sequence_key: Hashable | None = None

    def __post_init__(self) -> None:
        if self.operation not in _COLLECTIVES:
            raise ValueError(f"unsupported collective: {self.operation}")
        if self.cluster_axis not in _CANONICAL_AXES[self.operation]:
            expected = sorted(_CANONICAL_AXES[self.operation])
            raise ValueError(f"{self.operation} requires canonical cluster axis in {expected}, got {self.cluster_axis}")
        if self.geometry is None:
            raise ValueError("collective geometry must be resolved")
        try:
            hash(self.geometry)
            hash(self.sequence_key)
        except TypeError as exc:
            raise TypeError("geometry and sequence_key must be hashable") from exc


@dataclass(frozen=True)
class GalaxyCollectiveResources:
    """Static handles and buffers for one fully keyed collective."""

    key: GalaxyResourceKey
    topology: Any
    semaphore_handles: tuple[Any, ...]
    barrier_semaphore_handles: tuple[Any, ...]
    num_links: int = 1
    persistent_output_buffers: tuple[Any, ...] = ()
    intermediate_output_buffers: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "semaphore_handles", tuple(self.semaphore_handles))
        object.__setattr__(self, "barrier_semaphore_handles", tuple(self.barrier_semaphore_handles))
        object.__setattr__(self, "persistent_output_buffers", tuple(self.persistent_output_buffers))
        object.__setattr__(self, "intermediate_output_buffers", tuple(self.intermediate_output_buffers))
        if self.topology is None:
            raise ValueError("topology must be resolved")
        if not self.semaphore_handles:
            raise ValueError("at least one semaphore handle set is required")
        if not self.barrier_semaphore_handles:
            raise ValueError("at least one barrier semaphore handle is required")
        if self.num_links < 1:
            raise ValueError(f"num_links must be positive, got {self.num_links}")

    @property
    def operation(self) -> CollectiveName:
        return self.key.operation

    @property
    def cluster_axis(self) -> int:
        return self.key.cluster_axis

    @property
    def geometry(self) -> Hashable:
        return self.key.geometry

    @property
    def sequence_key(self) -> Hashable | None:
        return self.key.sequence_key


class GalaxyCCLResourceFactory(Protocol):
    """Injected boundary for hardware allocation and release."""

    def create(self, mesh_device: Any, mode: GalaxyMode, key: GalaxyResourceKey) -> GalaxyCollectiveResources:
        ...

    def release(self, resource: GalaxyCollectiveResources) -> None:
        ...


@dataclass(frozen=True)
class GalaxyModeResources:
    """Frozen CCL strategy for a mode, with borrowed resources and owned specs."""

    mode: GalaxyMode
    mesh_device: Any
    worker_sub_device_id: Any
    stall_group: tuple[Any, ...]
    borrowed_resources: tuple[GalaxyCollectiveResources, ...] = ()
    owned_resource_keys: tuple[GalaxyResourceKey, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stall_group", tuple(self.stall_group))
        object.__setattr__(self, "borrowed_resources", tuple(self.borrowed_resources))
        object.__setattr__(self, "owned_resource_keys", tuple(self.owned_resource_keys))
        if self.mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported Galaxy CCL mode: {self.mode}")
        if self.mesh_device is None:
            raise ValueError("mesh_device must be resolved")
        if self.worker_sub_device_id is None:
            raise ValueError("worker_sub_device_id must be resolved")
        if not self.stall_group:
            raise ValueError("stall_group must be resolved")
        keys = [resource.key for resource in self.borrowed_resources] + list(self.owned_resource_keys)
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate Galaxy CCL resource key in {self.mode} mode")
        if not keys:
            raise ValueError(f"at least one Galaxy CCL resource is required for {self.mode} mode")


@dataclass(frozen=True)
class GalaxyCCLConfig:
    """Complete immutable configuration for the shared Galaxy CCL owner."""

    mesh_device: Any
    architecture: Any
    prefill: GalaxyModeResources
    decode: GalaxyModeResources
    resource_factory: GalaxyCCLResourceFactory | None = None
    mesh_shape: tuple[int, int] = (8, 4)

    def __post_init__(self) -> None:
        _validate_wh_galaxy(self.mesh_device, self.mesh_shape, self.architecture)
        if self.prefill.mode != "prefill" or self.decode.mode != "decode":
            raise ValueError("prefill and decode resources must match their declared modes")
        if self.prefill.mesh_device != self.mesh_device or self.decode.mesh_device != self.mesh_device:
            raise ValueError("all Galaxy CCL resources must refer to the configured mesh")
        if (self.prefill.owned_resource_keys or self.decode.owned_resource_keys) and self.resource_factory is None:
            raise ValueError("owned Galaxy CCL resource keys require an injected resource factory")


@runtime_checkable
class GalaxyCCLCollaborator(Protocol):
    @property
    def mesh_device(self) -> Any:
        ...

    def context(self, mode: GalaxyMode) -> "GalaxyCCLContext":
        ...

    def activate(self, mode: GalaxyMode) -> "GalaxyCCLContext":
        ...


class GalaxyCCLContext:
    """Mode-bound borrowed view with deterministic per-key cycling."""

    def __init__(self, mode_resources: GalaxyModeResources, resources: tuple[GalaxyCollectiveResources, ...]):
        self._mode_resources = mode_resources
        self._resources = {resource.key: resource for resource in resources}
        self._semaphore_indices = dict.fromkeys(self._resources, 0)
        self._barrier_indices = dict.fromkeys(self._resources, 0)
        self._lock = Lock()

    @property
    def mode(self) -> GalaxyMode:
        return self._mode_resources.mode

    @property
    def mesh_device(self) -> Any:
        return self._mode_resources.mesh_device

    @property
    def worker_sub_device_id(self) -> Any:
        return self._mode_resources.worker_sub_device_id

    @property
    def stall_group(self) -> tuple[Any, ...]:
        return self._mode_resources.stall_group

    @property
    def resource_keys(self) -> tuple[GalaxyResourceKey, ...]:
        return tuple(self._resources)

    def resources(
        self,
        collective: CollectiveName,
        cluster_axis: int | None = None,
        geometry: Hashable | None = None,
        sequence_key: Hashable | None = None,
    ) -> GalaxyCollectiveResources:
        if collective not in _COLLECTIVES:
            raise ValueError(f"unsupported collective: {collective}")
        matches = [
            resource
            for key, resource in self._resources.items()
            if key.operation == collective
            and (cluster_axis is None or key.cluster_axis == cluster_axis)
            and (geometry is None or key.geometry == geometry)
            and (sequence_key is None or key.sequence_key == sequence_key)
        ]
        if cluster_axis is None and geometry is None and sequence_key is None and len(matches) > 1:
            conventional_axis = 0 if collective in ("all_reduce", "all_gather_concat") else 1
            conventional = [resource for resource in matches if resource.cluster_axis == conventional_axis]
            if len(conventional) == 1:
                matches = conventional
        if not matches:
            raise KeyError(
                f"no {collective} resources for axis={cluster_axis}, geometry={geometry}, sequence={sequence_key}"
            )
        if len(matches) != 1:
            raise ValueError(f"ambiguous {collective} resources; specify axis, geometry, and sequence key")
        return matches[0]

    def next_semaphore_handles(
        self,
        collective: CollectiveName,
        cluster_axis: int | None = None,
        geometry: Hashable | None = None,
        sequence_key: Hashable | None = None,
    ) -> Any:
        resource = self.resources(collective, cluster_axis, geometry, sequence_key)
        with self._lock:
            index = self._semaphore_indices[resource.key]
            self._semaphore_indices[resource.key] = (index + 1) % len(resource.semaphore_handles)
        return resource.semaphore_handles[index]

    def next_semaphore_window(
        self,
        collective: CollectiveName,
        cluster_axis: int | None = None,
        geometry: Hashable | None = None,
        sequence_key: Hashable | None = None,
        *,
        count: int = 2,
    ) -> list[Any]:
        resource = self.resources(collective, cluster_axis, geometry, sequence_key)
        if count < 1 or count > len(resource.semaphore_handles):
            raise ValueError(f"semaphore window count must be in [1, {len(resource.semaphore_handles)}]")
        with self._lock:
            index = self._semaphore_indices[resource.key]
            self._semaphore_indices[resource.key] = (index + 1) % len(resource.semaphore_handles)
        return [
            resource.semaphore_handles[(index + offset) % len(resource.semaphore_handles)] for offset in range(count)
        ]

    def next_barrier_semaphore_handle(
        self,
        collective: CollectiveName,
        cluster_axis: int | None = None,
        geometry: Hashable | None = None,
        sequence_key: Hashable | None = None,
    ) -> Any:
        resource = self.resources(collective, cluster_axis, geometry, sequence_key)
        with self._lock:
            index = self._barrier_indices[resource.key]
            self._barrier_indices[resource.key] = (index + 1) % len(resource.barrier_semaphore_handles)
        return resource.barrier_semaphore_handles[index]

    def reset_cycles(self) -> None:
        with self._lock:
            for key in self._resources:
                self._semaphore_indices[key] = 0
                self._barrier_indices[key] = 0

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis: int) -> Any:
        return self.next_semaphore_handles("all_gather", cluster_axis)

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis: int) -> Any:
        return self.next_semaphore_handles("reduce_scatter", cluster_axis)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis: int) -> Any:
        return self.next_barrier_semaphore_handle("all_gather", cluster_axis)


class GalaxyCCL:
    """Owner of factory-created CCL resources and contexts that borrow them."""

    def __init__(self, config: GalaxyCCLConfig):
        self.config = config
        self._contexts: dict[GalaxyMode, GalaxyCCLContext] = {}
        self._owned_resources: list[GalaxyCollectiveResources] = []
        self._borrowed_resources: list[GalaxyCollectiveResources] = []
        self._active_mode: GalaxyMode | None = None
        self._cleaned = False
        try:
            for mode_config in (config.prefill, config.decode):
                resources = list(mode_config.borrowed_resources)
                self._borrowed_resources.extend(resources)
                for key in mode_config.owned_resource_keys:
                    resource = config.resource_factory.create(config.mesh_device, mode_config.mode, key)
                    self._owned_resources.append(resource)
                    if resource.key != key:
                        raise ValueError(f"resource factory returned key {resource.key}, expected {key}")
                    resources.append(resource)
                self._contexts[mode_config.mode] = GalaxyCCLContext(mode_config, tuple(resources))
        except Exception:
            self._release_owned(suppress_errors=True)
            raise

    @property
    def mesh_device(self) -> Any:
        return self.config.mesh_device

    @property
    def active_mode(self) -> GalaxyMode | None:
        return self._active_mode

    @property
    def owned_resource_keys(self) -> tuple[GalaxyResourceKey, ...]:
        return tuple(resource.key for resource in self._owned_resources)

    @property
    def borrowed_resource_keys(self) -> tuple[GalaxyResourceKey, ...]:
        return tuple(resource.key for resource in self._borrowed_resources)

    def context(self, mode: GalaxyMode) -> GalaxyCCLContext:
        self._ensure_open()
        try:
            return self._contexts[mode]
        except KeyError as exc:
            raise ValueError(f"unsupported Galaxy CCL mode: {mode}") from exc

    def activate(self, mode: GalaxyMode) -> GalaxyCCLContext:
        context = self.context(mode)
        self._active_mode = mode
        return context

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis: int) -> Any:
        return self._active_context().get_and_cycle_ag_semaphore_handles(cluster_axis)

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis: int) -> Any:
        return self._active_context().get_and_cycle_rs_semaphore_handles(cluster_axis)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis: int) -> Any:
        return self._active_context().get_and_cycle_barrier_semaphore_handle(cluster_axis)

    def reset_cycles(self, mode: GalaxyMode | None = None) -> None:
        self._ensure_open()
        if mode is None:
            for context in self._contexts.values():
                context.reset_cycles()
            return
        self.context(mode).reset_cycles()

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._active_mode = None
        self._contexts.clear()
        error = self._release_owned(suppress_errors=False)
        self._borrowed_resources.clear()
        self._cleaned = True
        if error is not None:
            raise error

    def __enter__(self) -> "GalaxyCCL":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

    def _release_owned(self, *, suppress_errors: bool) -> Exception | None:
        first_error: Exception | None = None
        while self._owned_resources:
            resource = self._owned_resources.pop()
            try:
                self.config.resource_factory.release(resource)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        return None if suppress_errors else first_error

    def _active_context(self) -> GalaxyCCLContext:
        if self._active_mode is None:
            raise RuntimeError("Galaxy CCL mode must be activated at the operation boundary")
        return self._contexts[self._active_mode]

    def _ensure_open(self) -> None:
        if self._cleaned:
            raise RuntimeError("Galaxy CCL has been cleaned up")
