# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Production allocation boundary for WH Galaxy CCL and subdevice resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

import ttnn
from models.common.models.galaxy.ccl import (
    GalaxyCCL,
    GalaxyCCLConfig,
    GalaxyCollectiveResources,
    GalaxyMode,
    GalaxyModeResources,
    GalaxyResourceKey,
)
from models.common.modules.prefetcher import Prefetcher2DResourceOwner


@dataclass(frozen=True)
class GalaxyTensorSpec:
    """Exact device allocation required by one persistent CCL tensor."""

    shape: tuple[int, ...]
    dtype: Any
    layout: Any
    memory_config: Any
    mesh_mapper: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.shape or any(not isinstance(value, int) or value <= 0 for value in self.shape):
            raise ValueError(f"tensor shape must contain positive integers, got {self.shape}")
        if self.dtype is None or self.layout is None or self.memory_config is None:
            raise ValueError("tensor dtype, layout, and memory_config must be resolved")


@dataclass(frozen=True)
class GalaxyCollectivePlan:
    """Fully resolved allocation recipe for one operation/axis/geometry key."""

    key: GalaxyResourceKey
    topology: Any
    persistent_output_specs: tuple[GalaxyTensorSpec, ...]
    intermediate_output_specs: tuple[GalaxyTensorSpec, ...] = ()
    num_links: int = 1
    semaphore_slots: int = 2
    semaphores_per_slot: int = 1
    barrier_slots: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "persistent_output_specs", tuple(self.persistent_output_specs))
        object.__setattr__(self, "intermediate_output_specs", tuple(self.intermediate_output_specs))
        if self.topology is None:
            raise ValueError("collective topology must be resolved")
        for name in ("num_links", "semaphore_slots", "semaphores_per_slot", "barrier_slots"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.persistent_output_specs:
            raise ValueError("at least one persistent output tensor spec is required")
        if self.key.operation == "reduce_scatter" and not self.intermediate_output_specs:
            raise ValueError("reduce_scatter requires at least one intermediate output tensor spec")


@dataclass(frozen=True)
class GalaxyModePlan:
    """Subdevice and collective allocations for one execution mode."""

    mode: GalaxyMode
    sub_devices: tuple[Any, ...]
    worker_sub_device_id: Any
    stall_group: tuple[Any, ...]
    semaphore_cores: Any
    collectives: tuple[GalaxyCollectivePlan, ...]
    local_l1_size: int = 0
    #: The worker subdevice's core set. `ttnn.SubDevice` exposes no accessor, so
    #: it cannot be recovered from `sub_devices`; supply it here and the D3
    #: invariant below is enforced rather than merely documented.
    worker_cores: Any = None
    #: Opt in to a `semaphore_cores` narrower than the worker subdevice. Safe
    #: only for a collective that binds its semaphore to a grid it owns, the way
    #: the fused RMS all-gather does - see `_require_semaphore_cores_cover_workers`.
    allow_narrow_semaphore_cores: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sub_devices", tuple(self.sub_devices))
        object.__setattr__(self, "stall_group", tuple(self.stall_group))
        object.__setattr__(self, "collectives", tuple(self.collectives))
        if self.mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported Galaxy mode: {self.mode}")
        if not self.sub_devices or self.worker_sub_device_id is None or not self.stall_group:
            raise ValueError("subdevices, worker_sub_device_id, and stall_group must be resolved")
        if self.worker_sub_device_id not in self.stall_group:
            raise ValueError("stall_group must include worker_sub_device_id")
        if self.semaphore_cores is None:
            raise ValueError("semaphore_cores must be resolved")
        self._require_semaphore_cores_cover_workers()
        if not self.collectives:
            raise ValueError(f"at least one collective is required for {self.mode}")
        keys = tuple(plan.key for plan in self.collectives)
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate Galaxy resource key in {self.mode} plan")
        if self.local_l1_size < 0:
            raise ValueError("local_l1_size cannot be negative")

    def _require_semaphore_cores_cover_workers(self) -> None:
        """Reject a semaphore allocation narrower than the worker subdevice.

        Milestone A defect D3. The generic async CCLs (`all_gather_async`,
        `reduce_scatter_minimal_async`, `all_reduce_async`,
        `all_reduce_create_qkv_heads`) choose their sender worker cores from the
        worker subdevice minus the reserved output cores. A global semaphore
        allocated on a narrower set therefore leaves a sender polling an L1
        address its own core never had reserved or zeroed - which **hangs the
        collective indefinitely** rather than failing it. That cost four
        consecutive 2700 s timeouts to diagnose, after one process had passed.

        Narrowing is legitimate for a collective that binds its semaphore to a
        grid it owns, as the fused RMS all-gather does. That case must say so
        with `allow_narrow_semaphore_cores=True`; it is not inferable from the
        plan, because both forms key on the same `all_gather` operation name.

        The check is skipped when either side is not a `CoreRangeSet` - host
        tests legitimately build plans out of stand-in objects.
        """

        if self.worker_cores is None or self.allow_narrow_semaphore_cores:
            return
        subtract = getattr(self.worker_cores, "subtract", None)
        if not callable(subtract) or not hasattr(self.semaphore_cores, "num_cores"):
            return
        uncovered = subtract(self.semaphore_cores)
        if uncovered.num_cores():
            raise ValueError(
                f"{self.mode} semaphore_cores must cover the worker subdevice; "
                f"{uncovered.num_cores()} worker core(s) are outside it, starting at "
                f"{uncovered.bounding_box().start}. A sender on an uncovered core polls an L1 "
                "address that was never reserved or zeroed and hangs (Milestone A defect D3). "
                "Set allow_narrow_semaphore_cores=True only for a collective that binds its "
                "semaphore to a grid it owns."
            )


@dataclass(frozen=True)
class GalaxyResourcesConfig:
    """Complete production resource policy for one WH Galaxy mesh."""

    architecture: Any
    prefill: GalaxyModePlan
    decode: GalaxyModePlan
    mesh_shape: tuple[int, int] = (8, 4)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_shape", tuple(self.mesh_shape))
        if self.architecture != ttnn.device.Arch.WORMHOLE_B0:
            raise ValueError(f"Galaxy resources require Wormhole B0, got {self.architecture}")
        if self.mesh_shape != (8, 4):
            raise ValueError(f"Galaxy resources require logical mesh shape (8, 4), got {self.mesh_shape}")
        if self.prefill.mode != "prefill" or self.decode.mode != "decode":
            raise ValueError("prefill and decode plans must match their declared modes")


def _allocate_tensor(mesh_device: Any, spec: GalaxyTensorSpec) -> Any:
    if spec.mesh_mapper is not None:
        return ttnn.from_torch(
            torch.zeros(spec.shape, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=spec.dtype,
            layout=spec.layout,
            memory_config=spec.memory_config,
            mesh_mapper=spec.mesh_mapper,
        )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(spec.shape), spec.dtype, spec.layout, mesh_device, spec.memory_config
    )


def _deallocate_tensor(tensor: Any) -> None:
    deallocate = getattr(tensor, "deallocate", None)
    if callable(deallocate):
        deallocate(True)
    else:
        ttnn.deallocate(tensor)


@dataclass(frozen=True)
class GalaxyResourceBindings:
    """TTNN calls used by the production owner, injectable for host tests."""

    create_semaphore: Callable[[Any, Any, int], Any] = ttnn.create_global_semaphore
    reset_semaphore: Callable[[Any, int], None] = ttnn.reset_global_semaphore_value
    allocate_tensor: Callable[[Any, GalaxyTensorSpec], Any] = _allocate_tensor
    deallocate_tensor: Callable[[Any], None] = _deallocate_tensor
    synchronize: Callable[..., None] = ttnn.synchronize_device


class TTNNGalaxyCCLResourceFactory:
    """Allocate concrete TTNN resources for exact mode and geometry plans."""

    def __init__(
        self,
        mesh_device: Any,
        mode_plans: tuple[GalaxyModePlan, ...],
        *,
        bindings: GalaxyResourceBindings | None = None,
    ):
        self._mesh_device = mesh_device
        self._bindings = bindings or GalaxyResourceBindings()
        self._plans = {(mode.mode, plan.key): (mode, plan) for mode in mode_plans for plan in mode.collectives}
        self._allocations: dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}

    def create(self, mesh_device: Any, mode: GalaxyMode, key: GalaxyResourceKey) -> GalaxyCollectiveResources:
        if mesh_device is not self._mesh_device:
            raise ValueError("resource allocation requested for a different mesh")
        try:
            mode_plan, plan = self._plans[(mode, key)]
        except KeyError as exc:
            raise KeyError(f"no Galaxy allocation plan for mode={mode}, key={key}") from exc

        semaphores: list[Any] = []
        buffers: list[Any] = []
        try:
            semaphore_handles = tuple(
                self._allocate_semaphore_set(mode_plan.semaphore_cores, plan.semaphores_per_slot, semaphores)
                for _ in range(plan.semaphore_slots)
            )
            barrier_handles = tuple(
                self._allocate_semaphore_set(mode_plan.semaphore_cores, 1, semaphores)
                for _ in range(plan.barrier_slots)
            )
            persistent = tuple(self._allocate_buffer(spec, buffers) for spec in plan.persistent_output_specs)
            intermediate = tuple(self._allocate_buffer(spec, buffers) for spec in plan.intermediate_output_specs)
            resource = GalaxyCollectiveResources(
                key=key,
                topology=plan.topology,
                semaphore_handles=semaphore_handles,
                barrier_semaphore_handles=barrier_handles,
                num_links=plan.num_links,
                persistent_output_buffers=persistent,
                intermediate_output_buffers=intermediate,
            )
        except Exception:
            self._release_parts(tuple(semaphores), tuple(buffers), suppress_errors=True)
            raise

        self._allocations[id(resource)] = (tuple(semaphores), tuple(buffers))
        return resource

    def release(self, resource: GalaxyCollectiveResources) -> None:
        allocation = self._allocations.pop(id(resource), None)
        if allocation is None:
            return
        semaphores, buffers = allocation
        error = self._release_parts(semaphores, buffers, suppress_errors=False)
        if error is not None:
            raise error

    def _allocate_semaphore_set(self, cores: Any, count: int, owned: list[Any]) -> Any:
        handles = []
        for _ in range(count):
            handle = self._bindings.create_semaphore(self._mesh_device, cores, 0)
            handles.append(handle)
            owned.append(handle)
        return handles[0] if count == 1 else handles

    def _allocate_buffer(self, spec: GalaxyTensorSpec, owned: list[Any]) -> Any:
        tensor = self._bindings.allocate_tensor(self._mesh_device, spec)
        owned.append(tensor)
        return tensor

    def _release_parts(
        self, semaphores: tuple[Any, ...], buffers: tuple[Any, ...], *, suppress_errors: bool
    ) -> Exception | None:
        first_error: Exception | None = None
        for buffer in reversed(buffers):
            try:
                self._bindings.deallocate_tensor(buffer)
            except Exception as exc:
                first_error = first_error or exc
        # GlobalSemaphore owns its L1 allocation through RAII. Reset before dropping
        # factory ownership so stale values cannot survive an externally held handle.
        for semaphore in reversed(semaphores):
            try:
                self._bindings.reset_semaphore(semaphore, 0)
            except Exception as exc:
                first_error = first_error or exc
        return None if suppress_errors else first_error


class GalaxyResources:
    """Own Galaxy CCL allocations while borrowing Prefetcher2D mode resources."""

    def __init__(
        self,
        mesh_device: Any,
        config: GalaxyResourcesConfig,
        *,
        prefetcher: Prefetcher2DResourceOwner,
        bindings: GalaxyResourceBindings | None = None,
    ):
        self._mesh_device = mesh_device
        self.config = config
        self._prefetcher = prefetcher
        self._bindings = bindings or GalaxyResourceBindings()
        self._active_mode: GalaxyMode | None = None
        self._cleaned = False
        self._ccl: GalaxyCCL | None = None

        self._validate_mesh()
        self._validate_prefetcher()
        factory = TTNNGalaxyCCLResourceFactory(mesh_device, (config.prefill, config.decode), bindings=self._bindings)
        self._ccl = GalaxyCCL(
            GalaxyCCLConfig(
                mesh_device=mesh_device,
                architecture=config.architecture,
                mesh_shape=config.mesh_shape,
                prefill=self._ccl_mode_resources(config.prefill),
                decode=self._ccl_mode_resources(config.decode),
                resource_factory=factory,
            )
        )

    @property
    def ccl(self) -> "GalaxyResources":
        """Expose the owner itself as the lifecycle-aware CCL collaborator."""

        return self

    @property
    def prefetcher(self) -> Prefetcher2DResourceOwner:
        """Return the borrowed model-owned Prefetcher2D collaborator."""

        return self._prefetcher

    @property
    def mesh_device(self) -> Any:
        return self._mesh_device

    @property
    def active_mode(self) -> GalaxyMode | None:
        return self._active_mode

    def context(self, mode: GalaxyMode) -> Any:
        self._ensure_open()
        return self._ccl.context(mode)

    def activate(self, mode: GalaxyMode) -> Any:
        self._ensure_open()
        context = self._ccl.context(mode)
        if mode == self._active_mode:
            self._prefetcher.activate(mode)
            self._ccl.activate(mode)
            return context

        previous = self._active_mode
        if previous is not None:
            self._synchronize(previous)
        self._prefetcher.activate(mode)
        self._active_mode = mode
        self._ccl.activate(mode)
        return context

    def synchronize(self, mode: GalaxyMode) -> None:
        """Wait for the mode's worker without stalling the persistent decode sender."""

        self._ensure_open()
        self._synchronize(mode)

    def reset_cycles(self, mode: GalaxyMode | None = None) -> None:
        self._ensure_open()
        self._ccl.reset_cycles(mode)

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis: int) -> Any:
        return self._ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis: int) -> Any:
        return self._ccl.get_and_cycle_rs_semaphore_handles(cluster_axis)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis: int) -> Any:
        return self._ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        first_error: Exception | None = None

        def attempt(action: Callable[[], None]) -> None:
            nonlocal first_error
            try:
                action()
            except Exception as exc:
                first_error = first_error or exc

        if self._active_mode is not None:
            attempt(lambda: self._synchronize(self._active_mode))
        if self._ccl is not None:
            attempt(self._ccl.cleanup)
        self._active_mode = None
        self._cleaned = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "GalaxyResources":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()

    def _validate_mesh(self) -> None:
        if tuple(self._mesh_device.shape) != self.config.mesh_shape:
            raise ValueError("mesh device shape does not match Galaxy resource config")
        if self._mesh_device.get_num_devices() != 32:
            raise ValueError("Galaxy resources require exactly 32 devices")
        if self._mesh_device.arch() != self.config.architecture:
            raise ValueError("mesh device architecture does not match Galaxy resource config")

    def _validate_prefetcher(self) -> None:
        if self._prefetcher.mesh_device is not self._mesh_device:
            raise ValueError("Prefetcher2D belongs to a different mesh")
        for plan in (self.config.prefill, self.config.decode):
            self._prefetcher.borrow_context(
                plan.mode,
                sub_devices=plan.sub_devices,
                worker_sub_device_id=plan.worker_sub_device_id,
                stall_group=plan.stall_group,
                local_l1_size=plan.local_l1_size,
            )

    def _ccl_mode_resources(self, plan: GalaxyModePlan) -> GalaxyModeResources:
        return GalaxyModeResources(
            mode=plan.mode,
            mesh_device=self._mesh_device,
            worker_sub_device_id=plan.worker_sub_device_id,
            stall_group=plan.stall_group,
            owned_resource_keys=tuple(collective.key for collective in plan.collectives),
        )

    def _mode_plan(self, mode: GalaxyMode) -> GalaxyModePlan:
        if mode == "prefill":
            return self.config.prefill
        if mode == "decode":
            return self.config.decode
        raise ValueError(f"unsupported Galaxy mode: {mode}")

    def _synchronize(self, mode: GalaxyMode) -> None:
        self._bindings.synchronize(self._mesh_device, sub_device_ids=list(self._mode_plan(mode).stall_group))

    def _ensure_open(self) -> None:
        if self._cleaned:
            raise RuntimeError("Galaxy resources have been cleaned up")


def create_galaxy_resources(
    mesh_device: Any,
    *,
    config: GalaxyResourcesConfig,
    prefetcher: Prefetcher2DResourceOwner,
    bindings: GalaxyResourceBindings | None = None,
) -> GalaxyResources:
    """Allocate a fully resolved WH `(8, 4)` production resource owner."""

    return GalaxyResources(mesh_device, config, prefetcher=prefetcher, bindings=bindings)
