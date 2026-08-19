# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import ttnn
from models.common.models.galaxy import (
    GalaxyCollectivePlan,
    GalaxyModePlan,
    GalaxyResourceBindings,
    GalaxyResourceKey,
    GalaxyResourcesConfig,
    GalaxyTensorSpec,
    create_galaxy_resources,
)


class FakeMesh:
    def __init__(self, *, fail_stall_group=None):
        self.shape = (8, 4)
        self.events = []
        self._manager_index = 0
        self.fail_stall_group = fail_stall_group

    def get_num_devices(self):
        return 32

    def arch(self):
        return ttnn.device.Arch.WORMHOLE_B0

    def create_sub_device_manager(self, sub_devices, local_l1_size):
        manager = f"manager-{self._manager_index}"
        self._manager_index += 1
        self.events.append(("create-manager", manager, tuple(sub_devices), local_l1_size))
        return manager

    def load_sub_device_manager(self, manager):
        self.events.append(("load-manager", manager))

    def set_sub_device_stall_group(self, stall_group):
        self.events.append(("stall-group", tuple(stall_group)))
        if tuple(stall_group) == self.fail_stall_group:
            raise RuntimeError("stall group failure")

    def reset_sub_device_stall_group(self):
        self.events.append(("reset-stall-group",))

    def clear_loaded_sub_device_manager(self):
        self.events.append(("clear-manager",))

    def remove_sub_device_manager(self, manager):
        self.events.append(("remove-manager", manager))


class FakeTTNN:
    def __init__(self, *, fail_shape=None):
        self.events = []
        self.fail_shape = fail_shape
        self._semaphore_index = 0

    def create_semaphore(self, mesh, cores, initial_value):
        handle = f"sem-{self._semaphore_index}"
        self._semaphore_index += 1
        self.events.append(("create-semaphore", handle, mesh, cores, initial_value))
        return handle

    def reset_semaphore(self, handle, value):
        self.events.append(("reset-semaphore", handle, value))

    def allocate_tensor(self, mesh, spec):
        if spec.shape == self.fail_shape:
            raise RuntimeError("allocation failure")
        tensor = f"tensor-{spec.shape}"
        self.events.append(("allocate", tensor, mesh, spec))
        return tensor

    def deallocate_tensor(self, tensor):
        self.events.append(("deallocate", tensor))

    def synchronize(self, mesh, *, sub_device_ids):
        self.events.append(("synchronize", mesh, tuple(sub_device_ids)))

    def bindings(self):
        return GalaxyResourceBindings(
            create_semaphore=self.create_semaphore,
            reset_semaphore=self.reset_semaphore,
            allocate_tensor=self.allocate_tensor,
            deallocate_tensor=self.deallocate_tensor,
            synchronize=self.synchronize,
        )


class FakePrefetcher:
    def __init__(self, mesh, config, *, fail_mode=None):
        self.mesh_device = mesh
        self.events = []
        self.fail_mode = fail_mode
        self.cleaned = False
        self._plans = {plan.mode: plan for plan in (config.prefill, config.decode)}

    def borrow_context(self, mode, *, sub_devices, worker_sub_device_id, stall_group, local_l1_size):
        self.events.append(("borrow", mode))
        plan = self._plans[mode]
        requested = (tuple(sub_devices), worker_sub_device_id, tuple(stall_group), local_l1_size)
        expected = (plan.sub_devices, plan.worker_sub_device_id, plan.stall_group, plan.local_l1_size)
        if requested != expected:
            raise ValueError("incompatible prefetcher mode plan")
        return SimpleNamespace(
            mode=mode,
            mesh_device=self.mesh_device,
            worker_sub_device_id=worker_sub_device_id,
            stall_group=tuple(stall_group),
        )

    def activate(self, mode):
        self.events.append(("activate", mode))
        if mode == self.fail_mode:
            raise RuntimeError("prefetcher activation failure")
        plan = self._plans[mode]
        return SimpleNamespace(mode=mode, worker_sub_device_id=plan.worker_sub_device_id)

    def cleanup(self):
        self.cleaned = True


def tensor_spec(shape):
    return GalaxyTensorSpec(shape, "dtype", "layout", "memory-config")


def collective(operation, mode, *, shape, axis=None, semaphores_per_slot=1):
    axis = axis if axis is not None else (0 if operation == "all_reduce" else 1)
    key = GalaxyResourceKey(operation, axis, f"{mode}-{operation}", 2048 if mode == "prefill" else 32)
    intermediate = (tensor_spec((*shape[:-1], shape[-1] * 4)),) if operation == "reduce_scatter" else ()
    return GalaxyCollectivePlan(
        key=key,
        topology=ttnn.Topology.Linear,
        num_links=2,
        semaphore_slots=2,
        semaphores_per_slot=semaphores_per_slot,
        barrier_slots=2,
        persistent_output_specs=(tensor_spec(shape),),
        intermediate_output_specs=intermediate,
    )


def mode_plan(mode):
    sequence = 2048 if mode == "prefill" else 32
    return GalaxyModePlan(
        mode=mode,
        sub_devices=(f"{mode}-subdevice",),
        worker_sub_device_id=f"{mode}-worker",
        stall_group=(f"{mode}-worker",),
        semaphore_cores=f"{mode}-cores",
        collectives=(
            collective("reduce_scatter", mode, shape=(1, 1, sequence, 896), semaphores_per_slot=3),
            collective("all_gather", mode, shape=(1, 1, sequence, 3584)),
            collective("all_reduce", mode, shape=(1, 1, sequence, 8192)),
        ),
    )


def resource_config():
    return GalaxyResourcesConfig(
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=mode_plan("prefill"),
        decode=mode_plan("decode"),
    )


def create_owner(mesh, fake_ttnn, *, config=None, prefetcher=None):
    config = config or resource_config()
    prefetcher = prefetcher or FakePrefetcher(mesh, config)
    owner = create_galaxy_resources(
        mesh,
        config=config,
        prefetcher=prefetcher,
        bindings=fake_ttnn.bindings(),
    )
    return owner, prefetcher


def test_resource_config_is_explicit_frozen_and_rejects_incomplete_geometry():
    config = resource_config()
    with pytest.raises(FrozenInstanceError):
        config.mesh_shape = (4, 8)
    with pytest.raises(TypeError, match="config"):
        create_galaxy_resources(FakeMesh())
    with pytest.raises(ValueError, match="intermediate"):
        GalaxyCollectivePlan(
            key=GalaxyResourceKey("reduce_scatter", 1, "missing-intermediate"),
            topology=ttnn.Topology.Linear,
            persistent_output_specs=(tensor_spec((1, 1, 32, 896)),),
        )
    prefill = mode_plan("prefill")
    with pytest.raises(ValueError, match="stall_group must include"):
        GalaxyModePlan(
            mode="prefill",
            sub_devices=prefill.sub_devices,
            worker_sub_device_id="worker",
            stall_group=("different-worker",),
            semaphore_cores=prefill.semaphore_cores,
            collectives=prefill.collectives,
        )


def test_factory_allocates_exact_mode_geometry_and_semaphore_cardinality():
    mesh = FakeMesh()
    fake_ttnn = FakeTTNN()
    owner, prefetcher = create_owner(mesh, fake_ttnn)

    prefill = owner.context("prefill")
    rs_plan, ag_plan, ar_plan = resource_config().prefill.collectives
    rs = prefill.resources("reduce_scatter", 1, rs_plan.key.geometry, 2048)
    ag = prefill.resources("all_gather", 1, ag_plan.key.geometry, 2048)
    ar = prefill.resources("all_reduce", 0, ar_plan.key.geometry, 2048)

    assert len(rs.semaphore_handles) == 2
    assert all(len(handles) == 3 for handles in rs.semaphore_handles)
    assert rs.persistent_output_buffers == ("tensor-(1, 1, 2048, 896)",)
    assert rs.intermediate_output_buffers == ("tensor-(1, 1, 2048, 3584)",)
    assert ag.persistent_output_buffers == ("tensor-(1, 1, 2048, 3584)",)
    assert ar.persistent_output_buffers == ("tensor-(1, 1, 2048, 8192)",)
    assert owner.ccl is owner
    assert owner.prefetcher is prefetcher

    allocated_specs = [event[3] for event in fake_ttnn.events if event[0] == "allocate"]
    assert tensor_spec((1, 1, 2048, 896)) in allocated_specs
    assert tensor_spec((1, 1, 32, 8192)) in allocated_specs
    owner.cleanup()


def test_activation_serializes_mode_switches_and_cleanup_is_reverse_and_idempotent():
    mesh = FakeMesh()
    fake_ttnn = FakeTTNN()
    owner, prefetcher = create_owner(mesh, fake_ttnn)

    owner.activate("decode")
    owner.activate("decode")
    owner.activate("prefill")
    owner.cleanup()
    event_count = (len(prefetcher.events), len(fake_ttnn.events))
    owner.cleanup()

    assert [event for event in prefetcher.events if event[0] == "activate"] == [
        ("activate", "decode"),
        ("activate", "decode"),
        ("activate", "prefill"),
    ]
    assert [event for event in fake_ttnn.events if event[0] == "synchronize"] == [
        ("synchronize", mesh, ("decode-worker",)),
        ("synchronize", mesh, ("prefill-worker",)),
    ]
    assert mesh.events == []
    assert not prefetcher.cleaned
    assert event_count == (len(prefetcher.events), len(fake_ttnn.events))
    with pytest.raises(RuntimeError, match="cleaned up"):
        owner.context("decode")


def test_public_synchronize_is_worker_scoped_and_terminal_after_cleanup():
    mesh = FakeMesh()
    fake_ttnn = FakeTTNN()
    owner, _ = create_owner(mesh, fake_ttnn)

    owner.activate("decode")
    owner.synchronize("decode")
    owner.activate("prefill")
    owner.synchronize("prefill")

    explicit_waits = [event for event in fake_ttnn.events if event[0] == "synchronize"]
    assert explicit_waits[:3] == [
        ("synchronize", mesh, ("decode-worker",)),
        ("synchronize", mesh, ("decode-worker",)),
        ("synchronize", mesh, ("prefill-worker",)),
    ]
    owner.cleanup()
    with pytest.raises(RuntimeError, match="cleaned up"):
        owner.synchronize("decode")


def test_allocation_failure_rolls_back_only_ccl_resources():
    mesh = FakeMesh()
    fake_ttnn = FakeTTNN(fail_shape=(1, 1, 2048, 3584))
    config = resource_config()
    prefetcher = FakePrefetcher(mesh, config)

    with pytest.raises(RuntimeError, match="allocation failure"):
        create_galaxy_resources(mesh, config=config, prefetcher=prefetcher, bindings=fake_ttnn.bindings())

    assert ("deallocate", "tensor-(1, 1, 2048, 896)") in fake_ttnn.events
    assert any(event[0] == "reset-semaphore" for event in fake_ttnn.events)
    assert mesh.events == []
    assert not prefetcher.cleaned


def test_prefetcher_activation_failure_does_not_publish_ccl_mode():
    mesh = FakeMesh()
    fake_ttnn = FakeTTNN()
    config = resource_config()
    prefetcher = FakePrefetcher(mesh, config, fail_mode="decode")
    owner, _ = create_owner(mesh, fake_ttnn, config=config, prefetcher=prefetcher)

    with pytest.raises(RuntimeError, match="prefetcher activation failure"):
        owner.activate("decode")

    assert owner.active_mode is None
    assert owner._ccl.active_mode is None
    assert mesh.events == []
    owner.cleanup()


def test_incompatible_prefetcher_is_rejected_before_ccl_allocation():
    mesh = FakeMesh()
    fake_ttnn = FakeTTNN()
    config = resource_config()
    incompatible = FakePrefetcher(mesh, config)
    incompatible._plans["decode"] = mode_plan("prefill")

    with pytest.raises(ValueError, match="incompatible prefetcher"):
        create_galaxy_resources(
            mesh,
            config=config,
            prefetcher=incompatible,
            bindings=fake_ttnn.bindings(),
        )

    assert fake_ttnn.events == []
    assert mesh.events == []


def test_cleanup_continues_after_release_error():
    mesh = FakeMesh()
    fake_ttnn = FakeTTNN()
    original_deallocate = fake_ttnn.deallocate_tensor
    failed = False

    def fail_once(tensor):
        nonlocal failed
        original_deallocate(tensor)
        if not failed:
            failed = True
            raise RuntimeError("deallocate failure")

    bindings = fake_ttnn.bindings()
    bindings = GalaxyResourceBindings(
        create_semaphore=bindings.create_semaphore,
        reset_semaphore=bindings.reset_semaphore,
        allocate_tensor=bindings.allocate_tensor,
        deallocate_tensor=fail_once,
        synchronize=bindings.synchronize,
    )
    config = resource_config()
    prefetcher = FakePrefetcher(mesh, config)
    owner = create_galaxy_resources(mesh, config=config, prefetcher=prefetcher, bindings=bindings)

    with pytest.raises(RuntimeError, match="deallocate failure"):
        owner.cleanup()

    assert mesh.events == []
    assert not prefetcher.cleaned
    assert len([event for event in fake_ttnn.events if event[0] == "deallocate"]) == 8
