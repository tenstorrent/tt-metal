# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for composing concrete Prefetcher2D and GalaxyResources owners."""

from __future__ import annotations

from dataclasses import replace

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
from models.common.modules.prefetcher import Prefetcher2D, Prefetcher2DConfig, Prefetcher2DModeConfig


class FakeMesh:
    shape = (8, 4)

    def __init__(self, events):
        self.events = events
        self._manager_index = 0

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

    def reset_sub_device_stall_group(self):
        self.events.append(("reset-stall-group",))

    def clear_loaded_sub_device_manager(self):
        self.events.append(("clear-manager",))

    def remove_sub_device_manager(self, manager):
        self.events.append(("remove-manager", manager))


class FakeWeight:
    def __init__(self, mesh, address=0x1000, size=128):
        self._mesh = mesh
        self._address = address
        self._size = size

    def device(self):
        return self._mesh

    def buffer_address(self):
        return self._address

    def buffer_size(self):
        return self._size


class HostResources:
    def __init__(self, events):
        self.events = events
        self.deallocated = []
        self._semaphore_index = 0
        self._tensor_index = 0
        self._prefetch_index = 0
        self.fail_prefetch_start = False

    def create_global_cb(self, mesh, mapping, size):
        resource = ("global-cb", tuple(mapping), size)
        self.events.append(("create-global-cb", resource))
        return resource

    def create_metadata(self, addresses, **kwargs):
        resource = ("metadata", tuple(tuple(row) for row in addresses.tolist()))
        self.events.append(("create-metadata", resource))
        return resource

    def start_prefetch(self, context):
        self.events.append(("start-prefetch", context.mode))
        if self.fail_prefetch_start:
            raise RuntimeError("prefetch start failure")
        self._prefetch_index += 1
        return ("prefetch-result", self._prefetch_index)

    def stop_prefetch(self, mesh, result):
        self.events.append(("stop-prefetch", result))
        self.deallocate(result)
        return ("prefetch-sync", result[-1])

    def create_semaphore(self, mesh, cores, initial_value):
        handle = f"semaphore-{self._semaphore_index}"
        self._semaphore_index += 1
        self.events.append(("create-semaphore", handle, cores, initial_value))
        return handle

    def reset_semaphore(self, handle, value):
        self.events.append(("reset-semaphore", handle, value))

    def allocate_tensor(self, mesh, spec):
        tensor = ("ccl-tensor", self._tensor_index, spec.shape)
        self._tensor_index += 1
        self.events.append(("allocate-ccl", tensor))
        return tensor

    def deallocate(self, resource):
        self.deallocated.append(resource)
        self.events.append(("deallocate", resource))

    def synchronize(self, mesh, *, sub_device_ids):
        self.events.append(("synchronize", tuple(sub_device_ids)))

    def galaxy_bindings(self):
        return GalaxyResourceBindings(
            create_semaphore=self.create_semaphore,
            reset_semaphore=self.reset_semaphore,
            allocate_tensor=self.allocate_tensor,
            deallocate_tensor=self.deallocate,
            synchronize=self.synchronize,
        )

    def prefetcher_kwargs(self):
        return {
            "create_global_cb": self.create_global_cb,
            "create_address_metadata": self.create_metadata,
            "deallocate": self.deallocate,
            "dram_prefetch_start": self.start_prefetch,
            "dram_prefetch_stop": self.stop_prefetch,
        }


def mode_plan(mode):
    worker = f"{mode}-worker"
    return GalaxyModePlan(
        mode=mode,
        sub_devices=(f"{mode}-subdevice",),
        worker_sub_device_id=worker,
        stall_group=(worker,),
        semaphore_cores=f"{mode}-cores",
        collectives=(
            GalaxyCollectivePlan(
                key=GalaxyResourceKey("all_reduce", 0, f"{mode}-geometry"),
                topology=ttnn.Topology.Linear,
                persistent_output_specs=(GalaxyTensorSpec((1, 1, 32, 128), "dtype", "layout", "memory-config"),),
            ),
        ),
    )


def galaxy_config():
    return GalaxyResourcesConfig(
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=mode_plan("prefill"),
        decode=mode_plan("decode"),
    )


def create_composition(*, fail_prefetch_start=False, galaxy=None):
    events = []
    mesh = FakeMesh(events)
    host = HostResources(events)
    host.fail_prefetch_start = fail_prefetch_start
    galaxy = galaxy or galaxy_config()

    def prefetch_mode(plan):
        return Prefetcher2DModeConfig(
            mode=plan.mode,
            sub_devices=plan.sub_devices,
            worker_sub_device_id=plan.worker_sub_device_id,
            stall_group=plan.stall_group,
            local_l1_size=plan.local_l1_size,
        )

    prefetcher = Prefetcher2D(
        Prefetcher2DConfig(
            mesh_device=mesh,
            architecture=ttnn.device.Arch.WORMHOLE_B0,
            prefill=prefetch_mode(galaxy_config().prefill),
            decode=prefetch_mode(galaxy_config().decode),
            sender_receiver_mapping=(("sender", "receiver"),),
            global_cb_size=256,
            expected_weight_count=1,
            address_repeat_count=1,
            address_memory_config="address-memory-config",
            address_mesh_mapper="address-mesh-mapper",
        ),
        **host.prefetcher_kwargs(),
    )
    prefetcher.initialize()
    weight = FakeWeight(mesh)
    prefetcher.register_weight("projection", weight)
    prefetcher.seal()
    return events, mesh, host, weight, prefetcher, galaxy


def create_galaxy_owner(mesh, host, prefetcher, config):
    return create_galaxy_resources(
        mesh,
        config=config,
        prefetcher=prefetcher,
        bindings=host.galaxy_bindings(),
    )


def test_concrete_owners_share_contexts_and_serialize_repeated_decode_and_mode_switch():
    events, mesh, host, _, prefetcher, config = create_composition()
    resources = create_galaxy_owner(mesh, host, prefetcher, config)

    assert resources.prefetcher is prefetcher
    assert resources.context("decode").worker_sub_device_id == prefetcher.context("decode").worker_sub_device_id

    resources.activate("decode")
    resources.synchronize("decode")
    resources.activate("decode")
    resources.activate("prefill")

    lifecycle = [event[0] for event in events if event[0] in {"start-prefetch", "stop-prefetch", "synchronize"}]
    assert lifecycle == [
        "start-prefetch",
        "synchronize",
        "stop-prefetch",
        "start-prefetch",
        "synchronize",
        "stop-prefetch",
    ]
    assert [event for event in events if event[0] == "synchronize"] == [
        ("synchronize", ("decode-worker",)),
        ("synchronize", ("decode-worker",)),
    ]
    assert resources.active_mode == prefetcher.active_mode == "prefill"

    resources.cleanup()
    prefetcher.cleanup()


def test_galaxy_cleanup_releases_only_its_resources_before_prefetcher_cleanup():
    events, mesh, host, weight, prefetcher, config = create_composition()
    resources = create_galaxy_owner(mesh, host, prefetcher, config)
    resources.activate("decode")
    running_prefetch = prefetcher.prefetch_result

    resources.cleanup()

    assert prefetcher.active_mode == "decode"
    assert prefetcher.prefetch_result == running_prefetch
    assert not any(event[0] == "remove-manager" for event in events)
    assert len([resource for resource in host.deallocated if resource[0] == "ccl-tensor"]) == 2
    assert running_prefetch not in host.deallocated

    prefetcher.cleanup()
    prefetcher.cleanup()

    assert running_prefetch in host.deallocated
    assert weight not in host.deallocated
    assert [event for event in events if event[0] == "remove-manager"] == [
        ("remove-manager", "manager-1"),
        ("remove-manager", "manager-0"),
    ]


def test_concrete_prefetcher_policy_mismatch_is_rejected_before_ccl_allocation():
    _, mesh, host, _, prefetcher, config = create_composition()
    incompatible_decode = replace(config.decode, sub_devices=("other-subdevice",))
    incompatible = replace(config, decode=incompatible_decode)

    with pytest.raises(ValueError, match="do not match the Prefetcher2D subdevice policy"):
        create_galaxy_owner(mesh, host, prefetcher, incompatible)

    assert not any(event[0] == "allocate-ccl" for event in host.events)
    prefetcher.cleanup()


def test_concrete_prefetcher_activation_failure_is_not_published_by_galaxy_owner():
    _, mesh, host, _, prefetcher, config = create_composition(fail_prefetch_start=True)
    resources = create_galaxy_owner(mesh, host, prefetcher, config)

    with pytest.raises(RuntimeError, match="prefetch start failure"):
        resources.activate("decode")

    assert resources.active_mode is None
    assert resources._ccl.active_mode is None
    assert prefetcher.active_mode is None
    assert prefetcher.prefetch_result is None

    resources.cleanup()
    prefetcher.cleanup()
