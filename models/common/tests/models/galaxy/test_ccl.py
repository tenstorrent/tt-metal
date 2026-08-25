# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

import ttnn
from models.common.models.galaxy.ccl import (
    GalaxyCCL,
    GalaxyCCLCollaborator,
    GalaxyCCLConfig,
    GalaxyCollectiveResources,
    GalaxyModeResources,
    GalaxyResourceKey,
)


class FakeMesh:
    def __init__(self, shape=(8, 4), count=32, arch=ttnn.device.Arch.WORMHOLE_B0):
        self.shape = shape
        self._count = count
        self._arch = arch

    def get_num_devices(self):
        return self._count

    def arch(self):
        return self._arch


class FakeFactory:
    def __init__(self, fail_key=None):
        self.fail_key = fail_key
        self.created = []
        self.released = []

    def create(self, mesh, mode, key):
        self.created.append((mesh, mode, key))
        if key == self.fail_key:
            raise RuntimeError("factory failure")
        return collective(f"owned-{mode}-{len(self.created)}", key=key)

    def release(self, resource):
        self.released.append(resource)


def resource_key(operation, *, axis=1, geometry="hidden", sequence_key=None):
    return GalaxyResourceKey(operation, axis, geometry, sequence_key)


def collective(prefix, *, key=None):
    key = key or resource_key("all_gather")
    return GalaxyCollectiveResources(
        key=key,
        topology=ttnn.Topology.Linear,
        semaphore_handles=(f"{prefix}-s0", f"{prefix}-s1"),
        barrier_semaphore_handles=(f"{prefix}-b0", f"{prefix}-b1"),
        persistent_output_buffers=(f"{prefix}-persistent",),
        intermediate_output_buffers=(f"{prefix}-intermediate",),
    )


def mode_resources(mesh, mode, *, owned_keys=()):
    borrowed = (
        collective(f"{mode}-rs", key=resource_key("reduce_scatter", geometry="ffn")),
        collective(f"{mode}-ag", key=resource_key("all_gather", geometry="hidden")),
        collective(f"{mode}-attention-row", key=resource_key("all_gather", axis=0, geometry="heads")),
        collective(f"{mode}-ar", key=resource_key("all_reduce", geometry="hidden")),
        collective(f"{mode}-agc", key=resource_key("all_gather_concat", axis=0, geometry="logits")),
    )
    return GalaxyModeResources(
        mode=mode,
        mesh_device=mesh,
        worker_sub_device_id=f"{mode}-worker",
        stall_group=(f"{mode}-worker",),
        borrowed_resources=borrowed,
        owned_resource_keys=owned_keys,
    )


def config(mesh=None, *, factory=None, prefill_owned=(), decode_owned=()):
    mesh = FakeMesh() if mesh is None else mesh
    return GalaxyCCLConfig(
        mesh_device=mesh,
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=mode_resources(mesh, "prefill", owned_keys=prefill_owned),
        decode=mode_resources(mesh, "decode", owned_keys=decode_owned),
        resource_factory=factory,
    )


def test_config_is_frozen_and_normalizes_resource_sequences():
    cfg = config()
    with pytest.raises(FrozenInstanceError):
        cfg.mesh_shape = (4, 8)
    with pytest.raises(FrozenInstanceError):
        cfg.decode.mode = "prefill"

    resource = GalaxyCollectiveResources(
        key=resource_key("all_gather"),
        topology=ttnn.Topology.Linear,
        semaphore_handles=["s0", "s1"],
        barrier_semaphore_handles=["b0", "b1"],
        persistent_output_buffers=["output"],
    )
    assert resource.semaphore_handles == ("s0", "s1")
    assert resource.barrier_semaphore_handles == ("b0", "b1")
    assert resource.persistent_output_buffers == ("output",)


@pytest.mark.parametrize(
    "operation,axis",
    [
        ("reduce_scatter", 2),
        ("all_reduce", 2),
        ("all_gather_concat", 1),
        ("all_gather", 2),
    ],
)
def test_resource_key_validates_canonical_axes_but_allows_attention_axes(operation, axis):
    with pytest.raises(ValueError, match="canonical cluster axis"):
        resource_key(operation, axis=axis)

    assert resource_key("all_gather", axis=0, geometry="attention").cluster_axis == 0
    assert resource_key("all_gather", axis=1, geometry="attention").cluster_axis == 1
    assert resource_key("reduce_scatter", axis=0, geometry="mlp-output").cluster_axis == 0
    assert resource_key("all_reduce", axis=0, geometry="mlp-output").cluster_axis == 0
    assert resource_key("all_reduce_create_qkv_heads", axis=1, geometry="qkv").cluster_axis == 1


@pytest.mark.parametrize(
    "mesh,match",
    [
        (FakeMesh(shape=(4, 8)), "mesh device shape"),
        (FakeMesh(count=31), "exactly 32 devices"),
        (FakeMesh(arch=ttnn.device.Arch.BLACKHOLE), "architecture does not match"),
    ],
)
def test_config_fails_closed_for_non_wh_8x4_mesh(mesh, match):
    with pytest.raises(ValueError, match=match):
        config(mesh)


def test_config_rejects_duplicate_keys_and_missing_factory():
    mesh = FakeMesh()
    duplicate = collective("duplicate", key=resource_key("all_gather", geometry="hidden"))
    resources = mode_resources(mesh, "decode")
    with pytest.raises(ValueError, match="duplicate"):
        GalaxyModeResources(
            mode="decode",
            mesh_device=mesh,
            worker_sub_device_id="worker",
            stall_group=("worker",),
            borrowed_resources=resources.borrowed_resources + (duplicate,),
        )

    with pytest.raises(ValueError, match="resource factory"):
        config(decode_owned=(resource_key("all_gather", geometry="owned"),))


def test_context_keys_by_operation_axis_geometry_and_sequence():
    mesh = FakeMesh()
    seq_128 = resource_key("all_gather", geometry="prefill-attention", sequence_key=128)
    seq_4096 = resource_key("all_gather", geometry="prefill-attention", sequence_key=4096)
    prefill = mode_resources(mesh, "prefill")
    prefill = GalaxyModeResources(
        mode="prefill",
        mesh_device=mesh,
        worker_sub_device_id="prefill-worker",
        stall_group=("prefill-worker",),
        borrowed_resources=prefill.borrowed_resources
        + (collective("seq128", key=seq_128), collective("seq4096", key=seq_4096)),
    )
    cfg = GalaxyCCLConfig(
        mesh_device=mesh,
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=prefill,
        decode=mode_resources(mesh, "decode"),
    )
    context = GalaxyCCL(cfg).context("prefill")

    assert context.resources("all_gather", 0, "heads").cluster_axis == 0
    assert context.resources("all_gather", 1, "prefill-attention", 128).key == seq_128
    with pytest.raises(ValueError, match="ambiguous"):
        context.resources("all_gather", 1)
    with pytest.raises(KeyError, match="no all_gather"):
        context.resources("all_gather", 1, "prefill-attention", 256)


def test_context_cycles_independently_per_full_key_and_legacy_helpers_remain_usable():
    ccl = GalaxyCCL(config())
    assert isinstance(ccl, GalaxyCCLCollaborator)
    decode = ccl.context("decode")

    assert decode.next_semaphore_handles("all_gather", 1, "hidden") == "decode-ag-s0"
    assert decode.next_semaphore_handles("all_gather", 1, "hidden") == "decode-ag-s1"
    decode.reset_cycles()
    assert decode.next_semaphore_window("all_gather", 1, "hidden") == ["decode-ag-s0", "decode-ag-s1"]
    assert decode.next_semaphore_window("all_gather", 1, "hidden") == ["decode-ag-s1", "decode-ag-s0"]
    assert decode.next_semaphore_handles("all_gather", 0, "heads") == "decode-attention-row-s0"
    assert decode.next_barrier_semaphore_handle("all_gather", 1, "hidden") == "decode-ag-b0"
    decode.reset_cycles()
    assert decode.next_semaphore_handles("all_gather", 1, "hidden") == "decode-ag-s0"

    with pytest.raises(RuntimeError, match="activated"):
        ccl.get_and_cycle_ag_semaphore_handles(1)
    ccl.activate("decode")
    assert ccl.get_and_cycle_ag_semaphore_handles(1) == "decode-ag-s1"
    assert ccl.get_and_cycle_rs_semaphore_handles(1) == "decode-rs-s0"


def test_factory_resources_are_owned_borrowed_resources_are_not_and_cleanup_is_idempotent():
    factory = FakeFactory()
    owned_key = resource_key("all_gather", geometry="owned-geometry", sequence_key=2048)
    ccl = GalaxyCCL(config(factory=factory, decode_owned=(owned_key,)))

    assert ccl.owned_resource_keys == (owned_key,)
    assert owned_key not in ccl.borrowed_resource_keys
    owned_resource = ccl.context("decode").resources("all_gather", 1, "owned-geometry", 2048)
    ccl.cleanup()
    ccl.cleanup()

    assert factory.released == [owned_resource]
    with pytest.raises(RuntimeError, match="cleaned up"):
        ccl.context("decode")


def test_factory_failure_releases_already_created_resources_in_reverse_order():
    first = resource_key("all_gather", geometry="first-owned")
    failing = resource_key("all_gather", geometry="failing-owned")
    factory = FakeFactory(fail_key=failing)

    with pytest.raises(RuntimeError, match="factory failure"):
        GalaxyCCL(config(factory=factory, decode_owned=(first, failing)))

    assert [resource.key for resource in factory.released] == [first]


def test_context_rejects_unknown_mode_and_collective():
    ccl = GalaxyCCL(config())
    with pytest.raises(ValueError, match="mode"):
        ccl.context("training")
    with pytest.raises(ValueError, match="collective"):
        ccl.context("decode").resources("broadcast")
