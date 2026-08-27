# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ttnn
from models.common.modules.prefetcher.prefetcher_2d import Prefetcher2D, Prefetcher2DConfig, Prefetcher2DModeConfig


class FakeMesh:
    def __init__(self, shape=(8, 4), count=32, arch=ttnn.device.Arch.WORMHOLE_B0):
        self.shape = shape
        self._count = count
        self._arch = arch
        self.events = []
        self.fail_next_stall = False
        self.fail_remove = None

    def get_num_devices(self):
        return self._count

    def arch(self):
        return self._arch

    def create_sub_device_manager(self, subdevices, local_l1_size):
        manager = f"manager-{len([event for event in self.events if event[0] == 'create'])}"
        self.events.append(("create", tuple(subdevices), local_l1_size, manager))
        return manager

    def load_sub_device_manager(self, manager):
        self.events.append(("load", manager))

    def set_sub_device_stall_group(self, stall_group):
        self.events.append(("stall", tuple(stall_group)))
        if self.fail_next_stall:
            self.fail_next_stall = False
            raise RuntimeError("stall failure")

    def reset_sub_device_stall_group(self):
        self.events.append(("reset_stall",))

    def clear_loaded_sub_device_manager(self):
        self.events.append(("clear_manager",))

    def remove_sub_device_manager(self, manager):
        self.events.append(("remove", manager))
        if manager == self.fail_remove:
            raise RuntimeError(f"remove failure: {manager}")


class FakeTensor:
    def __init__(self, mesh, address, size=128):
        self._mesh = mesh
        self._address = address
        self._size = size

    def device(self):
        return self._mesh

    def buffer_address(self):
        return self._address

    def buffer_size(self):
        return self._size


class ResourceHarness:
    def __init__(self):
        self.created_cbs = []
        self.metadata = []
        self.deallocated = []
        self.prefetch_events = []
        self.fail_metadata_once = False
        self.fail_start_number = None
        self.fail_stop_once = False
        self.start_count = 0

    def create_global_cb(self, mesh, mapping, size):
        value = ("global-cb", len(self.created_cbs), tuple(mapping), size)
        self.created_cbs.append(value)
        return value

    def create_metadata(self, addresses, **kwargs):
        if self.fail_metadata_once:
            self.fail_metadata_once = False
            raise RuntimeError("metadata failure")
        value = {"addresses": addresses.clone(), **kwargs}
        self.metadata.append(value)
        return value

    def deallocate(self, resource):
        self.deallocated.append(resource)

    def start(self, context):
        self.start_count += 1
        self.prefetch_events.append(("start", context.mode, context.weights, context.global_cb))
        if self.start_count == self.fail_start_number:
            raise RuntimeError("prefetch start failure")
        return ("prefetch-result", self.start_count)

    def stop(self, mesh, result):
        self.prefetch_events.append(("stop", mesh, result))
        if self.fail_stop_once:
            self.fail_stop_once = False
            raise RuntimeError("prefetch stop failure")
        self.deallocate(result)
        return ("sync-result", result[-1])

    def kwargs(self):
        return {
            "create_global_cb": self.create_global_cb,
            "create_address_metadata": self.create_metadata,
            "deallocate": self.deallocate,
            "dram_prefetch_start": self.start,
            "dram_prefetch_stop": self.stop,
        }


@pytest.fixture
def resources():
    return ResourceHarness()


def make_config(mesh=None, expected_weight_count=2, global_cb_size=4096, defer_global_cb=False):
    mesh = FakeMesh() if mesh is None else mesh
    return Prefetcher2DConfig(
        mesh_device=mesh,
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=Prefetcher2DModeConfig(
            mode="prefill",
            sub_devices=("all-workers",),
            worker_sub_device_id="prefill-worker",
            stall_group=("prefill-worker",),
        ),
        decode=Prefetcher2DModeConfig(
            mode="decode",
            sub_devices=("senders", "workers"),
            worker_sub_device_id="decode-worker",
            stall_group=("prefetch-worker", "decode-worker"),
        ),
        sender_receiver_mapping=(("sender-0", "receiver-0"), ("sender-1", "receiver-1")),
        global_cb_size=global_cb_size,
        expected_weight_count=expected_weight_count,
        address_repeat_count=2,
        address_memory_config="address-memcfg",
        address_mesh_mapper="address-mapper",
        defer_global_cb=defer_global_cb,
    )


def initialized_owner(resources, expected_weight_count=2, **kwargs):
    owner = Prefetcher2D(make_config(expected_weight_count=expected_weight_count, **kwargs), **resources.kwargs())
    owner.initialize()
    return owner


def seal_one(owner, *, size=128):
    weight = FakeTensor(owner.config.mesh_device, 101, size)
    owner.register_weight("weight", weight)
    owner.seal()
    return weight


def test_config_is_frozen_and_fails_closed_for_wh_8x4():
    cfg = make_config()
    with pytest.raises(FrozenInstanceError):
        cfg.mesh_shape = (4, 8)

    with pytest.raises(ValueError, match="mesh device shape"):
        make_config(FakeMesh(shape=(4, 8)))
    with pytest.raises(ValueError, match="exactly 32 devices"):
        make_config(FakeMesh(count=16))
    with pytest.raises(ValueError, match="architecture does not match"):
        make_config(FakeMesh(arch=ttnn.device.Arch.BLACKHOLE))
    assert isinstance(cfg.prefill.sub_devices, tuple)
    assert isinstance(cfg.decode.stall_group, tuple)
    assert isinstance(cfg.sender_receiver_mapping, tuple)


def test_address_repeat_count_tracks_active_readers_not_dummy_global_cb_mappings():
    cfg = replace(
        make_config(),
        sender_receiver_mapping=(
            ("active-0", "receiver-0"),
            ("active-1", "receiver-1"),
            ("dummy", "remaining-workers"),
        ),
    )
    assert cfg.address_repeat_count == 2

    with pytest.raises(ValueError, match="cannot exceed"):
        replace(cfg, address_repeat_count=4)


def test_initialize_creates_only_both_managers_and_is_idempotent(resources):
    owner = initialized_owner(resources)
    owner.initialize()

    creates = [event for event in owner.config.mesh_device.events if event[0] == "create"]
    assert len(creates) == 2
    assert creates[0][1] == ("all-workers",)
    assert creates[1][1] == ("senders", "workers")
    assert resources.created_cbs == []


def test_registration_is_ordered_borrowed_and_compatibility_validated(resources):
    calls = []

    def validate(name, tensor, existing):
        calls.append((name, tensor, existing))
        if name == "bad":
            raise ValueError("incompatible prefetch weight")

    owner = Prefetcher2D(make_config(), validate_weight_compatibility=validate, **resources.kwargs())
    owner.initialize()
    mesh = owner.config.mesh_device
    first = FakeTensor(mesh, 101)
    second = FakeTensor(mesh, 202)
    owner.register_weight("layer.0.w1", first)
    with pytest.raises(ValueError, match="incompatible"):
        owner.register_weight("bad", second)
    owner.register_weight("layer.0.w2", second)
    prefill, decode = owner.seal()

    assert calls[-1] == ("layer.0.w2", second, (first,))
    assert owner.borrowed_weights == (first, second)
    assert prefill.weights == ()
    assert decode.weights == (first, second)
    assert dict(decode.weight_addresses) == {"layer.0.w1": 101, "layer.0.w2": 202}
    assert resources.metadata[0]["addresses"].tolist() == [[101, 202], [101, 202]]
    with pytest.raises(RuntimeError, match="sealed"):
        owner.register_weight("layer.1.w1", FakeTensor(mesh, 303))
    assert owner.seal() == (prefill, decode)


def test_seal_loads_decode_manager_before_global_cb_allocation(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101))

    owner.seal()

    assert owner.config.mesh_device.events[-2:] == [
        ("load", "manager-1"),
        ("stall", (ttnn.SubDeviceId(0), ttnn.SubDeviceId(1))),
    ]


def test_default_registration_rejects_duplicate_buffer_and_other_mesh(resources):
    owner = initialized_owner(resources)
    owner.register_weight("first", FakeTensor(owner.config.mesh_device, 101))
    with pytest.raises(ValueError, match="distinct device buffers"):
        owner.register_weight("alias", FakeTensor(owner.config.mesh_device, 101))
    with pytest.raises(TypeError, match="materialized"):
        owner.register_weight("lazy", object())
    with pytest.raises(ValueError, match="different mesh"):
        owner.register_weight("other", FakeTensor(FakeMesh(), 202))


def test_deferred_global_cb_is_allocated_on_first_decode_activation(resources):
    """`defer_global_cb` moves the allocation from `seal()` to `activate("decode")`.

    The global CB is ~774 kB of L1 per sender/receiver core and nothing frees it,
    so a prefill program needing static circular buffers on those cores cannot be
    placed while it is resident. Prefill never reads it, so it must not exist yet.

    Three properties matter, and all three are checked here:

    1. sealing allocates nothing, and prefill still gets `global_cb=None`;
    2. the first `activate("decode")` allocates exactly one buffer, *before* the
       prefetch program is started - the prefetch program is what reads it;
    3. the decode context object that modules captured at build time sees the
       buffer, not the `None` it was sealed with. Module configs hold the context
       *object* (`MLP2DConfig.decode_prefetch_context`), so binding a replacement
       context would silently leave every built module with no global CB.
    """

    owner = initialized_owner(resources, expected_weight_count=1, defer_global_cb=True)
    weight = FakeTensor(owner.config.mesh_device, 101, 128)
    owner.register_weight("weight", weight)
    prefill, decode = owner.seal()

    assert resources.created_cbs == []
    assert prefill.global_cb is None
    assert decode.global_cb is None

    # Prefill must be activatable with no buffer in existence at all.
    owner.activate("prefill")
    assert resources.created_cbs == []

    owner.activate("decode")
    assert len(resources.created_cbs) == 1
    assert decode.global_cb == resources.created_cbs[0]
    # The prefetch program was handed the buffer, not None.
    starts = [event for event in resources.prefetch_events if event[0] == "start"]
    assert starts[-1][3] == resources.created_cbs[0]

    # Re-activating decode must not allocate a second buffer.
    owner.activate("decode")
    assert len(resources.created_cbs) == 1
    owner.cleanup()


def test_seal_derives_cb_size_and_rejects_undersized_configuration(resources):
    derived = Prefetcher2D(make_config(expected_weight_count=2, global_cb_size=None), **resources.kwargs())
    derived.initialize()
    derived.register_weight("small", FakeTensor(derived.config.mesh_device, 101, size=128))
    derived.register_weight("large", FakeTensor(derived.config.mesh_device, 202, size=300))
    _, decode = derived.seal()
    assert derived.resolved_global_cb_size == 600
    assert decode.global_cb[-1] == 600

    configured = initialized_owner(resources, expected_weight_count=1, global_cb_size=128)
    configured.register_weight("large", FakeTensor(configured.config.mesh_device, 303, size=100))
    _, decode = configured.seal()
    assert configured.resolved_global_cb_size == 128
    assert decode.global_cb[-1] == 128


def test_seal_is_transactional_and_retryable_after_metadata_failure(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    owner.register_weight("weight", FakeTensor(owner.config.mesh_device, 101))
    resources.fail_metadata_once = True

    with pytest.raises(RuntimeError, match="metadata failure"):
        owner.seal()
    assert not owner.sealed
    assert resources.deallocated == []

    _, decode = owner.seal()
    assert owner.sealed
    assert decode.global_cb == resources.created_cbs[1]


def test_activation_starts_stops_and_releases_repeat_results(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    weight = seal_one(owner)

    decode = owner.activate("decode")
    first_result = owner.prefetch_result
    owner.activate("decode")
    second_result = owner.prefetch_result
    owner.activate("prefill")

    assert decode.worker_sub_device_id == "decode-worker"
    assert first_result != second_result
    assert owner.prefetch_result is None
    assert first_result in resources.deallocated
    assert second_result in resources.deallocated
    assert first_result not in owner.owned_resources
    assert second_result not in owner.owned_resources
    assert weight not in owner.owned_resources
    assert [event[0] for event in resources.prefetch_events] == ["start", "stop", "start", "stop"]
    stall_events = [event[1] for event in owner.config.mesh_device.events if event[0] == "stall"]
    assert stall_events[0] == (ttnn.SubDeviceId(0), ttnn.SubDeviceId(1))
    assert stall_events[1] == (ttnn.SubDeviceId(0), ttnn.SubDeviceId(1))
    assert stall_events[2] == owner.config.decode.stall_group


def test_default_ttnn_session_includes_addresses_and_deallocates_stop_sentinel(monkeypatch):
    start = MagicMock(return_value="sentinel")
    deallocate = MagicMock()
    monkeypatch.setattr(ttnn, "dram_prefetcher", start)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    owner = object.__new__(Prefetcher2D)
    owner.config = SimpleNamespace(prefetch_num_layers=1)
    context = SimpleNamespace(weights=("w1", "w2"), weight_address_metadata="addresses", global_cb="cb")

    assert owner._default_dram_prefetch_start(context) == "sentinel"
    start.assert_called_once_with(["w1", "w2", "addresses"], num_layers=1, global_cb="cb")
    assert Prefetcher2D._default_dram_prefetch_stop("mesh", "sentinel") is None
    deallocate.assert_called_once_with("sentinel")


def test_borrow_context_requires_exact_sealed_subdevice_policy(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)

    context = owner.borrow_context(
        "decode",
        sub_devices=("senders", "workers"),
        worker_sub_device_id="decode-worker",
        stall_group=("prefetch-worker", "decode-worker"),
        local_l1_size=0,
    )

    assert owner.mesh_device is owner.config.mesh_device
    assert context is owner.context("decode")
    with pytest.raises(ValueError, match="do not match"):
        owner.borrow_context(
            "decode",
            sub_devices=("workers",),
            worker_sub_device_id="decode-worker",
            stall_group=("prefetch-worker", "decode-worker"),
            local_l1_size=0,
        )


def test_failed_activation_rolls_back_mode_and_running_prefetch(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.activate("decode")
    previous_result = owner.prefetch_result
    resources.fail_start_number = 2

    with pytest.raises(RuntimeError, match="prefetch start failure"):
        owner.activate("decode")

    assert owner.active_mode == "decode"
    assert owner.prefetch_result is not None
    assert owner.prefetch_result != previous_result
    assert resources.start_count == 3


def test_failed_stop_preserves_active_session_ownership(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.activate("decode")
    active_result = owner.prefetch_result
    resources.fail_stop_once = True

    with pytest.raises(RuntimeError, match="prefetch stop failure"):
        owner.activate("prefill")

    assert owner.active_mode == "decode"
    assert owner.prefetch_result == active_result
    owner.cleanup()
    assert active_result in resources.deallocated


def test_failed_stall_transition_restores_previous_mode_without_publishing_target(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.activate("prefill")
    owner.config.mesh_device.fail_next_stall = True

    with pytest.raises(RuntimeError, match="stall failure"):
        owner.activate("decode")

    assert owner.active_mode == "prefill"
    assert owner.prefetch_result is None
    assert owner.config.mesh_device.events[-2:] == [
        ("load", "manager-0"),
        ("stall", ("prefill-worker",)),
    ]


def test_cleanup_is_idempotent_releases_owned_results_and_never_weights(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    weight = seal_one(owner)
    _, decode = owner.seal()
    owner.activate("decode")
    prefetch_result = owner.prefetch_result
    metadata = decode.weight_address_metadata

    owner.cleanup()
    owner.cleanup()

    events = owner.config.mesh_device.events
    assert events.count(("reset_stall",)) == 1
    assert events.count(("clear_manager",)) == 1
    assert events.count(("remove", "manager-1")) == 1
    assert events.count(("remove", "manager-0")) == 1
    assert prefetch_result in resources.deallocated
    assert metadata in resources.deallocated
    assert weight not in resources.deallocated
    with pytest.raises(RuntimeError, match="cleaned up"):
        owner.activate("decode")


def test_cleanup_continues_after_failure_and_remains_idempotent(resources):
    owner = initialized_owner(resources, expected_weight_count=1)
    seal_one(owner)
    owner.config.mesh_device.fail_remove = "manager-1"

    with pytest.raises(RuntimeError, match="remove failure"):
        owner.cleanup()
    owner.cleanup()
    assert ("remove", "manager-0") in owner.config.mesh_device.events


def test_context_manager_cleans_up_on_failure(resources):
    owner = Prefetcher2D(make_config(expected_weight_count=1), **resources.kwargs())
    with pytest.raises(RuntimeError, match="body failure"):
        with owner:
            raise RuntimeError("body failure")
    assert ("remove", "manager-1") in owner.config.mesh_device.events
    assert ("remove", "manager-0") in owner.config.mesh_device.events
