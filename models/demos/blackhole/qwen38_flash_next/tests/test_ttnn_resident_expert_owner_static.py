# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device lifetime contracts for capacity-bounded resident BF4 experts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import models.demos.blackhole.qwen38_flash_next.ttnn.bf4 as bf4_module
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4CleanupError,
    BF4TensorCleanupOutcome,
    Qwen38BF4ResidentSet,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    Qwen38ResidentBuildFailure,
    Qwen38TTNNBuilder,
    Qwen38TTNNMTPComponents,
    Qwen38TTNNTargetComponents,
)


class _Contract:
    @staticmethod
    def validate_mesh(_mesh) -> None:
        return None


class _MeshDevice:
    shape = (1, 4)

    @staticmethod
    def get_device_ids() -> tuple[int, int, int, int]:
        return (0, 1, 2, 3)

    @staticmethod
    def get_device_id(coordinate) -> int:
        row, column = tuple(coordinate)
        assert row == 0
        return column


class _LocalTensor:
    def __init__(self, mesh: _MeshDevice, physical_id: int, address: int) -> None:
        self._device = mesh
        self._coordinate = (0, physical_id)
        self._address = address

    def device(self) -> _MeshDevice:
        return self._device

    def device_coords(self) -> tuple[tuple[int, int]]:
        return (self._coordinate,)

    def buffer_address(self) -> int:
        return self._address


class _Topology:
    @staticmethod
    def distribution_shape() -> tuple[int, int]:
        return (1, 4)

    @staticmethod
    def mesh_coords() -> tuple[tuple[int, int], ...]:
        return tuple((0, column) for column in range(4))


class _Tensor:
    _next_graph_id = 1
    _next_backing_base = 1
    _mesh = _MeshDevice()

    def __init__(
        self,
        key: tuple[str, int],
        slot: int,
        *,
        backing_base: int | None = None,
        graph_id: int | None = None,
    ) -> None:
        self.key = key
        self.slot = slot
        self.graph_id = type(self)._next_graph_id if graph_id is None else graph_id
        if graph_id is None:
            type(self)._next_graph_id += 1
        self.backing_base = type(self)._next_backing_base if backing_base is None else backing_base
        if backing_base is None:
            type(self)._next_backing_base += 1
        self.locals = tuple(
            _LocalTensor(self._mesh, physical_id, self.backing_base * 0x10000 + physical_id * 0x1000)
            for physical_id in range(4)
        )

    def tensor_id(self) -> int:
        return self.graph_id

    def device(self) -> _MeshDevice:
        return self._mesh

    @staticmethod
    def tensor_topology() -> _Topology:
        return _Topology()

    def __repr__(self) -> str:
        return f"_Tensor(key={self.key!r}, slot={self.slot})"


@pytest.fixture(autouse=True)
def _exact_backing_api(monkeypatch):
    monkeypatch.setattr(bf4_module.ttnn, "get_device_tensors", lambda tensor: tensor.locals)


class _Cache:
    mesh_contract = _Contract()

    def __init__(self, *, fail_key: tuple[str, int] | None = None, malformed=None) -> None:
        self.fail_key = fail_key
        self.malformed = malformed
        self.loads: list[tuple[str, int]] = []

    @staticmethod
    def _validate_layer_request(namespace: str, layer_index: int) -> None:
        if namespace == "backbone" and 0 <= layer_index < 48:
            return
        if namespace == "mtp" and layer_index == 0:
            return
        raise ValueError(f"invalid synthetic BF4 key {(namespace, layer_index)}")

    def load_layer(self, _mesh, *, layer_index: int, namespace: str):
        key = (namespace, layer_index)
        self.loads.append(key)
        if key == self.fail_key:
            raise RuntimeError(f"injected load failure at {key}")
        if self.malformed is not None:
            return self.malformed(key)
        return (_Tensor(key, 0), _Tensor(key, 1))


class _ReadOnlyAdapterWithoutPrivateValidation:
    """The live diagnostic cache exposes load/verify, not production-cache internals."""

    mesh_contract = _Contract()

    def __init__(self) -> None:
        self.loads: list[tuple[str, int]] = []

    def load_layer(self, _mesh, *, layer_index: int, namespace: str):
        key = (namespace, layer_index)
        self.loads.append(key)
        return (_Tensor(key, 0), _Tensor(key, 1))


def _components(owner: Qwen38BF4ResidentSet):
    identity = object()
    target = Qwen38TTNNTargetComponents(
        identity=identity,
        bf4_cache=owner.cache,
        io_cache=object(),
        expert_streamer=owner,
        model_io=object(),
        layers=(),
        final_mixer=object(),
    )
    mtp = Qwen38TTNNMTPComponents(
        identity=identity,
        input_mixer=object(),
        decoder_layer=SimpleNamespace(expert_streamer=owner),
        final_mixer=object(),
    )
    return target, mtp


def _bare_builder(owner: Qwen38BF4ResidentSet) -> Qwen38TTNNBuilder:
    builder = object.__new__(Qwen38TTNNBuilder)
    builder.expert_streamer = owner
    builder._target_components_published = False
    builder._built_target = False
    builder._built_mtp = False
    builder._building_target = False
    builder._building_mtp = False
    builder._resident_build_failed = False
    builder._resident_build_failure = None
    return builder


def test_backbone_and_mtp_load_once_then_every_decode_borrow_has_zero_io(monkeypatch) -> None:
    cache = _Cache()
    owner = Qwen38BF4ResidentSet(cache, object())
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    owner.preload_backbone()
    owner.preload_backbone()
    owner.preload_mtp()
    owner.preload_all()
    expected = tuple(("backbone", layer) for layer in range(48)) + (("mtp", 0),)
    assert tuple(cache.loads) == expected
    assert owner.resident_layers == expected
    assert owner.cache_load_attempt_count == owner.cache_load_success_count == 49
    assert owner.registered_tensor_handle_count == owner.live_tensor_handle_count == 98
    assert tuple(
        (event.namespace, event.layer_index, event.outcome, event.error_type) for event in owner.preload_events
    ) == tuple(
        item
        for namespace, layer_index in expected
        for item in (
            (namespace, layer_index, "attempt", None),
            (namespace, layer_index, "success", None),
        )
    )

    for namespace, layer_index in expected:
        before = tuple(cache.loads)
        before_events = owner.preload_events
        with owner.layer(layer_index, namespace=namespace) as pair:
            assert tuple(tensor.key for tensor in pair) == ((namespace, layer_index),) * 2
        assert tuple(cache.loads) == before
        assert owner.preload_events == before_events
        assert released == []


def test_resident_owner_accepts_read_only_cache_adapter_without_private_validator(monkeypatch) -> None:
    cache = _ReadOnlyAdapterWithoutPrivateValidation()
    owner = Qwen38BF4ResidentSet(cache, object())
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    owner.preload_all()
    expected = tuple(("backbone", layer) for layer in range(48)) + (("mtp", 0),)
    with owner.layer(0, namespace="backbone") as pair:
        assert tuple(tensor.key for tensor in pair) == (("backbone", 0),) * 2
    with owner.layer(0, namespace="mtp") as pair:
        assert tuple(tensor.key for tensor in pair) == (("mtp", 0),) * 2

    assert tuple(cache.loads) == expected
    assert owner.resident_layers == expected
    assert owner.cache_load_attempt_count == owner.cache_load_success_count == 49
    owner.close()
    assert len(released) == 98


@pytest.mark.parametrize(
    ("invalid_key", "error_type"),
    (
        (("unsupported", 0), ValueError),
        (("backbone", 48), ValueError),
        (("mtp", 1), ValueError),
        (("backbone", True), TypeError),
    ),
)
def test_resident_owner_validates_every_key_before_cache_io(expect_error, invalid_key, error_type) -> None:
    cache = _ReadOnlyAdapterWithoutPrivateValidation()
    owner = Qwen38BF4ResidentSet(cache, object())

    with expect_error(error_type):
        owner._preload((("backbone", 0), invalid_key))

    assert cache.loads == []
    assert owner.preload_events == ()
    assert owner.owned_layers == ()


def test_shared_target_and_mtp_component_close_deallocates_each_tensor_once(monkeypatch) -> None:
    owner = Qwen38BF4ResidentSet(_Cache(), object())
    owner.preload_all()
    target, mtp = _components(owner)
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    target.close_resident_experts()
    mtp.close_resident_experts()

    assert owner.closed and not owner.poisoned
    assert len(released) == 98
    assert len({id(tensor) for tensor in released}) == 98
    assert owner.unreleased_tensor_slots == ()


@pytest.mark.parametrize(
    ("public_name", "private_name"),
    (
        ("build_target_components", "_build_target_components"),
        ("build_mtp_components", "_build_mtp_components"),
    ),
)
def test_builder_partial_target_or_mtp_failure_closes_resident_owner_once(
    expect_error, monkeypatch, public_name: str, private_name: str
) -> None:
    owner = Qwen38BF4ResidentSet(_Cache(), object())
    owner._preload((("backbone", 0),))
    builder = _bare_builder(owner)

    def fail_build():
        raise RuntimeError(f"injected {private_name} failure")

    setattr(builder, private_name, fail_build)
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    with expect_error(RuntimeError, match="injected"):
        getattr(builder, public_name)()
    assert owner.closed
    assert builder.resident_build_failure is not None
    assert builder.resident_build_failure[2:] == (True, False, "cleanup_enqueued_unfenced")
    assert len(released) == 2
    owner.close()
    assert len(released) == 2


def test_second_mtp_build_precondition_does_not_close_or_retire_published_owner(expect_error, monkeypatch) -> None:
    owner = Qwen38BF4ResidentSet(_Cache(), object())
    owner._preload((("backbone", 0), ("mtp", 0)))
    builder = _bare_builder(owner)
    builder._built_mtp = True
    released = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    with expect_error(RuntimeError, match="already constructed its MTP"):
        builder.build_mtp_components()

    assert not owner.closed and not owner.poisoned
    assert builder._resident_build_failed is False
    assert released == []


def test_mtp_first_then_target_failure_preserves_owner_and_retires_builder(expect_error, monkeypatch) -> None:
    owner = Qwen38BF4ResidentSet(_Cache(), object())
    owner._preload((("backbone", 0), ("mtp", 0)))
    builder = _bare_builder(owner)
    builder._built_mtp = True
    builder._build_target_components = lambda: (_ for _ in ()).throw(RuntimeError("injected target failure"))
    released = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        Qwen38ResidentBuildFailure, match="resident graph was published"
    ) as caught:  # allow-pytest.raises: inspect the captured exception object
        builder.build_target_components()

    assert caught.value.requires_process_termination is True
    assert not owner.closed and released == []
    assert builder.resident_build_failure is not None
    assert builder.resident_build_failure[2:] == (True, True, "not_attempted_published_graph")
    with expect_error(RuntimeError, match="builder was retired"):
        builder.build_target_components()
    assert released == []


def test_target_published_mid_mtp_failure_preserves_owner_and_rejects_retry(expect_error, monkeypatch) -> None:
    owner = Qwen38BF4ResidentSet(_Cache(), object())
    owner._preload((("backbone", 0),))
    builder = _bare_builder(owner)
    builder._built_target = True
    builder._build_mtp_components = lambda: (_ for _ in ()).throw(RuntimeError("injected mid-MTP failure"))
    released = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        Qwen38ResidentBuildFailure, match="mid-MTP failure"
    ) as caught:  # allow-pytest.raises: inspect the captured exception object
        builder.build_mtp_components()

    assert caught.value.unreleased_tensor_slots == owner.unreleased_tensor_slots
    assert not owner.closed and released == []
    with expect_error(RuntimeError, match="builder was retired"):
        builder.build_mtp_components()
    assert released == []


def test_malformed_duplicate_pair_is_registered_then_released_once(expect_error, monkeypatch) -> None:
    tensor = _Tensor(("backbone", 0), 0)
    owner = Qwen38BF4ResidentSet(_Cache(malformed=lambda _key: (tensor, tensor)), object())
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    with expect_error(RuntimeError, match="two new distinct"):
        owner._preload((("backbone", 0),))

    assert released == [tensor]
    assert owner.owned_layers == ()
    assert owner.unreleased_tensor_slots == ()
    assert not owner.poisoned


def test_distinct_graph_wrappers_with_one_physical_backing_are_released_once(expect_error, monkeypatch) -> None:
    first = _Tensor(("backbone", 0), 0, backing_base=700)
    alias = _Tensor(("backbone", 0), 1, backing_base=700)
    assert first.tensor_id() != alias.tensor_id()
    owner = Qwen38BF4ResidentSet(_Cache(malformed=lambda _key: (first, alias)), object())
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    with expect_error(RuntimeError, match="two new distinct"):
        owner._preload((("backbone", 0),))

    assert released == [first]
    assert owner.unreleased_tensor_slots == ()
    assert owner.live_tensor_handle_count == 0


def test_cache_cleanup_deduplicates_distinct_wrappers_by_physical_backing(monkeypatch) -> None:
    first = _Tensor(("backbone", 0), 0, backing_base=701)
    alias = _Tensor(("backbone", 0), 1, backing_base=701)
    assert first.tensor_id() != alias.tensor_id()
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    outcomes = bf4_module._deallocate_all((first, alias))

    assert released == [first]
    assert len(outcomes) == 2
    assert all(outcome.released for outcome in outcomes)


def test_equal_graph_ids_with_distinct_physical_backings_are_both_released(monkeypatch) -> None:
    first = _Tensor(("backbone", 0), 0, backing_base=702, graph_id=77)
    second = _Tensor(("backbone", 0), 1, backing_base=703, graph_id=77)
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    outcomes = bf4_module._deallocate_all((first, second))

    assert released == [first, second]
    assert all(outcome.released for outcome in outcomes)


def test_alias_wrapper_repeated_across_layer_keys_does_not_double_release(expect_error, monkeypatch) -> None:
    first_pair = (
        _Tensor(("backbone", 0), 0, backing_base=710),
        _Tensor(("backbone", 0), 1, backing_base=711),
    )
    second_pair = (
        _Tensor(("backbone", 1), 0, backing_base=710),
        _Tensor(("backbone", 1), 1, backing_base=712),
    )

    class AliasingCache(_Cache):
        def load_layer(self, _mesh, *, layer_index: int, namespace: str):
            self.loads.append((namespace, layer_index))
            return first_pair if layer_index == 0 else second_pair

    owner = Qwen38BF4ResidentSet(AliasingCache(), object())
    released: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", released.append)

    owner._preload((("backbone", 0),))
    with expect_error(RuntimeError, match="two new distinct"):
        owner._preload((("backbone", 1),))
    owner.close()

    assert [tensor.backing_base for tensor in released].count(710) == 1
    assert sorted(tensor.backing_base for tensor in released) == [710, 711, 712]


def test_malformed_return_cleanup_failure_retains_poisoned_handle_without_retry(expect_error, monkeypatch) -> None:
    tensor = _Tensor(("backbone", 0), 0)
    owner = Qwen38BF4ResidentSet(_Cache(malformed=lambda _key: [tensor]), object())
    calls: list[_Tensor] = []

    def fail_release(value: _Tensor) -> None:
        calls.append(value)
        raise RuntimeError("injected malformed cleanup failure")

    monkeypatch.setattr(bf4_module.ttnn, "deallocate", fail_release)
    with expect_error(BF4CleanupError, match="partial cleanup was incomplete"):
        owner._preload((("backbone", 0),))

    assert owner.poisoned and not owner.closed
    assert owner.unreleased_tensor_slots == (("backbone", 0, 0, True, "RuntimeError"),)
    with expect_error(BF4CleanupError, match="owner cleanup was incomplete"):
        owner.close()
    assert calls == [tensor]


def test_cache_partial_load_cleanup_failure_is_adopted_without_second_release(expect_error, monkeypatch) -> None:
    tensor = _Tensor(("backbone", 0), 0)
    release_error = RuntimeError("injected cache cleanup failure")

    class FailedCache(_Cache):
        def load_layer(self, _mesh, *, layer_index: int, namespace: str):
            key = (namespace, layer_index)
            self.loads.append(key)
            raise BF4CleanupError(
                "injected second BF4 load validation failure",
                primary_error=RuntimeError("injected second load validation failure"),
                cleanup_errors=(release_error,),
                tensor_cleanup_outcomes=(
                    BF4TensorCleanupOutcome(0, tensor, True, False, release_error),
                    BF4TensorCleanupOutcome(1, None, False, False, None),
                ),
            )

    owner = Qwen38BF4ResidentSet(FailedCache(), object())
    unexpected_releases = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", unexpected_releases.append)

    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        BF4CleanupError, match="partial cleanup was incomplete"
    ) as caught:  # allow-pytest.raises: inspect the captured exception object
        owner._preload((("backbone", 0),))

    assert unexpected_releases == []
    assert caught.value.unreleased_tensors == (tensor,)
    assert owner.poisoned and not owner.closed
    assert owner.unreleased_tensor_slots == (("backbone", 0, 0, True, "RuntimeError"),)
    assert tuple((event.outcome, event.error_type) for event in owner.preload_events) == (
        ("attempt", None),
        ("failure", "BF4CleanupError"),
    )
    with expect_error(BF4CleanupError):
        owner.close()
    assert unexpected_releases == []


@pytest.mark.parametrize(
    "outcomes",
    (
        lambda first, second, error: (
            BF4TensorCleanupOutcome(1, first, True, False, error),
            BF4TensorCleanupOutcome(0, second, True, False, error),
        ),
        lambda first, _second, error: (
            BF4TensorCleanupOutcome(0, first, True, False, error),
            BF4TensorCleanupOutcome(
                1,
                _Tensor(("backbone", 0), 1, backing_base=first.backing_base),
                True,
                False,
                error,
            ),
        ),
        lambda first, second, error: (
            BF4TensorCleanupOutcome(0, first, True, False, error),
            BF4TensorCleanupOutcome(0, second, True, False, error),
        ),
    ),
)
def test_malformed_or_aliasing_cache_cleanup_ledger_is_terminally_retained(expect_error, monkeypatch, outcomes) -> None:
    first = _Tensor(("backbone", 0), 0)
    second = _Tensor(("backbone", 0), 1)
    release_error = RuntimeError("injected uncertain cache release")

    class FailedCache(_Cache):
        def load_layer(self, _mesh, *, layer_index: int, namespace: str):
            key = (namespace, layer_index)
            self.loads.append(key)
            raise BF4CleanupError(
                "injected malformed cleanup ledger",
                primary_error=RuntimeError("load failed"),
                cleanup_errors=(release_error,),
                tensor_cleanup_outcomes=outcomes(first, second, release_error),
            )

    owner = Qwen38BF4ResidentSet(FailedCache(), object())
    unexpected_releases: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", unexpected_releases.append)

    with expect_error(BF4CleanupError):
        owner._preload((("backbone", 0),))

    assert unexpected_releases == []
    assert owner.poisoned and not owner.closed
    assert owner.live_tensor_handle_count >= 1
    assert owner.unreleased_tensor_slots


def test_contradictory_released_outcome_with_live_wrapper_is_poisoned_and_retained(expect_error, monkeypatch) -> None:
    tensor = _Tensor(("backbone", 0), 0)
    release_error = RuntimeError("injected cache cleanup failure")

    class FailedCache(_Cache):
        def load_layer(self, _mesh, *, layer_index: int, namespace: str):
            raise BF4CleanupError(
                "injected contradictory cleanup ledger",
                primary_error=RuntimeError("load failed"),
                cleanup_errors=(release_error,),
                tensor_cleanup_outcomes=(
                    BF4TensorCleanupOutcome(0, tensor, True, True, None),
                    BF4TensorCleanupOutcome(1, None, False, False, None),
                ),
            )

    owner = Qwen38BF4ResidentSet(FailedCache(), object())
    unexpected_releases: list[_Tensor] = []
    monkeypatch.setattr(bf4_module.ttnn, "deallocate", unexpected_releases.append)

    with expect_error(BF4CleanupError, match="malformed"):
        owner._preload((("backbone", 0),))

    assert unexpected_releases == []
    assert owner.poisoned
    assert owner.unreleased_tensor_slots == (("backbone", 0, 0, True, "RuntimeError"),)


def test_load_n_failure_with_cleanup_failure_attempts_all_and_retains_only_failure(expect_error, monkeypatch) -> None:
    owner = Qwen38BF4ResidentSet(_Cache(fail_key=("backbone", 2)), object())
    calls: list[_Tensor] = []

    def release(value: _Tensor) -> None:
        calls.append(value)
        if value.key == ("backbone", 0) and value.slot == 0:
            raise RuntimeError("injected rollback release failure")

    monkeypatch.setattr(bf4_module.ttnn, "deallocate", release)
    with expect_error(BF4CleanupError, match="injected load failure"):
        owner._preload(tuple(("backbone", layer) for layer in range(4)))

    assert [(tensor.key, tensor.slot) for tensor in calls] == [
        (("backbone", 1), 1),
        (("backbone", 1), 0),
        (("backbone", 0), 1),
        (("backbone", 0), 0),
    ]
    assert owner.unreleased_tensor_slots == (("backbone", 0, 0, True, "RuntimeError"),)
    assert tuple((event.layer_index, event.outcome) for event in owner.preload_events) == (
        (0, "attempt"),
        (0, "success"),
        (1, "attempt"),
        (1, "success"),
        (2, "attempt"),
        (2, "failure"),
    )
    with expect_error(BF4CleanupError):
        owner.close()
    assert len(calls) == 4


def test_close_middle_tensor_failure_attempts_every_handle_once_and_keeps_ledger(expect_error, monkeypatch) -> None:
    owner = Qwen38BF4ResidentSet(_Cache(), object())
    owner._preload((("backbone", 0), ("backbone", 1)))
    calls: list[_Tensor] = []

    def release(value: _Tensor) -> None:
        calls.append(value)
        if value.key == ("backbone", 1) and value.slot == 0:
            raise RuntimeError("injected middle release failure")

    monkeypatch.setattr(bf4_module.ttnn, "deallocate", release)
    with expect_error(BF4CleanupError, match="owner cleanup was incomplete"):
        owner.close()

    assert [(tensor.key, tensor.slot) for tensor in calls] == [
        (("backbone", 1), 1),
        (("backbone", 1), 0),
        (("backbone", 0), 1),
        (("backbone", 0), 0),
    ]
    assert owner.unreleased_tensor_slots == (("backbone", 1, 0, True, "RuntimeError"),)
    with expect_error(BF4CleanupError):
        owner.close()
    assert len(calls) == 4
