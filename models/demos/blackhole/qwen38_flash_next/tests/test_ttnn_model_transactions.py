# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device fault injection for TTNN model state and tensor ownership."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import models.demos.blackhole.qwen38_flash_next.ttnn.layer as layer_module
import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, Qwen38TTNNGDNSnapshot, Qwen38TTNNGDNState
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    BACKBONE_LAYERS,
    Qwen38TTNNDecoderLayer,
    Qwen38TTNNDecoderLayerAux,
    Qwen38TTNNDecoderLayerSnapshot,
    Qwen38TTNNDecoderLayerState,
    Qwen38TTNNLayerCleanupError,
    Qwen38TTNNLayerNamespace,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    Qwen38TTNNCleanupError,
    Qwen38TTNNModelPoisonedError,
    Qwen38TTNNRoPE,
    Qwen38TTNNRoPEInputs,
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelSnapshot,
    Qwen38TTNNTextModelState,
    _host_rope,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import Qwen38TTNNPLE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38TTNNQSA, Qwen38TTNNQSAState


class _FakeTensor:
    _next_id = 1

    def __init__(self, name: str, shape=(1, 4, 1, 640), *, layout=None) -> None:
        self.name = name
        self.shape = shape
        self.dtype = ttnn.bfloat16
        self.layout = ttnn.TILE_LAYOUT if layout is None else layout
        self._id = _FakeTensor._next_id
        _FakeTensor._next_id += 1

    def tensor_id(self) -> int:
        return self._id


class _FakeMeshContract:
    def validate_tensor(self, tensor, **kwargs) -> None:
        del tensor, kwargs


def _layer_state(index: int, position: int) -> SimpleNamespace:
    return SimpleNamespace(
        namespace=Qwen38TTNNLayerNamespace.BACKBONE,
        layer_index=index,
        position=position,
    )


class _FakeLayer:
    def __init__(self, index: int, log: list[tuple[str, int]]) -> None:
        self.layer_index = index
        self.layer_type = Qwen38TTNNLayerType.QSA if index % 4 == 3 else Qwen38TTNNLayerType.GDN
        self.log = log
        self.preflight_failure: str | None = None
        self.mutation_failure: str | None = None
        self.snapshot_failure = False
        self.cleanup_failure = False
        self.release_failure = False
        self.forward_failure = False
        self.forward_kwargs = None

    def allocate_state(self):
        self.log.append(("allocate", self.layer_index))
        return _layer_state(self.layer_index, 0)

    def reset_state(self, state):
        self.log.append(("reset", self.layer_index))
        return _layer_state(self.layer_index, 0)

    def validate_state(self, state) -> None:
        self.log.append(("validate_state", self.layer_index))
        assert state.layer_index == self.layer_index

    def validate_snapshot(self, snapshot) -> None:
        self.log.append(("validate_snapshot", self.layer_index))
        assert snapshot.layer_index == self.layer_index
        assert snapshot.active

    def validate_restore_pair(self, state, snapshot) -> None:
        del state, snapshot
        self.log.append(("preflight_restore", self.layer_index))
        if self.preflight_failure == "restore":
            raise RuntimeError(f"restore preflight {self.layer_index}")

    def validate_commit_pair(self, state, snapshot) -> None:
        del state, snapshot
        self.log.append(("preflight_commit", self.layer_index))
        if self.preflight_failure == "commit":
            raise RuntimeError(f"commit preflight {self.layer_index}")

    def restore_state(self, state, snapshot):
        self.log.append(("restore", self.layer_index))
        if self.mutation_failure == "restore":
            raise RuntimeError(f"restore mutation {self.layer_index}")
        snapshot.active = False
        return _layer_state(self.layer_index, snapshot.position)

    def commit_state(self, state, snapshot):
        self.log.append(("commit", self.layer_index))
        snapshot.active = False
        if self.mutation_failure == "commit" or self.cleanup_failure:
            raise RuntimeError(f"commit mutation {self.layer_index}")
        return state

    def snapshot_state(self, state):
        self.log.append(("snapshot", self.layer_index))
        if self.snapshot_failure:
            raise RuntimeError(f"snapshot creation {self.layer_index}")
        return SimpleNamespace(layer_index=self.layer_index, position=state.position, active=True)

    def release_state(self, state) -> None:
        del state
        self.log.append(("release", self.layer_index))
        if self.release_failure:
            raise RuntimeError(f"release {self.layer_index}")

    def forward_decode(self, residual, state, **kwargs):
        del residual
        self.log.append(("forward", self.layer_index))
        self.forward_kwargs = kwargs
        phase_observer = kwargs.get("phase_observer")
        if phase_observer is not None:
            phase_observer("before-fake-stage")
            phase_observer("after-fake-stage")
        if self.forward_failure:
            raise RuntimeError(f"forward {self.layer_index}")
        return SimpleNamespace(
            residual_sharded=_FakeTensor(f"residual-{self.layer_index}"),
            state=_layer_state(self.layer_index, state.position + 1),
            aux=Qwen38TTNNDecoderLayerAux(routing=None, selection=None, reused_qsa_selection=False),
        )


def _model_owner(*, position: int = 0, register_snapshot: bool = False):
    log: list[tuple[str, int]] = []
    layers = tuple(_FakeLayer(index, log) for index in range(BACKBONE_LAYERS))
    owner = Qwen38TTNNTextModel.__new__(Qwen38TTNNTextModel)
    owner.layers = layers
    owner.mesh_contract = _FakeMeshContract()
    owner._state_owner = object()
    owner._poisoned_error = None
    state = Qwen38TTNNTextModelState(
        position=position,
        layers=tuple(_layer_state(index, position) for index in range(BACKBONE_LAYERS)),
        _owner=owner._state_owner,
    )
    snapshots = tuple(
        SimpleNamespace(layer_index=index, position=position, active=True) for index in range(BACKBONE_LAYERS)
    )
    snapshot = Qwen38TTNNTextModelSnapshot(position, snapshots, owner._state_owner)
    owner._active_snapshot = snapshot if register_snapshot else None
    return owner, layers, state, snapshot, log


@pytest.mark.parametrize("operation", ("restore", "commit"))
def test_all_48_pairs_are_preflighted_before_late_snapshot_failure(operation: str, expect_error) -> None:
    owner, layers, state, snapshot, log = _model_owner(position=9, register_snapshot=True)
    layers[-1].preflight_failure = operation

    with expect_error(RuntimeError, f"{operation} preflight 47"):
        getattr(owner, f"{operation}_state")(state, snapshot)

    assert [index for action, index in log if action == "validate_state"] == list(range(BACKBONE_LAYERS))
    assert [index for action, index in log if action == "validate_snapshot"] == list(range(BACKBONE_LAYERS))
    assert [index for action, index in log if action == f"preflight_{operation}"] == list(range(BACKBONE_LAYERS))
    assert not any(action == operation for action, _ in log)
    assert snapshot.active
    assert owner._active_snapshot is snapshot
    assert not owner.poisoned


@pytest.mark.parametrize("operation", ("restore", "commit"))
def test_mutation_failure_consumes_outer_snapshot_and_permanently_poisons_owner(operation: str, expect_error) -> None:
    owner, layers, state, snapshot, log = _model_owner(position=5, register_snapshot=True)
    layers[37].mutation_failure = operation

    with expect_error(Qwen38TTNNModelPoisonedError, f"poisoned after {operation}_state"):
        getattr(owner, f"{operation}_state")(state, snapshot)

    poison = owner.poisoned_error
    assert poison is not None
    assert poison.operation == f"{operation}_state"
    assert poison.processed_layers == 37
    assert not snapshot.active
    assert owner._active_snapshot is None
    assert owner.poisoned_error is poison
    mutation_count = len([item for item in log if item[0] == operation])
    with expect_error(Qwen38TTNNModelPoisonedError, f"poisoned after {operation}_state"):
        owner.snapshot_state(state)
    assert owner.poisoned_error is poison
    assert len([item for item in log if item[0] == operation]) == mutation_count


def test_late_snapshot_and_cleanup_failures_attempt_every_prior_layer_and_poison(expect_error) -> None:
    owner, layers, state, _, log = _model_owner(position=2)
    layers[47].snapshot_failure = True
    layers[31].cleanup_failure = True

    with expect_error(Qwen38TTNNModelPoisonedError, "poisoned after snapshot_state cleanup"):
        owner.snapshot_state(state)

    assert owner.poisoned_error is not None
    assert isinstance(owner.poisoned_error.original_cause, Qwen38TTNNCleanupError)
    assert [index for action, index in log if action == "snapshot"] == list(range(BACKBONE_LAYERS))
    assert [index for action, index in log if action == "commit"] == list(reversed(range(47)))
    assert owner.poisoned


def test_state_release_attempts_every_layer_once_and_poison_blocks_double_release(expect_error) -> None:
    owner, layers, state, _, log = _model_owner(position=0)
    layers[29].release_failure = True

    with expect_error(Qwen38TTNNModelPoisonedError, "poisoned after release_state"):
        owner.release_state(state)

    assert [index for action, index in log if action == "release"] == list(reversed(range(BACKBONE_LAYERS)))
    with expect_error(Qwen38TTNNModelPoisonedError, "poisoned after release_state"):
        owner.release_state(state)
    assert [index for action, index in log if action == "release"] == list(reversed(range(BACKBONE_LAYERS)))


def test_model_snapshot_position_mismatch_is_rejected_before_any_mutation(expect_error) -> None:
    owner, _, state, snapshot, log = _model_owner(position=11, register_snapshot=True)
    snapshot.layers[-1].position = 10

    with expect_error(ValueError, "target snapshot 47 position 10"):
        owner.commit_state(state, snapshot)

    assert not any(action == "commit" for action, _ in log)
    assert snapshot.active
    assert not owner.poisoned


@pytest.mark.parametrize("operation", ("restore", "commit"))
def test_model_allows_only_one_registered_snapshot_and_clears_it_on_consume(operation, expect_error) -> None:
    owner, _, state, _, log = _model_owner(position=4)
    snapshot = owner.snapshot_state(state)
    assert owner._active_snapshot is snapshot
    snapshot_count = len([item for item in log if item[0] == "snapshot"])

    with expect_error(RuntimeError, "create another snapshot"):
        owner.snapshot_state(state)

    assert len([item for item in log if item[0] == "snapshot"]) == snapshot_count
    consumed = getattr(owner, f"{operation}_state")(state, snapshot)
    assert consumed.position == state.position
    assert not snapshot.active
    assert owner._active_snapshot is None


def test_active_snapshot_blocks_allocate_reset_release_and_prefill_before_mutation(expect_error) -> None:
    owner, _, state, _, log = _model_owner(position=2, register_snapshot=True)

    with expect_error(RuntimeError, "allocate another state"):
        owner.allocate_state()
    with expect_error(RuntimeError, "reset state"):
        owner.reset_state(state)
    with expect_error(RuntimeError, "release state"):
        owner.release_state(state)
    with expect_error(RuntimeError, "run serial prefill"):
        owner.prefill_serial(torch.tensor([[7]], dtype=torch.long), state=state)

    assert not any(action in {"allocate", "reset", "release", "forward"} for action, _ in log)


def test_speculative_retain_phase_is_preflighted_before_embedding(expect_error) -> None:
    owner, _, state, _, _ = _model_owner(position=6, register_snapshot=True)
    embed_calls = []
    owner._embed_residual = lambda host_token: embed_calls.append(host_token)

    with expect_error(RuntimeError, "retain_input_state must be true"):
        owner.forward_decode(7, state, retain_input_state=False)
    assert embed_calls == []
    assert owner._active_snapshot is not None and owner._active_snapshot.active

    advanced = Qwen38TTNNTextModelState(
        7,
        tuple(_layer_state(index, 7) for index in range(BACKBONE_LAYERS)),
        owner._state_owner,
    )
    with expect_error(RuntimeError, "retain_input_state must be false"):
        owner.forward_decode(7, advanced, retain_input_state=True)
    assert embed_calls == []
    assert owner._active_snapshot is not None and owner._active_snapshot.active

    owner, _, state, _, _ = _model_owner(position=6)
    owner._embed_residual = lambda host_token: embed_calls.append(host_token)
    with expect_error(RuntimeError, "requires an active text-model snapshot"):
        owner.forward_decode(7, state, retain_input_state=True)
    assert embed_calls == []


def test_decode_phase_observer_is_rejected_before_state_or_device_work(expect_error) -> None:
    owner, _, state, _, log = _model_owner(position=6)
    embed_calls = []
    owner._embed_residual = lambda host_token: embed_calls.append(host_token)

    with expect_error(TypeError, "decode phase observer must be callable"):
        owner.forward_decode(7, state, phase_observer=object())

    assert embed_calls == []
    assert not any(action == "forward" for action, _ in log)
    assert not owner.poisoned


def test_unregistered_snapshot_is_rejected_before_child_transaction_mutation(expect_error) -> None:
    owner, _, state, snapshot, log = _model_owner(position=3)
    with expect_error(ValueError, "not the active transaction"):
        owner.restore_state(state, snapshot)
    assert not any(action == "restore" for action, _ in log)
    assert snapshot.active


def test_post_preflight_speculative_setup_failure_consumes_snapshot_and_poisons(expect_error) -> None:
    owner, _, state, snapshot, _ = _model_owner(position=3, register_snapshot=True)

    def fail_embedding(host_token):
        del host_token
        raise RuntimeError("injected speculative embedding failure")

    owner._embed_residual = fail_embedding
    with expect_error(Qwen38TTNNModelPoisonedError, "speculative decode setup"):
        owner.forward_decode(7, state, retain_input_state=True)

    assert owner.poisoned
    assert owner._active_snapshot is None
    assert not snapshot.active


def test_real_qsa_view_tracker_restores_then_commits_one_protected_branch(monkeypatch) -> None:
    import models.demos.blackhole.qwen38_flash_next.ttnn.qsa as qsa_module

    qsa = Qwen38TTNNQSA.__new__(Qwen38TTNNQSA)
    qsa.layer_index = 3
    qsa.mesh_contract = _FakeMeshContract()
    qsa._live_epochs = {1}
    qsa._live_views = {10: 1}
    qsa._protected_views = set()
    qsa._trace_input_retention_active = False
    qsa._trace_retained_inputs = {}
    monkeypatch.setattr(qsa_module, "_deallocate", lambda *tensors: None)

    packed = _FakeTensor("packed", (1, 1, 262144, 512), layout=ttnn.ROW_MAJOR_LAYOUT)
    compressed = _FakeTensor("compressed", (1, 1, 65536, 128))
    staging = _FakeTensor("staging", (1, 1, 32, 512), layout=ttnn.ROW_MAJOR_LAYOUT)

    def view(view_id: int, *, owns_staging: bool) -> Qwen38TTNNQSAState:
        return Qwen38TTNNQSAState(
            layer_index=3,
            epoch=1,
            view_id=view_id,
            next_position=0,
            compressed_blocks=0,
            raw_tail_count=0,
            raw_index_tail=None,
            packed_kv_cache=packed,
            compressed_index_cache=compressed,
            kv_staging=staging,
            kv_staging_owned=owns_staging,
        )

    checkpoint = view(10, owns_staging=True)
    assert qsa.checkpoint_state(checkpoint) is checkpoint
    branch = view(11, owns_staging=False)
    qsa._live_views[11] = 1
    qsa._validate_transaction_pair(checkpoint, branch)
    restored = qsa.restore_state(branch, checkpoint)
    assert restored is checkpoint
    assert qsa._live_views == {10: 1}
    assert qsa._protected_views == set()

    assert qsa.checkpoint_state(restored) is restored
    committed = view(12, owns_staging=False)
    qsa._live_views[12] = 1
    qsa._validate_transaction_pair(restored, committed)
    assert qsa.commit_state(restored, committed) is committed
    assert qsa._live_views == {12: 1}
    assert qsa._protected_views == set()


def test_rope_cleanup_retry_releases_only_the_failed_alias_group(monkeypatch, expect_error) -> None:
    first = _FakeTensor("first")
    flaky = _FakeTensor("flaky")
    last = _FakeTensor("last")
    attempts = defaultdict(int)

    def deallocate(tensor) -> None:
        attempts[tensor.name] += 1
        if tensor is flaky and attempts[tensor.name] == 1:
            raise RuntimeError("injected deallocate failure")

    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    inputs = Qwen38TTNNRoPEInputs(3, first, flaky, first, last)

    with expect_error(Qwen38TTNNCleanupError, "QSA RoPE inputs cleanup failed"):
        inputs.deallocate()
    assert inputs.active
    assert dict(attempts) == {"first": 1, "flaky": 1, "last": 1}

    inputs.deallocate()
    assert not inputs.active
    assert dict(attempts) == {"first": 1, "flaky": 2, "last": 1}
    with expect_error(RuntimeError, "already deallocated"):
        inputs.deallocate()
    assert dict(attempts) == {"first": 1, "flaky": 2, "last": 1}


@pytest.mark.parametrize("position, block_start", ((2, None), (3, 0), (7, 4)))
def test_rope_uploads_block_start_only_when_position_closes_four_token_block(position, block_start) -> None:
    rope = Qwen38TTNNRoPE.__new__(Qwen38TTNNRoPE)
    rope.inverse_frequency = 1.0 / (float(10_000_000) ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    uploaded = []

    def upload(host):
        uploaded.append(host.clone())
        return host

    rope._upload = upload
    inputs = rope.for_position(position)
    expected_cos, expected_sin = _host_rope(position, rope.inverse_frequency)
    assert torch.equal(inputs.cos, expected_cos)
    assert torch.equal(inputs.sin, expected_sin)
    if block_start is None:
        assert inputs.block_start_cos is None and inputs.block_start_sin is None
        assert len(uploaded) == 2
    else:
        block_cos, block_sin = _host_rope(block_start, rope.inverse_frequency)
        assert torch.equal(inputs.block_start_cos, block_cos)
        assert torch.equal(inputs.block_start_sin, block_sin)
        assert len(uploaded) == 4


def test_late_forward_failure_poison_blocks_stranded_state_reuse(monkeypatch, expect_error) -> None:
    owner, layers, state, _, log = _model_owner(position=3)
    layers[-1].forward_failure = True
    initial = _FakeTensor("initial")

    class FakeRoPEInputs:
        def __init__(self) -> None:
            self.cos = _FakeTensor("cos", shape=(1, 1, 1, 64))
            self.sin = _FakeTensor("sin", shape=(1, 1, 1, 64))
            self.block_start_cos = _FakeTensor("block-cos", shape=(1, 1, 1, 64))
            self.block_start_sin = _FakeTensor("block-sin", shape=(1, 1, 1, 64))
            self.active = True
            self.release_count = 0

        def deallocate(self) -> None:
            self.release_count += 1
            self.active = False

    rope_inputs = FakeRoPEInputs()
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: None)
    owner._embed_residual = lambda host_token: initial
    owner.rope = SimpleNamespace(for_position=lambda position: rope_inputs)

    with expect_error(Qwen38TTNNModelPoisonedError, "poisoned after forward_decode"):
        owner.forward_decode(17, state, return_logits=False)

    poison = owner.poisoned_error
    assert poison is not None
    assert poison.operation == "forward_decode"
    assert poison.processed_layers == BACKBONE_LAYERS - 1
    assert rope_inputs.release_count == 1
    assert [index for action, index in log if action == "forward"] == list(range(BACKBONE_LAYERS))
    for index, layer in enumerate(layers):
        assert "phase_observer" not in layer.forward_kwargs
        assert layer.forward_kwargs["token_id"].shape == (1, 1)
        if layer.layer_type is Qwen38TTNNLayerType.QSA:
            assert layer.forward_kwargs["cos"] is rope_inputs.cos
            assert layer.forward_kwargs["sin"] is rope_inputs.sin
            assert layer.forward_kwargs["block_start_cos"] is rope_inputs.block_start_cos
            assert layer.forward_kwargs["block_start_sin"] is rope_inputs.block_start_sin
        else:
            assert layer.forward_kwargs["cos"] is None
            assert layer.forward_kwargs["sin"] is None
    with expect_error(Qwen38TTNNModelPoisonedError, "poisoned after forward_decode"):
        owner.reset_state(state)
    assert owner.poisoned_error is poison


def test_decode_phase_observer_brackets_all_layers_final_mixer_and_lm_head_without_tensors(monkeypatch) -> None:
    owner, _, state, _, log = _model_owner(position=3)
    initial = _FakeTensor("initial")
    hidden = _FakeTensor("hidden", shape=(1, 1, 1, 640))
    logits_tensor = _FakeTensor("logits")
    logits = SimpleNamespace(tensor=logits_tensor)
    released = []

    class FakeRoPEInputs:
        def __init__(self) -> None:
            self.cos = _FakeTensor("cos", shape=(1, 1, 1, 64))
            self.sin = _FakeTensor("sin", shape=(1, 1, 1, 64))
            self.block_start_cos = _FakeTensor("block-cos", shape=(1, 1, 1, 64))
            self.block_start_sin = _FakeTensor("block-sin", shape=(1, 1, 1, 64))
            self.active = True

        def deallocate(self) -> None:
            assert self.active
            self.active = False

    rope_inputs = FakeRoPEInputs()
    owner._embed_residual = lambda host_token: initial
    owner.rope = SimpleNamespace(for_position=lambda position: rope_inputs)
    owner.final_mixer = lambda residual: hidden

    class FakeLMHead:
        def __call__(self, residual):
            return logits

        def greedy_token(self, actual_logits):
            assert actual_logits is logits
            return torch.tensor([[[15]]], dtype=torch.int64)

    owner.model_io = SimpleNamespace(lm_head=FakeLMHead())
    monkeypatch.setattr(ttnn, "deallocate", released.append)
    phases = []

    def observe(phase: str) -> None:
        assert type(phase) is str
        phases.append(phase)

    output = owner.forward_decode(17, state, resolve_greedy=True, phase_observer=observe)

    expected_layers = [
        phase
        for layer_index in range(BACKBONE_LAYERS)
        for phase in (
            f"before-layer-{layer_index}",
            f"layer-{layer_index}-before-fake-stage",
            f"layer-{layer_index}-after-fake-stage",
            f"after-layer-{layer_index}",
        )
    ]
    assert phases == [
        *expected_layers,
        "before-final-mixer",
        "after-final-mixer",
        "before-lm-head",
        "after-lm-head",
        "before-greedy-resolve",
        "after-greedy-resolve",
    ]
    assert [index for action, index in log if action == "forward"] == list(range(BACKBONE_LAYERS))
    assert not rope_inputs.active
    output.release_tensors()
    assert released == [output.hyper_residual_sharded, hidden, logits_tensor]


def test_decode_phase_observer_failure_preserves_final_mixer_cleanup_ownership(monkeypatch, expect_error) -> None:
    owner, _, state, _, _ = _model_owner(position=3)
    hidden = _FakeTensor("hidden", shape=(1, 1, 1, 640))
    released = []

    class FakeRoPEInputs:
        def __init__(self) -> None:
            self.cos = _FakeTensor("cos", shape=(1, 1, 1, 64))
            self.sin = _FakeTensor("sin", shape=(1, 1, 1, 64))
            self.block_start_cos = _FakeTensor("block-cos", shape=(1, 1, 1, 64))
            self.block_start_sin = _FakeTensor("block-sin", shape=(1, 1, 1, 64))
            self.active = True

        def deallocate(self) -> None:
            self.active = False

    owner._embed_residual = lambda host_token: _FakeTensor("initial")
    owner.rope = SimpleNamespace(for_position=lambda position: FakeRoPEInputs())
    owner.final_mixer = lambda residual: hidden
    owner.model_io = SimpleNamespace(lm_head=lambda residual: None)
    monkeypatch.setattr(ttnn, "deallocate", released.append)

    def fail_after_mixer(phase: str) -> None:
        if phase == "after-final-mixer":
            raise RuntimeError("injected nonsemantic observer failure")

    with expect_error(Qwen38TTNNModelPoisonedError, "poisoned after forward_decode"):
        owner.forward_decode(17, state, return_logits=False, phase_observer=fail_after_mixer)

    assert [tensor.name for tensor in released] == ["residual-47", "hidden"]
    assert owner.poisoned


def test_internal_phase_observer_failure_is_deferred_until_layer_output_has_cleanup_ownership(
    monkeypatch, expect_error
) -> None:
    owner, _, state, _, log = _model_owner(position=3)
    released = []

    class FakeRoPEInputs:
        def __init__(self) -> None:
            self.cos = _FakeTensor("cos", shape=(1, 1, 1, 64))
            self.sin = _FakeTensor("sin", shape=(1, 1, 1, 64))
            self.block_start_cos = _FakeTensor("block-cos", shape=(1, 1, 1, 64))
            self.block_start_sin = _FakeTensor("block-sin", shape=(1, 1, 1, 64))
            self.active = True

        def deallocate(self) -> None:
            self.active = False

    owner._embed_residual = lambda host_token: _FakeTensor("initial")
    owner.rope = SimpleNamespace(for_position=lambda position: FakeRoPEInputs())
    monkeypatch.setattr(ttnn, "deallocate", released.append)

    def fail_inside_layer(phase: str) -> None:
        if phase == "layer-0-after-fake-stage":
            raise RuntimeError("injected internal observer failure")

    with expect_error(Qwen38TTNNModelPoisonedError, "injected internal observer failure"):
        owner.forward_decode(17, state, return_logits=False, phase_observer=fail_inside_layer)

    assert [index for action, index in log if action == "forward"] == [0]
    assert [tensor.name for tensor in released] == ["residual-0"]
    assert owner.poisoned_error is not None
    assert owner.poisoned_error.processed_layers == 1


def test_decoder_layer_internal_phase_observer_brackets_exact_ownership_stages(monkeypatch) -> None:
    operations = []
    phases = []

    class FakeGR:
        def __init__(self, label: str) -> None:
            self.label = label

        def read(self, residual):
            operations.append(f"{self.label}-read")
            return object(), SimpleNamespace(residual=object(), injection=object())

        def write(self, hidden, state):
            operations.append(f"{self.label}-write")
            return object()

    class FakeStreamer:
        released = False

        class Lease:
            def __init__(self, owner) -> None:
                self.owner = owner

            def __enter__(self):
                operations.append("expert-stream-acquire")
                return (object(), object())

            def __exit__(self, error_type, error, traceback) -> None:
                del error_type, error, traceback
                operations.append("expert-stream-release")
                self.owner.released = True

        def layer(self, layer_index: int, *, namespace: str):
            assert (layer_index, namespace) == (1, "backbone")
            return self.Lease(self)

    attention = Qwen38TTNNGDN.__new__(Qwen38TTNNGDN)
    attention.forward_decode = lambda hidden, state: (
        operations.append("gdn-forward") or SimpleNamespace(hidden_sharded=object(), state=state)
    )

    def moe_forward(hidden, packed_w01, packed_w2, *, return_routing, phase_observer=None):
        del hidden, packed_w01, packed_w2, return_routing
        operations.append("moe-forward")
        if phase_observer is not None:
            phase_observer("before-moe-compute-launch")
            phase_observer("after-moe-compute-launch")
        return SimpleNamespace(hidden_sharded=object(), routing=object())

    mlp = SimpleNamespace(forward=moe_forward)
    streamer = FakeStreamer()
    layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    layer.layer_index = 1
    layer.layer_type = Qwen38TTNNLayerType.GDN
    layer.attention = attention
    layer.attention_gr = FakeGR("attention-gr")
    layer.mlp = mlp
    layer.mlp_gr = FakeGR("mlp-gr")
    layer.expert_streamer = streamer
    layer.ple = object()
    layer._validate_residual = lambda residual, label: None
    layer._validate_block = lambda hidden, label: None
    layer._validate_state = lambda state: None
    layer._apply_ple = lambda residual, state, *, token_id=None, prepared_ple=None: (
        operations.append("ple") or residual,
        state.ple,
    )
    monkeypatch.setattr(layer_module, "_deallocate_unique", lambda *tensors: None)

    def observe(phase: str) -> None:
        assert type(phase) is str
        if phase == "after-expert-stream-release":
            assert streamer.released
        phases.append(phase)

    state = Qwen38TTNNDecoderLayerState(
        Qwen38TTNNLayerNamespace.BACKBONE,
        1,
        0,
        object(),
        object(),
    )
    result = layer.forward_decode(object(), state, token_id=torch.tensor([[17]]), phase_observer=observe)

    assert phases == [
        "before-ple",
        "after-ple",
        "before-attention-gr-read",
        "after-attention-gr-read",
        "before-gdn",
        "after-gdn",
        "before-attention-gr-write",
        "after-attention-gr-write",
        "before-mlp-gr-read",
        "after-mlp-gr-read",
        "before-expert-stream-acquire",
        "after-expert-stream-acquire",
        "before-moe-forward",
        "before-moe-compute-launch",
        "after-moe-compute-launch",
        "after-moe-forward",
        "before-expert-stream-release",
        "after-expert-stream-release",
        "before-mlp-gr-write",
        "after-mlp-gr-write",
        "before-state-update",
        "after-state-update",
    ]
    assert operations == [
        "ple",
        "attention-gr-read",
        "gdn-forward",
        "attention-gr-write",
        "mlp-gr-read",
        "expert-stream-acquire",
        "moe-forward",
        "expert-stream-release",
        "mlp-gr-write",
    ]
    assert result.state.position == 1


def test_decoder_layer_phase_observer_is_rejected_before_state_or_tensor_validation(expect_error) -> None:
    layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    validation = []
    layer._validate_residual = lambda residual, label: validation.append("residual")
    layer._validate_state = lambda state: validation.append("state")

    with expect_error(TypeError, match="decoder-layer phase observer must be callable"):
        layer.forward_decode(object(), object(), phase_observer=object())

    assert validation == []


def test_layer_preflight_rejects_wrong_gdn_qsa_and_ple_sources(monkeypatch, expect_error) -> None:
    mesh_contract = object()
    monkeypatch.setattr(Qwen38TTNNGDN, "_validate_state", lambda self, state: None)
    monkeypatch.setattr(Qwen38TTNNGDNSnapshot, "validate", lambda self, contract: None)
    monkeypatch.setattr(Qwen38TTNNQSA, "_validate_state", lambda self, state: None)
    monkeypatch.setattr(Qwen38TTNNQSA, "_validate_transaction_pair", lambda self, checkpoint, current: None)

    gdn = Qwen38TTNNGDN.__new__(Qwen38TTNNGDN)
    gdn_layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    gdn_layer.mesh_contract = mesh_contract
    gdn_layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    gdn_layer.layer_index = 0
    gdn_layer.layer_type = Qwen38TTNNLayerType.GDN
    gdn_layer.attention = gdn
    gdn_layer.ple = None
    gdn_state = Qwen38TTNNGDNState(0, object(), (object(),) * 4, object(), object(), mesh_contract)
    current = Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.BACKBONE, 0, 0, gdn_state, None)
    gdn_snapshot = Qwen38TTNNGDNSnapshot(0, object(), (object(),) * 4, captured=True)
    snapshot = Qwen38TTNNDecoderLayerSnapshot(
        Qwen38TTNNLayerNamespace.BACKBONE,
        0,
        0,
        gdn_state,
        gdn_snapshot,
        None,
        None,
    )
    gdn_layer.validate_restore_pair(current, snapshot)
    snapshot.source_attention = Qwen38TTNNGDNState(0, object(), (object(),) * 4, object(), object(), mesh_contract)
    with expect_error(ValueError, "snapshotted fixed-address"):
        gdn_layer.validate_restore_pair(current, snapshot)
    with expect_error(TypeError, "non-GDN"):
        gdn_layer.validate_state(Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.BACKBONE, 0, 0, object(), None))

    qsa = Qwen38TTNNQSA.__new__(Qwen38TTNNQSA)
    qsa_layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    qsa_layer.mesh_contract = mesh_contract
    qsa_layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    qsa_layer.layer_index = 3
    qsa_layer.layer_type = Qwen38TTNNLayerType.QSA
    qsa_layer.attention = qsa
    qsa_layer.ple = None
    qsa_state = Qwen38TTNNQSAState(3, 1, 1, 0, 0, 0, None, object(), object(), object(), True)
    qsa_current = Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.BACKBONE, 3, 0, qsa_state, None)
    qsa_snapshot = Qwen38TTNNDecoderLayerSnapshot(
        Qwen38TTNNLayerNamespace.BACKBONE,
        3,
        0,
        qsa_state,
        qsa_state,
        None,
        None,
    )
    qsa_layer.validate_commit_pair(qsa_current, qsa_snapshot)
    qsa_snapshot.attention = Qwen38TTNNQSAState(3, 1, 2, 0, 0, 0, None, object(), object(), object(), True)
    with expect_error(ValueError, "immutable source view"):
        qsa_layer.validate_commit_pair(qsa_current, qsa_snapshot)

    ple_layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    ple_layer.mesh_contract = mesh_contract
    ple_layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    ple_layer.layer_index = 1
    ple_layer.layer_type = Qwen38TTNNLayerType.GDN
    ple_layer.attention = gdn
    ple_layer.ple = Qwen38TTNNPLE.__new__(Qwen38TTNNPLE)
    ple_gdn_state = Qwen38TTNNGDNState(1, object(), (object(),) * 4, object(), object(), mesh_contract)
    with expect_error(TypeError, "Qwen38TTNNPLEState"):
        ple_layer.validate_state(
            Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.BACKBONE, 1, 0, ple_gdn_state, object())
        )


def test_layer_allocate_validation_failure_releases_ple_then_attention(expect_error) -> None:
    released = []

    class FakeAttention:
        def allocate_state(self):
            return object()

        def release_state(self, state) -> None:
            del state
            released.append("attention")

    class FakePLEState:
        def deallocate(self) -> None:
            released.append("PLE")

    class FakePLE:
        def allocate_state(self):
            return FakePLEState()

    layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    layer.attention = FakeAttention()
    layer.ple = FakePLE()
    layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    layer.layer_index = 1
    layer._validate_state = lambda state: (_ for _ in ()).throw(RuntimeError("injected validation failure"))

    with expect_error(RuntimeError, "injected validation failure"):
        layer.allocate_state()

    assert released == ["PLE", "attention"]


@pytest.mark.parametrize("operation", ("restore", "commit"))
def test_layer_postmutation_validation_failure_still_consumes_and_cleans_snapshot(
    monkeypatch, operation, expect_error
) -> None:
    released = []
    mesh_contract = object()
    attention = Qwen38TTNNGDN.__new__(Qwen38TTNNGDN)
    state = Qwen38TTNNGDNState(0, object(), (object(),) * 4, object(), object(), mesh_contract)
    attention_snapshot = Qwen38TTNNGDNSnapshot(0, object(), (object(),) * 4, captured=True)
    snapshot = Qwen38TTNNDecoderLayerSnapshot(
        Qwen38TTNNLayerNamespace.BACKBONE,
        0,
        0,
        state,
        attention_snapshot,
        None,
        None,
    )
    current = Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.BACKBONE, 0, 1, state, None)
    layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    layer.attention = attention
    layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    layer.layer_index = 0
    layer.ple = None
    setattr(layer, f"validate_{operation}_pair", lambda state, snapshot: None)
    layer._validate_state = lambda state: (_ for _ in ()).throw(RuntimeError("injected result validation failure"))
    monkeypatch.setattr(
        Qwen38TTNNGDNSnapshot,
        "deallocate",
        lambda self: released.append("GDN snapshot"),
    )
    monkeypatch.setattr(Qwen38TTNNGDNState, "restore_from", lambda self, snapshot: None)

    with expect_error(RuntimeError, "injected result validation failure"):
        getattr(layer, f"{operation}_state")(current, snapshot)

    assert not snapshot.active
    assert snapshot.attention is None
    assert released == ["GDN snapshot"]


def test_layer_postmutation_validation_and_cleanup_failures_keep_both_causes(monkeypatch) -> None:
    mesh_contract = object()
    attention = Qwen38TTNNGDN.__new__(Qwen38TTNNGDN)
    state = Qwen38TTNNGDNState(0, object(), (object(),) * 4, object(), object(), mesh_contract)
    attention_snapshot = Qwen38TTNNGDNSnapshot(0, object(), (object(),) * 4, captured=True)
    snapshot = Qwen38TTNNDecoderLayerSnapshot(
        Qwen38TTNNLayerNamespace.BACKBONE,
        0,
        0,
        state,
        attention_snapshot,
        None,
        None,
    )
    current = Qwen38TTNNDecoderLayerState(Qwen38TTNNLayerNamespace.BACKBONE, 0, 1, state, None)
    layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    layer.attention = attention
    layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    layer.layer_index = 0
    layer.ple = None
    layer.validate_commit_pair = lambda state, snapshot: None
    validation_error = RuntimeError("injected result validation failure")
    layer._validate_state = lambda state: (_ for _ in ()).throw(validation_error)
    monkeypatch.setattr(
        Qwen38TTNNGDNSnapshot,
        "deallocate",
        lambda self: (_ for _ in ()).throw(RuntimeError("injected cleanup failure")),
    )

    raised = None
    try:
        layer.commit_state(current, snapshot)
    except Qwen38TTNNLayerCleanupError as error:
        raised = error

    assert raised is not None
    assert raised.primary is validation_error
    assert raised.__cause__ is validation_error
    assert not snapshot.active
    assert snapshot.attention is attention_snapshot
