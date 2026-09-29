# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device execution, transaction, and ownership tests for fixed-five TTNN."""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import models.demos.blackhole.qwen38_flash_next.ttnn.fixed_five as fixed_module
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38TTNNBuiltTarget, Qwen38TTNNTargetComponents
from models.demos.blackhole.qwen38_flash_next.ttnn.fixed_five import (
    TARGET_ROWS,
    Qwen38TTNNFixedFiveTarget,
    _Qwen38FixedFiveTransaction,
    validate_fixed_five_static_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    BACKBONE_LAYERS,
    Qwen38TTNNDecoderLayerState,
    Qwen38TTNNLayerNamespace,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    Qwen38TTNNModelPoisonedError,
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelOutput,
    Qwen38TTNNTextModelState,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import Qwen38TTNNMoEResult
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_decode import (
    Qwen38FixedFiveTarget,
    Qwen38TargetCommitMode,
    Qwen38TargetVerification,
)

IDENTITY_KEY = "9" * 64


class _FakeTensor:
    _ids = itertools.count(1)

    def __init__(self, name: str, shape=(1, 1, 1, 640), *, layout=None, dtype=None) -> None:
        self.name = name
        self.shape = tuple(shape)
        self.layout = fixed_module.ttnn.TILE_LAYOUT if layout is None else layout
        self.dtype = fixed_module.ttnn.bfloat16 if dtype is None else dtype
        self._id = next(self._ids)
        self.released = False

    def tensor_id(self) -> int:
        return self._id


class _MeshContract:
    def __init__(self) -> None:
        self.validations: list[tuple[_FakeTensor, dict]] = []

    def validate_tensor(self, tensor, **kwargs) -> None:
        self.validations.append((tensor, kwargs))


def _model_state(owner: object, position: int) -> Qwen38TTNNTextModelState:
    return Qwen38TTNNTextModelState(
        position=position,
        layers=tuple(
            Qwen38TTNNDecoderLayerState(
                namespace=Qwen38TTNNLayerNamespace.BACKBONE,
                layer_index=index,
                position=position,
                attention=SimpleNamespace(position=position),
                ple=None,
            )
            for index in range(BACKBONE_LAYERS)
        ),
        _owner=owner,
    )


def _bare_model(position: int = 8) -> tuple[Qwen38TTNNTextModel, Qwen38TTNNTextModelState]:
    model = object.__new__(Qwen38TTNNTextModel)
    model._state_owner = object()
    model._poisoned_error = None
    model._active_snapshot = None
    model._runtime_owner = None
    model.mesh_contract = _MeshContract()
    model.mesh_device = object()
    model.layers = ()
    model._validate_state = mock.Mock(side_effect=lambda state: None)
    model._validate_hidden = mock.Mock(side_effect=lambda tensor, **kwargs: None)
    model._preflight_transaction = mock.Mock()
    model.claim_runtime_owner = mock.Mock()
    model.release_runtime_owner = mock.Mock()
    model.transfer_runtime_owner = mock.Mock()
    model.release_state = mock.Mock()
    state = _model_state(model._state_owner, position)
    return model, state


def _bare_adapter(model: Qwen38TTNNTextModel) -> Qwen38TTNNFixedFiveTarget:
    adapter = object.__new__(Qwen38TTNNFixedFiveTarget)
    adapter.model = model
    adapter.components = None
    adapter._identity_key = IDENTITY_KEY
    adapter._transaction_owner = object()
    adapter._active_transaction = None
    adapter._poisoned_error = None
    adapter._closed = False
    adapter.row5_moes = ()
    return adapter


def _install_tensor_ops(monkeypatch) -> list[str]:
    releases: list[str] = []

    def deallocate(tensor) -> None:
        if tensor.released:
            raise AssertionError(f"double release of {tensor.name}")
        tensor.released = True
        releases.append(tensor.name)

    def concat(tensors, *, dim, memory_config):
        del memory_config
        shape = list(tensors[0].shape)
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        return _FakeTensor("concat", shape)

    def to_layout(tensor, layout, *, memory_config, pad_value=None):
        del memory_config, pad_value
        return _FakeTensor(f"{tensor.name}-layout", tensor.shape, layout=layout, dtype=tensor.dtype)

    def tensor_slice(tensor, start, end, *, memory_config):
        del memory_config
        return _FakeTensor(
            f"{tensor.name}-slice-{start[2]}",
            tuple(end[index] - start[index] for index in range(len(start))),
            layout=tensor.layout,
            dtype=tensor.dtype,
        )

    monkeypatch.setattr(fixed_module.ttnn, "deallocate", deallocate)
    monkeypatch.setattr(fixed_module.ttnn, "concat", concat)
    monkeypatch.setattr(fixed_module.ttnn, "to_layout", to_layout)
    monkeypatch.setattr(fixed_module.ttnn, "slice", tensor_slice)
    return releases


def test_static_contract_implements_protocol_without_public_serial_verification() -> None:
    validate_fixed_five_static_contract()
    assert TARGET_ROWS == 5
    assert issubclass(Qwen38TTNNFixedFiveTarget, Qwen38FixedFiveTarget)
    assert not Qwen38TTNNFixedFiveTarget.__abstractmethods__

    verify_tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen38TTNNFixedFiveTarget.verify_five)))
    verify_attributes = {node.attr for node in ast.walk(verify_tree) if isinstance(node, ast.Attribute)}
    assert "forward_decode" not in verify_attributes
    assert "greedy_step" not in verify_attributes

    layer_tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen38TTNNFixedFiveTarget._forward_layer_five)))
    calls = [
        node for node in ast.walk(layer_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert sum(call.func.attr == "forward" for call in calls) == 1
    assert sum(call.func.attr == "layer" for call in calls) == 1
    assert sum(call.func.attr == "concat" for call in calls) == 1


class _Clone:
    instances: list["_Clone"] = []

    def __init__(
        self,
        mesh_device,
        mesh_contract,
        weights,
        *,
        tt_ccl,
        collective_topology,
        rows,
        synchronization_policy,
    ) -> None:
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        self.tt_ccl = tt_ccl
        self.collective_topology = collective_topology
        self.rows = rows
        self.synchronization_policy = synchronization_policy
        self.release_calls = 0
        self.actual_release_attempts = 0
        self.owned = True
        self.fail_release_once = False
        self.instances.append(self)

    def release_owned_buffers(self) -> None:
        self.release_calls += 1
        if not self.owned:
            return
        self.actual_release_attempts += 1
        if self.fail_release_once:
            self.fail_release_once = False
            raise RuntimeError("injected private-buffer release failure")
        self.owned = False


def test_constructor_builds_48_weight_borrowing_row5_clones_and_cleanup_is_retryable(expect_error, monkeypatch) -> None:
    _Clone.instances = []
    monkeypatch.setattr(fixed_module, "Qwen38TTNNMoE", _Clone)
    model, _ = _bare_model()
    streamer = object()
    tt_ccl = object()
    layers = tuple(
        SimpleNamespace(
            mlp=SimpleNamespace(
                weights=object(),
                tt_ccl=tt_ccl,
                collective_topology="linear",
                rows=1,
                synchronization_policy=object(),
                mesh_device=model.mesh_device,
                mesh_contract=model.mesh_contract,
            ),
            expert_streamer=streamer,
        )
        for _ in range(BACKBONE_LAYERS)
    )
    model.layers = layers
    components = Qwen38TTNNTargetComponents(
        identity=SimpleNamespace(key=IDENTITY_KEY),
        bf4_cache=None,
        io_cache=None,
        expert_streamer=streamer,
        model_io=None,
        layers=layers,
        final_mixer=None,
    )
    target = Qwen38TTNNBuiltTarget(model=model, components=components)

    adapter = Qwen38TTNNFixedFiveTarget(target)

    assert len(adapter.row5_moes) == BACKBONE_LAYERS
    assert all(clone.rows == TARGET_ROWS for clone in adapter.row5_moes)
    assert all(clone.weights is layer.mlp.weights for clone, layer in zip(adapter.row5_moes, layers))
    assert all(clone.mesh_device is model.mesh_device for clone in adapter.row5_moes)
    assert all(clone.mesh_contract is model.mesh_contract for clone in adapter.row5_moes)
    assert all(clone.tt_ccl is tt_ccl for clone in adapter.row5_moes)
    assert all(
        clone.synchronization_policy is layer.mlp.synchronization_policy
        for clone, layer in zip(adapter.row5_moes, layers)
    )
    assert model._fixed_five_adapter is adapter

    adapter.row5_moes[17].fail_release_once = True
    with expect_error(  # allow-pytest.raises: cleanup fault-injection assertion
        fixed_module.Qwen38FixedFiveCleanupError, match="layer 17"
    ):
        adapter.release_owned_buffers()
    assert not adapter._closed
    assert all(clone.release_calls == 1 for clone in adapter.row5_moes)

    adapter.release_owned_buffers()
    assert adapter._closed
    assert model._fixed_five_adapter is None
    assert adapter.row5_moes[17].release_calls == 2
    assert all(clone.release_calls == 2 for clone in adapter.row5_moes)
    assert all(
        clone.actual_release_attempts == (2 if index == 17 else 1) for index, clone in enumerate(adapter.row5_moes)
    )
    calls_after_close = tuple(clone.release_calls for clone in adapter.row5_moes)
    adapter.close()
    assert tuple(clone.release_calls for clone in adapter.row5_moes) == calls_after_close


class _GR:
    def __init__(self, prefix: str, log: list[tuple]) -> None:
        self.prefix = prefix
        self.log = log

    def read(self, residual):
        self.log.append((f"{self.prefix}_read", residual.name))
        return _FakeTensor(f"{self.prefix}-input"), SimpleNamespace(
            residual=_FakeTensor(f"{self.prefix}-residual"),
            injection=_FakeTensor(f"{self.prefix}-injection"),
        )

    def write(self, hidden, state):
        del state
        self.log.append((f"{self.prefix}_write", hidden.name))
        shape = (1, 4, 1, 640)
        return _FakeTensor(f"{self.prefix}-output", shape)


class _Attention:
    def __init__(self, kind: Qwen38TTNNLayerType, log: list[tuple]) -> None:
        self.kind = kind
        self.log = log

    def forward_decode(self, hidden, state, **kwargs):
        self.log.append(
            (
                "attention",
                state.position,
                kwargs.get("position"),
                kwargs.get("retain_input_state"),
                tuple(sorted(kwargs)),
            )
        )
        return SimpleNamespace(
            hidden_sharded=_FakeTensor("attention-output"),
            state=SimpleNamespace(position=state.position + 1),
            selection=object() if self.kind is Qwen38TTNNLayerType.QSA else None,
        )


class _Layer:
    def __init__(self, kind: Qwen38TTNNLayerType, log: list[tuple]) -> None:
        self.layer_index = 3 if kind is Qwen38TTNNLayerType.QSA else 0
        self.layer_type = kind
        self.namespace = Qwen38TTNNLayerNamespace.BACKBONE
        self.attention = _Attention(kind, log)
        self.attention_gr = _GR("attn", log)
        self.mlp_gr = _GR("mlp", log)
        self.expert_streamer = _Streamer(log)
        self.log = log

    def _apply_ple(self, residual, state, *, token_id):
        self.log.append(("ple", state.position, int(token_id.item())))
        return residual, None

    @staticmethod
    def _validate_block(tensor, *, label):
        assert tensor.shape == (1, 1, 1, 640), label

    @staticmethod
    def _validate_residual(tensor, *, label):
        assert tensor.shape == (1, 4, 1, 640), label

    @staticmethod
    def validate_state(state):
        assert state.attention.position == state.position


class _Streamer:
    def __init__(self, log: list[tuple]) -> None:
        self.log = log

    @contextmanager
    def layer(self, layer_index, *, namespace):
        self.log.append(("stream_enter", layer_index, namespace))
        yield (_FakeTensor("bf4-w01"), _FakeTensor("bf4-w2"))
        self.log.append(("stream_exit", layer_index, namespace))


class _Row5:
    def __init__(self, log: list[tuple]) -> None:
        self.log = log
        self.calls = 0

    def forward(self, hidden, w01, w2):
        self.calls += 1
        self.log.append(("moe", hidden.shape, w01.name, w2.name))
        return Qwen38TTNNMoEResult(_FakeTensor("moe-five", (1, 1, 5, 640)))


@pytest.mark.parametrize("kind", (Qwen38TTNNLayerType.GDN, Qwen38TTNNLayerType.QSA))
def test_each_layer_advances_attention_serially_but_streams_and_runs_moe_once(monkeypatch, kind) -> None:
    releases = _install_tensor_ops(monkeypatch)
    model, _ = _bare_model(position=12)
    log: list[tuple] = []
    layer = _Layer(kind, log)
    model.layers = (layer,)
    adapter = _bare_adapter(model)
    row5 = _Row5(log)
    adapter.row5_moes = (row5,)
    residuals = [_FakeTensor(f"input-{row}", (1, 4, 1, 640)) for row in range(TARGET_ROWS)]
    state = Qwen38TTNNDecoderLayerState(
        namespace=Qwen38TTNNLayerNamespace.BACKBONE,
        layer_index=layer.layer_index,
        position=12,
        attention=SimpleNamespace(position=12),
        ple=None,
    )
    tokens = tuple(torch.tensor([[100 + row]], dtype=torch.long) for row in range(TARGET_ROWS))
    ropes = [SimpleNamespace(cos=object(), sin=object(), block_start_cos=None, block_start_sin=None)] * TARGET_ROWS

    outputs, next_state = adapter._forward_layer_five(0, residuals, state, tokens, ropes)

    assert [entry[1] for entry in log if entry[0] == "attention"] == [12, 13, 14, 15, 16]
    if kind is Qwen38TTNNLayerType.QSA:
        attention = [entry for entry in log if entry[0] == "attention"]
        assert [entry[2] for entry in attention] == [12, 13, 14, 15, 16]
        assert [entry[3] for entry in attention] == [True, False, False, False, False]
    else:
        assert all(entry[2] is None and entry[3] is None for entry in log if entry[0] == "attention")
    assert [entry[0] for entry in log].count("stream_enter") == 1
    assert [entry[0] for entry in log].count("stream_exit") == 1
    assert row5.calls == 1
    assert [entry for entry in log if entry[0] == "moe"][0][1] == (1, 1, 5, 640)
    assert [entry[0] for entry in log].count("mlp_write") == TARGET_ROWS
    assert len(outputs) == TARGET_ROWS and all(output.shape == (1, 4, 1, 640) for output in outputs)
    assert next_state.position == 17
    assert "concat" in releases
    assert "moe-five" in releases
    expected_names = {
        "concat",
        "moe-five",
        *(f"moe-five-layout-slice-{row}-layout" for row in range(TARGET_ROWS)),
    }
    topology_checks = [
        (tensor.shape, kwargs) for tensor, kwargs in model.mesh_contract.validations if tensor.name in expected_names
    ]
    assert [shape for shape, _ in topology_checks].count((1, 1, 5, 640)) == 2
    assert [shape for shape, _ in topology_checks].count((1, 1, 1, 640)) == TARGET_ROWS
    assert all(kwargs["placement"] is fixed_module.TensorPlacement.HIDDEN_SHARDED for _, kwargs in topology_checks)
    assert all(kwargs["shard_dim"] == 3 for _, kwargs in topology_checks)


class _Rope:
    def __init__(self, position: int) -> None:
        self.position = position
        self.active = True

    def deallocate(self) -> None:
        assert self.active
        self.active = False


def test_verify_five_opens_one_model_transaction_and_visits_all_48_layers_without_ordinary_decode(
    monkeypatch,
) -> None:
    releases = _install_tensor_ops(monkeypatch)
    model, state = _bare_model(position=24)
    model.layers = tuple(SimpleNamespace() for _ in range(BACKBONE_LAYERS))
    snapshot = SimpleNamespace(position=24, active=True)
    model.snapshot_state = mock.Mock(return_value=snapshot)
    model._embed_residual = mock.Mock(
        side_effect=[_FakeTensor(f"embedding-{row}", (1, 4, 1, 640)) for row in range(TARGET_ROWS)]
    )
    ropes = [_Rope(24 + row) for row in range(TARGET_ROWS)]
    model.rope = SimpleNamespace(for_position=mock.Mock(side_effect=ropes))
    model.forward_decode = mock.Mock(side_effect=AssertionError("ordinary verification fallback"))
    adapter = _bare_adapter(model)
    layer_calls: list[tuple[int, tuple[int, ...], int, tuple[int, ...]]] = []

    def run_layer(self, layer_index, residuals, layer_state, host_tokens, rope_inputs):
        del self
        layer_calls.append(
            (
                layer_index,
                tuple(int(token.item()) for token in host_tokens),
                layer_state.position,
                tuple(rope.position for rope in rope_inputs),
            )
        )
        next_residuals = [_FakeTensor(f"layer-{layer_index}-row-{row}", (1, 4, 1, 640)) for row in range(TARGET_ROWS)]
        next_state = Qwen38TTNNDecoderLayerState(
            namespace=Qwen38TTNNLayerNamespace.BACKBONE,
            layer_index=layer_index,
            position=layer_state.position + TARGET_ROWS,
            attention=SimpleNamespace(position=layer_state.position + TARGET_ROWS),
            ple=None,
        )
        return next_residuals, next_state

    adapter._forward_layer_five = run_layer.__get__(adapter, Qwen38TTNNFixedFiveTarget)
    adapter._resolve_five_logits = mock.Mock(return_value=(201, 202, 203, 204, 205))

    verification = adapter.verify_five((101, 102, 103, 104, 105), state, base_position=24)

    model.snapshot_state.assert_called_once_with(state)
    model.forward_decode.assert_not_called()
    assert [call[0] for call in layer_calls] == list(range(BACKBONE_LAYERS))
    assert all(call[1] == (101, 102, 103, 104, 105) for call in layer_calls)
    assert all(call[2] == 24 for call in layer_calls)
    assert all(call[3] == (24, 25, 26, 27, 28) for call in layer_calls)
    assert verification.target_token_ids == (201, 202, 203, 204, 205)
    assert verification.speculative_position == 29
    assert verification.transaction is adapter._active_transaction
    assert verification.transaction.branch_state.position == 29
    assert all(not rope.active for rope in ropes)

    model.restore_state = mock.Mock(return_value=state)
    rollback = adapter.abort(verification)
    assert rollback.state is state and rollback.position == 24
    assert set(releases) == {f"layer-47-row-{row}" for row in range(TARGET_ROWS)}


def test_five_logits_resolve_only_small_greedy_candidates_and_keep_roots_live(monkeypatch) -> None:
    releases = _install_tensor_ops(monkeypatch)
    model, _ = _bare_model()
    adapter = _bare_adapter(model)
    roots = [_FakeTensor(f"root-{row}", (1, 4, 1, 640)) for row in range(TARGET_ROWS)]
    model.final_mixer = mock.Mock(side_effect=[_FakeTensor(f"hidden-{row}") for row in range(TARGET_ROWS)])

    class _Head:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, hidden):
            del hidden
            row = self.calls
            self.calls += 1
            return SimpleNamespace(tensor=_FakeTensor(f"logits-{row}", (1, 1, 1, 62080)))

        @staticmethod
        def greedy_token(logits):
            row = int(logits.tensor.name.rsplit("-", 1)[1])
            return torch.tensor([[[301 + row]]], dtype=torch.long)

    head = _Head()
    model.model_io = SimpleNamespace(lm_head=head)

    predictions = adapter._resolve_five_logits(roots)

    assert predictions == (301, 302, 303, 304, 305)
    assert head.calls == TARGET_ROWS
    assert set(releases) == {
        *(f"hidden-{row}" for row in range(TARGET_ROWS)),
        *(f"logits-{row}" for row in range(TARGET_ROWS)),
    }
    assert all(not root.released for root in roots)


def _verification_fixture(position: int = 20):
    model, state = _bare_model(position)
    adapter = _bare_adapter(model)
    roots = [_FakeTensor(f"root-{index}", (1, 4, 1, 640)) for index in range(TARGET_ROWS)]
    snapshot = SimpleNamespace(active=True, position=position)
    branch = _model_state(model._state_owner, position + TARGET_ROWS)
    transaction = _Qwen38FixedFiveTransaction(
        owner=adapter._transaction_owner,
        base_position=position,
        input_token_ids=(10, 11, 12, 13, 14),
        snapshot=snapshot,
        branch_state=branch,
        roots=list(roots),
    )
    verification = Qwen38TargetVerification(
        base_position=position,
        input_token_ids=transaction.input_token_ids,
        target_token_ids=(21, 22, 23, 24, 25),
        target_hyper_residuals=tuple(roots),
        speculative_position=position + TARGET_ROWS,
        transaction=transaction,
    )
    adapter._active_transaction = transaction
    return adapter, model, state, branch, transaction, verification, roots


@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_prefix_commit_restores_replays_exact_inputs_and_transfers_ordered_root_prefix(monkeypatch, count) -> None:
    releases = _install_tensor_ops(monkeypatch)
    adapter, model, state, branch, transaction, verification, roots = _verification_fixture()
    restored = state
    committed_state = _model_state(model._state_owner, state.position + count)
    model.restore_state = mock.Mock(return_value=restored)
    model.commit_state = mock.Mock()
    replay = mock.Mock(return_value=committed_state)
    adapter._replay_prefix = replay

    result = adapter.commit_prefix(verification, count)

    model.restore_state.assert_called_once_with(branch, transaction.snapshot)
    model.commit_state.assert_not_called()
    replay.assert_called_once_with(restored, transaction.input_token_ids[:count])
    assert result.state is committed_state
    assert result.position == state.position + count
    assert result.committed_hyper_residuals == tuple(roots[:count])
    assert all(actual is expected for actual, expected in zip(result.committed_hyper_residuals, roots[:count]))
    assert result.target_hyper_residual is roots[count - 1]
    assert all(root.released == (index >= count) for index, root in enumerate(roots))
    assert result.mode is Qwen38TargetCommitMode.RESTORE_REPLAY
    assert result.replayed_input_token_ids == transaction.input_token_ids[:count]
    assert set(releases) == {f"root-{index}" for index in range(count, TARGET_ROWS)}
    assert transaction.roots[:count] == [None] * count
    assert not transaction.active and adapter._active_transaction is None


def test_full_commit_keeps_live_branch_and_never_replays(monkeypatch) -> None:
    _install_tensor_ops(monkeypatch)
    adapter, model, _, branch, transaction, verification, roots = _verification_fixture()
    model.commit_state = mock.Mock(return_value=branch)
    model.restore_state = mock.Mock()
    adapter._replay_prefix = mock.Mock()

    result = adapter.commit_prefix(verification, TARGET_ROWS)

    model.commit_state.assert_called_once_with(branch, transaction.snapshot)
    model.restore_state.assert_not_called()
    adapter._replay_prefix.assert_not_called()
    assert result.state is branch
    assert result.committed_hyper_residuals == tuple(roots)
    assert all(actual is expected for actual, expected in zip(result.committed_hyper_residuals, roots))
    assert result.target_hyper_residual is roots[-1]
    assert all(not root.released for root in roots)
    assert transaction.roots == [None] * TARGET_ROWS
    assert result.mode is Qwen38TargetCommitMode.PREFIX_TRANSACTION
    assert result.replayed_input_token_ids == ()


def test_abort_restores_base_and_releases_all_five_roots(monkeypatch) -> None:
    releases = _install_tensor_ops(monkeypatch)
    adapter, model, state, branch, transaction, verification, roots = _verification_fixture()
    model.restore_state = mock.Mock(return_value=state)

    result = adapter.abort(verification)

    model.restore_state.assert_called_once_with(branch, transaction.snapshot)
    assert result.state is state and result.position == state.position
    assert all(root.released for root in roots)
    assert set(releases) == {f"root-{index}" for index in range(TARGET_ROWS)}
    assert not transaction.active and adapter._active_transaction is None


def test_replay_consumes_only_committed_prefix_and_releases_every_duplicate_output(monkeypatch) -> None:
    _install_tensor_ops(monkeypatch)
    model, state = _bare_model(position=30)
    adapter = _bare_adapter(model)
    calls: list[tuple[int, dict]] = []

    def forward(token, current, **kwargs):
        calls.append((token, kwargs))
        next_state = _model_state(model._state_owner, current.position + 1)
        return Qwen38TTNNTextModelOutput(
            input_token_id=token,
            position=current.position,
            hyper_residual_sharded=None,
            hidden_sharded=None,
            logits=None,
            greedy_token=None,
            state=next_state,
            layer_aux=(),
        )

    model.forward_decode = forward
    result = adapter._replay_prefix(state, (41, 42, 43))

    assert result.position == 33
    assert [token for token, _ in calls] == [41, 42, 43]
    assert all(
        kwargs
        == {
            "return_logits": False,
            "resolve_greedy": False,
            "retain_hidden": False,
            "retain_hyper_residual": False,
            "retain_input_state": False,
            "return_routing": False,
        }
        for _, kwargs in calls
    )


def test_invalid_commit_counts_and_foreign_transactions_fail_before_mutation(expect_error, monkeypatch) -> None:
    _install_tensor_ops(monkeypatch)
    adapter, model, _, _, transaction, verification, _ = _verification_fixture()
    model.restore_state = mock.Mock()
    model.commit_state = mock.Mock()

    for count in (False, 0, 6, 1.0):
        with expect_error(  # allow-pytest.raises: pure invalid-input assertion
            ValueError, match="committed_input_count"
        ):
            adapter.commit_prefix(verification, count)
    foreign = Qwen38TargetVerification(
        base_position=verification.base_position,
        input_token_ids=verification.input_token_ids,
        target_token_ids=verification.target_token_ids,
        target_hyper_residuals=verification.target_hyper_residuals,
        speculative_position=verification.speculative_position,
        transaction=object(),
    )
    with expect_error(ValueError, match="another adapter"):  # allow-pytest.raises: pure ownership assertion
        adapter.abort(foreign)
    model.restore_state.assert_not_called()
    model.commit_state.assert_not_called()
    assert transaction.active


def test_commit_rejects_aliasing_prefix_and_suffix_roots_before_state_mutation(expect_error, monkeypatch) -> None:
    _install_tensor_ops(monkeypatch)
    adapter, model, _, _, transaction, verification, roots = _verification_fixture()
    transaction.roots[-1] = roots[0]
    aliased = Qwen38TargetVerification(
        base_position=verification.base_position,
        input_token_ids=verification.input_token_ids,
        target_token_ids=verification.target_token_ids,
        target_hyper_residuals=tuple(transaction.roots),  # type: ignore[arg-type]
        speculative_position=verification.speculative_position,
        transaction=transaction,
    )
    model.restore_state = mock.Mock()
    model.commit_state = mock.Mock()

    with expect_error(RuntimeError, match="five distinct live buffers"):  # allow-pytest.raises: preflight guard
        adapter.commit_prefix(aliased, 2)

    model.restore_state.assert_not_called()
    model.commit_state.assert_not_called()
    assert transaction.active and adapter._active_transaction is transaction
    assert all(not root.released for root in roots)


def test_failure_after_transaction_mutation_poison_is_sticky(monkeypatch) -> None:
    _install_tensor_ops(monkeypatch)
    adapter, model, _, branch, transaction, verification, _ = _verification_fixture()
    failure = RuntimeError("injected restore mutation failure")
    model.restore_state = mock.Mock(side_effect=failure)

    def mark(operation, processed_layers, cause):
        model._poisoned_error = Qwen38TTNNModelPoisonedError(operation, processed_layers, cause)
        raise model._poisoned_error

    model._mark_poisoned = mark

    with pytest.raises(  # allow-pytest.raises: poison identity is asserted below
        Qwen38TTNNModelPoisonedError, match="fixed_five.commit_prefix"
    ) as raised:
        adapter.commit_prefix(verification, 2)

    assert adapter.poisoned
    assert adapter.poisoned_error is raised.value
    assert not transaction.active and adapter._active_transaction is None
    model.restore_state.assert_called_once_with(branch, transaction.snapshot)
    with pytest.raises(Qwen38TTNNModelPoisonedError) as repeated:  # allow-pytest.raises: sticky identity assertion
        adapter.state_position(_model_state(model._state_owner, 20))
    assert repeated.value is raised.value


def test_prefix_replay_failure_is_terminal_and_never_releases_any_root_twice(monkeypatch) -> None:
    releases = _install_tensor_ops(monkeypatch)
    adapter, model, state, _, transaction, verification, roots = _verification_fixture()
    model.restore_state = mock.Mock(return_value=state)
    adapter._replay_prefix = mock.Mock(side_effect=RuntimeError("injected ordinary replay failure"))

    def mark(operation, processed_layers, cause):
        model._poisoned_error = Qwen38TTNNModelPoisonedError(operation, processed_layers, cause)
        raise model._poisoned_error

    model._mark_poisoned = mark

    with pytest.raises(  # allow-pytest.raises: terminal replay fault assertion
        Qwen38TTNNModelPoisonedError, match="fixed_five.commit_prefix"
    ) as raised:
        adapter.commit_prefix(verification, 2)

    assert adapter.poisoned_error is raised.value
    assert not transaction.active and adapter._active_transaction is None
    assert all(not root.released for root in roots)
    assert transaction.roots == roots
    releases_after_failure = tuple(releases)
    with pytest.raises(Qwen38TTNNModelPoisonedError) as repeated:  # allow-pytest.raises: sticky poison assertion
        adapter.commit_prefix(verification, 2)
    assert repeated.value is raised.value
    assert tuple(releases) == releases_after_failure


def test_suffix_cleanup_failure_retains_committed_prefix_and_never_double_releases(monkeypatch) -> None:
    adapter, model, state, _, transaction, verification, roots = _verification_fixture()
    model.restore_state = mock.Mock(return_value=state)
    adapter._replay_prefix = mock.Mock(return_value=_model_state(model._state_owner, state.position + 2))
    attempts: list[str] = []

    def deallocate(tensor) -> None:
        attempts.append(tensor.name)
        if tensor.name == "root-3":
            raise RuntimeError("injected suffix cleanup failure")
        if tensor.released:
            raise AssertionError(f"double release of {tensor.name}")
        tensor.released = True

    monkeypatch.setattr(fixed_module.ttnn, "deallocate", deallocate)

    def mark(operation, processed_layers, cause):
        model._poisoned_error = Qwen38TTNNModelPoisonedError(operation, processed_layers, cause)
        raise model._poisoned_error

    model._mark_poisoned = mark

    with pytest.raises(  # allow-pytest.raises: terminal suffix-cleanup assertion
        Qwen38TTNNModelPoisonedError, match="fixed_five.commit_prefix"
    ) as raised:
        adapter.commit_prefix(verification, 2)

    assert adapter.poisoned_error is raised.value
    assert attempts == ["root-2", "root-3", "root-4"]
    assert all(not root.released for root in roots[:2])
    assert roots[2].released and not roots[3].released and roots[4].released
    assert transaction.roots[:2] == roots[:2]
    attempts_after_failure = tuple(attempts)
    with pytest.raises(Qwen38TTNNModelPoisonedError) as repeated:  # allow-pytest.raises: sticky poison assertion
        adapter.commit_prefix(verification, 2)
    assert repeated.value is raised.value
    assert tuple(attempts) == attempts_after_failure


def test_abort_cleanup_failure_attempts_every_root_once_then_becomes_terminal(monkeypatch) -> None:
    adapter, model, state, _, transaction, verification, roots = _verification_fixture()
    model.restore_state = mock.Mock(return_value=state)
    attempts: list[str] = []

    def deallocate(tensor) -> None:
        attempts.append(tensor.name)
        if tensor.name == "root-2":
            raise RuntimeError("injected root cleanup failure")
        if tensor.released:
            raise AssertionError(f"double release of {tensor.name}")
        tensor.released = True

    monkeypatch.setattr(fixed_module.ttnn, "deallocate", deallocate)

    def mark(operation, processed_layers, cause):
        model._poisoned_error = Qwen38TTNNModelPoisonedError(operation, processed_layers, cause)
        raise model._poisoned_error

    model._mark_poisoned = mark

    with pytest.raises(  # allow-pytest.raises: terminal abort-cleanup assertion
        Qwen38TTNNModelPoisonedError, match="fixed_five.abort"
    ) as raised:
        adapter.abort(verification)

    assert adapter.poisoned_error is raised.value
    assert not transaction.active and adapter._active_transaction is None
    assert attempts == [f"root-{index}" for index in range(TARGET_ROWS)]
    assert not roots[2].released
    assert all(root.released for index, root in enumerate(roots) if index != 2)
    attempts_after_failure = tuple(attempts)
    with pytest.raises(Qwen38TTNNModelPoisonedError) as repeated:  # allow-pytest.raises: sticky poison assertion
        adapter.abort(verification)
    assert repeated.value is raised.value
    assert tuple(attempts) == attempts_after_failure
