# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-resident five-position target verification for Qwen3.8 TTNN.

The public ordinary target deliberately owns only one-token decode.  This
adapter adds the different execution boundary required by four-step MTP:
five consecutive target positions are advanced transactionally while each
layer's routed and shared MoE work is executed once at height five.

Attention remains serial within a layer.  This is required for GDN recurrent
and convolution state, QSA cache/tail/index state, and the layer-1 PLE token
history.  After the five attention GR reads, their MLP inputs are concatenated
and passed through one ``rows=5`` MoE which borrows the ordinary layer's exact
resident router/shared weights and streams the layer's BF4_B routed weights
once.  Its five outputs complete the MLP GR writes in token order.

The model-wide transaction is retained until the controller selects a prefix.
All five positions can commit the live branch directly.  Prefixes one through
four restore the real GDN/QSA/PLE snapshot and deterministically replay only
the committed inputs through ordinary decode.  The exact ordered prefix of
originally verified hyper residuals remains transferred to MTP by identity;
only the uncommitted suffix is released.  Deterministic replay retains no
duplicate activations.

There is no serialized public verification fallback in :meth:`verify_five`.
Any exception after the target snapshot is active poisons the underlying model
and this adapter.  Recovery is process teardown under the existing lease, not
a reset or an attempted continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38TTNNBuiltTarget, Qwen38TTNNTargetComponents
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    BACKBONE_LAYERS,
    BLOCK_LOCAL_SHAPE,
    RESIDUAL_LOCAL_SHAPE,
    Qwen38TTNNDecoderLayerState,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    MAX_CONTEXT,
    VOCAB_SIZE,
    Qwen38TTNNModelPoisonedError,
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelOutput,
    Qwen38TTNNTextModelSnapshot,
    Qwen38TTNNTextModelState,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import Qwen38TTNNMoE, Qwen38TTNNMoEResult
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_decode import (
    VERIFY_POSITIONS,
    Qwen38FixedFiveTarget,
    Qwen38StateRollback,
    Qwen38TargetCommit,
    Qwen38TargetCommitMode,
    Qwen38TargetVerification,
)

TARGET_ROWS = 5


class Qwen38FixedFiveCleanupError(RuntimeError):
    """One or more independent adapter-owned releases failed."""

    def __init__(self, label: str, errors: list[BaseException], *, primary: BaseException | None = None) -> None:
        self.label = label
        self.errors = tuple(errors)
        self.primary = primary
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{label} cleanup failed for {len(errors)} resource(s): {detail}")


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _tensor_key(tensor: Any) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _release_slots(
    slots: list[Any | None],
    indices: tuple[int, ...] | range | None = None,
    *,
    label: str,
    primary: BaseException | None = None,
) -> None:
    """Release selected alias groups and clear successful slots immediately."""

    selected = tuple(range(len(slots))) if indices is None else tuple(dict.fromkeys(indices))
    if any(index < 0 or index >= len(slots) for index in selected):
        raise IndexError(f"{label} selected an out-of-range tensor slot")
    groups: dict[tuple[str, int], list[int]] = {}
    for index in selected:
        tensor = slots[index]
        if tensor is not None:
            groups.setdefault(_tensor_key(tensor), []).append(index)
    errors: list[BaseException] = []
    for key, group in groups.items():
        tensor = slots[group[0]]
        aliases = [
            index for index, candidate in enumerate(slots) if candidate is not None and _tensor_key(candidate) == key
        ]
        try:
            ttnn.deallocate(tensor)
        except BaseException as error:
            errors.append(error)
        else:
            for index in aliases:
                slots[index] = None
    if errors:
        raise Qwen38FixedFiveCleanupError(label, errors, primary=primary) from primary


def _release_rope_inputs(rope_inputs: list[Any], *, primary: BaseException | None = None) -> None:
    errors: list[BaseException] = []
    for rope in rope_inputs:
        if not getattr(rope, "active", True):
            continue
        try:
            rope.deallocate()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise Qwen38FixedFiveCleanupError("fixed-five RoPE inputs", errors, primary=primary) from primary


def _normalize_token_ids(values: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    if not isinstance(values, tuple) or len(values) != TARGET_ROWS:
        raise TypeError("fixed-five target requires an exact tuple of five token IDs")
    normalized: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < VOCAB_SIZE:
            raise ValueError(f"fixed-five token {index} must be an integer in [0,{VOCAB_SIZE}), got {value!r}")
        normalized.append(value)
    return tuple(normalized)  # type: ignore[return-value]


def _require_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= TARGET_ROWS:
        raise ValueError(f"committed_input_count must be an integer in [1,{TARGET_ROWS}], got {value!r}")
    return value


def _require_identity_key(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"fixed-five target identity must be lowercase 64-hex, got {value!r}")
    return value


@dataclass
class _Qwen38FixedFiveTransaction:
    """Opaque, one-shot owner behind :class:`Qwen38TargetVerification`."""

    owner: object = field(repr=False)
    base_position: int
    input_token_ids: tuple[int, int, int, int, int]
    snapshot: Qwen38TTNNTextModelSnapshot = field(repr=False)
    branch_state: Qwen38TTNNTextModelState | None = field(default=None, repr=False)
    roots: list[Any | None] = field(default_factory=lambda: [None] * TARGET_ROWS, repr=False)
    active: bool = True


class Qwen38TTNNFixedFiveTarget(Qwen38FixedFiveTarget):
    """Concrete fixed-five verifier for one exact built 48-layer target."""

    def __init__(self, built_target: Qwen38TTNNBuiltTarget) -> None:
        if type(built_target) is not Qwen38TTNNBuiltTarget:
            raise TypeError("fixed-five target requires the exact Qwen38TTNNBuiltTarget owner")
        if type(built_target.model) is not Qwen38TTNNTextModel:
            raise TypeError("fixed-five target requires the exact Qwen38TTNNTextModel")
        if type(built_target.components) is not Qwen38TTNNTargetComponents:
            raise TypeError("fixed-five target requires exact inspectable target components")
        model = built_target.model
        components = built_target.components
        if model.layers is not components.layers or len(model.layers) != BACKBONE_LAYERS:
            raise ValueError("fixed-five target layers are not the exact 48-layer built-target tuple")
        if any(layer.expert_streamer is not components.expert_streamer for layer in model.layers):
            raise ValueError("fixed-five target requires the built target's single BF4_B streamer")
        for layer_index, layer in enumerate(model.layers):
            ordinary_moe = layer.mlp
            if getattr(ordinary_moe, "rows", None) != 1:
                raise ValueError(f"target layer {layer_index} does not own an exact rows=1 ordinary MoE")
            if (
                ordinary_moe.mesh_device is not model.mesh_device
                or ordinary_moe.mesh_contract is not model.mesh_contract
            ):
                raise ValueError(f"target layer {layer_index} ordinary MoE belongs to another live mesh")
        if getattr(model, "_fixed_five_adapter", None) is not None:
            raise RuntimeError("the TTNN target already has a fixed-five adapter")

        self.model = model
        self.components = components
        self._identity_key = _require_identity_key(components.identity.key)
        self._transaction_owner = object()
        self._active_transaction: _Qwen38FixedFiveTransaction | None = None
        self._poisoned_error: BaseException | None = None
        self._closed = False
        self.row5_moes: tuple[Qwen38TTNNMoE, ...] = ()

        row5: list[Qwen38TTNNMoE] = []
        try:
            for layer in model.layers:
                clone = Qwen38TTNNMoE(
                    model.mesh_device,
                    model.mesh_contract,
                    layer.mlp.weights,
                    tt_ccl=layer.mlp.tt_ccl,
                    collective_topology=layer.mlp.collective_topology,
                    rows=TARGET_ROWS,
                    synchronization_policy=layer.mlp.synchronization_policy,
                )
                if (
                    clone.weights is not layer.mlp.weights
                    or clone.rows != TARGET_ROWS
                    or clone.mesh_device is not model.mesh_device
                    or clone.mesh_contract is not model.mesh_contract
                    or clone.tt_ccl is not layer.mlp.tt_ccl
                    or clone.collective_topology != layer.mlp.collective_topology
                    or clone.synchronization_policy is not layer.mlp.synchronization_policy
                    or layer.expert_streamer is not components.expert_streamer
                ):
                    raise RuntimeError(
                        "rows=5 MoE did not preserve the ordinary layer's weights, mesh, topology, and streamer"
                    )
                row5.append(clone)
        except BaseException as error:
            cleanup_errors: list[BaseException] = []
            for clone in reversed(row5):
                try:
                    clone.release_owned_buffers()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise Qwen38FixedFiveCleanupError(
                    "partial rows=5 MoE construction", cleanup_errors, primary=error
                ) from error
            raise
        self.row5_moes = tuple(row5)
        model._fixed_five_adapter = self

    @property
    def identity_key(self) -> str:
        return self._identity_key

    @property
    def allocated_context(self) -> int:
        return self.model.allocated_context

    @property
    def poisoned(self) -> bool:
        return self._poisoned_error is not None or self.model.poisoned

    @property
    def poisoned_error(self) -> BaseException | None:
        return self._poisoned_error or self.model.poisoned_error

    def _require_healthy(self) -> None:
        error = self.poisoned_error
        if error is not None:
            raise error
        if self._closed:
            raise RuntimeError("fixed-five target adapter is closed")

    def _poison(self, operation: str, processed_layers: int, cause: BaseException) -> NoReturn:
        transaction = self._active_transaction
        if transaction is not None:
            transaction.active = False
            self._active_transaction = None
        if self._poisoned_error is None:
            try:
                self.model._mark_poisoned(f"fixed_five.{operation}", processed_layers, cause)
            except Qwen38TTNNModelPoisonedError as error:
                self._poisoned_error = error
            except BaseException as error:
                self._poisoned_error = error
        raise self._poisoned_error from cause

    def claim_runtime_owner(self, owner: object) -> None:
        self._require_healthy()
        self.model.claim_runtime_owner(owner)

    def release_runtime_owner(self, owner: object) -> None:
        self._require_healthy()
        if self._active_transaction is not None:
            raise RuntimeError("cannot release runtime ownership during fixed-five verification")
        self.model.release_runtime_owner(owner)

    def transfer_runtime_owner(self, current_owner: object, next_owner: object) -> None:
        self._require_healthy()
        if self._active_transaction is not None:
            raise RuntimeError("cannot transfer runtime ownership during fixed-five verification")
        self.model.transfer_runtime_owner(current_owner, next_owner)

    def state_position(self, state: Any) -> int:
        self._require_healthy()
        self.model._validate_state(state)
        return state.position

    def release_state(self, state: Any) -> None:
        self._require_healthy()
        if self._active_transaction is not None:
            raise RuntimeError("cannot release target state during fixed-five verification")
        self.model.release_state(state)

    def release_owned_buffers(self) -> None:
        """Release only the 48 rows=5 mapping/combine buffer pairs.

        Resident router/shared weights and the ordinary rows=1 MoE buffers are
        borrowed and are never released here.  Failed clones remain reachable
        so a healthy caller can retry only their still-owned buffers.
        """

        if self._closed:
            return
        self._require_healthy()
        if self._active_transaction is not None:
            raise RuntimeError("cannot release rows=5 buffers during active verification")
        errors: list[BaseException] = []
        for layer_index, clone in reversed(tuple(enumerate(self.row5_moes))):
            try:
                clone.release_owned_buffers()
            except BaseException as error:
                wrapped = RuntimeError(f"rows=5 MoE layer {layer_index}: {type(error).__name__}: {error}")
                wrapped.__cause__ = error
                errors.append(wrapped)
        if errors:
            raise Qwen38FixedFiveCleanupError("rows=5 MoE buffers", errors)
        self._closed = True
        if getattr(self.model, "_fixed_five_adapter", None) is self:
            self.model._fixed_five_adapter = None

    close = release_owned_buffers

    def _validate_residual(self, tensor: Any, *, label: str) -> None:
        if _shape(tensor) != RESIDUAL_LOCAL_SHAPE:
            raise RuntimeError(f"{label} must have local shape {RESIDUAL_LOCAL_SHAPE}, got {_shape(tensor)}")
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{label} must be TILE BFLOAT16")
        self.model.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_block(self, tensor: Any, *, label: str) -> None:
        if _shape(tensor) != BLOCK_LOCAL_SHAPE:
            raise RuntimeError(f"{label} must have local shape {BLOCK_LOCAL_SHAPE}, got {_shape(tensor)}")
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{label} must be TILE BFLOAT16")
        self.model.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_five_blocks(self, tensor: Any, *, label: str) -> None:
        expected = (1, 1, TARGET_ROWS, BLOCK_LOCAL_SHAPE[-1])
        if _shape(tensor) != expected or tensor.dtype != ttnn.bfloat16:
            raise RuntimeError(f"{label} must be BFLOAT16 local shape {expected}, got {_shape(tensor)}")
        self.model.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _serial_attention_and_mlp_read(
        self,
        layer: Any,
        residual: Any,
        state: Qwen38TTNNDecoderLayerState,
        *,
        host_token: torch.Tensor,
        rope: Any,
        retain_input_state: bool,
    ) -> tuple[Any, Any, Qwen38TTNNDecoderLayerState]:
        """Advance one row through PLE/attention and stop at the MLP GR read."""

        residual, ple_state = layer._apply_ple(residual, state, token_id=host_token)
        attention_input, attention_gr_state = layer.attention_gr.read(residual)
        layer._validate_block(attention_input, label="fixed-five attention GR read")
        if layer.layer_type is Qwen38TTNNLayerType.GDN:
            attention_result = layer.attention.forward_decode(attention_input, state.attention)
        else:
            attention_result = layer.attention.forward_decode(
                attention_input,
                state.attention,
                cos=rope.cos,
                sin=rope.sin,
                block_start_cos=rope.block_start_cos,
                block_start_sin=rope.block_start_sin,
                position=state.position,
                reuse_selection=None,
                retain_input_state=retain_input_state,
            )
        attention_slots = [attention_input]
        _release_slots(attention_slots, label="fixed-five attention input")
        layer._validate_block(attention_result.hidden_sharded, label="fixed-five attention output")
        residual = layer.attention_gr.write(attention_result.hidden_sharded, attention_gr_state)
        attention_write_slots = [
            attention_result.hidden_sharded,
            attention_gr_state.residual,
            attention_gr_state.injection,
        ]
        _release_slots(attention_write_slots, label="fixed-five attention GR write")
        layer._validate_residual(residual, label="fixed-five post-attention residual")

        mlp_input, mlp_gr_state = layer.mlp_gr.read(residual)
        layer._validate_block(mlp_input, label="fixed-five MLP GR read")
        next_state = Qwen38TTNNDecoderLayerState(
            namespace=layer.namespace,
            layer_index=layer.layer_index,
            position=state.position + 1,
            attention=attention_result.state,
            ple=ple_state,
        )
        layer.validate_state(next_state)
        return mlp_input, mlp_gr_state, next_state

    def _split_five_moe_rows(self, combined: Any) -> list[Any]:
        """Materialize independent TILE rows before releasing the height-five tensor."""

        self._validate_five_blocks(combined, label="fixed-five MoE output")
        row_major = ttnn.to_layout(combined, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        combined_slots = [combined]
        _release_slots(combined_slots, label="fixed-five tiled MoE output")
        rows: list[Any] = []
        try:
            for row in range(TARGET_ROWS):
                row_rm = ttnn.slice(
                    row_major,
                    (0, 0, row, 0),
                    (1, 1, row + 1, BLOCK_LOCAL_SHAPE[-1]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                tiled = ttnn.to_layout(
                    row_rm,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    pad_value=0.0,
                )
                rm_slot = [row_rm]
                _release_slots(rm_slot, label=f"fixed-five MoE RM row {row}")
                self._validate_block(tiled, label=f"fixed-five MoE row {row}")
                rows.append(tiled)
        except BaseException as error:
            slots = [row_major, *rows]
            _release_slots(slots, label="partial fixed-five MoE row split", primary=error)
            raise
        rm_slot = [row_major]
        _release_slots(rm_slot, label="fixed-five MoE RM batch")
        return rows

    def _forward_layer_five(
        self,
        layer_index: int,
        residuals: list[Any],
        state: Qwen38TTNNDecoderLayerState,
        host_tokens: tuple[torch.Tensor, ...],
        rope_inputs: list[Any],
    ) -> tuple[list[Any], Qwen38TTNNDecoderLayerState]:
        layer = self.model.layers[layer_index]
        row5_moe = self.row5_moes[layer_index]
        mlp_inputs: list[Any] = []
        mlp_write_states: list[Any] = []
        serial_state = state
        for row in range(TARGET_ROWS):
            mlp_input, write_state, serial_state = self._serial_attention_and_mlp_read(
                layer,
                residuals[row],
                serial_state,
                host_token=host_tokens[row],
                rope=rope_inputs[row],
                retain_input_state=row == 0,
            )
            mlp_inputs.append(mlp_input)
            mlp_write_states.append(write_state)

        combined = ttnn.concat(mlp_inputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._validate_five_blocks(combined, label="fixed-five concatenated MLP input")
        mlp_input_slots = list(mlp_inputs)
        _release_slots(mlp_input_slots, label="fixed-five serial MLP inputs")
        with layer.expert_streamer.layer(layer.layer_index, namespace=layer.namespace.value) as packed_experts:
            if not isinstance(packed_experts, tuple) or len(packed_experts) != 2:
                raise RuntimeError("BF4 streamer must yield exactly (packed_w0_w1, packed_w2)")
            result = row5_moe.forward(combined, packed_experts[0], packed_experts[1])
        combined_slot = [combined]
        _release_slots(combined_slot, label="fixed-five concatenated MLP input")
        if type(result) is not Qwen38TTNNMoEResult or result.routing is not None:
            raise RuntimeError("rows=5 MoE must return one hidden batch without retained routing")
        moe_rows = self._split_five_moe_rows(result.hidden_sharded)

        next_residuals: list[Any] = []
        for row, (moe_row, write_state) in enumerate(zip(moe_rows, mlp_write_states)):
            layer._validate_block(moe_row, label=f"fixed-five routed/shared MoE row {row}")
            residual = layer.mlp_gr.write(moe_row, write_state)
            write_slots = [moe_row, write_state.residual, write_state.injection]
            _release_slots(write_slots, label=f"fixed-five MLP GR write row {row}")
            layer._validate_residual(residual, label=f"fixed-five layer {layer_index} row {row}")
            next_residuals.append(residual)
        if serial_state.position != state.position + TARGET_ROWS:
            raise RuntimeError(
                f"fixed-five layer {layer_index} advanced to {serial_state.position}, "
                f"expected {state.position + TARGET_ROWS}"
            )
        return next_residuals, serial_state

    def _resolve_five_logits(self, roots: list[Any]) -> tuple[int, int, int, int, int]:
        target_tokens: list[int] = []
        for row, residual in enumerate(roots):
            owned: list[Any | None] = [None, None]
            try:
                hidden = self.model.final_mixer(residual)
                owned[0] = hidden
                self.model._validate_hidden(hidden, label=f"fixed-five terminal row {row}")
                logits = self.model.model_io.lm_head(hidden)
                owned[1] = logits.tensor
                greedy = self.model.model_io.lm_head.greedy_token(logits)
                if greedy.device.type != "cpu" or tuple(greedy.shape) != (1, 1, 1):
                    raise RuntimeError("fixed-five greedy result must be one CPU token [1,1,1]")
                value = int(greedy.item())
                if not 0 <= value < VOCAB_SIZE:
                    raise RuntimeError(f"fixed-five target returned out-of-range token {value}")
                target_tokens.append(value)
                _release_slots(owned, label=f"fixed-five row {row} logits")
            except BaseException as error:
                _release_slots(owned, label=f"failed fixed-five row {row} logits", primary=error)
                raise
        return tuple(target_tokens)  # type: ignore[return-value]

    def verify_five(
        self,
        input_token_ids: tuple[int, int, int, int, int],
        state: Any,
        *,
        base_position: int,
    ) -> Qwen38TargetVerification:
        """Run one five-position target branch with one rows=5 MoE per layer."""

        self._require_healthy()
        if self._active_transaction is not None:
            raise RuntimeError("a fixed-five target verification is already active")
        tokens = _normalize_token_ids(input_token_ids)
        self.model._validate_state(state)
        if isinstance(base_position, bool) or not isinstance(base_position, int):
            raise TypeError("fixed-five base_position must be an integer")
        if state.position != base_position:
            raise ValueError(
                f"fixed-five target state is at position {state.position}, not requested base {base_position}"
            )
        allocated_context = getattr(self.model, "allocated_context", MAX_CONTEXT)
        if not 0 <= base_position or base_position + TARGET_ROWS > allocated_context:
            raise ValueError(f"five-position target from {base_position} exceeds allocated context {allocated_context}")

        snapshot = self.model.snapshot_state(state)
        transaction = _Qwen38FixedFiveTransaction(
            owner=self._transaction_owner,
            base_position=base_position,
            input_token_ids=tokens,
            snapshot=snapshot,
        )
        self._active_transaction = transaction
        processed_layers = 0
        rope_inputs: list[Any] = []
        try:
            host_tokens = tuple(torch.tensor([[token]], dtype=torch.long) for token in tokens)
            residuals = [self.model._embed_residual(host_token) for host_token in host_tokens]
            for row, residual in enumerate(residuals):
                self._validate_residual(residual, label=f"fixed-five embedding row {row}")
            rope_inputs = [self.model.rope.for_position(base_position + row) for row in range(TARGET_ROWS)]

            next_layers: list[Qwen38TTNNDecoderLayerState] = []
            for layer_index, layer_state in enumerate(state.layers):
                residuals, next_state = self._forward_layer_five(
                    layer_index,
                    residuals,
                    layer_state,
                    host_tokens,
                    rope_inputs,
                )
                next_layers.append(next_state)
                processed_layers += 1
            _release_rope_inputs(rope_inputs)
            rope_inputs = []

            branch_state = Qwen38TTNNTextModelState(
                position=base_position + TARGET_ROWS,
                layers=tuple(next_layers),
                _owner=self.model._state_owner,
            )
            self.model._validate_state(branch_state)
            target_tokens = self._resolve_five_logits(residuals)
            transaction.branch_state = branch_state
            transaction.roots = list(residuals)
            verification = Qwen38TargetVerification(
                base_position=base_position,
                input_token_ids=tokens,
                target_token_ids=target_tokens,
                target_hyper_residuals=tuple(residuals),  # type: ignore[arg-type]
                speculative_position=base_position + TARGET_ROWS,
                transaction=transaction,
            )
            return verification
        except BaseException as error:
            # A snapshot is active and a component may have mutated fixed
            # state.  Do not guess at cleanup/rollback safety here.
            if rope_inputs:
                try:
                    _release_rope_inputs(rope_inputs, primary=error)
                except BaseException as cleanup_error:
                    error = cleanup_error
            self._poison("verify_five", processed_layers, error)

    def _require_verification(
        self,
        verification: Qwen38TargetVerification,
    ) -> _Qwen38FixedFiveTransaction:
        self._require_healthy()
        if type(verification) is not Qwen38TargetVerification:
            raise TypeError("fixed-five adapter requires Qwen38TargetVerification")
        transaction = verification.transaction
        if not isinstance(transaction, _Qwen38FixedFiveTransaction) or transaction.owner is not self._transaction_owner:
            raise ValueError("fixed-five verification belongs to another adapter")
        if self._active_transaction is not transaction:
            raise ValueError("fixed-five verification is not this adapter's active transaction")
        if not transaction.active:
            raise RuntimeError("fixed-five verification was already committed or aborted")
        if transaction.branch_state is None:
            raise RuntimeError("fixed-five transaction has no completed target branch")
        if (
            verification.base_position != transaction.base_position
            or verification.input_token_ids != transaction.input_token_ids
            or verification.speculative_position != transaction.base_position + TARGET_ROWS
        ):
            raise ValueError("fixed-five verification metadata differs from its opaque transaction")
        roots = tuple(verification.target_hyper_residuals)
        if len(roots) != TARGET_ROWS or any(root is None for root in roots):
            raise ValueError("fixed-five verification must retain five hyper residuals")
        if any(root is not transaction.roots[index] for index, root in enumerate(roots)):
            raise ValueError("fixed-five verification root ownership differs from its transaction")
        return transaction

    def preflight_commit(self, verification: Qwen38TargetVerification, committed_input_count: int) -> None:
        count = _require_count(committed_input_count)
        transaction = self._require_verification(verification)
        branch_state = transaction.branch_state
        operation = "commit" if count == TARGET_ROWS else "restore"
        self.model._preflight_transaction(branch_state, transaction.snapshot, operation=operation)
        for row, root in enumerate(transaction.roots):
            self._validate_residual(root, label=f"fixed-five retained root {row}")
        root_keys = tuple(_tensor_key(root) for root in transaction.roots)
        if len(set(root_keys)) != TARGET_ROWS:
            raise RuntimeError("fixed-five target roots must be five distinct live buffers")

    def _consume_transaction(self, transaction: _Qwen38FixedFiveTransaction) -> None:
        transaction.active = False
        self._active_transaction = None

    @staticmethod
    def _committed_root_prefix(
        transaction: _Qwen38FixedFiveTransaction,
        count: int,
    ) -> tuple[Any, ...]:
        roots = tuple(transaction.roots[:count])
        if len(roots) != count or any(root is None for root in roots):
            raise RuntimeError("committed fixed-five root prefix was already released")
        return roots

    @staticmethod
    def _detach_committed_root_prefix(
        transaction: _Qwen38FixedFiveTransaction,
        roots: tuple[Any, ...],
    ) -> None:
        for index, root in enumerate(roots):
            if transaction.roots[index] is not root:
                raise RuntimeError("committed fixed-five root prefix changed before ownership transfer")
        for index in range(len(roots)):
            transaction.roots[index] = None

    def _replay_prefix(
        self,
        state: Qwen38TTNNTextModelState,
        tokens: tuple[int, ...],
    ) -> Qwen38TTNNTextModelState:
        current = state
        for offset, token in enumerate(tokens):
            output = self.model.forward_decode(
                token,
                current,
                return_logits=False,
                resolve_greedy=False,
                retain_hidden=False,
                retain_hyper_residual=False,
                retain_input_state=False,
                return_routing=False,
            )
            if type(output) is not Qwen38TTNNTextModelOutput:
                raise TypeError("ordinary deterministic replay returned a foreign model output")
            current = output.state
            if current.position != state.position + offset + 1:
                raise RuntimeError("ordinary deterministic replay advanced to the wrong target position")
            output.release_tensors()
        return current

    def commit_prefix(
        self,
        verification: Qwen38TargetVerification,
        committed_input_count: int,
    ) -> Qwen38TargetCommit:
        count = _require_count(committed_input_count)
        transaction = self._require_verification(verification)
        self.preflight_commit(verification, count)
        committed = transaction.input_token_ids[:count]
        try:
            if count == TARGET_ROWS:
                state = self.model.commit_state(transaction.branch_state, transaction.snapshot)
                mode = Qwen38TargetCommitMode.PREFIX_TRANSACTION
                replayed: tuple[int, ...] = ()
                self._consume_transaction(transaction)
            else:
                restored = self.model.restore_state(transaction.branch_state, transaction.snapshot)
                self._consume_transaction(transaction)
                # Preserve every committed verification-branch root identity.
                # They are activations, not mutable cache state; replay does
                # not retain duplicate roots while rebuilding state exactly.
                state = self._replay_prefix(restored, committed)
                mode = Qwen38TargetCommitMode.RESTORE_REPLAY
                replayed = committed
            expected_position = transaction.base_position + count
            self.model._validate_state(state)
            if state.position != expected_position:
                raise RuntimeError(f"fixed-five commit advanced to {state.position}, expected {expected_position}")
            committed_roots = self._committed_root_prefix(transaction, count)
            _release_slots(
                transaction.roots,
                range(count, TARGET_ROWS),
                label="uncommitted fixed-five root suffix",
            )
            result = Qwen38TargetCommit(
                state=state,
                position=expected_position,
                committed_input_token_ids=committed,
                committed_hyper_residuals=committed_roots,
                mode=mode,
                replayed_input_token_ids=replayed,
            )
            self._detach_committed_root_prefix(transaction, committed_roots)
            return result
        except BaseException as error:
            self._poison("commit_prefix", BACKBONE_LAYERS, error)

    def abort(self, verification: Qwen38TargetVerification) -> Qwen38StateRollback:
        transaction = self._require_verification(verification)
        self.model._preflight_transaction(transaction.branch_state, transaction.snapshot, operation="restore")
        try:
            restored = self.model.restore_state(transaction.branch_state, transaction.snapshot)
            self._consume_transaction(transaction)
            _release_slots(transaction.roots, label="aborted fixed-five roots")
            self.model._validate_state(restored)
            if restored.position != transaction.base_position:
                raise RuntimeError(
                    f"fixed-five abort restored position {restored.position}, expected {transaction.base_position}"
                )
            return Qwen38StateRollback(state=restored, position=transaction.base_position)
        except BaseException as error:
            self._poison("abort", BACKBONE_LAYERS, error)


def validate_fixed_five_static_contract() -> None:
    """No-device guard for exact verifier geometry and protocol inheritance."""

    if (TARGET_ROWS, VERIFY_POSITIONS, BACKBONE_LAYERS, BLOCK_LOCAL_SHAPE, RESIDUAL_LOCAL_SHAPE) != (
        5,
        5,
        48,
        (1, 1, 1, 640),
        (1, 4, 1, 640),
    ):
        raise RuntimeError("Qwen3.8 fixed-five geometry drifted")
    if not issubclass(Qwen38TTNNFixedFiveTarget, Qwen38FixedFiveTarget):
        raise RuntimeError("device fixed-five adapter no longer implements the strict MTP target protocol")


validate_fixed_five_static_contract()


__all__ = [
    "Qwen38FixedFiveCleanupError",
    "Qwen38TTNNFixedFiveTarget",
    "TARGET_ROWS",
    "validate_fixed_five_static_contract",
]
