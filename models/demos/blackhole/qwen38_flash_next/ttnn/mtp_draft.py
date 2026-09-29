# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Concrete shifted-state TTNN MTP draft owner for Qwen3.8-Flash-Next.

The live seed is produced by target-conditioned shifted extend.  Prompt
bootstrap pairs ``h_i`` with ``t_(i+1)`` and pairs the final prompt root with
the pending token.  Every speculative round starts from that precomputed d1
and performs exactly three decoder-layer calls.  Post-verification alignment
restores the provisional branch and evaluates each authoritative
``(emitted_token, committed_target_root)`` pair with a fresh QSA selection;
only the final row becomes the next seed.

The released decoder layer normally scopes its one-slot BF4_B expert load inside
``forward_decode``.  A capacity-qualified resident builder instead supplies
the same TP4/EP4 BF4 pair through the target-shared built-graph owner.
This adapter reports the selected policy without changing tensor identity or
substituting BF8.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, NoReturn, Sequence

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import Qwen38BF4ResidentSet
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import Qwen38TTNNLMHead, Qwen38TTNNTokenEmbedding
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import Qwen38TTNNFinalMixer
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    Qwen38TTNNDecoderLayer,
    Qwen38TTNNDecoderLayerResult,
    Qwen38TTNNDecoderLayerSnapshot,
    Qwen38TTNNDecoderLayerState,
    Qwen38TTNNLayerNamespace,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNRoPE
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import Qwen38TTNNRouting
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp import Qwen38TTNNMTPInput
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_decode import (
    DRAFT_EXTENSION_CALLS,
    DRAFT_STEPS,
    MAX_CONTEXT,
    MTP_LAYER_INDEX,
    VERIFY_POSITIONS,
    VOCAB_SIZE,
    Qwen38MTPAlignmentCommit,
    Qwen38MTPAlignmentStep,
    Qwen38MTPDraftBatch,
    Qwen38MTPDraftEngine,
    Qwen38MTPDraftStep,
    Qwen38MTPQSASelectionProof,
    Qwen38MTPSeed,
    Qwen38StateRollback,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    COMPRESS_RATIO,
    MAX_SPECULATIVE_STEPS,
    TOKEN_BUDGET,
    Qwen38TTNNQSA,
    Qwen38TTNNQSASelection,
)

RESIDUAL_LOCAL_SHAPE = (1, 4, 1, 640)
EMBEDDING_LOCAL_SHAPE = (1, 1, 1, 640)
BF4_STREAMED_LOAD_POLICY = "per-layer-call-serialized"
BF4_RESIDENT_LOAD_POLICY = "build-time-resident-components-owned"
BF4_LOAD_POLICY = BF4_STREAMED_LOAD_POLICY
MTP_BOOTSTRAP_CONTRACT = "shifted-target-conditioned-seed-v1"


class Qwen38TTNNMTPDraftError(RuntimeError):
    pass


class Qwen38TTNNMTPDraftCleanupError(Qwen38TTNNMTPDraftError):
    def __init__(self, label: str, errors: Sequence[BaseException], *, primary: BaseException | None = None) -> None:
        self.label = label
        self.errors = tuple(errors)
        self.primary = primary
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in self.errors)
        super().__init__(f"{label} cleanup failed for {len(self.errors)} resource(s): {detail}")


class Qwen38TTNNMTPAlignmentError(Qwen38TTNNMTPDraftError):
    pass


class Qwen38TTNNMTPDraftPoisonedError(Qwen38TTNNMTPDraftError):
    def __init__(self, operation: str, cause: BaseException, cleanup_errors: Sequence[BaseException] = ()) -> None:
        self.operation = operation
        self.original_cause = cause
        self.cleanup_errors = tuple(cleanup_errors)
        cleanup = ""
        if self.cleanup_errors:
            cleanup = "; rollback diagnostics: " + "; ".join(
                f"{type(error).__name__}: {error}" for error in self.cleanup_errors
            )
        super().__init__(
            f"TTNN MTP draft owner was poisoned during {operation}: {type(cause).__name__}: {cause}{cleanup}; "
            "terminate the task-owned process and release its lease without resetting hardware"
        )


def _tensor_key(tensor: Any) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _release_slots(slots: list[Any | None], *, label: str, primary: BaseException | None = None) -> None:
    groups: dict[tuple[str, int], list[int]] = {}
    for index, tensor in enumerate(slots):
        if tensor is not None:
            groups.setdefault(_tensor_key(tensor), []).append(index)
    errors: list[BaseException] = []
    for indices in groups.values():
        try:
            ttnn.deallocate(slots[indices[0]])
        except BaseException as error:
            errors.append(error)
        else:
            for index in indices:
                slots[index] = None
    if errors:
        raise Qwen38TTNNMTPDraftCleanupError(label, errors, primary=primary) from primary


def _require_identity_key(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"MTP identity must be lowercase 64-hex, got {value!r}")
    return value


def _require_token(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < VOCAB_SIZE:
        raise ValueError(f"{label} must be an integer in [0,{VOCAB_SIZE}), got {value!r}")
    return value


def _require_position(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_CONTEXT:
        raise ValueError(f"{label} must be an integer in [0,{MAX_CONTEXT}], got {value!r}")
    return value


@dataclass
class _SeedOwner:
    owner: Any
    state: Qwen38TTNNDecoderLayerState
    residual: Any
    selection: Qwen38TTNNQSASelection
    active: bool = True
    seed: Qwen38MTPSeed | None = None


@dataclass
class _DraftTransaction:
    owner: Any
    seed_owner: _SeedOwner
    snapshot: Qwen38TTNNDecoderLayerSnapshot | Any
    current_state: Qwen38TTNNDecoderLayerState
    terminal_residual: list[Any | None]
    steps: tuple[Qwen38MTPDraftStep, ...]
    active: bool = True
    batch: Qwen38MTPDraftBatch | None = None


@dataclass
class _RunProgress:
    state: Qwen38TTNNDecoderLayerState
    terminal_residual: list[Any | None] = field(default_factory=lambda: [None])
    selection: Qwen38TTNNQSASelection | None = None
    prediction: int | None = None
    alignment_steps: list[Qwen38MTPAlignmentStep] = field(default_factory=list)
    draft_steps: list[Qwen38MTPDraftStep] = field(default_factory=list)


@dataclass(frozen=True)
class Qwen38TTNNMTPExtensionPositionOneObservation:
    """Borrowed terminal tensors for one synchronous diagnostic readback."""

    position: int
    input_token_id: int
    predicted_token_id: int
    hidden_sharded: Any
    logits_sharded: Any
    routing: Qwen38TTNNRouting


@dataclass(frozen=True)
class Qwen38TTNNMTPExtensionPositionOneResult:
    """Host result after the one-row branch has been restored to ReadySeed."""

    position: int
    input_token_id: int
    predicted_token_id: int
    qsa_selection: Qwen38MTPQSASelectionProof
    observer_result: Any


class _RunFailure(RuntimeError):
    def __init__(
        self, cause: BaseException, progress: _RunProgress, cleanup_errors: Sequence[BaseException] = ()
    ) -> None:
        self.cause = cause
        self.progress = progress
        self.cleanup_errors = tuple(cleanup_errors)
        super().__init__(str(cause))


class _RowFailure(RuntimeError):
    def __init__(
        self,
        cause: BaseException,
        state: Qwen38TTNNDecoderLayerState,
        residual: Any | None,
        cleanup_errors: Sequence[BaseException] = (),
    ) -> None:
        self.cause = cause
        self.state = state
        self.residual = residual
        self.cleanup_errors = tuple(cleanup_errors)
        super().__init__(str(cause))


class Qwen38TTNNMTPDraftEngine(Qwen38MTPDraftEngine):
    """Exclusive concrete owner of the released one-layer MTP state."""

    def __init__(self, built_target: Any, mtp_components: Any) -> None:
        from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38TTNNBuiltTarget, Qwen38TTNNMTPComponents

        if not isinstance(built_target, Qwen38TTNNBuiltTarget):
            raise TypeError("MTP draft engine requires the exact Qwen38TTNNBuiltTarget")
        if not isinstance(mtp_components, Qwen38TTNNMTPComponents):
            raise TypeError("MTP draft engine requires exact Qwen38TTNNMTPComponents")
        target = built_target.components
        model = built_target.model
        if target.identity is not mtp_components.identity:
            raise ValueError("target and MTP must share the same live build identity")
        if model.final_mixer is not target.final_mixer or tuple(model.layers) != tuple(target.layers):
            raise ValueError("built target model/component object graph is inconsistent")
        if (
            model.model_io is not target.model_io
            or target.model_io.embedding.weights is not target.model_io.lm_head.weights
        ):
            raise ValueError("MTP requires the target's exact shared embedding/LM-head owner")
        if model.mesh_device is not target.model_io.embedding.mesh_device:
            raise ValueError("target model and embedding use different live mesh objects")
        input_mixer = mtp_components.input_mixer
        layer = mtp_components.decoder_layer
        final_mixer = mtp_components.final_mixer
        if not isinstance(input_mixer, Qwen38TTNNMTPInput):
            raise TypeError("MTP components contain a non-Qwen38 input mixer")
        if not isinstance(layer, Qwen38TTNNDecoderLayer):
            raise TypeError("MTP components contain a non-Qwen38 decoder layer")
        if not isinstance(final_mixer, Qwen38TTNNFinalMixer):
            raise TypeError("MTP components contain a non-Qwen38 final mixer")
        if (
            layer.namespace is not Qwen38TTNNLayerNamespace.MTP
            or layer.layer_index != MTP_LAYER_INDEX
            or layer.layer_type is not Qwen38TTNNLayerType.QSA
            or not isinstance(layer.attention, Qwen38TTNNQSA)
            or layer.ple is not None
        ):
            raise ValueError("released MTP decoder must be namespace=mtp QSA layer 0 without PLE")
        if layer.expert_streamer is not target.expert_streamer or layer.expert_streamer._active is not None:
            raise ValueError("MTP must share the idle one-slot BF4_B expert streamer")
        if final_mixer.weights.namespace != Qwen38TTNNLayerNamespace.MTP.value:
            raise ValueError("MTP draft engine requires the checkpoint MTP final mixer")
        for label, component in (("input mixer", input_mixer), ("final mixer", final_mixer)):
            if component.mesh_device is not model.mesh_device or component.mesh_contract != model.mesh_contract:
                raise ValueError(f"MTP {label} belongs to a different physical mesh")
        if layer.mesh_contract != model.mesh_contract or layer.attention.mesh_device is not model.mesh_device:
            raise ValueError("MTP decoder belongs to a different physical mesh")
        if not isinstance(model.rope, Qwen38TTNNRoPE):
            raise TypeError("built target must expose exact Qwen38 TTNN RoPE")
        if model.rope.mesh_device is not model.mesh_device or model.rope.mesh_contract != model.mesh_contract:
            raise ValueError("target RoPE belongs to a different physical mesh")
        self._initialize_owner(
            identity_key=mtp_components.identity.key,
            embedding=target.model_io.embedding,
            lm_head=target.model_io.lm_head,
            input_mixer=input_mixer,
            decoder_layer=layer,
            final_mixer=final_mixer,
            rope=model.rope,
        )
        self.mesh_device = model.mesh_device
        self.mesh_contract = model.mesh_contract
        self.identity = mtp_components.identity

    def _initialize_owner(
        self,
        *,
        identity_key: str,
        embedding: Qwen38TTNNTokenEmbedding | Any,
        lm_head: Qwen38TTNNLMHead | Any,
        input_mixer: Qwen38TTNNMTPInput | Any,
        decoder_layer: Qwen38TTNNDecoderLayer | Any,
        final_mixer: Qwen38TTNNFinalMixer | Any,
        rope: Qwen38TTNNRoPE | Any,
    ) -> None:
        self._identity_key = _require_identity_key(identity_key)
        self.embedding = embedding
        self.lm_head = lm_head
        self.input_mixer = input_mixer
        self.decoder_layer = decoder_layer
        self.final_mixer = final_mixer
        self.rope = rope
        capacity = getattr(getattr(decoder_layer, "attention", None), "allocated_context", MAX_CONTEXT)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 0 < capacity <= MAX_CONTEXT:
            raise ValueError(f"MTP decoder allocated context must be in [1,{MAX_CONTEXT}], got {capacity!r}")
        self._allocated_context = capacity
        self._runtime_owner: object | None = None
        self._live_state: Qwen38TTNNDecoderLayerState | None = None
        self._live_seed: Qwen38MTPSeed | None = None
        self._active_transaction: _DraftTransaction | None = None
        self._poisoned_error: Qwen38TTNNMTPDraftPoisonedError | None = None
        self._poisoned_device_owners: list[Any] = []
        self._released_tensor_keys: set[tuple[str, int]] = set()
        self._bootstrap_phase_observer: Callable[[str], None] | None = None

    @property
    def identity_key(self) -> str:
        return self._identity_key

    @property
    def allocated_context(self) -> int:
        return self._allocated_context

    @property
    def bf4_load_policy(self) -> str:
        owner = getattr(self.decoder_layer, "expert_streamer", None)
        return BF4_RESIDENT_LOAD_POLICY if isinstance(owner, Qwen38BF4ResidentSet) else BF4_STREAMED_LOAD_POLICY

    @property
    def bootstrap_contract(self) -> str:
        return MTP_BOOTSTRAP_CONTRACT

    @property
    def poisoned(self) -> bool:
        return self._poisoned_error is not None

    @property
    def poisoned_error(self) -> Qwen38TTNNMTPDraftPoisonedError | None:
        return self._poisoned_error

    @contextmanager
    def observe_bootstrap_phases(self, observer: Callable[[str], None]) -> Iterator[None]:
        """Report nonsemantic shifted-bootstrap boundaries for one scoped diagnostic."""

        if not callable(observer):
            raise TypeError("bootstrap phase observer must be callable")
        if self._bootstrap_phase_observer is not None:
            raise RuntimeError("a bootstrap phase observer is already active")
        self._bootstrap_phase_observer = observer
        try:
            yield
        finally:
            self._bootstrap_phase_observer = None

    def _observe_bootstrap_phase(self, phase: str) -> None:
        observer = self._bootstrap_phase_observer
        if observer is not None:
            observer(phase)

    def _require_healthy(self) -> None:
        if self._poisoned_error is not None:
            raise self._poisoned_error from self._poisoned_error.original_cause

    def _poison(self, operation: str, cause: BaseException, cleanup_errors: Sequence[BaseException] = ()) -> NoReturn:
        if self._poisoned_error is None:
            self._poisoned_error = Qwen38TTNNMTPDraftPoisonedError(operation, cause, cleanup_errors)
        raise self._poisoned_error from cause

    def _require_runtime_owner(self, operation: str) -> None:
        if self._runtime_owner is None:
            raise RuntimeError(f"cannot {operation}: MTP engine has no runtime owner")

    def _require_no_transaction(self, operation: str) -> None:
        if self._active_transaction is not None:
            raise RuntimeError(f"cannot {operation} while an MTP draft transaction is active")

    def claim_runtime_owner(self, owner: object) -> None:
        self._require_healthy()
        self._require_no_transaction("claim runtime ownership")
        if owner is None or self._runtime_owner is not None:
            raise RuntimeError("MTP runtime owner token is invalid or already claimed")
        self._runtime_owner = owner

    def release_runtime_owner(self, owner: object) -> None:
        self._require_healthy()
        self._require_no_transaction("release runtime ownership")
        if self._runtime_owner is not owner:
            raise ValueError("runtime owner token does not own this MTP engine")
        self._runtime_owner = None

    def transfer_runtime_owner(self, current_owner: object, next_owner: object) -> None:
        self._require_healthy()
        self._require_no_transaction("transfer runtime ownership")
        if self._runtime_owner is not current_owner or next_owner is None or next_owner is current_owner:
            raise ValueError("invalid MTP runtime owner transfer")
        self._runtime_owner = next_owner

    def _validate_state(self, state: Any, *, require_live: bool = True) -> Qwen38TTNNDecoderLayerState:
        if type(state) is not Qwen38TTNNDecoderLayerState:
            raise TypeError("MTP state must be exact Qwen38TTNNDecoderLayerState")
        if state.namespace is not Qwen38TTNNLayerNamespace.MTP or state.layer_index != MTP_LAYER_INDEX:
            raise ValueError("MTP state must belong to namespace=mtp layer 0")
        _require_position(state.position, label="MTP state position")
        self.decoder_layer.validate_state(state)
        if require_live and state is not self._live_state:
            raise ValueError("MTP state is not the one live state owned by this engine")
        return state

    def allocate_state(self) -> Qwen38TTNNDecoderLayerState:
        self._require_healthy()
        self._require_runtime_owner("allocate MTP state")
        self._require_no_transaction("allocate state")
        if self._live_state is not None:
            raise RuntimeError("MTP engine already owns a live state")
        state = None
        try:
            state = self.decoder_layer.allocate_state()
            self._validate_state(state, require_live=False)
            if state.position != 0:
                raise RuntimeError("fresh MTP state must start at position zero")
        except BaseException as error:
            if state is not None:
                try:
                    self.decoder_layer.release_state(state)
                except BaseException as cleanup_error:
                    self._poison("MTP state allocation", error, (cleanup_error,))
            raise
        self._live_state = state
        return state

    def raw_state_position(self, state: Any) -> int:
        self._require_healthy()
        return self._validate_state(state).position

    def state_position(self, state: Any) -> int:
        return self.raw_state_position(state)

    def release_state(self, state: Any) -> None:
        self._require_healthy()
        self._require_runtime_owner("release unseeded state")
        self._require_no_transaction("release unseeded state")
        state = self._validate_state(state)
        if self._live_seed is not None:
            raise RuntimeError("seeded MTP state must be released through release_seed")
        try:
            self.decoder_layer.release_state(state)
        except BaseException as error:
            self._poison("MTP state release", error)
        self._live_state = None

    def _validate_residual(self, residual: Any, *, label: str) -> None:
        if residual is None:
            raise ValueError(f"{label} cannot be None")
        if _tensor_key(residual) in self._released_tensor_keys:
            raise RuntimeError(f"{label} was already released")
        self.input_mixer._validate_input(residual, label="hidden residual", shape=RESIDUAL_LOCAL_SHAPE)

    def _normalize_roots(self, roots: tuple[Any, ...], *, count: int) -> tuple[Any, ...]:
        if type(roots) is not tuple or len(roots) != count:
            raise ValueError("MTP alignment requires one target root per consumed token")
        if any(root is None for root in roots) or len({_tensor_key(root) for root in roots}) != count:
            raise ValueError("MTP target roots must be non-null distinct tensor owners")
        for index, root in enumerate(roots):
            self._validate_residual(root, label=f"target root {index}")
        return roots

    def _resolve_token(self, logits: Any, *, label: str) -> int:
        token = self.lm_head.greedy_token(logits)
        if (
            not isinstance(token, torch.Tensor)
            or token.device.type != "cpu"
            or tuple(token.shape) != (1, 1, 1)
            or token.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError(f"{label} must resolve to one CPU int token [1,1,1]")
        return _require_token(int(token.item()), label=label)

    def _embed_token(self, token_id: int) -> Any:
        uploaded = self.embedding.upload_tokens(torch.tensor([[token_id]], dtype=torch.int64))
        token_slots = [uploaded.tensor]
        embedding = None
        try:
            embedding = self.embedding(uploaded)
        except BaseException as error:
            if getattr(self.embedding, "poisoned", False):
                retained = list(getattr(self.embedding, "poisoned_device_owners", ()))
                retained_ids = {id(owner) for owner in retained}
                for tensor in token_slots:
                    if tensor is None:
                        continue
                    if id(tensor) not in retained_ids:
                        retained.append(tensor)
                        retained_ids.add(id(tensor))
                    try:
                        local_tensors = tuple(ttnn.get_device_tensors(tensor))
                    except BaseException:
                        local_tensors = ()
                    for local_tensor in local_tensors:
                        if id(local_tensor) not in retained_ids:
                            retained.append(local_tensor)
                            retained_ids.add(id(local_tensor))
                self._poisoned_device_owners.extend(retained)
                token_slots[:] = [None] * len(token_slots)
                self._poison("MTP token embedding asynchronous chain", error)
            cleanup: list[BaseException] = []
            try:
                _release_slots(token_slots, label="failed MTP uploaded token", primary=error)
            except BaseException as cleanup_error:
                cleanup.append(cleanup_error)
            if embedding is not None:
                try:
                    _release_slots([embedding], label="failed MTP embedding", primary=error)
                except BaseException as cleanup_error:
                    cleanup.append(cleanup_error)
            if cleanup:
                raise Qwen38TTNNMTPDraftCleanupError("MTP embedding", cleanup, primary=error) from error
            raise
        try:
            _release_slots(token_slots, label="MTP uploaded token")
        except BaseException as error:
            # A throwing device deallocation has uncertain ownership.  Do not
            # retry that token allocation; clean only the independently-owned
            # embedding and propagate the original cleanup failure.
            token_slots[0] = None
            cleanup: list[BaseException] = []
            try:
                _release_slots([embedding], label="MTP embedding after token cleanup failure", primary=error)
            except BaseException as cleanup_error:
                cleanup.append(cleanup_error)
            if cleanup:
                raise Qwen38TTNNMTPDraftCleanupError("MTP uploaded token", cleanup, primary=error) from error
            raise
        return embedding

    @staticmethod
    def _selection_key(selection: Qwen38TTNNQSASelection) -> tuple[str, int] | None:
        return None if selection.complete_indices is None else _tensor_key(selection.complete_indices)

    def _selection_proof(
        self,
        result: Qwen38TTNNDecoderLayerResult,
        *,
        position: int,
        reuse: Qwen38TTNNQSASelection | None,
    ) -> tuple[Qwen38TTNNQSASelection, Qwen38MTPQSASelectionProof]:
        if type(result) is not Qwen38TTNNDecoderLayerResult:
            raise TypeError("MTP decoder returned a non-Qwen38 result")
        selection = result.aux.selection
        if type(selection) is not Qwen38TTNNQSASelection:
            raise TypeError("MTP QSA did not return exact selection metadata")
        attention = result.state.attention
        result_view = getattr(attention, "view_id", None)
        if getattr(attention, "last_selection", None) is not selection:
            raise RuntimeError("MTP selection is not owned by returned state metadata")
        if result.aux.reused_qsa_selection is not (reuse is not None):
            raise RuntimeError("MTP QSA reuse flag differs from supplied selection")
        if selection.layer_index != MTP_LAYER_INDEX or selection.epoch <= 0:
            raise RuntimeError("MTP selection has wrong layer or epoch")
        if not isinstance(result_view, int) or result_view <= 0:
            raise RuntimeError("MTP result has invalid QSA view")
        if reuse is None:
            if selection.source_view_id != result_view or selection.source_position != position:
                raise RuntimeError("fresh MTP QSA selection has wrong source row/view")
            if selection.complete_indices is not None and not selection.owns_complete_indices:
                raise RuntimeError("fresh MTP QSA result must own selected complete indices")
        else:
            if (
                selection.source_view_id != reuse.source_view_id
                or selection.source_position != reuse.source_position
                or selection.tail_start != reuse.tail_start
                or selection.complete_token_count != reuse.complete_token_count
                or self._selection_key(selection) != self._selection_key(reuse)
            ):
                raise RuntimeError("speculative QSA reuse changed the seed selection")
            if selection.owns_complete_indices:
                raise RuntimeError("speculative branch must borrow seed complete indices")
            if not 0 <= position - selection.source_position <= MAX_SPECULATIVE_STEPS:
                raise RuntimeError("speculative QSA reuse exceeds four-step distance")
        if (
            selection.complete_token_count % COMPRESS_RATIO
            or not 0 <= selection.complete_token_count <= TOKEN_BUDGET
            or (selection.complete_indices is None) != (selection.complete_token_count == 0)
        ):
            raise RuntimeError("MTP QSA complete selection metadata is inconsistent")
        expected_valid = selection.complete_token_count + position + 1 - selection.tail_start
        if selection.valid_token_count != expected_valid or selection.sparse_indices is None:
            raise RuntimeError("MTP QSA causal tail materialization is inconsistent")
        proof = Qwen38MTPQSASelectionProof(
            layer_index=selection.layer_index,
            epoch=selection.epoch,
            source_view_id=selection.source_view_id,
            result_view_id=result_view,
            source_position=selection.source_position,
            tail_start=selection.tail_start,
            complete_token_count=selection.complete_token_count,
            complete_indices_key=self._selection_key(selection),
            valid_token_count=selection.valid_token_count,
        )
        return selection, proof

    def _execute_row(
        self,
        token_id: int,
        root: Any,
        state: Qwen38TTNNDecoderLayerState,
        *,
        position: int,
        reuse: Qwen38TTNNQSASelection | None,
        retain_input_state: bool,
        row_observer: Callable[[Qwen38TTNNMTPExtensionPositionOneObservation], Any] | None = None,
    ) -> tuple[Qwen38TTNNDecoderLayerState, Any, Qwen38TTNNQSASelection, Qwen38MTPQSASelectionProof, int]:
        if row_observer is not None and not callable(row_observer):
            raise TypeError("MTP row observer must be callable or None")
        embedding = mixed = rope_inputs = hidden = logits = residual = routing = None
        next_state = state
        phase = f"ready-seed-mtp-row-{position}"
        try:
            self._observe_bootstrap_phase(f"before-{phase}-embedding")
            embedding = self._embed_token(token_id)
            self._observe_bootstrap_phase(f"after-{phase}-embedding")
            self._observe_bootstrap_phase(f"before-{phase}-input-mixer")
            mixed = self.input_mixer(embedding, root)
            self._observe_bootstrap_phase(f"after-{phase}-input-mixer")
            embedding_slot = [embedding]
            try:
                self._observe_bootstrap_phase(f"before-{phase}-embedding-release")
                _release_slots(embedding_slot, label=f"MTP embedding at position {position}")
                self._observe_bootstrap_phase(f"after-{phase}-embedding-release")
            except BaseException:
                embedding = None
                raise
            embedding = embedding_slot[0]
            self._observe_bootstrap_phase(f"before-{phase}-rope")
            rope_inputs = self.rope.for_position(position)
            self._observe_bootstrap_phase(f"after-{phase}-rope")
            if rope_inputs.position != position:
                raise RuntimeError("MTP RoPE uploader returned wrong position")
            try:
                self._observe_bootstrap_phase(f"before-{phase}-decoder-layer")
                result = self.decoder_layer.forward_decode(
                    mixed,
                    state,
                    token_id=None,
                    cos=rope_inputs.cos,
                    sin=rope_inputs.sin,
                    block_start_cos=rope_inputs.block_start_cos,
                    block_start_sin=rope_inputs.block_start_sin,
                    reuse_qsa_selection=reuse,
                    retain_input_state=retain_input_state,
                    return_routing=row_observer is not None,
                )
                self._observe_bootstrap_phase(f"after-{phase}-decoder-layer")
            except BaseException:
                mixed = None
                raise
            mixed = None
            next_state = result.state
            residual = result.residual_sharded
            routing = result.aux.routing
            if (routing is None) is (row_observer is not None):
                raise RuntimeError("MTP decoder routing retention differs from the row-observer request")
            if routing is not None and type(routing) is not Qwen38TTNNRouting:
                raise TypeError("MTP decoder returned non-exact routing metadata")
            self._validate_state(next_state, require_live=False)
            if next_state.position != position + 1:
                raise RuntimeError("MTP decoder advanced to wrong position")
            self._validate_residual(residual, label="MTP recurrent residual")
            if _tensor_key(residual) == _tensor_key(root):
                raise RuntimeError("MTP output aliases borrowed target/recurrent residual")
            owned_rope_inputs = rope_inputs
            rope_inputs = None
            self._observe_bootstrap_phase(f"before-{phase}-rope-release")
            owned_rope_inputs.deallocate()
            self._observe_bootstrap_phase(f"after-{phase}-rope-release")
            self._observe_bootstrap_phase(f"before-{phase}-selection-proof")
            selection, proof = self._selection_proof(result, position=position, reuse=reuse)
            self._observe_bootstrap_phase(f"after-{phase}-selection-proof")
            self._observe_bootstrap_phase(f"before-{phase}-final-mixer")
            hidden = self.final_mixer(residual)
            self._observe_bootstrap_phase(f"after-{phase}-final-mixer")
            self._observe_bootstrap_phase(f"before-{phase}-lm-head")
            logits = self.lm_head(hidden)
            self._observe_bootstrap_phase(f"after-{phase}-lm-head")
            self._observe_bootstrap_phase(f"before-{phase}-resolve-token")
            prediction = self._resolve_token(logits, label=f"MTP prediction at position {position}")
            self._observe_bootstrap_phase(f"after-{phase}-resolve-token")
            if row_observer is not None:
                row_observer(
                    Qwen38TTNNMTPExtensionPositionOneObservation(
                        position=position,
                        input_token_id=token_id,
                        predicted_token_id=prediction,
                        hidden_sharded=hidden,
                        logits_sharded=logits,
                        routing=routing,
                    )
                )
            terminal_slots = [
                hidden,
                getattr(logits, "tensor", None),
                None if routing is None else routing.scores,
                None if routing is None else routing.indices,
            ]
            try:
                self._observe_bootstrap_phase(f"before-{phase}-terminal-release")
                _release_slots(terminal_slots, label=f"MTP terminal outputs at position {position}")
                self._observe_bootstrap_phase(f"after-{phase}-terminal-release")
            except BaseException:
                hidden = logits = routing = None
                raise
            hidden = terminal_slots[0]
            logits = routing = None
            return next_state, residual, selection, proof, prediction
        except BaseException as error:
            cleanup: list[BaseException] = []
            try:
                _release_slots(
                    [
                        embedding,
                        mixed,
                        hidden,
                        getattr(logits, "tensor", None),
                        None if routing is None else routing.scores,
                        None if routing is None else routing.indices,
                    ],
                    label=f"failed MTP row {position}",
                    primary=error,
                )
            except BaseException as cleanup_error:
                cleanup.append(cleanup_error)
            if rope_inputs is not None:
                try:
                    rope_inputs.deallocate()
                except BaseException as cleanup_error:
                    cleanup.append(cleanup_error)
            raise _RowFailure(error, next_state, residual, cleanup) from error

    def _run_authoritative(
        self,
        tokens: tuple[int, ...],
        roots: tuple[Any, ...],
        state: Qwen38TTNNDecoderLayerState,
        *,
        base_position: int,
    ) -> _RunProgress:
        progress = _RunProgress(state)
        for index, (token, root) in enumerate(zip(tokens, roots)):
            old_residual = progress.terminal_residual[0]
            try:
                next_state, residual, selection, proof, prediction = self._execute_row(
                    token,
                    root,
                    progress.state,
                    position=base_position + index,
                    reuse=None,
                    retain_input_state=index == 0,
                )
            except _RowFailure as failure:
                progress.state = failure.state
                if failure.residual is not None:
                    progress.terminal_residual[0] = failure.residual
                raise _RunFailure(failure.cause, progress, failure.cleanup_errors) from failure
            if old_residual is not None:
                try:
                    _release_slots([old_residual], label="superseded authoritative MTP residual")
                except BaseException as error:
                    progress.state = next_state
                    progress.terminal_residual[0] = residual
                    raise _RunFailure(error, progress) from error
            progress.state = next_state
            progress.terminal_residual[0] = residual
            progress.selection = selection
            progress.prediction = prediction
            progress.alignment_steps.append(Qwen38MTPAlignmentStep(index, base_position + index, token, proof))
        return progress

    def _run_extensions(self, seed_owner: _SeedOwner, seed: Qwen38MTPSeed) -> _RunProgress:
        progress = _RunProgress(seed_owner.state)
        prior_residual = seed_owner.residual
        owns_prior = False
        selection = seed_owner.selection
        token = seed.first_draft_token_id
        for extension in range(DRAFT_EXTENSION_CALLS):
            position = seed.position + extension
            try:
                next_state, residual, next_selection, proof, prediction = self._execute_row(
                    token,
                    prior_residual,
                    progress.state,
                    position=position,
                    reuse=selection,
                    retain_input_state=extension == 0,
                )
            except _RowFailure as failure:
                progress.state = failure.state
                if failure.residual is not None:
                    progress.terminal_residual[0] = failure.residual
                raise _RunFailure(failure.cause, progress, failure.cleanup_errors) from failure
            if owns_prior:
                try:
                    _release_slots([prior_residual], label="superseded speculative MTP residual")
                except BaseException as error:
                    progress.state = next_state
                    progress.terminal_residual[0] = residual
                    raise _RunFailure(error, progress) from error
            progress.state = next_state
            progress.terminal_residual[0] = residual
            selection = next_selection
            progress.selection = selection
            progress.prediction = prediction
            progress.draft_steps.append(
                Qwen38MTPDraftStep(
                    step_index=extension + 1,
                    position=position,
                    input_token_id=token,
                    predicted_token_id=prediction,
                    qsa_selection=proof,
                    reused_qsa_selection=True,
                    reused_from_view_id=seed.qsa_selection.source_view_id,
                    precomputed=False,
                )
            )
            prior_residual = residual
            owns_prior = True
            token = prediction
        return progress

    def _rollback(
        self,
        progress: _RunProgress,
        snapshot: Qwen38TTNNDecoderLayerSnapshot | Any,
        *,
        label: str,
    ) -> tuple[Qwen38TTNNDecoderLayerState | None, list[BaseException]]:
        errors: list[BaseException] = []
        try:
            _release_slots(progress.terminal_residual, label=f"{label} residual")
        except BaseException as error:
            errors.append(error)
        restored = None
        try:
            restored = self.decoder_layer.restore_state(progress.state, snapshot)
            self._validate_state(restored, require_live=False)
        except BaseException as error:
            errors.append(error)
        return restored, errors

    def _release_consumed(self, tensors: tuple[Any, ...], *, label: str) -> None:
        keys = tuple(_tensor_key(tensor) for tensor in tensors)
        slots = list(tensors)
        _release_slots(slots, label=label)
        self._released_tensor_keys.update(keys)

    def _make_seed(
        self,
        *,
        state: Qwen38TTNNDecoderLayerState,
        residual: Any,
        selection: Qwen38TTNNQSASelection,
        proof: Qwen38MTPQSASelectionProof,
        current_token_id: int,
        first_draft_token_id: int,
    ) -> Qwen38MTPSeed:
        owner = _SeedOwner(self, state, residual, selection)
        seed = Qwen38MTPSeed(
            position=state.position,
            current_token_id=current_token_id,
            first_draft_token_id=first_draft_token_id,
            state=state,
            recurrent_residual=residual,
            qsa_selection=proof,
            transaction=owner,
        )
        owner.seed = seed
        self._live_state = state
        self._live_seed = seed
        return seed

    def _publish_committed_seed(self, *, operation: str, **seed_fields: Any) -> Qwen38MTPSeed:
        """Publish a replacement seed after its cache commit, or poison.

        Once ``commit_state`` succeeds there is no valid rollback snapshot.
        Seed construction and validation therefore belong to the committed
        transaction boundary: any failure is terminal and must never leave a
        reusable, apparently healthy engine around the committed state.
        """

        try:
            seed = self._make_seed(**seed_fields)
            self._validate_seed(seed)
        except BaseException as error:
            self._poison(operation, error)
        return seed

    def _validate_seed(self, seed: Any) -> _SeedOwner:
        if type(seed) is not Qwen38MTPSeed:
            raise TypeError("MTP operation requires exact Qwen38MTPSeed")
        owner = seed.transaction
        if type(owner) is not _SeedOwner or owner.owner is not self or owner.seed is not seed:
            raise ValueError("MTP seed was not created by this engine")
        if not owner.active or self._live_seed is not seed:
            raise RuntimeError("MTP seed is not the live active seed (it may be released or consumed)")
        if owner.state is not seed.state or owner.residual is not seed.recurrent_residual:
            raise RuntimeError("MTP seed ownership record drifted")
        self._validate_state(seed.state)
        self._validate_residual(seed.recurrent_residual, label="MTP seed residual")
        if seed.position != seed.state.position or seed.position == 0:
            raise Qwen38TTNNMTPAlignmentError("MTP seed must follow at least one target-conditioned shifted row")
        return owner

    def bootstrap_shifted_prefill(
        self,
        consumed_token_ids: tuple[int, ...],
        pending_token_id: int,
        target_hyper_residuals: tuple[Any, ...],
        state: Any,
    ) -> Qwen38MTPSeed:
        self._require_healthy()
        self._require_runtime_owner("bootstrap shifted MTP state")
        self._require_no_transaction("bootstrap shifted MTP state")
        state = self._validate_state(state)
        if self._live_seed is not None or state.position != 0:
            raise Qwen38TTNNMTPAlignmentError("shifted prompt bootstrap requires one fresh position-zero state")
        if type(consumed_token_ids) is not tuple or not consumed_token_ids:
            raise ValueError("shifted bootstrap requires at least one consumed target token")
        consumed = tuple(
            _require_token(token, label=f"consumed target token {index}")
            for index, token in enumerate(consumed_token_ids)
        )
        pending = _require_token(pending_token_id, label="pending target token")
        roots = self._normalize_roots(target_hyper_residuals, count=len(consumed))
        if len(consumed) > self.allocated_context:
            raise ValueError("shifted bootstrap exceeds allocated context")
        shifted = (*consumed[1:], pending)
        return self.bootstrap_shifted_prefill_rows(zip(shifted, roots), state)

    @staticmethod
    def _normalize_shifted_row(row: Any, *, index: int) -> tuple[int, Any]:
        if not isinstance(row, tuple) or len(row) != 2:
            raise TypeError(f"shifted bootstrap row {index} must be exactly (token_id, target_root)")
        token, root = row
        return _require_token(token, label=f"shifted bootstrap token {index}"), root

    def bootstrap_shifted_prefill_rows(
        self,
        shifted_rows: Iterable[tuple[int, Any]],
        state: Any,
    ) -> Qwen38MTPSeed:
        """Stream shifted prompt rows with at most one new target root retained.

        Yielded root ownership transfers to this engine.  A root is released
        after its cache row and prediction succeed, before the next row is
        requested.  Any later failure poisons the engine: already-consumed
        roots remain released, while the failing/unrequested suffix remains
        untouched.  Post-verification alignment intentionally uses the
        stronger all-roots-live-until-seed-validation transaction instead.
        """

        self._require_healthy()
        self._require_runtime_owner("stream shifted MTP bootstrap")
        self._require_no_transaction("stream shifted MTP bootstrap")
        state = self._validate_state(state)
        if self._live_seed is not None or state.position != 0:
            raise Qwen38TTNNMTPAlignmentError("shifted prompt bootstrap requires one fresh position-zero state")
        try:
            iterator = iter(shifted_rows)
        except TypeError as error:
            raise TypeError("shifted bootstrap rows must be iterable") from error
        self._observe_bootstrap_phase("before-ready-seed-request-shifted-row-0")
        try:
            first = next(iterator)
        except StopIteration as error:
            self._observe_bootstrap_phase("after-ready-seed-shifted-rows-empty")
            raise ValueError("shifted bootstrap requires at least one target-conditioned row") from error
        self._observe_bootstrap_phase("after-ready-seed-request-shifted-row-0")
        first = self._normalize_shifted_row(first, index=0)
        if first[1] is None:
            raise ValueError("shifted bootstrap target root 0 cannot be None")
        self._validate_residual(first[1], label="shifted bootstrap root 0")

        progress = _RunProgress(state)
        seen_roots: set[tuple[str, int]] = set()
        last_token = last_prediction = None
        outstanding_root: Any | None = first[1]
        try:
            self._observe_bootstrap_phase("before-ready-seed-snapshot-state")
            snapshot = self.decoder_layer.snapshot_state(state)
            self._observe_bootstrap_phase("after-ready-seed-snapshot-state")
            pending_rows = [first]
            row_index = 0
            while pending_rows:
                token, root = pending_rows.pop()
                if row_index >= self.allocated_context:
                    raise ValueError("shifted bootstrap exceeds allocated context")
                root_key = _tensor_key(root)
                if root_key in seen_roots:
                    raise ValueError("shifted bootstrap roots must have distinct tensor ownership")
                self._validate_residual(root, label=f"shifted bootstrap root {row_index}")
                seen_roots.add(root_key)
                outstanding_root = root
                old_residual = progress.terminal_residual[0]
                try:
                    self._observe_bootstrap_phase(f"before-ready-seed-execute-mtp-row-{row_index}")
                    next_state, residual, selection, proof, prediction = self._execute_row(
                        token,
                        root,
                        progress.state,
                        position=row_index,
                        reuse=None,
                        retain_input_state=row_index == 0,
                    )
                    self._observe_bootstrap_phase(f"after-ready-seed-execute-mtp-row-{row_index}")
                except _RowFailure as failure:
                    progress.state = failure.state
                    if failure.residual is not None:
                        progress.terminal_residual[0] = failure.residual
                    raise _RunFailure(failure.cause, progress, failure.cleanup_errors) from failure
                progress.state = next_state
                progress.terminal_residual[0] = residual
                progress.selection = selection
                progress.prediction = prediction
                if old_residual is not None:
                    _release_slots([old_residual], label="superseded streamed-bootstrap MTP residual")
                progress.alignment_steps.append(Qwen38MTPAlignmentStep(row_index, row_index, token, proof))
                transferred_root = outstanding_root
                outstanding_root = None
                self._observe_bootstrap_phase(f"before-ready-seed-release-target-root-{row_index}")
                self._release_consumed(
                    (transferred_root,),
                    label=f"shifted-bootstrap target root {row_index}",
                )
                self._observe_bootstrap_phase(f"after-ready-seed-release-target-root-{row_index}")
                last_token, last_prediction = token, prediction
                row_index += 1
                self._observe_bootstrap_phase(f"before-ready-seed-request-shifted-row-{row_index}")
                try:
                    next_row = next(iterator)
                except StopIteration:
                    self._observe_bootstrap_phase("after-ready-seed-shifted-rows-exhausted")
                    break
                self._observe_bootstrap_phase(f"after-ready-seed-request-shifted-row-{row_index}")
                pending_rows.append(self._normalize_shifted_row(next_row, index=row_index))
            self._observe_bootstrap_phase("before-ready-seed-validate-commit-pair")
            self.decoder_layer.validate_commit_pair(progress.state, snapshot)
            self._observe_bootstrap_phase("after-ready-seed-validate-commit-pair")
            self._observe_bootstrap_phase("before-ready-seed-commit-state")
            committed = self.decoder_layer.commit_state(progress.state, snapshot)
            self._observe_bootstrap_phase("after-ready-seed-commit-state")
            self._validate_state(committed, require_live=False)
            if (
                progress.selection is None
                or progress.prediction is None
                or last_token is None
                or last_prediction is None
            ):
                raise RuntimeError("shifted bootstrap committed without a final MTP seed output")
            final_proof = progress.alignment_steps[-1].qsa_selection
        except _RunFailure as failure:
            root_cleanup: list[BaseException] = []
            if outstanding_root is not None:
                try:
                    self._release_consumed((outstanding_root,), label="failed shifted-bootstrap target root")
                except BaseException as cleanup_error:
                    root_cleanup.append(cleanup_error)
            restored, cleanup = self._rollback(failure.progress, snapshot, label="failed shifted bootstrap")
            if restored is not None:
                self._live_state = restored
            self._poison(
                "shifted prompt bootstrap",
                failure.cause,
                (*failure.cleanup_errors, *root_cleanup, *cleanup),
            )
        except BaseException as error:
            cleanup: list[BaseException] = []
            if outstanding_root is not None:
                try:
                    self._release_consumed((outstanding_root,), label="failed shifted-bootstrap target root")
                except BaseException as cleanup_error:
                    cleanup.append(cleanup_error)
            if "snapshot" in locals() and getattr(snapshot, "active", False):
                _, rollback = self._rollback(progress, snapshot, label="failed streamed bootstrap")
                cleanup.extend(rollback)
            self._poison("shifted prompt bootstrap commit", error, cleanup)
        self._observe_bootstrap_phase("before-ready-seed-publish-committed-seed")
        seed = self._publish_committed_seed(
            operation="publish shifted-bootstrap MTP seed",
            state=committed,
            residual=progress.terminal_residual[0],
            selection=progress.selection,
            proof=final_proof,
            current_token_id=last_token,
            first_draft_token_id=last_prediction,
        )
        self._observe_bootstrap_phase("after-ready-seed-publish-committed-seed")
        return seed

    def draft_four(self, seed: Qwen38MTPSeed) -> Qwen38MTPDraftBatch:
        self._require_healthy()
        self._require_runtime_owner("draft four tokens")
        self._require_no_transaction("draft four tokens")
        owner = self._validate_seed(seed)
        if seed.position + DRAFT_EXTENSION_CALLS > self.allocated_context:
            raise ValueError("three MTP extensions exceed allocated context")
        try:
            snapshot = self.decoder_layer.snapshot_state(seed.state)
            progress = self._run_extensions(owner, seed)
        except _RunFailure as failure:
            restored, cleanup = self._rollback(failure.progress, snapshot, label="failed speculative MTP branch")
            if restored is not None:
                self._live_state = seed.state
            self._poison("three-call MTP extension", failure.cause, (*failure.cleanup_errors, *cleanup))
        except BaseException as error:
            self._poison("MTP draft snapshot", error)
        seed_step = Qwen38MTPDraftStep(
            step_index=0,
            position=seed.position - 1,
            input_token_id=seed.current_token_id,
            predicted_token_id=seed.first_draft_token_id,
            qsa_selection=seed.qsa_selection,
            reused_qsa_selection=False,
            reused_from_view_id=None,
            precomputed=True,
        )
        steps = (seed_step, *progress.draft_steps)
        transaction = _DraftTransaction(
            self,
            owner,
            snapshot,
            progress.state,
            progress.terminal_residual,
            steps,
        )
        batch = Qwen38MTPDraftBatch(seed.position, seed, steps, progress.state.position, transaction)
        transaction.batch = batch
        self._active_transaction = transaction
        return batch

    def diagnose_extension_position_one(
        self,
        seed: Qwen38MTPSeed,
        observer: Callable[[Qwen38TTNNMTPExtensionPositionOneObservation], Any],
    ) -> Qwen38TTNNMTPExtensionPositionOneResult:
        """Execute and restore only the first recurrent row after ReadySeed.

        The observer borrows the retained MoE route, terminal hidden tensor,
        and vocabulary-sharded logits until it returns.  Production row
        execution is unchanged; this path only retains those outputs long
        enough for synchronous diagnostic readback.  No draft transaction is
        published and no downstream speculative row is evaluated.
        """

        self._require_healthy()
        self._require_runtime_owner("diagnose MTP extension position one")
        self._require_no_transaction("diagnose MTP extension position one")
        if not callable(observer):
            raise TypeError("position-one diagnostic observer must be callable")
        owner = self._validate_seed(seed)
        if seed.position != 1:
            raise Qwen38TTNNMTPAlignmentError(
                f"position-one diagnostic requires ReadySeed position 1, got {seed.position}"
            )
        observed: list[Any] = []

        def capture(value: Qwen38TTNNMTPExtensionPositionOneObservation) -> None:
            if observed:
                raise RuntimeError("position-one diagnostic observer was invoked more than once")
            observed.append(observer(value))

        try:
            snapshot = self.decoder_layer.snapshot_state(seed.state)
            next_state, residual, _selection, proof, prediction = self._execute_row(
                seed.first_draft_token_id,
                owner.residual,
                seed.state,
                position=seed.position,
                reuse=owner.selection,
                retain_input_state=True,
                row_observer=capture,
            )
            progress = _RunProgress(next_state, [residual])
        except _RowFailure as failure:
            progress = _RunProgress(failure.state, [failure.residual])
            restored, cleanup = self._rollback(progress, snapshot, label="failed position-one diagnostic")
            if restored is not None:
                self._live_state = seed.state
            self._poison(
                "MTP extension position-one diagnostic",
                failure.cause,
                (*failure.cleanup_errors, *cleanup),
            )
        except BaseException as error:
            self._poison("MTP extension position-one diagnostic snapshot", error)

        restored, cleanup = self._rollback(progress, snapshot, label="position-one diagnostic branch")
        if restored is None or cleanup:
            self._poison(
                "restore MTP extension position-one diagnostic",
                cleanup[0] if cleanup else RuntimeError("MTP diagnostic restore returned no state"),
                cleanup[1:],
            )
        try:
            self.decoder_layer.validate_state(seed.state)
            if not observed:
                raise RuntimeError("position-one diagnostic observer was not invoked")
        except BaseException as error:
            self._poison("validate restored MTP position-one ReadySeed", error)
        self._live_state = seed.state
        owner.state = seed.state
        return Qwen38TTNNMTPExtensionPositionOneResult(
            position=seed.position,
            input_token_id=seed.first_draft_token_id,
            predicted_token_id=prediction,
            qsa_selection=proof,
            observer_result=observed[0],
        )

    def _validate_transaction(self, batch: Any) -> _DraftTransaction:
        if type(batch) is not Qwen38MTPDraftBatch:
            raise TypeError("MTP transaction requires exact Qwen38MTPDraftBatch")
        transaction = batch.transaction
        if (
            type(transaction) is not _DraftTransaction
            or transaction.owner is not self
            or transaction.batch is not batch
        ):
            raise ValueError("MTP batch was not created by this engine")
        if not transaction.active or self._active_transaction is not transaction:
            raise RuntimeError("MTP draft transaction is no longer active")
        if transaction.seed_owner is not self._validate_seed(batch.seed):
            raise RuntimeError("MTP draft transaction lost its seed owner")
        return transaction

    def _normalize_alignment(
        self,
        emitted_token_ids: tuple[int, ...],
        target_hyper_residuals: tuple[Any, ...],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        if type(emitted_token_ids) is not tuple or not 1 <= len(emitted_token_ids) <= VERIFY_POSITIONS:
            raise ValueError("MTP alignment requires one through five emitted tokens")
        tokens = tuple(
            _require_token(token, label=f"authoritative emitted token {index}")
            for index, token in enumerate(emitted_token_ids)
        )
        roots = self._normalize_roots(target_hyper_residuals, count=len(tokens))
        return tokens, roots

    def preflight_alignment(
        self,
        batch: Qwen38MTPDraftBatch,
        emitted_token_ids: tuple[int, ...],
        target_hyper_residuals: tuple[Any, ...],
    ) -> None:
        self._require_healthy()
        self._require_runtime_owner("preflight MTP alignment")
        transaction = self._validate_transaction(batch)
        tokens, _ = self._normalize_alignment(emitted_token_ids, target_hyper_residuals)
        if batch.base_position + len(tokens) > self.allocated_context:
            raise ValueError("authoritative MTP alignment exceeds allocated context")
        self.decoder_layer.validate_restore_pair(transaction.current_state, transaction.snapshot)

    def _restore_draft(self, transaction: _DraftTransaction) -> Qwen38TTNNDecoderLayerState:
        restored, cleanup = self._rollback(
            _RunProgress(transaction.current_state, transaction.terminal_residual),
            transaction.snapshot,
            label="provisional MTP branch",
        )
        if restored is None or cleanup:
            transaction.active = False
            self._active_transaction = None
            self._poison(
                "restore provisional MTP branch",
                cleanup[0] if cleanup else RuntimeError("MTP restore returned no state"),
                cleanup[1:],
            )
        self._live_state = restored
        return restored

    def _align_from_seed(
        self,
        seed: Qwen38MTPSeed,
        seed_owner: _SeedOwner,
        tokens: tuple[int, ...],
        roots: tuple[Any, ...],
        state: Qwen38TTNNDecoderLayerState,
    ) -> tuple[Qwen38MTPSeed, tuple[Qwen38MTPAlignmentStep, ...]]:
        try:
            snapshot = self.decoder_layer.snapshot_state(state)
            progress = self._run_authoritative(tokens, roots, state, base_position=seed.position)
        except _RunFailure as failure:
            restored, cleanup = self._rollback(failure.progress, snapshot, label="failed authoritative alignment")
            if restored is not None:
                self._live_state = restored
            self._poison("target-conditioned MTP alignment", failure.cause, (*failure.cleanup_errors, *cleanup))
        except BaseException as error:
            self._poison("authoritative MTP alignment snapshot", error)
        try:
            self.decoder_layer.validate_commit_pair(progress.state, snapshot)
            committed = self.decoder_layer.commit_state(progress.state, snapshot)
            self._validate_state(committed, require_live=False)
            if committed.position != seed.position + len(tokens):
                raise RuntimeError("authoritative MTP alignment committed wrong position")
            if progress.selection is None or progress.prediction is None:
                raise RuntimeError("authoritative alignment committed without a final MTP seed output")
            final_proof = progress.alignment_steps[-1].qsa_selection
        except BaseException as error:
            self._poison("finalize authoritative MTP alignment", error)
        next_seed = self._publish_committed_seed(
            operation="publish authoritative MTP seed",
            state=committed,
            residual=progress.terminal_residual[0],
            selection=progress.selection,
            proof=final_proof,
            current_token_id=tokens[-1],
            first_draft_token_id=progress.prediction,
        )
        # Validate the complete replacement seed while every transferred root
        # and the old seed residual are still live.  Consumption is last.
        try:
            self._release_consumed((seed_owner.residual, *roots), label="consumed MTP seed and target roots")
        except BaseException as error:
            self._poison("consume authoritative MTP inputs", error)
        seed_owner.active = False
        return next_seed, tuple(progress.alignment_steps)

    def commit_alignment(
        self,
        batch: Qwen38MTPDraftBatch,
        emitted_token_ids: tuple[int, ...],
        target_hyper_residuals: tuple[Any, ...],
    ) -> Qwen38MTPAlignmentCommit:
        self.preflight_alignment(batch, emitted_token_ids, target_hyper_residuals)
        transaction = self._validate_transaction(batch)
        tokens, roots = self._normalize_alignment(emitted_token_ids, target_hyper_residuals)
        restored = self._restore_draft(transaction)
        transaction.active = False
        self._active_transaction = None
        next_seed, steps = self._align_from_seed(batch.seed, transaction.seed_owner, tokens, roots, restored)
        return Qwen38MTPAlignmentCommit(
            state=next_seed.state,
            position=next_seed.position,
            aligned_token_ids=tokens,
            consumed_hyper_residuals=roots,
            alignment_steps=steps,
            consumed_seed=batch.seed,
            seed=next_seed,
        )

    def advance_seed(
        self,
        seed: Qwen38MTPSeed,
        emitted_token_ids: tuple[int, ...],
        target_hyper_residuals: tuple[Any, ...],
    ) -> Qwen38MTPSeed:
        self._require_healthy()
        self._require_runtime_owner("advance MTP seed from ordinary decode")
        self._require_no_transaction("advance MTP seed from ordinary decode")
        owner = self._validate_seed(seed)
        tokens, roots = self._normalize_alignment(emitted_token_ids, target_hyper_residuals)
        if seed.position + len(tokens) > self.allocated_context:
            raise ValueError("ordinary target-conditioned MTP alignment exceeds allocated context")
        next_seed, _ = self._align_from_seed(seed, owner, tokens, roots, seed.state)
        return next_seed

    def advance_seed_rows(
        self,
        seed: Qwen38MTPSeed,
        shifted_rows: Iterable[tuple[int, Any]],
    ) -> Qwen38MTPSeed:
        """Stream ordinary target-conditioned rows with one-root residency.

        This is the long-suffix counterpart to
        :meth:`bootstrap_shifted_prefill_rows`.  It intentionally is not used
        by fixed-five commit, whose one-to-five accepted roots remain live as
        a single atomic ownership tuple until its replacement seed validates.
        """

        self._require_healthy()
        self._require_runtime_owner("stream ordinary MTP alignment")
        self._require_no_transaction("stream ordinary MTP alignment")
        owner = self._validate_seed(seed)
        try:
            iterator = iter(shifted_rows)
        except TypeError as error:
            raise TypeError("ordinary shifted alignment rows must be iterable") from error
        try:
            first = self._normalize_shifted_row(next(iterator), index=0)
        except StopIteration as error:
            raise ValueError("ordinary shifted alignment requires at least one row") from error
        self._validate_residual(first[1], label="ordinary shifted target root 0")
        outstanding_root: Any | None = first[1]
        progress = _RunProgress(seed.state)
        seen_roots: set[tuple[str, int]] = set()
        last_token = last_prediction = None
        try:
            snapshot = self.decoder_layer.snapshot_state(seed.state)
            pending_rows = [first]
            row_index = 0
            while pending_rows:
                token, root = pending_rows.pop()
                position = seed.position + row_index
                if position >= self.allocated_context:
                    raise ValueError("streamed ordinary MTP alignment exceeds allocated context")
                root_key = _tensor_key(root)
                if root_key in seen_roots:
                    raise ValueError("ordinary shifted roots must have distinct tensor ownership")
                self._validate_residual(root, label=f"ordinary shifted target root {row_index}")
                seen_roots.add(root_key)
                outstanding_root = root
                old_residual = progress.terminal_residual[0]
                try:
                    next_state, residual, selection, proof, prediction = self._execute_row(
                        token,
                        root,
                        progress.state,
                        position=position,
                        reuse=None,
                        retain_input_state=row_index == 0,
                    )
                except _RowFailure as failure:
                    progress.state = failure.state
                    if failure.residual is not None:
                        progress.terminal_residual[0] = failure.residual
                    raise _RunFailure(failure.cause, progress, failure.cleanup_errors) from failure
                progress.state = next_state
                progress.terminal_residual[0] = residual
                progress.selection = selection
                progress.prediction = prediction
                if old_residual is not None:
                    _release_slots([old_residual], label="superseded streamed ordinary MTP residual")
                progress.alignment_steps.append(Qwen38MTPAlignmentStep(row_index, position, token, proof))
                transferred_root = outstanding_root
                outstanding_root = None
                self._release_consumed(
                    (transferred_root,),
                    label=f"ordinary shifted target root {row_index}",
                )
                last_token, last_prediction = token, prediction
                row_index += 1
                try:
                    next_row = next(iterator)
                except StopIteration:
                    break
                pending_rows.append(self._normalize_shifted_row(next_row, index=row_index))
            self.decoder_layer.validate_commit_pair(progress.state, snapshot)
            committed = self.decoder_layer.commit_state(progress.state, snapshot)
            self._validate_state(committed, require_live=False)
            if (
                progress.selection is None
                or progress.prediction is None
                or last_token is None
                or last_prediction is None
            ):
                raise RuntimeError("streamed ordinary alignment committed without a final MTP seed output")
            final_proof = progress.alignment_steps[-1].qsa_selection
        except _RunFailure as failure:
            cleanup: list[BaseException] = list(failure.cleanup_errors)
            if outstanding_root is not None:
                try:
                    self._release_consumed((outstanding_root,), label="failed ordinary shifted target root")
                except BaseException as cleanup_error:
                    cleanup.append(cleanup_error)
            restored, rollback = self._rollback(failure.progress, snapshot, label="failed ordinary alignment")
            cleanup.extend(rollback)
            if restored is not None:
                self._live_state = restored
            self._poison("stream ordinary target-conditioned MTP alignment", failure.cause, cleanup)
        except BaseException as error:
            cleanup = []
            if outstanding_root is not None:
                try:
                    self._release_consumed((outstanding_root,), label="failed ordinary shifted target root")
                except BaseException as cleanup_error:
                    cleanup.append(cleanup_error)
            if "snapshot" in locals() and getattr(snapshot, "active", False):
                restored, rollback = self._rollback(progress, snapshot, label="failed streamed ordinary alignment")
                cleanup.extend(rollback)
                if restored is not None:
                    self._live_state = restored
            self._poison("stream ordinary target-conditioned MTP alignment", error, cleanup)
        next_seed = self._publish_committed_seed(
            operation="publish streamed-alignment MTP seed",
            state=committed,
            residual=progress.terminal_residual[0],
            selection=progress.selection,
            proof=final_proof,
            current_token_id=last_token,
            first_draft_token_id=last_prediction,
        )
        try:
            self._release_consumed((owner.residual,), label="consumed prior MTP seed residual")
        except BaseException as error:
            self._poison("consume prior streamed MTP seed", error)
        owner.active = False
        return next_seed

    def abort(self, batch: Qwen38MTPDraftBatch) -> Qwen38StateRollback:
        self._require_healthy()
        self._require_runtime_owner("abort MTP draft")
        transaction = self._validate_transaction(batch)
        restored = self._restore_draft(transaction)
        transaction.active = False
        self._active_transaction = None
        # QSA restore returns the protected seed view.  Keep the public seed's
        # exact wrapper identity live; it references that same restored view.
        try:
            self.decoder_layer.validate_state(batch.seed.state)
        except BaseException as error:
            self._poison("validate restored MTP seed", error)
        self._live_state = batch.seed.state
        transaction.seed_owner.state = batch.seed.state
        return Qwen38StateRollback(batch.seed.state, restored.position)

    def release_seed(self, seed: Qwen38MTPSeed) -> None:
        self._require_healthy()
        self._require_runtime_owner("release MTP seed")
        self._require_no_transaction("release MTP seed")
        owner = self._validate_seed(seed)
        try:
            self._release_consumed((seed.recurrent_residual,), label="MTP seed residual")
            self.decoder_layer.release_state(seed.state)
        except BaseException as error:
            self._poison("MTP seed release", error)
        owner.active = False
        self._live_seed = None
        self._live_state = None


def validate_mtp_draft_static_contract() -> None:
    if (
        DRAFT_STEPS,
        DRAFT_EXTENSION_CALLS,
        VERIFY_POSITIONS,
        MTP_LAYER_INDEX,
        MAX_SPECULATIVE_STEPS,
        RESIDUAL_LOCAL_SHAPE,
        EMBEDDING_LOCAL_SHAPE,
        BF4_STREAMED_LOAD_POLICY,
        BF4_RESIDENT_LOAD_POLICY,
        MTP_BOOTSTRAP_CONTRACT,
    ) != (
        4,
        3,
        5,
        0,
        4,
        (1, 4, 1, 640),
        (1, 1, 1, 640),
        "per-layer-call-serialized",
        "build-time-resident-components-owned",
        "shifted-target-conditioned-seed-v1",
    ):
        raise RuntimeError("Qwen3.8 concrete shifted MTP contract drifted")
    if Qwen38TTNNMTPDraftEngine.__abstractmethods__:
        raise RuntimeError("concrete Qwen3.8 MTP engine does not implement its full protocol")


validate_mtp_draft_static_contract()


__all__ = [
    "BF4_LOAD_POLICY",
    "BF4_RESIDENT_LOAD_POLICY",
    "BF4_STREAMED_LOAD_POLICY",
    "MTP_BOOTSTRAP_CONTRACT",
    "Qwen38TTNNMTPAlignmentError",
    "Qwen38TTNNMTPDraftCleanupError",
    "Qwen38TTNNMTPDraftEngine",
    "Qwen38TTNNMTPDraftError",
    "Qwen38TTNNMTPDraftPoisonedError",
    "validate_mtp_draft_static_contract",
]
