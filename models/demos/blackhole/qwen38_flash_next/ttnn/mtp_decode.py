# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact shifted-seed four-step MTP controller for Qwen3.8-Flash-Next.

The target state position counts consumed target inputs; its pending token has
been emitted but is not consumed.  MTP state is shifted by one token.  At a
target position ``p`` a live :class:`Qwen38MTPSeed` already represents the MTP
row ``(pending, target_hyper[p-1])`` at MTP position ``p-1`` and owns its
precomputed first draft ``d1``, terminal MTP residual, and QSA selection.

A four-token round therefore executes exactly three new MTP calls, for
``d1->d2``, ``d2->d3``, and ``d3->d4``.  One fixed-five target call verifies
``[pending,d1,d2,d3,d4]``.  If ``c`` target inputs commit, the MTP state is
restored to the seed and authoritatively aligned with ``c`` pairs
``(emitted[j], target_hyper[j])``.  The final aligned row produces the next
seed.  All committed target roots are consumed only after that alignment
commits; no stale target root survives in controller state.

This module is control/ownership code.  It cannot emulate a missing fixed-five
target operation and never substitutes five public one-token target calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NoReturn

DRAFT_STEPS = 4
DRAFT_EXTENSION_CALLS = DRAFT_STEPS - 1
VERIFY_POSITIONS = DRAFT_STEPS + 1
TP_SIZE = 4
VOCAB_SIZE = 248_320
MAX_CONTEXT = 262_144
QSA_COMPRESS_RATIO = 4
QSA_TOKEN_BUDGET = 2_048
QSA_MAX_REUSE_DISTANCE = 4
MTP_LAYER_INDEX = 0


class Qwen38SpeculativeStatus(str, Enum):
    READY = "ready"
    FINISHED = "finished"
    POISONED = "poisoned"
    HANDED_OFF = "handed_off"
    CLOSED = "closed"


class Qwen38SpeculativeStopReason(str, Enum):
    EOS = "eos"


class Qwen38SpeculativeTokenSource(str, Enum):
    ACCEPTED_DRAFT = "accepted_draft"
    TARGET_REPLACEMENT = "target_replacement"
    TARGET_BONUS = "target_bonus"


class Qwen38TargetCommitMode(str, Enum):
    PREFIX_TRANSACTION = "prefix_transaction"
    RESTORE_REPLAY = "restore_replay"


class Qwen38SpeculativeDecodeError(RuntimeError):
    pass


class Qwen38SpeculativeDecodeUnavailableError(Qwen38SpeculativeDecodeError):
    pass


class Qwen38SpeculativeDecodePoisonedError(Qwen38SpeculativeDecodeError):
    def __init__(self, operation: str, cause: BaseException, cleanup_errors: Sequence[BaseException] = ()) -> None:
        self.operation = operation
        self.cause = cause
        self.cleanup_errors = tuple(cleanup_errors)
        cleanup = ""
        if self.cleanup_errors:
            cleanup = "; rollback diagnostics: " + "; ".join(
                f"{type(error).__name__}: {error}" for error in self.cleanup_errors
            )
        super().__init__(
            f"TTNN MTP decode was poisoned during {operation}: {type(cause).__name__}: {cause}{cleanup}; "
            "terminate the task-owned process and release its lease without resetting hardware"
        )


@dataclass(frozen=True)
class Qwen38MTPQSASelectionProof:
    """Host proof for a selection and the state view produced by its call."""

    layer_index: int
    epoch: int
    source_view_id: int
    result_view_id: int
    source_position: int
    tail_start: int
    complete_token_count: int
    complete_indices_key: tuple[str, int] | None
    valid_token_count: int


@dataclass(frozen=True)
class Qwen38MTPSeed:
    """Live shifted MTP state with one already-computed draft token."""

    position: int
    current_token_id: int
    first_draft_token_id: int
    state: Any = field(repr=False, compare=False)
    recurrent_residual: Any = field(repr=False, compare=False)
    qsa_selection: Qwen38MTPQSASelectionProof
    transaction: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class Qwen38MTPDraftStep:
    step_index: int
    position: int
    input_token_id: int
    predicted_token_id: int
    qsa_selection: Qwen38MTPQSASelectionProof
    reused_qsa_selection: bool
    reused_from_view_id: int | None
    precomputed: bool


@dataclass(frozen=True)
class Qwen38MTPDraftBatch:
    """Seed plus three provisional MTP extensions."""

    base_position: int
    seed: Qwen38MTPSeed
    steps: tuple[Qwen38MTPDraftStep, ...]
    speculative_position: int
    transaction: Any = field(repr=False, compare=False)

    @property
    def current_token_id(self) -> int:
        return self.seed.current_token_id

    @property
    def draft_token_ids(self) -> tuple[int, int, int, int]:
        values = tuple(step.predicted_token_id for step in self.steps)
        if len(values) != DRAFT_STEPS:
            raise RuntimeError(f"MTP batch has {len(values)} drafts, expected {DRAFT_STEPS}")
        return values  # type: ignore[return-value]


@dataclass(frozen=True)
class Qwen38MTPAlignmentStep:
    """One authoritative target-conditioned MTP row."""

    step_index: int
    position: int
    input_token_id: int
    qsa_selection: Qwen38MTPQSASelectionProof


@dataclass(frozen=True)
class Qwen38TargetVerification:
    base_position: int
    input_token_ids: tuple[int, int, int, int, int]
    target_token_ids: tuple[int, int, int, int, int]
    target_hyper_residuals: tuple[Any, Any, Any, Any, Any] = field(repr=False, compare=False)
    speculative_position: int = 0
    transaction: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Qwen38StateRollback:
    state: Any = field(repr=False, compare=False)
    position: int


@dataclass(frozen=True)
class Qwen38TargetCommit:
    """Target prefix owner; all retained roots transfer as one ordered tuple."""

    state: Any = field(repr=False, compare=False)
    position: int
    committed_input_token_ids: tuple[int, ...]
    committed_hyper_residuals: tuple[Any, ...] = field(repr=False, compare=False)
    mode: Qwen38TargetCommitMode
    replayed_input_token_ids: tuple[int, ...] = ()

    @property
    def target_hyper_residual(self) -> Any:
        """Compatibility view only; ownership belongs to the full tuple."""

        if not self.committed_hyper_residuals:
            raise RuntimeError("target commit has no committed hyper-residual")
        return self.committed_hyper_residuals[-1]


@dataclass(frozen=True)
class Qwen38MTPAlignmentCommit:
    """Committed authoritative alignment and its replacement live seed."""

    state: Any = field(repr=False, compare=False)
    position: int
    aligned_token_ids: tuple[int, ...]
    consumed_hyper_residuals: tuple[Any, ...] = field(repr=False, compare=False)
    alignment_steps: tuple[Qwen38MTPAlignmentStep, ...]
    consumed_seed: Qwen38MTPSeed = field(repr=False, compare=False)
    seed: Qwen38MTPSeed


@dataclass(frozen=True)
class Qwen38SpeculativeEmission:
    token_id: int
    source: Qwen38SpeculativeTokenSource


@dataclass(frozen=True)
class Qwen38GreedyResolution:
    matched_draft_depth: int
    accepted_draft_count: int
    emissions: tuple[Qwen38SpeculativeEmission, ...]
    committed_input_count: int
    stop_reason: Qwen38SpeculativeStopReason | None


@dataclass(frozen=True)
class Qwen38SpeculativeRound:
    round_index: int
    base_position: int
    next_position: int
    input_token_ids: tuple[int, int, int, int, int]
    draft_token_ids: tuple[int, int, int, int]
    target_token_ids: tuple[int, int, int, int, int]
    matched_draft_depth: int
    accepted_draft_count: int
    committed_input_count: int
    emissions: tuple[Qwen38SpeculativeEmission, ...]
    pending_token_id: int
    stop_reason: Qwen38SpeculativeStopReason | None
    target_commit_mode: Qwen38TargetCommitMode
    mtp_extension_count: int
    mtp_alignment_count: int

    @property
    def mtp_replay_count(self) -> int:
        """Compatibility name for the authoritative alignment row count."""

        return self.mtp_alignment_count


@dataclass(frozen=True)
class Qwen38SpeculativeHandoff:
    identity_key: str
    target_state: Any = field(repr=False, compare=False)
    mtp_seed: Qwen38MTPSeed
    position: int
    pending_token_id: int
    finished: bool
    rounds: tuple[Qwen38SpeculativeRound, ...]

    @property
    def mtp_state(self) -> Any:
        return self.mtp_seed.state


class Qwen38FixedFiveTarget(ABC):
    """Strict device adapter for one true five-position target operation.

    A successful ``commit_prefix(..., c)`` transfers the exact ordered root
    prefix ``verification.target_hyper_residuals[:c]`` and releases only the
    suffix.  ``abort`` releases all five roots.  A serial public decode loop is
    not a valid implementation of ``verify_five``.
    """

    @property
    @abstractmethod
    def identity_key(self) -> str:
        ...

    @property
    @abstractmethod
    def allocated_context(self) -> int:
        ...

    @property
    @abstractmethod
    def poisoned(self) -> bool:
        ...

    @property
    @abstractmethod
    def poisoned_error(self) -> BaseException | None:
        ...

    @abstractmethod
    def claim_runtime_owner(self, owner: object) -> None:
        ...

    @abstractmethod
    def release_runtime_owner(self, owner: object) -> None:
        ...

    @abstractmethod
    def transfer_runtime_owner(self, current_owner: object, next_owner: object) -> None:
        ...

    @abstractmethod
    def state_position(self, state: Any) -> int:
        ...

    @abstractmethod
    def verify_five(
        self,
        input_token_ids: tuple[int, int, int, int, int],
        state: Any,
        *,
        base_position: int,
    ) -> Qwen38TargetVerification:
        ...

    @abstractmethod
    def preflight_commit(self, verification: Qwen38TargetVerification, committed_input_count: int) -> None:
        ...

    @abstractmethod
    def commit_prefix(self, verification: Qwen38TargetVerification, committed_input_count: int) -> Qwen38TargetCommit:
        ...

    @abstractmethod
    def abort(self, verification: Qwen38TargetVerification) -> Qwen38StateRollback:
        ...

    @abstractmethod
    def release_state(self, state: Any) -> None:
        ...


class Qwen38MTPDraftEngine(ABC):
    """Strict shifted-state MTP adapter.

    Bootstrap pairs ``target_hyper[i]`` with ``consumed[i+1]`` and pairs the
    final root with ``pending``.  ``draft_four`` consumes no seed ownership and
    executes three extensions.  ``commit_alignment`` restores the provisional
    branch, pairs every emitted token with its corresponding committed target
    root, commits the state, creates a replacement seed, then consumes the old
    seed and all roots.  ``advance_seed`` is the same authoritative operation
    without a speculative branch and supports ordinary/speculative handoff.
    """

    @property
    @abstractmethod
    def identity_key(self) -> str:
        ...

    @property
    @abstractmethod
    def allocated_context(self) -> int:
        ...

    @property
    @abstractmethod
    def poisoned(self) -> bool:
        ...

    @property
    @abstractmethod
    def poisoned_error(self) -> BaseException | None:
        ...

    @abstractmethod
    def claim_runtime_owner(self, owner: object) -> None:
        ...

    @abstractmethod
    def release_runtime_owner(self, owner: object) -> None:
        ...

    @abstractmethod
    def transfer_runtime_owner(self, current_owner: object, next_owner: object) -> None:
        ...

    @abstractmethod
    def state_position(self, state: Any) -> int:
        ...

    @abstractmethod
    def bootstrap_shifted_prefill(
        self,
        consumed_token_ids: tuple[int, ...],
        pending_token_id: int,
        target_hyper_residuals: tuple[Any, ...],
        state: Any,
    ) -> Qwen38MTPSeed:
        ...

    @abstractmethod
    def bootstrap_shifted_prefill_rows(
        self,
        shifted_rows: Iterable[tuple[int, Any]],
        state: Any,
    ) -> Qwen38MTPSeed:
        """Consume target-conditioned shifted rows incrementally.

        Each yielded root transfers to the engine for that row and may be
        released before the iterator is advanced.  This bounds prompt
        bootstrap residency without weakening post-verification transactions.
        """

        ...

    @abstractmethod
    def draft_four(self, seed: Qwen38MTPSeed) -> Qwen38MTPDraftBatch:
        ...

    @abstractmethod
    def preflight_alignment(
        self,
        batch: Qwen38MTPDraftBatch,
        emitted_token_ids: tuple[int, ...],
        target_hyper_residuals: tuple[Any, ...],
    ) -> None:
        ...

    @abstractmethod
    def commit_alignment(
        self,
        batch: Qwen38MTPDraftBatch,
        emitted_token_ids: tuple[int, ...],
        target_hyper_residuals: tuple[Any, ...],
    ) -> Qwen38MTPAlignmentCommit:
        ...

    @abstractmethod
    def advance_seed(
        self,
        seed: Qwen38MTPSeed,
        emitted_token_ids: tuple[int, ...],
        target_hyper_residuals: tuple[Any, ...],
    ) -> Qwen38MTPSeed:
        ...

    @abstractmethod
    def advance_seed_rows(
        self,
        seed: Qwen38MTPSeed,
        shifted_rows: Iterable[tuple[int, Any]],
    ) -> Qwen38MTPSeed:
        """Incrementally align ordinary target rows with bounded root residency."""

        ...

    @abstractmethod
    def abort(self, batch: Qwen38MTPDraftBatch) -> Qwen38StateRollback:
        ...

    @abstractmethod
    def release_seed(self, seed: Qwen38MTPSeed) -> None:
        ...

    @abstractmethod
    def release_state(self, state: Any) -> None:
        ...


def _require_token(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < VOCAB_SIZE:
        raise ValueError(f"{label} must be an integer in [0,{VOCAB_SIZE}), got {value!r}")
    return value


def _require_position(value: int, *, label: str, maximum: int = MAX_CONTEXT) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [0,{maximum}], got {value!r}")
    return value


def _require_allocated_context(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_CONTEXT:
        raise ValueError(f"{label} must be an integer in [1,{MAX_CONTEXT}], got {value!r}")
    return value


def _normalize_tokens(values: Sequence[int], *, length: int | None, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be an integer sequence")
    result = tuple(values)
    if length is not None and len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} tokens, got {len(result)}")
    return tuple(_require_token(value, label=f"{label}[{index}]") for index, value in enumerate(result))


def _normalize_eos(values: Sequence[int]) -> tuple[int, ...]:
    result = _normalize_tokens(values, length=None, label="EOS token IDs")
    if not result:
        raise ValueError("at least one pinned EOS token ID is required")
    if len(set(result)) != len(result):
        raise ValueError(f"EOS token IDs must be unique, got {result}")
    return result


def _require_identity_key(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be lowercase 64-hex, got {value!r}")
    return value


def resolve_greedy_five(
    draft_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    *,
    eos_token_ids: Sequence[int],
    diagnostic_exact_token_budget: bool = False,
) -> Qwen38GreedyResolution:
    """Resolve one fixed-five window, optionally treating EOS as an ordinary token.

    ``diagnostic_exact_token_budget`` exists only for bounded mechanical timing
    diagnostics that must execute their full requested token budget.  Normal
    decode retains EOS termination by default.
    """

    if type(diagnostic_exact_token_budget) is not bool:
        raise TypeError("diagnostic_exact_token_budget must be boolean")
    drafts = _normalize_tokens(draft_token_ids, length=4, label="MTP draft tokens")
    targets = _normalize_tokens(target_token_ids, length=5, label="target verifier tokens")
    eos = frozenset(_normalize_eos(eos_token_ids))
    terminal_eos = frozenset() if diagnostic_exact_token_budget else eos
    accepted: list[Qwen38SpeculativeEmission] = []
    matched = 0
    terminal_match = False
    for index, draft in enumerate(drafts):
        if draft != targets[index]:
            break
        matched += 1
        accepted.append(Qwen38SpeculativeEmission(draft, Qwen38SpeculativeTokenSource.ACCEPTED_DRAFT))
        if draft in terminal_eos:
            terminal_match = True
            break
    if terminal_match:
        emissions = tuple(accepted)
    elif matched < DRAFT_STEPS:
        emissions = (
            *accepted,
            Qwen38SpeculativeEmission(targets[matched], Qwen38SpeculativeTokenSource.TARGET_REPLACEMENT),
        )
    else:
        emissions = (*accepted, Qwen38SpeculativeEmission(targets[4], Qwen38SpeculativeTokenSource.TARGET_BONUS))
    first_eos = next((index for index, item in enumerate(emissions) if item.token_id in terminal_eos), None)
    if first_eos is not None:
        emissions = emissions[: first_eos + 1]
    if not 1 <= len(emissions) <= VERIFY_POSITIONS:
        raise AssertionError("greedy five-position resolution emitted an invalid token count")
    return Qwen38GreedyResolution(
        matched_draft_depth=matched,
        accepted_draft_count=sum(item.source is Qwen38SpeculativeTokenSource.ACCEPTED_DRAFT for item in emissions),
        emissions=emissions,
        committed_input_count=len(emissions),
        stop_reason=Qwen38SpeculativeStopReason.EOS if first_eos is not None else None,
    )


def _selection_signature(proof: Qwen38MTPQSASelectionProof) -> tuple[Any, ...]:
    return (
        proof.layer_index,
        proof.epoch,
        proof.source_position,
        proof.tail_start,
        proof.complete_token_count,
        proof.complete_indices_key,
    )


def _validate_qsa_proof(
    proof: Qwen38MTPQSASelectionProof,
    *,
    position: int,
    label: str,
    fresh: bool,
) -> None:
    if type(proof) is not Qwen38MTPQSASelectionProof:
        raise TypeError(f"{label} must carry the exact Qwen38 MTP QSA proof")
    if proof.layer_index != MTP_LAYER_INDEX:
        raise ValueError(f"{label} QSA layer must be MTP layer 0")
    for name in ("epoch", "source_view_id", "result_view_id"):
        value = getattr(proof, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} QSA {name} must be a positive integer")
    if fresh:
        if proof.source_position != position or proof.source_view_id != proof.result_view_id:
            raise ValueError(f"{label} fresh QSA selection must originate from its result view and row")
    elif not 0 <= position - proof.source_position <= QSA_MAX_REUSE_DISTANCE:
        raise ValueError(f"{label} QSA reuse distance is outside [0,{QSA_MAX_REUSE_DISTANCE}]")
    context_length = position + 1
    if not 0 <= proof.tail_start <= context_length:
        raise ValueError(f"{label} QSA tail starts outside its context")
    if (
        isinstance(proof.complete_token_count, bool)
        or not isinstance(proof.complete_token_count, int)
        or proof.complete_token_count % QSA_COMPRESS_RATIO
        or not 0 <= proof.complete_token_count <= QSA_TOKEN_BUDGET
    ):
        raise ValueError(f"{label} has invalid complete-token count")
    if (proof.complete_indices_key is None) is not (proof.complete_token_count == 0):
        raise ValueError(f"{label} complete-index allocation/count disagree")
    expected_valid = proof.complete_token_count + context_length - proof.tail_start
    if proof.valid_token_count != expected_valid or expected_valid <= 0:
        raise ValueError(f"{label} QSA valid-token count does not match complete blocks plus causal tail")


def _validate_seed(seed: Qwen38MTPSeed, *, label: str) -> None:
    if type(seed) is not Qwen38MTPSeed:
        raise TypeError(f"{label} must be the exact Qwen38 MTP seed")
    position = _require_position(seed.position, label=f"{label} position")
    if position == 0:
        raise ValueError(f"{label} requires at least one target-conditioned shifted row")
    _require_token(seed.current_token_id, label=f"{label} current token")
    _require_token(seed.first_draft_token_id, label=f"{label} first draft")
    if seed.state is None or seed.recurrent_residual is None or seed.transaction is None:
        raise ValueError(f"{label} is missing live state, residual, or owner")
    _validate_qsa_proof(seed.qsa_selection, position=position - 1, label=f"{label} selection", fresh=True)


def _validate_draft_steps(batch: Qwen38MTPDraftBatch) -> tuple[int, int, int, int]:
    seed = batch.seed
    base = batch.base_position
    steps = tuple(batch.steps)
    if len(steps) != DRAFT_STEPS:
        raise ValueError(f"MTP batch has {len(steps)} draft records, expected four")
    seen_results: set[int] = set()
    signature = _selection_signature(seed.qsa_selection)
    seed_source = seed.qsa_selection.source_view_id
    drafts: list[int] = []
    for index, step in enumerate(steps):
        if type(step) is not Qwen38MTPDraftStep:
            raise TypeError(f"MTP draft record {index} has the wrong type")
        expected_position = base - 1 if index == 0 else base + index - 1
        expected_input = seed.current_token_id if index == 0 else drafts[index - 1]
        if step.step_index != index or step.position != expected_position or step.input_token_id != expected_input:
            raise ValueError(f"MTP draft record {index} has wrong index, position, or recurrent input")
        _require_token(step.predicted_token_id, label=f"MTP draft {index}")
        if index == 0:
            if not step.precomputed or step.reused_qsa_selection or step.reused_from_view_id is not None:
                raise ValueError("MTP d1 must be the precomputed seed row, not a new recurrent call")
            if step.predicted_token_id != seed.first_draft_token_id or step.qsa_selection != seed.qsa_selection:
                raise ValueError("MTP d1 record does not preserve the exact seed")
            _validate_qsa_proof(step.qsa_selection, position=expected_position, label="MTP seed", fresh=True)
        else:
            if step.precomputed or not step.reused_qsa_selection or step.reused_from_view_id != seed_source:
                raise ValueError(f"MTP draft extension {index} did not reuse the latest seed-owned selection")
            _validate_qsa_proof(
                step.qsa_selection, position=expected_position, label=f"MTP extension {index}", fresh=False
            )
            if step.qsa_selection.source_view_id != seed_source:
                raise ValueError(f"MTP draft extension {index} changed the QSA selection source view")
            if _selection_signature(step.qsa_selection) != signature:
                raise ValueError(f"MTP draft extension {index} changed the frozen QSA complete selection")
            if step.qsa_selection.valid_token_count != seed.qsa_selection.valid_token_count + index:
                raise ValueError(f"MTP draft extension {index} did not grow only the causal tail")
        result_view = step.qsa_selection.result_view_id
        if result_view in seen_results:
            raise ValueError(f"MTP draft extension {index} reused a QSA result view")
        seen_results.add(result_view)
        drafts.append(step.predicted_token_id)
    return tuple(drafts)  # type: ignore[return-value]


class Qwen38SpeculativeDecodeController:
    def __init__(
        self,
        target: Qwen38FixedFiveTarget,
        mtp: Qwen38MTPDraftEngine,
        *,
        target_state: Any,
        mtp_seed: Qwen38MTPSeed,
        pending_token_id: int,
        eos_token_ids: Sequence[int],
        diagnostic_exact_token_budget: bool = False,
        previous_target_owner: object | None = None,
        previous_mtp_owner: object | None = None,
    ) -> None:
        if not isinstance(target, Qwen38FixedFiveTarget):
            raise TypeError("target must implement the strict fixed-five Qwen38 TTNN protocol")
        if not isinstance(mtp, Qwen38MTPDraftEngine):
            raise TypeError("mtp must implement the strict shifted-seed Qwen38 TTNN MTP protocol")
        if target is mtp:
            raise ValueError("target and MTP require distinct state owners")
        _validate_seed(mtp_seed, label="MTP handoff seed")
        pending = _require_token(pending_token_id, label="pending token ID")
        if mtp_seed.current_token_id != pending:
            raise ValueError("MTP seed current token does not match the target pending token")
        target_key = _require_identity_key(target.identity_key, label="target identity key")
        mtp_key = _require_identity_key(mtp.identity_key, label="MTP identity key")
        if target_key != mtp_key:
            raise ValueError("target and MTP components do not share one live build identity")
        if target.poisoned or mtp.poisoned:
            raise Qwen38SpeculativeDecodeError("cannot claim a poisoned target or MTP backend")
        target_capacity = _require_allocated_context(target.allocated_context, label="target allocated context")
        mtp_capacity = _require_allocated_context(mtp.allocated_context, label="MTP allocated context")
        if target_capacity != mtp_capacity:
            raise ValueError(
                f"target allocated context {target_capacity} differs from MTP allocated context {mtp_capacity}"
            )
        if type(diagnostic_exact_token_budget) is not bool:
            raise TypeError("diagnostic_exact_token_budget must be boolean")
        self.target, self.mtp = target, mtp
        self.allocated_context = target_capacity
        self.identity_key = target_key
        self.eos_token_ids = _normalize_eos(eos_token_ids)
        self._eos_token_set = frozenset(self.eos_token_ids)
        self._diagnostic_exact_token_budget = diagnostic_exact_token_budget
        self._target_state = target_state
        self._mtp_seed = mtp_seed
        self._pending_token_id = pending
        self._runtime_owner = object()
        self._status = Qwen38SpeculativeStatus.READY
        self._poison_error: Qwen38SpeculativeDecodePoisonedError | None = None
        self._rounds: list[Qwen38SpeculativeRound] = []
        self._owns_runtime = False
        acquired: list[tuple[Any, object | None]] = []
        try:
            for backend, previous in ((target, previous_target_owner), (mtp, previous_mtp_owner)):
                if previous is None:
                    backend.claim_runtime_owner(self._runtime_owner)
                else:
                    backend.transfer_runtime_owner(previous, self._runtime_owner)
                acquired.append((backend, previous))
            target_position = _require_position(
                target.state_position(target_state),
                label="target state position",
                maximum=self.allocated_context,
            )
            mtp_position = _require_position(
                mtp.state_position(mtp_seed.state),
                label="MTP state position",
                maximum=self.allocated_context,
            )
            if target_position != mtp_position or target_position != mtp_seed.position:
                raise ValueError(
                    f"target position {target_position}, MTP state {mtp_position}, and seed {mtp_seed.position} are not aligned"
                )
            if target_position >= self.allocated_context:
                raise ValueError("speculative handoff cannot start at exhausted allocated context")
            self._position = target_position
            self._owns_runtime = True
            if not self._diagnostic_exact_token_budget and pending in self._eos_token_set:
                self._status = Qwen38SpeculativeStatus.FINISHED
        except BaseException as error:
            cleanup: list[BaseException] = []
            for backend, previous in reversed(acquired):
                try:
                    if previous is None:
                        backend.release_runtime_owner(self._runtime_owner)
                    else:
                        backend.transfer_runtime_owner(self._runtime_owner, previous)
                except BaseException as cleanup_error:
                    cleanup.append(cleanup_error)
            if cleanup:
                raise Qwen38SpeculativeDecodeError(
                    "controller construction failed and runtime ownership could not be restored: "
                    + "; ".join(str(item) for item in cleanup)
                ) from error
            raise

    @property
    def status(self) -> Qwen38SpeculativeStatus:
        return self._status

    @property
    def position(self) -> int:
        self._assert_live("read position")
        return self._position

    @property
    def pending_token_id(self) -> int:
        self._assert_live("read pending token")
        return self._pending_token_id

    @property
    def rounds(self) -> tuple[Qwen38SpeculativeRound, ...]:
        return tuple(self._rounds)

    @property
    def poison_error(self) -> Qwen38SpeculativeDecodePoisonedError | None:
        return self._poison_error

    def _backend_poison_cause(self) -> BaseException | None:
        for label, backend in (("target", self.target), ("MTP", self.mtp)):
            try:
                if backend.poisoned:
                    return backend.poisoned_error or RuntimeError(f"{label} reports poison without an exception")
            except BaseException as error:
                return RuntimeError(f"failed to inspect {label} poison state: {error}")
        return None

    def _poison(self, operation: str, error: BaseException, cleanup_errors: Sequence[BaseException] = ()) -> NoReturn:
        if self._poison_error is None:
            self._poison_error = Qwen38SpeculativeDecodePoisonedError(
                operation, self._backend_poison_cause() or error, cleanup_errors
            )
        self._status = Qwen38SpeculativeStatus.POISONED
        raise self._poison_error from error

    def _assert_live(self, operation: str) -> None:
        if self._status is Qwen38SpeculativeStatus.POISONED:
            assert self._poison_error is not None
            raise self._poison_error
        if self._status is Qwen38SpeculativeStatus.HANDED_OFF:
            raise Qwen38SpeculativeDecodeError(f"cannot {operation}: runtime state was handed off")
        if self._status is Qwen38SpeculativeStatus.CLOSED:
            raise Qwen38SpeculativeDecodeError(f"cannot {operation}: controller is closed")
        cause = self._backend_poison_cause()
        if cause is not None:
            self._poison(operation, cause)

    def _assert_ready(self) -> None:
        self._assert_live("run an MTP round")
        if self._status is not Qwen38SpeculativeStatus.READY:
            raise Qwen38SpeculativeDecodeError(f"cannot run an MTP round while status is {self._status.value}")
        if self._position + VERIFY_POSITIONS > self.allocated_context:
            raise Qwen38SpeculativeDecodeUnavailableError(
                "fixed-five target verification exceeds allocated context; handoff to ordinary decode"
            )

    def _validate_batch(self, batch: Qwen38MTPDraftBatch) -> tuple[int, int, int, int]:
        if type(batch) is not Qwen38MTPDraftBatch or batch.seed is not self._mtp_seed:
            raise ValueError("MTP transaction did not retain the exact live seed")
        if batch.base_position != self._position or batch.speculative_position != self._position + 3:
            raise ValueError("MTP transaction reports the wrong base or three-extension position")
        if batch.transaction is None:
            raise ValueError("MTP transaction requires an opaque owner")
        return _validate_draft_steps(batch)

    def _validate_verification(self, verification: Any, expected_inputs: tuple[int, ...]) -> tuple[int, ...]:
        if type(verification) is not Qwen38TargetVerification:
            raise TypeError("target returned a non-Qwen38 fixed-five transaction")
        if verification.base_position != self._position or verification.input_token_ids != expected_inputs:
            raise ValueError("fixed-five target reports wrong inputs or base position")
        targets = _normalize_tokens(verification.target_token_ids, length=5, label="target predictions")
        if len(verification.target_hyper_residuals) != 5 or any(
            root is None for root in verification.target_hyper_residuals
        ):
            raise ValueError("fixed-five target must retain five distinct root rows")
        if len({id(root) for root in verification.target_hyper_residuals}) != 5:
            raise ValueError("fixed-five target root rows must have distinct ownership identities")
        if verification.speculative_position != self._position + 5 or verification.transaction is None:
            raise ValueError("fixed-five target transaction has wrong position or no owner")
        return targets

    def _validate_target_commit(
        self, commit: Any, verification: Qwen38TargetVerification, committed_inputs: tuple[int, ...]
    ) -> None:
        if type(commit) is not Qwen38TargetCommit:
            raise TypeError("target commit returned the wrong owner type")
        expected_position = self._position + len(committed_inputs)
        if commit.position != expected_position or commit.committed_input_token_ids != committed_inputs:
            raise ValueError("target commit reports wrong input prefix or position")
        expected_roots = verification.target_hyper_residuals[: len(committed_inputs)]
        if len(commit.committed_hyper_residuals) != len(expected_roots) or any(
            actual is not expected for actual, expected in zip(commit.committed_hyper_residuals, expected_roots)
        ):
            raise ValueError("target commit did not transfer the exact committed target root prefix")
        if commit.target_hyper_residual is not expected_roots[-1]:
            raise ValueError("target compatibility root does not expose the final prefix identity")
        if type(commit.mode) is not Qwen38TargetCommitMode:
            raise TypeError("target commit must disclose its prefix transaction mode")
        if commit.mode is Qwen38TargetCommitMode.RESTORE_REPLAY:
            if commit.replayed_input_token_ids != committed_inputs:
                raise ValueError("target restore/replay did not replay exact prefix")
        elif commit.replayed_input_token_ids:
            raise ValueError("target prefix transaction cannot report replayed tokens")
        if self.target.state_position(commit.state) != expected_position:
            raise RuntimeError("target committed state is misaligned")

    def _validate_alignment(
        self,
        commit: Any,
        batch: Qwen38MTPDraftBatch,
        emitted: tuple[int, ...],
        roots: tuple[Any, ...],
    ) -> None:
        if type(commit) is not Qwen38MTPAlignmentCommit:
            raise TypeError("MTP alignment returned the wrong commit owner")
        expected_position = self._position + len(emitted)
        if commit.position != expected_position or commit.aligned_token_ids != emitted:
            raise ValueError("MTP alignment reports wrong emitted tokens or position")
        if commit.consumed_seed is not batch.seed:
            raise ValueError("MTP alignment did not consume the exact old seed")
        if len(commit.consumed_hyper_residuals) != len(roots) or any(
            actual is not expected for actual, expected in zip(commit.consumed_hyper_residuals, roots)
        ):
            raise ValueError("MTP alignment did not consume every transferred target root in order")
        steps = tuple(commit.alignment_steps)
        if len(steps) != len(emitted):
            raise ValueError("MTP alignment row count differs from committed target prefix")
        for index, step in enumerate(steps):
            if type(step) is not Qwen38MTPAlignmentStep:
                raise TypeError("MTP alignment returned a wrong row record")
            position = self._position + index
            if step.step_index != index or step.position != position or step.input_token_id != emitted[index]:
                raise ValueError("MTP authoritative alignment row order changed")
            _validate_qsa_proof(step.qsa_selection, position=position, label=f"MTP alignment {index}", fresh=True)
        _validate_seed(commit.seed, label="replacement MTP seed")
        if commit.seed.state is not commit.state or commit.seed.position != expected_position:
            raise ValueError("replacement MTP seed/state/position disagree")
        if commit.seed.current_token_id != emitted[-1]:
            raise ValueError("replacement MTP seed does not carry the final emitted token")
        if commit.seed.qsa_selection != steps[-1].qsa_selection:
            raise ValueError("replacement MTP seed did not retain the final authoritative QSA selection")
        if self.mtp.state_position(commit.state) != expected_position:
            raise RuntimeError("MTP aligned state is misaligned")

    def _abort_transactions(self, verification: Any | None, batch: Any | None) -> list[BaseException]:
        errors: list[BaseException] = []
        if verification is not None:
            try:
                rollback = self.target.abort(verification)
                if type(rollback) is not Qwen38StateRollback or rollback.position != self._position:
                    raise RuntimeError("target abort did not restore base position")
            except BaseException as error:
                errors.append(error)
        if batch is not None:
            try:
                rollback = self.mtp.abort(batch)
                if type(rollback) is not Qwen38StateRollback or rollback.position != self._position:
                    raise RuntimeError("MTP abort did not restore seed position")
            except BaseException as error:
                errors.append(error)
        return errors

    def step(self) -> Qwen38SpeculativeRound:
        self._assert_ready()
        batch = None
        verification = None
        commit_started = False
        try:
            batch = self.mtp.draft_four(self._mtp_seed)
            drafts = self._validate_batch(batch)
            verify_inputs = (self._pending_token_id, *drafts)
            verification = self.target.verify_five(verify_inputs, self._target_state, base_position=self._position)
            targets = self._validate_verification(verification, verify_inputs)
            resolution = resolve_greedy_five(
                drafts,
                targets,
                eos_token_ids=self.eos_token_ids,
                diagnostic_exact_token_budget=self._diagnostic_exact_token_budget,
            )
            count = resolution.committed_input_count
            committed_inputs = verify_inputs[:count]
            emitted = tuple(item.token_id for item in resolution.emissions)
            roots = verification.target_hyper_residuals[:count]
            if self.target.preflight_commit(verification, count) is not None:
                raise RuntimeError("target preflight must return None")
            if self.mtp.preflight_alignment(batch, emitted, roots) is not None:
                raise RuntimeError("MTP alignment preflight must return None")
            commit_started = True
            target_commit = self.target.commit_prefix(verification, count)
            self._validate_target_commit(target_commit, verification, committed_inputs)
            alignment = self.mtp.commit_alignment(batch, emitted, target_commit.committed_hyper_residuals)
            self._validate_alignment(alignment, batch, emitted, target_commit.committed_hyper_residuals)
            self._target_state = target_commit.state
            self._mtp_seed = alignment.seed
            self._position = target_commit.position
            self._pending_token_id = emitted[-1]
            if resolution.stop_reason is Qwen38SpeculativeStopReason.EOS:
                self._status = Qwen38SpeculativeStatus.FINISHED
            event = Qwen38SpeculativeRound(
                round_index=len(self._rounds),
                base_position=verification.base_position,
                next_position=self._position,
                input_token_ids=verify_inputs,
                draft_token_ids=drafts,
                target_token_ids=targets,  # type: ignore[arg-type]
                matched_draft_depth=resolution.matched_draft_depth,
                accepted_draft_count=resolution.accepted_draft_count,
                committed_input_count=count,
                emissions=resolution.emissions,
                pending_token_id=self._pending_token_id,
                stop_reason=resolution.stop_reason,
                target_commit_mode=target_commit.mode,
                mtp_extension_count=DRAFT_EXTENSION_CALLS,
                mtp_alignment_count=len(alignment.alignment_steps),
            )
            self._rounds.append(event)
            return event
        except Qwen38SpeculativeDecodePoisonedError:
            raise
        except BaseException as error:
            cleanup = [] if commit_started else self._abort_transactions(verification, batch)
            self._poison("shifted-seed fixed-five speculative step", error, cleanup)

    def handoff(self, *, next_target_owner: object, next_mtp_owner: object) -> Qwen38SpeculativeHandoff:
        self._assert_live("handoff runtime state")
        if next_target_owner is None or next_mtp_owner is None:
            raise TypeError("handoff successor owner tokens cannot be None")
        target_transferred = False
        try:
            self.target.transfer_runtime_owner(self._runtime_owner, next_target_owner)
            target_transferred = True
            self.mtp.transfer_runtime_owner(self._runtime_owner, next_mtp_owner)
        except BaseException as error:
            cleanup: list[BaseException] = []
            if target_transferred:
                try:
                    self.target.transfer_runtime_owner(next_target_owner, self._runtime_owner)
                except BaseException as cleanup_error:
                    cleanup.append(cleanup_error)
            self._poison("runtime-owner handoff", error, cleanup)
        result = Qwen38SpeculativeHandoff(
            identity_key=self.identity_key,
            target_state=self._target_state,
            mtp_seed=self._mtp_seed,
            position=self._position,
            pending_token_id=self._pending_token_id,
            finished=self._status is Qwen38SpeculativeStatus.FINISHED,
            rounds=tuple(self._rounds),
        )
        self._target_state = None
        self._mtp_seed = None
        self._owns_runtime = False
        self._status = Qwen38SpeculativeStatus.HANDED_OFF
        return result

    def close(self) -> None:
        if self._status is Qwen38SpeculativeStatus.CLOSED:
            return
        self._assert_live("close")
        try:
            self.target.release_state(self._target_state)
            self.mtp.release_seed(self._mtp_seed)
            self.target.release_runtime_owner(self._runtime_owner)
            self.mtp.release_runtime_owner(self._runtime_owner)
        except BaseException as error:
            self._poison("close", error)
        self._target_state = None
        self._mtp_seed = None
        self._owns_runtime = False
        self._status = Qwen38SpeculativeStatus.CLOSED


def validate_mtp_decode_static_contract() -> None:
    if (DRAFT_STEPS, DRAFT_EXTENSION_CALLS, VERIFY_POSITIONS, TP_SIZE, VOCAB_SIZE, MTP_LAYER_INDEX) != (
        4,
        3,
        5,
        4,
        248_320,
        0,
    ):
        raise RuntimeError("Qwen3.8 shifted MTP geometry drifted")
    target_methods = {
        "allocated_context",
        "claim_runtime_owner",
        "release_runtime_owner",
        "transfer_runtime_owner",
        "state_position",
        "verify_five",
        "preflight_commit",
        "commit_prefix",
        "abort",
        "release_state",
    }
    mtp_methods = {
        "allocated_context",
        "claim_runtime_owner",
        "release_runtime_owner",
        "transfer_runtime_owner",
        "state_position",
        "bootstrap_shifted_prefill",
        "bootstrap_shifted_prefill_rows",
        "draft_four",
        "preflight_alignment",
        "commit_alignment",
        "advance_seed",
        "advance_seed_rows",
        "abort",
        "release_seed",
        "release_state",
    }
    if not target_methods.issubset(Qwen38FixedFiveTarget.__abstractmethods__):
        raise RuntimeError("fixed-five target protocol lost a required method")
    if not mtp_methods.issubset(Qwen38MTPDraftEngine.__abstractmethods__):
        raise RuntimeError("shifted MTP protocol lost a required method")


validate_mtp_decode_static_contract()


__all__ = [
    "DRAFT_EXTENSION_CALLS",
    "DRAFT_STEPS",
    "VERIFY_POSITIONS",
    "Qwen38FixedFiveTarget",
    "Qwen38GreedyResolution",
    "Qwen38MTPAlignmentCommit",
    "Qwen38MTPAlignmentStep",
    "Qwen38MTPDraftBatch",
    "Qwen38MTPDraftEngine",
    "Qwen38MTPDraftStep",
    "Qwen38MTPQSASelectionProof",
    "Qwen38MTPSeed",
    "Qwen38SpeculativeDecodeController",
    "Qwen38SpeculativeDecodeError",
    "Qwen38SpeculativeDecodePoisonedError",
    "Qwen38SpeculativeDecodeUnavailableError",
    "Qwen38SpeculativeEmission",
    "Qwen38SpeculativeHandoff",
    "Qwen38SpeculativeRound",
    "Qwen38SpeculativeStatus",
    "Qwen38SpeculativeStopReason",
    "Qwen38SpeculativeTokenSource",
    "Qwen38StateRollback",
    "Qwen38TargetCommit",
    "Qwen38TargetCommitMode",
    "Qwen38TargetVerification",
    "resolve_greedy_five",
    "validate_mtp_decode_static_contract",
]
