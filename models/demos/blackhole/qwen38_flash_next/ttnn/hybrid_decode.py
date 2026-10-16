# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact ordinary/MTP TTNN session owner for Qwen3.8-Flash-Next.

This module joins the already-built 48-layer target, the checkpoint's one
shifted MTP layer, and the true fixed-five target verifier.  Ordinary mode can
use the provenance-bound TP4 host sampler, including the official thinking and
non-thinking profiles.  MTP remains exact-greedy only; stochastic speculative
acceptance is rejected rather than approximated.

The target cache position counts consumed tokens.  ``pending_token_id`` is the
last emitted token and is not yet in the target cache.  The shifted MTP seed is
kept at the same position and has already consumed the target-conditioned pair
for that pending token, so it owns a precomputed first draft.  A speculative
round consequently performs exactly three proposal extensions, verifies five
target rows, and authoritatively aligns one through five committed rows.

Prompt and ordinary alignment are streamed.  The MTP engine consumes and
releases one yielded target hyper-residual before requesting the next target
row; this owner never retains a prompt-sized tuple of roots.  Ownership leaves
``Qwen38TTNNTextModelOutput`` only through its public
``take_hyper_residual()`` method.

Multi-turn reuse is fail closed.  A newly rendered full token history may
reuse caches only when its prefix exactly equals ``consumed_token_ids`` and
the remaining suffix begins with ``pending_token_id``.  A caller must request
``cache_policy=RESET`` to handle any template/tokenization drift; reset then
rebuilds both target and MTP state from position zero.

No function in this module opens a mesh, discovers a device, changes
visibility, acquires a lease, or resets hardware.  Construction assumes an
already-qualified, already-open physical 1x4 mesh held by the process-wide
lease owner.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, NoReturn

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import Qwen38BF4ResidentSet
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    Qwen38BuildProvenance,
    Qwen38LiveBuildIdentity,
    Qwen38TTNNBuilder,
    Qwen38TTNNBuiltTarget,
    Qwen38TTNNMTPComponents,
    Qwen38TTNNTargetComponents,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, TP_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.fixed_five import Qwen38TTNNFixedFiveTarget
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import BACKBONE_LAYERS
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    MAX_CONTEXT,
    VOCAB_SIZE,
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelOutput,
    Qwen38TTNNTextModelState,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_decode import (
    DRAFT_EXTENSION_CALLS,
    VERIFY_POSITIONS,
    Qwen38MTPSeed,
    Qwen38SpeculativeDecodeController,
    Qwen38SpeculativeRound,
    Qwen38SpeculativeStatus,
    Qwen38SpeculativeTokenSource,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_draft import Qwen38TTNNMTPDraftEngine
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38SampledTokens,
    Qwen38SamplingParameters,
    Qwen38SamplingProfile,
    Qwen38TTNNHostSampler,
)

Clock = Callable[[], int]
Synchronize = Callable[[], None]
DIAGNOSTIC_EXACT_TOKEN_BUDGET = 32

# A decoder layer emits exactly ten ownership-stage pairs.  GDN and QSA are
# mutually exclusive attention branches, so eleven ownership-stage names are
# valid globally.  The nested MoE observer adds ten mechanics-only stages.
_READY_SEED_LAYER_INTERNAL_STAGES = frozenset(
    {
        "ple",
        "attention-gr-read",
        "gdn",
        "qsa",
        "attention-gr-write",
        "mlp-gr-read",
        "expert-stream-acquire",
        "moe-forward",
        "expert-stream-release",
        "mlp-gr-write",
        "state-update",
        "hidden-all-gather",
        "router-logits",
        "router-topk",
        "shared-partial",
        "routed-dispatch",
        "moe-compute-launch",
        "selective-reduce",
        "partial-combine",
        "output-reduce-scatter",
        "output-release",
    }
)


def _translate_ready_seed_target_phase(phase: str, *, row_index: int) -> str:
    boundary, separator, component = phase.partition("-")
    if separator == "-" and boundary in {"before", "after"} and component:
        return f"{boundary}-ready-seed-target-row-{row_index}-{component}"

    parts = phase.split("-", 3)
    if len(parts) == 4 and parts[0] == "layer" and parts[2] in {"before", "after"}:
        layer_text, boundary, stage = parts[1:]
        if layer_text.isdecimal():
            layer_index = int(layer_text)
            if (
                layer_text == str(layer_index)
                and 0 <= layer_index < BACKBONE_LAYERS
                and stage in _READY_SEED_LAYER_INTERNAL_STAGES
            ):
                return f"{boundary}-ready-seed-target-row-{row_index}-layer-{layer_index}-{stage}"
    raise ValueError(f"invalid target decode phase {phase!r}")


class Qwen38HybridMode(str, Enum):
    ORDINARY = "ordinary"
    MTP = "mtp"


class Qwen38HybridCachePolicy(str, Enum):
    MATCH = "match"
    RESET = "reset"


class Qwen38HybridSessionStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FINISHED = "finished"
    POISONED = "poisoned"
    CLOSED = "closed"


class Qwen38HybridStopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONTEXT_LENGTH = "context_length"


class Qwen38HybridFallbackReason(str, Enum):
    OUTPUT_BUDGET = "remaining_output_budget_below_fixed_five"
    CONTEXT_TAIL = "native_context_tail_below_fixed_five"


class Qwen38HybridTimingPhase(str, Enum):
    TTFT = "ttft"
    ORDINARY_DECODE = "ordinary_decode"
    SPECULATIVE_DECODE = "speculative_decode"


class Qwen38HybridTokenSource(str, Enum):
    TARGET_GREEDY = "target_greedy"
    TARGET_SAMPLED = "target_sampled"
    ORDINARY_GREEDY = "ordinary_greedy"
    ORDINARY_SAMPLED = "ordinary_sampled"
    ACCEPTED_DRAFT = "accepted_draft"
    TARGET_REPLACEMENT = "target_replacement"
    TARGET_BONUS = "target_bonus"


class Qwen38HybridDecodeError(RuntimeError):
    """Base error for ordinary sampled decode and exact-greedy MTP decode."""


class Qwen38HybridCacheMismatchError(Qwen38HybridDecodeError):
    """A rendered history cannot be spliced onto the current device caches."""


class Qwen38HybridDecodePoisonedError(Qwen38HybridDecodeError):
    """A device transaction may be partially mutated and cannot be reused."""

    def __init__(
        self,
        operation: str,
        cause: BaseException,
        *,
        partial_session: "Qwen38TTNNHybridDecodeSession | None" = None,
    ) -> None:
        self.operation = operation
        self.cause = cause
        self.partial_session = partial_session
        super().__init__(
            f"Qwen3.8 hybrid TTNN decode was poisoned during {operation}: "
            f"{type(cause).__name__}: {cause}; terminate the task-owned process "
            "and release its lease without resetting hardware"
        )


@dataclass(frozen=True)
class Qwen38HybridCleanupFailure:
    resource: str
    error: BaseException = field(repr=False, compare=False)

    @property
    def description(self) -> str:
        return f"{self.resource}: {type(self.error).__name__}: {self.error}"


class Qwen38HybridCleanupError(Qwen38HybridDecodeError):
    """Best-effort cleanup left explicit task-owned resources unresolved."""

    def __init__(self, operation: str, failures: Sequence[Qwen38HybridCleanupFailure]) -> None:
        values = tuple(failures)
        if not values:
            raise ValueError("hybrid cleanup error requires at least one failure")
        self.operation = operation
        self.failures = values
        super().__init__(f"{operation} cleanup failed: " + "; ".join(item.description for item in values))


class Qwen38HybridConstructionError(Qwen38HybridDecodeError):
    """Construction failed after creating an inspectable partial session."""

    def __init__(
        self,
        cause: BaseException,
        *,
        cleanup_failures: Sequence[Qwen38HybridCleanupFailure],
        unreleased_resources: Sequence[str],
        partial_session: "Qwen38TTNNHybridDecodeSession",
    ) -> None:
        self.cause = cause
        self.cleanup_failures = tuple(cleanup_failures)
        self.unreleased_resources = tuple(unreleased_resources)
        self.partial_session = partial_session
        cleanup = "none" if not self.cleanup_failures else "; ".join(item.description for item in self.cleanup_failures)
        super().__init__(
            f"Qwen3.8 hybrid TTNN construction failed: {type(cause).__name__}: {cause}; "
            f"cleanup failures: {cleanup}; unreleased: {self.unreleased_resources}"
        )


class Qwen38HybridFactoryError(Qwen38HybridDecodeError):
    """Builder failure retaining every complete object created before it."""

    def __init__(
        self,
        cause: BaseException,
        *,
        builder: Qwen38TTNNBuilder,
        built_target: Qwen38TTNNBuiltTarget | None,
        mtp_components: Qwen38TTNNMTPComponents | None,
        resident_cleanup_error: BaseException | None = None,
    ) -> None:
        self.cause = cause
        self.builder = builder
        self.built_target = built_target
        self.mtp_components = mtp_components
        self.resident_cleanup_error = resident_cleanup_error
        cleanup = (
            "none"
            if resident_cleanup_error is None
            else f"{type(resident_cleanup_error).__name__}: {resident_cleanup_error}"
        )
        super().__init__(
            f"Qwen3.8 hybrid builder failed: {type(cause).__name__}: {cause}; "
            f"resident cleanup failure: {cleanup}; complete built objects remain attached "
            "for process-level reporting/mesh teardown"
        )


@dataclass(frozen=True)
class Qwen38HybridSessionIdentity:
    identity_key: str
    provenance: Qwen38BuildProvenance
    physical_ids: tuple[int, int, int, int]
    mesh_shape: tuple[int, int]
    collective_topology: str
    dram_bank_ring_order: tuple[int, ...]
    ring_size: int


@dataclass(frozen=True)
class Qwen38HybridModeCapabilities:
    mode: Qwen38HybridMode
    greedy: bool
    official_thinking_sampling: bool
    official_non_thinking_sampling: bool
    speculative: bool


@dataclass(frozen=True)
class Qwen38HybridTiming:
    """Non-overlapping synchronized host-wall timing domains.

    ``serialized_target_ns`` is measured inside the streamed row iterator and
    excludes the separately measured host-sampler call.  ``mtp_authoritative_ns``
    is the enclosing bootstrap/alignment call minus both nested target and
    sampler time.  A speculative controller call cannot be split
    without producer-side instrumentation, so it is retained only as
    ``speculative_transaction_ns`` and is never presented as device time.
    ``end_to_end_ns`` closes after state/token publication and primary step
    packaging, immediately before final timing replacement and metrics-list
    insertion.
    """

    phase: Qwen38HybridTimingPhase
    requested_mode: Qwen38HybridMode
    executed_mode: Qwen38HybridMode
    target_rows: int
    mtp_extension_calls: int
    mtp_alignment_rows: int
    fallback_reason: Qwen38HybridFallbackReason | None
    pre_sync_ns: int
    serialized_target_ns: int
    mtp_authoritative_ns: int
    speculative_transaction_ns: int
    post_sync_ns: int
    sampling_ns: int
    end_to_end_ns: int

    @property
    def explicit_sync_ns(self) -> int:
        return self.pre_sync_ns + self.post_sync_ns

    @property
    def serialized_host_call_ns(self) -> int:
        return (
            self.serialized_target_ns + self.mtp_authoritative_ns + self.speculative_transaction_ns + self.sampling_ns
        )

    @property
    def host_overhead_ns(self) -> int:
        return self.end_to_end_ns - self.explicit_sync_ns - self.serialized_host_call_ns


@dataclass(frozen=True)
class Qwen38HybridToken:
    request_id: int
    token_index: int
    token_id: int
    source: Qwen38HybridTokenSource
    cache_position: int
    stop_reason: Qwen38HybridStopReason | None


@dataclass(frozen=True)
class Qwen38HybridStep:
    request_id: int
    step_index: int
    base_position: int
    next_position: int
    requested_mode: Qwen38HybridMode
    executed_mode: Qwen38HybridMode
    tokens: tuple[Qwen38HybridToken, ...]
    accepted_draft_count: int
    committed_input_count: int
    fallback_reason: Qwen38HybridFallbackReason | None
    sampling_parameters: Qwen38SamplingParameters
    timing: Qwen38HybridTiming
    sampling_result: Qwen38SampledTokens | None = field(default=None, repr=False, compare=False)
    speculative_round: Qwen38SpeculativeRound | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _Qwen38HybridTestCollaborators:
    """Private no-device seam; production always constructs validated owners."""

    fixed_target: Qwen38TTNNFixedFiveTarget
    mtp_engine: Qwen38TTNNMTPDraftEngine
    host_sampler: Qwen38TTNNHostSampler


def _require_identity_key(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase 64-hex, got {value!r}")
    return value


def _normalize_physical_ids(values: Sequence[int]) -> tuple[int, int, int, int]:
    result = tuple(values)
    if (
        len(result) != TP_SIZE
        or len(set(result)) != TP_SIZE
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result)
    ):
        raise ValueError(f"hybrid decode requires four distinct nonnegative physical IDs, got {result}")
    return result  # type: ignore[return-value]


def _normalize_tokens(values: torch.Tensor | Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{label} must be CPU int32/int64")
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] <= 0:
            raise ValueError(f"{label} must be nonempty true-global-B1 [1,sequence]")
        result = tuple(int(value) for value in values[0].tolist())
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{label} must be a CPU tensor or flat integer sequence")
        result = tuple(values)
        if not result:
            raise ValueError(f"{label} cannot be empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
            raise TypeError(f"{label} must be a flat integer sequence")
    if min(result) < 0 or max(result) >= VOCAB_SIZE:
        raise IndexError(f"{label} contains a token outside [0,{VOCAB_SIZE})")
    return result


def _normalize_eos(values: Sequence[int]) -> tuple[int, ...]:
    result = _normalize_tokens(values, label="EOS token IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"EOS token IDs must be unique, got {result}")
    return result


def _normalize_max_new_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"max_new_tokens must be a positive integer, got {value!r}")
    return value


def _normalize_mode(value: Qwen38HybridMode | str) -> Qwen38HybridMode:
    try:
        return value if type(value) is Qwen38HybridMode else Qwen38HybridMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"mode must be one of {[item.value for item in Qwen38HybridMode]}, got {value!r}") from error


def _normalize_cache_policy(value: Qwen38HybridCachePolicy | str) -> Qwen38HybridCachePolicy:
    try:
        return value if type(value) is Qwen38HybridCachePolicy else Qwen38HybridCachePolicy(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"cache_policy must be one of {[item.value for item in Qwen38HybridCachePolicy]}, got {value!r}"
        ) from error


def _duration(start: int, end: int, *, label: str) -> int:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise RuntimeError(f"{label} clock must return integer nanoseconds")
    if end < start:
        raise RuntimeError(f"{label} monotonic clock moved backwards: {start} -> {end}")
    return end - start


def _source(value: Qwen38SpeculativeTokenSource) -> Qwen38HybridTokenSource:
    mapping = {
        Qwen38SpeculativeTokenSource.ACCEPTED_DRAFT: Qwen38HybridTokenSource.ACCEPTED_DRAFT,
        Qwen38SpeculativeTokenSource.TARGET_REPLACEMENT: Qwen38HybridTokenSource.TARGET_REPLACEMENT,
        Qwen38SpeculativeTokenSource.TARGET_BONUS: Qwen38HybridTokenSource.TARGET_BONUS,
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported speculative token source {value!r}") from error


class Qwen38TTNNHybridDecodeSession:
    """Exclusive ordinary/sampled and exact-greedy MTP owner for one open TP4 mesh."""

    def __init__(
        self,
        built_target: Qwen38TTNNBuiltTarget,
        mtp_components: Qwen38TTNNMTPComponents,
        *,
        expected_provenance: Qwen38BuildProvenance,
        expected_physical_ids: Sequence[int],
        expected_identity_key: str,
        eos_token_ids: Sequence[int],
        synchronize: Synchronize | None = None,
        clock_ns: Clock = time.perf_counter_ns,
        _test_collaborators: _Qwen38HybridTestCollaborators | None = None,
        _owns_built_graph: bool = False,
    ) -> None:
        physical_ids = _normalize_physical_ids(expected_physical_ids)
        identity_key = _require_identity_key(expected_identity_key, label="expected identity key")
        eos_ids = _normalize_eos(eos_token_ids)
        if type(expected_provenance) is not Qwen38BuildProvenance:
            raise TypeError("expected_provenance must be the exact validated Qwen38BuildProvenance")
        if type(built_target) is not Qwen38TTNNBuiltTarget:
            raise TypeError("hybrid decode requires the exact Qwen38TTNNBuiltTarget")
        if type(built_target.model) is not Qwen38TTNNTextModel:
            raise TypeError("hybrid decode requires the exact 48-layer Qwen38TTNNTextModel")
        if type(built_target.components) is not Qwen38TTNNTargetComponents:
            raise TypeError("hybrid decode requires exact inspectable target components")
        if type(mtp_components) is not Qwen38TTNNMTPComponents:
            raise TypeError("hybrid decode requires exact inspectable MTP components")
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if _test_collaborators is not None and type(_test_collaborators) is not _Qwen38HybridTestCollaborators:
            raise TypeError("_test_collaborators is a private no-device test seam")
        if type(_owns_built_graph) is not bool:
            raise TypeError("_owns_built_graph must be an explicit bool")

        identity = built_target.components.identity
        if type(identity) is not Qwen38LiveBuildIdentity or mtp_components.identity is not identity:
            raise ValueError("target and MTP components must share one exact live build identity object")
        if identity.provenance != expected_provenance:
            raise ValueError("built target provenance differs from independently admitted launch provenance")
        if identity.physical_ids != physical_ids or identity.mesh_shape != MESH_SHAPE:
            raise ValueError("built target physical TP4 placement differs from the admitted launch")
        if identity.key != identity_key:
            raise ValueError(f"built target identity {identity.key} differs from expected {identity_key}")

        model = built_target.model
        allocated_context = getattr(model, "allocated_context", MAX_CONTEXT)
        if (
            isinstance(allocated_context, bool)
            or not isinstance(allocated_context, int)
            or not 0 < allocated_context <= MAX_CONTEXT
        ):
            raise ValueError(f"hybrid target allocated context is invalid: {allocated_context!r}")
        if model.layers is not built_target.components.layers or len(model.layers) != BACKBONE_LAYERS:
            raise ValueError("built target does not expose its exact ordered 48-layer component tuple")
        if model.model_io is not built_target.components.model_io:
            raise ValueError("built target does not expose its inspectable shared model-I/O owner")
        target_expert_owner = built_target.components.expert_streamer
        mtp_expert_owner = getattr(mtp_components.decoder_layer, "expert_streamer", None)
        if _test_collaborators is None and mtp_expert_owner is not target_expert_owner:
            raise ValueError("target and MTP components do not share one exact BF4 expert owner")
        resident_expert_owner = target_expert_owner if isinstance(target_expert_owner, Qwen38BF4ResidentSet) else None
        if resident_expert_owner is not None:
            if mtp_expert_owner is not resident_expert_owner:
                raise ValueError("resident target and MTP components do not share one exact BF4 owner")
            if resident_expert_owner.closed or resident_expert_owner.poisoned:
                raise RuntimeError("hybrid decode received a closed or poisoned resident BF4 owner")
        self.built_target = built_target
        self.mtp_components = mtp_components
        self.model = model
        self.allocated_context = allocated_context
        self.fixed_target: Qwen38TTNNFixedFiveTarget | None = None
        self.mtp_engine: Qwen38TTNNMTPDraftEngine | None = None
        self.host_sampler: Qwen38TTNNHostSampler | None = None
        self.identity = Qwen38HybridSessionIdentity(
            identity_key=identity.key,
            provenance=identity.provenance,
            physical_ids=identity.physical_ids,
            mesh_shape=identity.mesh_shape,
            collective_topology=identity.collective_topology,
            dram_bank_ring_order=identity.dram_bank_ring_order,
            ring_size=identity.ring_size,
        )
        self.eos_token_ids = eos_ids
        self._eos_token_set = frozenset(eos_ids)
        self._synchronize = synchronize or (lambda: ttnn.synchronize_device(self.model.mesh_device))
        self._clock_ns = clock_ns
        self._runtime_owner = object()
        self._status = Qwen38HybridSessionStatus.IDLE
        self._mode = Qwen38HybridMode.ORDINARY
        self._poison_error: Qwen38HybridDecodePoisonedError | None = None
        self._target_state: Qwen38TTNNTextModelState | None = None
        self._raw_mtp_state: Any | None = None
        self._mtp_seed: Qwen38MTPSeed | None = None
        self._pending_mtp_seed: Qwen38MTPSeed | None = None
        self._controller: Qwen38SpeculativeDecodeController | None = None
        self._pending_token_id: int | None = None
        self._consumed_token_ids: tuple[int, ...] = ()
        self._request_id = 0
        self._request_generated = 0
        self._request_max_new_tokens = 0
        self._request_diagnostic_exact_token_budget = False
        self._request_step_index = 0
        self._sampling_parameters = Qwen38SamplingParameters.greedy()
        self._request_generator: torch.Generator | None = None
        self._metrics: list[Qwen38HybridStep] = []
        self._fallback_count = 0
        self._reset_count = 0
        self._fixed_claimed = False
        self._mtp_claimed = False
        self._fixed_open = False
        self._sampler_open = False
        self._resident_expert_owner = resident_expert_owner
        self._owns_built_graph = _owns_built_graph
        self._resident_expert_owner_open = resident_expert_owner is not None and _owns_built_graph
        self._cleanup_failures: list[Qwen38HybridCleanupFailure] = []
        self._unreleased_resources: set[str] = set()
        if self._resident_expert_owner_open:
            self._unreleased_resources.add("resident BF4 experts")
        self._ready_seed_phase_observer: Callable[[str], None] | None = None

        try:
            mesh_contract = getattr(model, "mesh_contract", None)
            if mesh_contract is None or not callable(getattr(mesh_contract, "validate_mesh", None)):
                raise TypeError("built target does not expose its validated physical mesh contract")
            mesh_contract.validate_mesh(model.mesh_device)
            if tuple(mesh_contract.physical_ids) != physical_ids:
                raise ValueError("target mesh contract physical order differs from the admitted launch")
            if _test_collaborators is None:
                sampler = Qwen38TTNNHostSampler(
                    model.model_io.lm_head,
                    identity,
                    expected_provenance=expected_provenance,
                    expected_identity_key=identity_key,
                    eos_token_ids=eos_ids,
                    clock_ns=clock_ns,
                )
                self.host_sampler = sampler
                self._sampler_open = True
                self._unreleased_resources.add("host sampler")
                fixed = Qwen38TTNNFixedFiveTarget(built_target)
                self.fixed_target = fixed
                self._fixed_open = True
                self._unreleased_resources.add("fixed-five private buffers")
                mtp = Qwen38TTNNMTPDraftEngine(built_target, mtp_components)
                self.mtp_engine = mtp
            else:
                fixed = _test_collaborators.fixed_target
                mtp = _test_collaborators.mtp_engine
                sampler = _test_collaborators.host_sampler
                self.fixed_target = fixed
                self.mtp_engine = mtp
                self.host_sampler = sampler
                self._fixed_open = True
                self._sampler_open = True
                self._unreleased_resources.update(("host sampler", "fixed-five private buffers"))

            if type(sampler) is not Qwen38TTNNHostSampler:
                raise TypeError("host sampler must be the exact provenance-bound TP4 sampler")
            if (
                sampler.identity_key != identity_key
                or sampler.live_identity is not identity
                or sampler.provenance != expected_provenance
                or sampler.lm_head is not model.model_io.lm_head
            ):
                raise ValueError("host sampler object graph/provenance differs from target/MTP identity")
            if (
                type(fixed) is not Qwen38TTNNFixedFiveTarget
                or fixed.model is not model
                or fixed.components is not built_target.components
            ):
                raise TypeError("fixed-five target must be the exact adapter bound to this built target")
            if type(mtp) is not Qwen38TTNNMTPDraftEngine:
                raise TypeError("MTP engine must be the exact shifted-seed TTNN draft engine")
            if _test_collaborators is None and mtp.allocated_context != self.allocated_context:
                raise ValueError("target and MTP QSA caches have different allocated capacities")
            if fixed.identity_key != identity_key or mtp.identity_key != identity_key:
                raise ValueError("target/fixed-five/MTP runtime identities are not identical")
            if getattr(mtp, "identity", None) is not identity:
                raise ValueError("MTP engine is not bound to the target's exact live identity object")
            if _test_collaborators is None and (
                sampler.mesh_device is not model.mesh_device
                or sampler.mesh_contract is not model.mesh_contract
                or mtp.mesh_device is not model.mesh_device
                or mtp.mesh_contract is not model.mesh_contract
                or mtp.embedding is not built_target.components.model_io.embedding
                or mtp.lm_head is not built_target.components.model_io.lm_head
                or mtp.input_mixer is not mtp_components.input_mixer
                or mtp.decoder_layer is not mtp_components.decoder_layer
                or mtp.final_mixer is not mtp_components.final_mixer
            ):
                raise ValueError("production sampler/MTP object graph differs from the exact target mesh/components")

            self.fixed_target.claim_runtime_owner(self._runtime_owner)
            self._fixed_claimed = True
            self._unreleased_resources.add("target runtime claim")
            self.mtp_engine.claim_runtime_owner(self._runtime_owner)
            self._mtp_claimed = True
            self._unreleased_resources.add("MTP runtime claim")
            target_state = self.model.allocate_state()
            self._target_state = target_state
            self._unreleased_resources.add("target state")
            self._validate_target_state(target_state, expected=0)
            mtp_state = self.mtp_engine.allocate_state()
            self._raw_mtp_state = mtp_state
            self._unreleased_resources.add("fresh MTP state")
            if self.mtp_engine.state_position(mtp_state) != 0:
                raise RuntimeError("fresh MTP state did not start at position zero")
            self._timed_sync()
            self._raise_if_backend_poisoned("allocate target and MTP state")
        except BaseException as error:
            cleanup_failures = self._unwind_failed_construction()
            self._status = Qwen38HybridSessionStatus.POISONED
            construction_error = Qwen38HybridConstructionError(
                error,
                cleanup_failures=cleanup_failures,
                unreleased_resources=sorted(self._unreleased_resources),
                partial_session=self,
            )
            self._poison_error = Qwen38HybridDecodePoisonedError(
                "construction",
                construction_error,
                partial_session=self,
            )
            raise construction_error from error

    def _unwind_failed_construction(self) -> tuple[Qwen38HybridCleanupFailure, ...]:
        """Best-effort reverse unwind while the failed object remains inspectable."""

        failures: list[Qwen38HybridCleanupFailure] = []

        def attempt(resource: str, action: Callable[[], None], clear: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as cleanup_error:
                failures.append(Qwen38HybridCleanupFailure(resource, cleanup_error))
                self._unreleased_resources.add(resource)
            else:
                clear()
                self._unreleased_resources.discard(resource)

        if self._raw_mtp_state is not None and self.mtp_engine is not None:
            state = self._raw_mtp_state
            attempt(
                "fresh MTP state",
                lambda: self.mtp_engine.release_state(state),
                lambda: setattr(self, "_raw_mtp_state", None),
            )
        if self._target_state is not None and self.fixed_target is not None:
            state = self._target_state
            attempt(
                "target state",
                lambda: self.fixed_target.release_state(state),
                lambda: setattr(self, "_target_state", None),
            )
        if (
            self._mtp_claimed
            and self.mtp_engine is not None
            and self._raw_mtp_state is None
            and self._mtp_seed is None
            and self._pending_mtp_seed is None
        ):
            attempt(
                "MTP runtime claim",
                lambda: self.mtp_engine.release_runtime_owner(self._runtime_owner),
                lambda: setattr(self, "_mtp_claimed", False),
            )
        if self._fixed_claimed and self.fixed_target is not None and self._target_state is None:
            attempt(
                "target runtime claim",
                lambda: self.fixed_target.release_runtime_owner(self._runtime_owner),
                lambda: setattr(self, "_fixed_claimed", False),
            )
        if self._fixed_open and self.fixed_target is not None and not self._fixed_claimed:
            attempt(
                "fixed-five private buffers",
                self.fixed_target.close,
                lambda: setattr(self, "_fixed_open", False),
            )
        if self._sampler_open and self.host_sampler is not None:
            attempt(
                "host sampler",
                self.host_sampler.close,
                lambda: setattr(self, "_sampler_open", False),
            )
        # Construction failed after runtime/device ownership may have changed.
        # Static resident weights stay registered for process/mesh teardown;
        # enqueueing their deallocation here would weaken poison safety.
        self._cleanup_failures.extend(failures)
        return tuple(failures)

    @classmethod
    def from_builder(
        cls,
        builder: Qwen38TTNNBuilder,
        *,
        expected_provenance: Qwen38BuildProvenance,
        expected_physical_ids: Sequence[int],
        expected_identity_key: str,
        eos_token_ids: Sequence[int],
        synchronize: Synchronize | None = None,
        clock_ns: Clock = time.perf_counter_ns,
    ) -> "Qwen38TTNNHybridDecodeSession":
        """Build target then MTP on the caller's already-open admitted mesh."""

        if type(builder) is not Qwen38TTNNBuilder:
            raise TypeError("from_builder requires the exact Qwen38TTNNBuilder")
        if type(expected_provenance) is not Qwen38BuildProvenance:
            raise TypeError("expected_provenance must be the exact validated Qwen38BuildProvenance")
        physical_ids = _normalize_physical_ids(expected_physical_ids)
        key = _require_identity_key(expected_identity_key, label="expected identity key")
        eos_ids = _normalize_eos(eos_token_ids)
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if builder.provenance != expected_provenance:
            raise ValueError("builder provenance differs from independently admitted launch provenance")
        if builder.identity.physical_ids != physical_ids or builder.identity.key != key:
            raise ValueError("builder live identity differs from the admitted physical TP4 launch")
        built_target = None
        mtp_components = None
        try:
            built_target = builder.build_target()
            mtp_components = builder.build_mtp_components()
            return cls(
                built_target,
                mtp_components,
                expected_provenance=expected_provenance,
                expected_physical_ids=physical_ids,
                expected_identity_key=key,
                eos_token_ids=eos_ids,
                synchronize=synchronize,
                clock_ns=clock_ns,
                _owns_built_graph=True,
            )
        except Qwen38HybridConstructionError:
            # The construction error already retains its partial session and
            # therefore the complete target/MTP object graph.
            raise
        except BaseException as error:
            resident_cleanup_error = None
            owner = getattr(builder, "expert_streamer", None)
            resident_graph_is_published = builder._resident_graph_is_published()
            build_failure = builder.resident_build_failure
            requires_process_termination = (
                isinstance(build_failure, tuple) and len(build_failure) == 5 and build_failure[2] is True
            )
            if (
                built_target is not None
                and not resident_graph_is_published
                and not requires_process_termination
                and isinstance(owner, Qwen38BF4ResidentSet)
                and not owner.closed
                and not owner.poisoned
            ):
                try:
                    owner.close()
                except BaseException as cleanup_error:
                    resident_cleanup_error = cleanup_error
            raise Qwen38HybridFactoryError(
                error,
                builder=builder,
                built_target=built_target,
                mtp_components=mtp_components,
                resident_cleanup_error=resident_cleanup_error,
            ) from error

    @property
    def status(self) -> Qwen38HybridSessionStatus:
        return self._status

    @property
    def mode(self) -> Qwen38HybridMode:
        return self._mode

    @property
    def pending_token_id(self) -> int | None:
        return self._pending_token_id

    @property
    def consumed_token_ids(self) -> tuple[int, ...]:
        return self._consumed_token_ids

    @property
    def transcript_token_ids(self) -> tuple[int, ...]:
        """Exact consumed prefix plus the one emitted, unconsumed token."""

        if self._pending_token_id is None:
            return self._consumed_token_ids
        return (*self._consumed_token_ids, self._pending_token_id)

    @property
    def metrics(self) -> tuple[Qwen38HybridStep, ...]:
        return tuple(self._metrics)

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def sampling_parameters(self) -> Qwen38SamplingParameters:
        return self._sampling_parameters

    @property
    def cleanup_failures(self) -> tuple[Qwen38HybridCleanupFailure, ...]:
        return tuple(self._cleanup_failures)

    @property
    def unreleased_resources(self) -> tuple[str, ...]:
        return tuple(sorted(self._unreleased_resources))

    @property
    def requires_process_termination(self) -> bool:
        """Whether ledgers are report-only and no resource retry is permitted."""

        return self._status is Qwen38HybridSessionStatus.POISONED

    @property
    def owns_built_graph(self) -> bool:
        return self._owns_built_graph

    @property
    def borrowed_resident_expert_owner(self) -> Qwen38BF4ResidentSet | None:
        return self._resident_expert_owner

    @staticmethod
    def capabilities(mode: Qwen38HybridMode | str) -> Qwen38HybridModeCapabilities:
        selected = _normalize_mode(mode)
        if selected is Qwen38HybridMode.ORDINARY:
            return Qwen38HybridModeCapabilities(
                mode=selected,
                greedy=True,
                official_thinking_sampling=True,
                official_non_thinking_sampling=True,
                speculative=False,
            )
        return Qwen38HybridModeCapabilities(
            mode=selected,
            greedy=True,
            official_thinking_sampling=False,
            official_non_thinking_sampling=False,
            speculative=True,
        )

    @property
    def poison_error(self) -> Qwen38HybridDecodePoisonedError | None:
        return self._poison_error

    @property
    def state_position(self) -> int:
        self._assert_usable("read state position")
        if self._controller is not None:
            return self._controller.position
        return self._require_target_state().position

    def _backend_poison_cause(self) -> BaseException | None:
        for label, backend in (
            ("target", self.model),
            ("fixed-five target", self.fixed_target),
            ("MTP engine", self.mtp_engine),
            ("MTP controller", self._controller),
        ):
            if backend is None:
                continue
            try:
                poisoned = bool(getattr(backend, "poisoned", False))
                if label == "MTP controller":
                    poisoned = getattr(backend, "status", None) is Qwen38SpeculativeStatus.POISONED
            except BaseException as error:
                return RuntimeError(f"failed to inspect {label} poison state: {error}")
            if not poisoned:
                continue
            try:
                cause = getattr(backend, "poisoned_error", None)
                if cause is None:
                    cause = getattr(backend, "poison_error", None)
            except BaseException as error:
                return RuntimeError(f"failed to inspect {label} poison cause: {error}")
            return cause if isinstance(cause, BaseException) else RuntimeError(f"{label} reports poison")
        return None

    def _raise_if_backend_poisoned(self, operation: str) -> None:
        cause = self._backend_poison_cause()
        if cause is not None:
            self._poison(operation, cause)

    def _poison(self, operation: str, error: BaseException) -> NoReturn:
        if self._poison_error is None:
            self._poison_error = Qwen38HybridDecodePoisonedError(
                operation,
                self._backend_poison_cause() or error,
                partial_session=self,
            )
        self._status = Qwen38HybridSessionStatus.POISONED
        raise self._poison_error from error

    def _assert_usable(self, operation: str) -> None:
        if self._status is Qwen38HybridSessionStatus.POISONED:
            assert self._poison_error is not None
            raise self._poison_error
        if self._status is Qwen38HybridSessionStatus.CLOSED:
            raise Qwen38HybridDecodeError(f"cannot {operation}: hybrid session is closed")
        self._raise_if_backend_poisoned(operation)

    def _require_target_state(self) -> Qwen38TTNNTextModelState:
        if self._target_state is None:
            raise RuntimeError("hybrid owner does not currently hold the target state")
        return self._target_state

    def _require_raw_mtp_state(self) -> Any:
        if self._raw_mtp_state is None:
            raise RuntimeError("hybrid owner does not currently hold a fresh MTP state")
        return self._raw_mtp_state

    def _require_seed(self) -> Qwen38MTPSeed:
        if self._pending_mtp_seed is not None:
            raise RuntimeError("hybrid owner has an unpublished replacement MTP ReadySeed")
        if self._mtp_seed is None:
            raise RuntimeError("hybrid owner does not currently hold an MTP ReadySeed")
        return self._mtp_seed

    def _require_no_pending_seed(self, operation: str) -> None:
        if self._pending_mtp_seed is not None:
            raise RuntimeError(f"cannot {operation}: an unpublished replacement MTP ReadySeed remains ledger-owned")

    def _validate_target_state(self, state: Any, *, expected: int) -> None:
        if type(state) is not Qwen38TTNNTextModelState:
            raise TypeError("target returned a non-Qwen38 TTNN model state")
        if state._owner is not self.model._state_owner:
            raise ValueError("target returned state from another model owner")
        if state.position != expected:
            raise RuntimeError(f"target state advanced to {state.position}, expected {expected}")

    def _validate_seed(self, seed: Any, *, expected_position: int, expected_pending: int) -> Qwen38MTPSeed:
        if type(seed) is not Qwen38MTPSeed:
            raise TypeError("MTP engine returned a non-Qwen38 ReadySeed")
        if seed.position != expected_position or seed.current_token_id != expected_pending:
            raise RuntimeError(
                f"MTP ReadySeed reports position/token ({seed.position},{seed.current_token_id}), "
                f"expected ({expected_position},{expected_pending})"
            )
        if self.mtp_engine.state_position(seed.state) != expected_position:
            raise RuntimeError("MTP ReadySeed state and metadata positions differ")
        if seed.recurrent_residual is None or seed.transaction is None:
            raise RuntimeError("MTP ReadySeed is missing its residual or ownership token")
        if isinstance(seed.first_draft_token_id, bool) or not 0 <= seed.first_draft_token_id < VOCAB_SIZE:
            raise RuntimeError("MTP ReadySeed carries an invalid first draft")
        return seed

    def _publish_pending_seed(self, *, expected_position: int, expected_pending: int) -> Qwen38MTPSeed:
        seed = self._pending_mtp_seed
        if seed is None:
            raise RuntimeError("MTP engine returned no replacement ReadySeed")
        validated = self._validate_seed(
            seed,
            expected_position=expected_position,
            expected_pending=expected_pending,
        )
        self._mtp_seed = validated
        self._pending_mtp_seed = None
        self._unreleased_resources.discard("pending MTP ReadySeed")
        self._unreleased_resources.add("MTP ReadySeed")
        return validated

    def _timed_sync(self) -> int:
        started = self._clock_ns()
        self._synchronize()
        ended = self._clock_ns()
        return _duration(started, ended, label="mesh synchronize")

    @contextmanager
    def observe_ready_seed_phases(self, observer: Callable[[str], None]) -> Iterator[None]:
        """Report nonsemantic position-one bootstrap boundaries for one scoped diagnostic."""

        self._assert_usable("observe ReadySeed phases")
        if not callable(observer):
            raise TypeError("ReadySeed phase observer must be callable")
        if self._ready_seed_phase_observer is not None:
            raise RuntimeError("a ReadySeed phase observer is already active")
        if self.mtp_engine is None:
            raise RuntimeError("cannot observe ReadySeed phases without the MTP engine")
        self._ready_seed_phase_observer = observer
        try:
            with self.mtp_engine.observe_bootstrap_phases(observer):
                yield
        finally:
            self._ready_seed_phase_observer = None

    def _observe_ready_seed_phase(self, phase: str) -> None:
        observer = self._ready_seed_phase_observer
        if observer is not None:
            observer(phase)

    def _snapshot_request_generator(self) -> torch.Tensor | None:
        if self._request_generator is None:
            return None
        return self._request_generator.get_state().clone()

    def _restore_request_generator(self, snapshot: torch.Tensor | None, cause: BaseException) -> BaseException:
        if snapshot is None:
            return cause
        generator = self._request_generator
        if generator is None:
            cleanup_error = RuntimeError(
                f"request generator disappeared while rolling back {type(cause).__name__}: {cause}"
            )
            failure = Qwen38HybridCleanupFailure("request CPU generator state", cleanup_error)
            self._cleanup_failures.append(failure)
            self._unreleased_resources.add(failure.resource)
            return Qwen38HybridCleanupError("request RNG rollback", (failure,))
        try:
            generator.set_state(snapshot)
        except BaseException as cleanup_error:
            failure = Qwen38HybridCleanupFailure("request CPU generator state", cleanup_error)
            self._cleanup_failures.append(failure)
            self._unreleased_resources.add(failure.resource)
            wrapped = Qwen38HybridCleanupError(
                "request RNG rollback",
                (failure,),
            )
            wrapped.__cause__ = cause
            return wrapped
        return cause

    @staticmethod
    def _release_detached_tensor(tensor: Any, *, label: str) -> None:
        allocated = getattr(tensor, "is_allocated", None)
        if not callable(allocated):
            raise TypeError(f"detached {label} has no retry-safe is_allocated() query")
        if not bool(allocated()):
            return
        try:
            ttnn.deallocate(tensor)
        except BaseException:
            if not bool(allocated()):
                return
            raise
        if bool(allocated()):
            raise RuntimeError(f"detached {label} remains allocated after deallocate")

    @classmethod
    def _release_taken_logits(cls, logits: Any) -> None:
        cls._release_detached_tensor(logits.tensor, label="sampled logits")

    def _target_row(
        self,
        token_id: int,
        state: Qwen38TTNNTextModelState,
        *,
        resolve_token: bool,
        token_history: tuple[int, ...],
        phase_observer: Callable[[str], None] | None = None,
    ) -> tuple[Qwen38TTNNTextModelState, Any, int | None, int, int, Qwen38SampledTokens | None]:
        """Run one target row and publicly transfer its retained root."""

        started = self._clock_ns()
        output: Qwen38TTNNTextModelOutput | None = None
        taken_logits = None
        detached_root = None
        sampling_ns = 0
        sampled_result: Qwen38SampledTokens | None = None
        stochastic = self._sampling_parameters.profile is not Qwen38SamplingProfile.GREEDY
        try:
            output = self.model.forward_decode(
                token_id,
                state,
                return_logits=resolve_token and stochastic,
                resolve_greedy=resolve_token and not stochastic,
                retain_hidden=False,
                retain_hyper_residual=True,
                retain_input_state=False,
                return_routing=False,
                phase_observer=phase_observer,
            )
            if type(output) is not Qwen38TTNNTextModelOutput:
                raise TypeError("target forward returned the wrong output owner type")
            if output.input_token_id != token_id or output.position != state.position:
                raise RuntimeError("target forward returned mismatched token/position metadata")
            self._validate_target_state(output.state, expected=state.position + 1)
            if len(output.layer_aux) != BACKBONE_LAYERS:
                raise RuntimeError(f"target returned {len(output.layer_aux)} layer records, expected 48")
            if output.hidden_sharded is not None:
                raise RuntimeError("target row retained an unexpected hidden tensor")
            if output.hyper_residual_sharded is None:
                raise RuntimeError("target row did not retain its MTP hyper-residual")
            prediction = None
            if resolve_token and not stochastic:
                if output.logits is not None:
                    raise RuntimeError("greedy target row unexpectedly retained full logits")
                host = output.greedy_token
                if (
                    not isinstance(host, torch.Tensor)
                    or host.device.type != "cpu"
                    or host.dtype not in (torch.int32, torch.int64)
                    or tuple(host.shape) != (1, 1, 1)
                ):
                    raise RuntimeError("target greedy row must return one CPU int32/int64 token [1,1,1]")
                prediction = int(host.item())
                if not 0 <= prediction < VOCAB_SIZE:
                    raise RuntimeError(f"target greedy row returned invalid token {prediction}")
            elif resolve_token:
                if output.greedy_token is not None or output.logits is None:
                    raise RuntimeError("sampled target row must retain logits without resolving greedy")
                taken_logits = output.take_logits()
                sample_started = self._clock_ns()
                sampled_result = self.host_sampler.sample(
                    taken_logits,
                    self._sampling_parameters,
                    token_histories=token_history,
                    source_identity_key=self.identity.identity_key,
                    generator=self._request_generator,
                )
                sample_ended = self._clock_ns()
                sampling_ns = _duration(sample_started, sample_ended, label="ordinary host sampling")
                if type(sampled_result) is not Qwen38SampledTokens or tuple(sampled_result.token_ids.shape) != (
                    1,
                    1,
                    1,
                ):
                    raise RuntimeError("host sampler did not return one exact sampled token")
                prediction = int(sampled_result.token_ids.item())
                if not 0 <= prediction < VOCAB_SIZE:
                    raise RuntimeError(f"host sampler returned invalid token {prediction}")
                self._release_taken_logits(taken_logits)
                taken_logits = None
            elif output.greedy_token is not None or output.logits is not None:
                raise RuntimeError("non-final target row unexpectedly resolved a greedy token")

            detached_root = output.take_hyper_residual()
            if detached_root is None or output.hyper_residual_sharded is not None:
                raise RuntimeError("public target-root ownership transfer did not detach the output slot")
            if output.active:
                output.release_tensors()
            ended = self._clock_ns()
            result = (
                output.state,
                detached_root,
                prediction,
                _duration(started, ended, label="serialized target row"),
                sampling_ns,
                sampled_result,
            )
            detached_root = None
            return result
        except BaseException as error:
            cleanup_failures: list[Qwen38HybridCleanupFailure] = []
            if detached_root is not None:
                try:
                    self._release_detached_tensor(detached_root, label="target hyper-residual")
                except BaseException as cleanup_error:
                    cleanup_failures.append(Qwen38HybridCleanupFailure("detached target hyper-residual", cleanup_error))
            if taken_logits is not None:
                try:
                    self._release_taken_logits(taken_logits)
                except BaseException as cleanup_error:
                    cleanup_failures.append(Qwen38HybridCleanupFailure("detached sampled logits", cleanup_error))
            if output is not None and output.active:
                try:
                    output.release_tensors()
                except BaseException as cleanup_error:
                    cleanup_failures.append(Qwen38HybridCleanupFailure("model-output residual tensors", cleanup_error))
            if cleanup_failures:
                wrapped = Qwen38HybridCleanupError("failed target row", cleanup_failures)
                wrapped.__cause__ = error
                raise wrapped from error
            raise

    def _stream_shifted_rows(
        self,
        consumed_rows: tuple[int, ...],
        *,
        initial_state: Qwen38TTNNTextModelState,
        progress: dict[str, Any],
    ) -> Iterable[tuple[int, Any]]:
        """Interleave one target row with one MTP row (one live root max)."""

        def rows() -> Iterator[tuple[int, Any]]:
            current = initial_state
            for index, token_id in enumerate(consumed_rows):
                detached_root = None
                try:
                    final = index == len(consumed_rows) - 1
                    history = (*self._consumed_token_ids, *consumed_rows[: index + 1])
                    target_phase_observer = None
                    if self._ready_seed_phase_observer is not None:

                        def target_phase_observer(phase: str, *, row_index: int = index) -> None:
                            self._observe_ready_seed_phase(
                                _translate_ready_seed_target_phase(phase, row_index=row_index)
                            )

                    self._observe_ready_seed_phase(f"before-ready-seed-target-row-{index}")
                    next_state, detached_root, prediction, target_ns, sampling_ns, sampled_result = self._target_row(
                        token_id,
                        current,
                        resolve_token=final,
                        token_history=history,
                        phase_observer=target_phase_observer,
                    )
                    self._observe_ready_seed_phase(f"after-ready-seed-target-row-{index}")
                    current = next_state
                    # Each target row consumes its predecessor on successful
                    # return.  Publish the replacement immediately so a later
                    # iterator/MTP failure cannot leave only the consumed
                    # predecessor in the session cleanup ledger.
                    self._target_state = current
                    progress["target_state"] = current
                    progress["target_rows"] += 1
                    progress["target_ns"] += target_ns
                    progress["sampling_ns"] += sampling_ns
                    if sampled_result is not None:
                        if progress["sampling_result"] is not None:
                            raise RuntimeError("one streamed target request produced more than one sampling result")
                        progress["sampling_result"] = sampled_result
                    if final:
                        if prediction is None:
                            raise AssertionError("final streamed target row did not produce pending token")
                        shifted_token = prediction
                        progress["pending"] = prediction
                    else:
                        shifted_token = consumed_rows[index + 1]
                    # Creating the pair and clearing this local slot is the
                    # transfer boundary.  From the ensuing yield onward the
                    # engine owns and must release the root before requesting
                    # another row.
                    shifted_row = (shifted_token, detached_root)
                    detached_root = None
                    self._observe_ready_seed_phase(f"before-ready-seed-yield-shifted-row-{index}")
                    yield shifted_row
                    self._observe_ready_seed_phase(f"after-ready-seed-yield-shifted-row-{index}")
                finally:
                    if detached_root is not None:
                        self._release_detached_tensor(detached_root, label="unyielded target hyper-residual")

        return rows()

    def _make_timing(
        self,
        *,
        phase: Qwen38HybridTimingPhase,
        requested_mode: Qwen38HybridMode,
        executed_mode: Qwen38HybridMode,
        target_rows: int,
        mtp_extension_calls: int,
        mtp_alignment_rows: int,
        fallback_reason: Qwen38HybridFallbackReason | None,
        pre_sync_ns: int,
        serialized_target_ns: int,
        mtp_authoritative_ns: int,
        speculative_transaction_ns: int,
        sampling_ns: int,
        post_sync_ns: int,
    ) -> Qwen38HybridTiming:
        return Qwen38HybridTiming(
            phase=phase,
            requested_mode=requested_mode,
            executed_mode=executed_mode,
            target_rows=target_rows,
            mtp_extension_calls=mtp_extension_calls,
            mtp_alignment_rows=mtp_alignment_rows,
            fallback_reason=fallback_reason,
            pre_sync_ns=pre_sync_ns,
            serialized_target_ns=serialized_target_ns,
            mtp_authoritative_ns=mtp_authoritative_ns,
            speculative_transaction_ns=speculative_transaction_ns,
            post_sync_ns=post_sync_ns,
            sampling_ns=sampling_ns,
            # Finalized by _record_step only after state/token publication and
            # primary result packaging have completed.
            end_to_end_ns=0,
        )

    def _detach_controller(self) -> None:
        controller = self._controller
        if controller is None:
            return
        self._require_no_pending_seed("handoff the speculative controller")
        handoff = controller.handoff(
            next_target_owner=self._runtime_owner,
            next_mtp_owner=self._runtime_owner,
        )
        self._controller = None
        self._target_state = handoff.target_state
        self._pending_mtp_seed = handoff.mtp_seed
        self._unreleased_resources.discard("MTP ReadySeed")
        self._unreleased_resources.add("pending MTP ReadySeed")
        if handoff.identity_key != self.identity.identity_key:
            raise RuntimeError("MTP controller handoff changed live build identity")
        if handoff.pending_token_id != self._pending_token_id:
            raise RuntimeError("MTP controller handoff changed pending token metadata")
        seed = self._publish_pending_seed(
            expected_position=handoff.position,
            expected_pending=handoff.pending_token_id,
        )
        self._validate_target_state(self._target_state, expected=handoff.position)
        self._mtp_seed = seed
        self._unreleased_resources.discard("speculative controller")

    def _ensure_controller(self) -> Qwen38SpeculativeDecodeController:
        if self._controller is not None:
            return self._controller
        self._require_no_pending_seed("construct a speculative controller")
        target_state = self._require_target_state()
        seed = self._require_seed()
        pending = self._pending_token_id
        if pending is None:
            raise RuntimeError("cannot construct speculative controller without a pending token")
        controller = Qwen38SpeculativeDecodeController(
            self.fixed_target,
            self.mtp_engine,
            target_state=target_state,
            mtp_seed=seed,
            pending_token_id=pending,
            eos_token_ids=self.eos_token_ids,
            diagnostic_exact_token_budget=self._request_diagnostic_exact_token_budget,
            previous_target_owner=self._runtime_owner,
            previous_mtp_owner=self._runtime_owner,
        )
        self._controller = controller
        self._target_state = None
        self._mtp_seed = None
        self._unreleased_resources.add("speculative controller")
        if controller.position != target_state.position or controller.pending_token_id != pending:
            raise RuntimeError("new speculative controller changed state position or pending token")
        return controller

    def set_mode(self, mode: Qwen38HybridMode | str) -> None:
        """Switch execution mode without changing any target/MTP cache row."""

        self._assert_usable("switch decode mode")
        selected = _normalize_mode(mode)
        if selected is Qwen38HybridMode.MTP and self._sampling_parameters.profile is not Qwen38SamplingProfile.GREEDY:
            raise Qwen38HybridDecodeError(
                "MTP mode supports exact greedy verification only; stochastic sampling is rejected"
            )
        if selected is Qwen38HybridMode.ORDINARY and self._controller is not None:
            try:
                self._detach_controller()
            except BaseException as error:
                self._poison("handoff speculative state to ordinary mode", error)
        self._mode = selected

    def _cache_suffix(
        self,
        full_ids: tuple[int, ...],
        *,
        cache_policy: Qwen38HybridCachePolicy,
    ) -> tuple[int, ...]:
        if not self._consumed_token_ids:
            return full_ids
        if cache_policy is Qwen38HybridCachePolicy.RESET:
            self.reset()
            return full_ids
        prefix_length = len(self._consumed_token_ids)
        if len(full_ids) <= prefix_length or full_ids[:prefix_length] != self._consumed_token_ids:
            raise Qwen38HybridCacheMismatchError(
                "rendered history does not exactly preserve the consumed target-token prefix; "
                "request cache_policy='reset' for a healthy full re-prefill"
            )
        suffix = full_ids[prefix_length:]
        if self._pending_token_id is None or suffix[0] != self._pending_token_id:
            raise Qwen38HybridCacheMismatchError(
                f"rendered continuation must begin with pending token {self._pending_token_id}, got {suffix[0]}; "
                "request cache_policy='reset' for a healthy full re-prefill"
            )
        return suffix

    def _bootstrap_or_extend(
        self,
        rows: tuple[int, ...],
        *,
        initial: bool,
        requested_mode: Qwen38HybridMode,
    ) -> tuple[int, Qwen38HybridTiming, Qwen38SampledTokens | None, int]:
        target_state = self._require_target_state()
        base_position = target_state.position
        if base_position + len(rows) > self.allocated_context:
            raise ValueError("rendered prompt suffix exceeds allocated cache capacity")
        progress: dict[str, Any] = {
            "target_state": target_state,
            "target_rows": 0,
            "target_ns": 0,
            "sampling_ns": 0,
            "sampling_result": None,
            "pending": None,
        }
        started = self._clock_ns()
        self._observe_ready_seed_phase("before-ready-seed-pre-synchronize")
        pre_sync_ns = self._timed_sync()
        self._observe_ready_seed_phase("after-ready-seed-pre-synchronize")
        operation_started = self._clock_ns()
        streamed = self._stream_shifted_rows(rows, initial_state=target_state, progress=progress)
        self._require_no_pending_seed("bootstrap or advance MTP alignment")
        if initial:
            self._observe_ready_seed_phase("before-ready-seed-mtp-bootstrap")
            self._pending_mtp_seed = self.mtp_engine.bootstrap_shifted_prefill_rows(
                streamed,
                self._require_raw_mtp_state(),
            )
            self._observe_ready_seed_phase("after-ready-seed-mtp-bootstrap")
            self._raw_mtp_state = None
            self._unreleased_resources.discard("fresh MTP state")
        else:
            self._pending_mtp_seed = self.mtp_engine.advance_seed_rows(self._require_seed(), streamed)
            self._mtp_seed = None
            self._unreleased_resources.discard("MTP ReadySeed")
        self._unreleased_resources.add("pending MTP ReadySeed")
        operation_ended = self._clock_ns()
        self._observe_ready_seed_phase("before-ready-seed-post-synchronize")
        post_sync_ns = self._timed_sync()
        self._observe_ready_seed_phase("after-ready-seed-post-synchronize")

        if progress["target_rows"] != len(rows) or progress["pending"] is None:
            raise RuntimeError("streamed target/MTP alignment did not consume every requested target row")
        next_position = base_position + len(rows)
        pending = int(progress["pending"])
        self._validate_target_state(progress["target_state"], expected=next_position)
        self._observe_ready_seed_phase("before-ready-seed-publish-pending-seed")
        seed = self._publish_pending_seed(expected_position=next_position, expected_pending=pending)
        self._observe_ready_seed_phase("after-ready-seed-publish-pending-seed")
        outer_ns = _duration(operation_started, operation_ended, label="streamed target/MTP alignment")
        target_ns = int(progress["target_ns"])
        sampling_ns = int(progress["sampling_ns"])
        serialized_target_ns = target_ns - sampling_ns
        mtp_ns = outer_ns - target_ns
        if serialized_target_ns < 0 or mtp_ns < 0:
            raise RuntimeError("streamed target timing exceeds its enclosing target/MTP operation")
        self._target_state = progress["target_state"]
        self._mtp_seed = seed
        timing = self._make_timing(
            phase=Qwen38HybridTimingPhase.TTFT,
            requested_mode=requested_mode,
            executed_mode=Qwen38HybridMode.ORDINARY,
            target_rows=len(rows),
            mtp_extension_calls=0,
            mtp_alignment_rows=len(rows),
            fallback_reason=None,
            pre_sync_ns=pre_sync_ns,
            serialized_target_ns=serialized_target_ns,
            mtp_authoritative_ns=mtp_ns,
            speculative_transaction_ns=0,
            sampling_ns=sampling_ns,
            post_sync_ns=post_sync_ns,
        )
        return pending, timing, progress["sampling_result"], started

    def _stop_reason(self, pending: int) -> Qwen38HybridStopReason | None:
        if not self._request_diagnostic_exact_token_budget and pending in self._eos_token_set:
            return Qwen38HybridStopReason.EOS
        if self._request_generated >= self._request_max_new_tokens:
            return Qwen38HybridStopReason.MAX_NEW_TOKENS
        if self.state_position >= self.allocated_context:
            return Qwen38HybridStopReason.CONTEXT_LENGTH
        return None

    def _record_step(
        self,
        *,
        base_position: int,
        next_position: int,
        requested_mode: Qwen38HybridMode,
        executed_mode: Qwen38HybridMode,
        emissions: Sequence[tuple[int, Qwen38HybridTokenSource]],
        accepted_draft_count: int,
        committed_input_count: int,
        fallback_reason: Qwen38HybridFallbackReason | None,
        timing: Qwen38HybridTiming,
        measurement_started_ns: int,
        sampling_result: Qwen38SampledTokens | None = None,
        speculative_round: Qwen38SpeculativeRound | None = None,
    ) -> Qwen38HybridStep:
        values = tuple(emissions)
        if not values:
            raise RuntimeError("a hybrid decode step must emit at least one token")
        if next_position - base_position != committed_input_count:
            raise RuntimeError("hybrid step position delta differs from committed target input count")
        self._request_generated += len(values)
        pending = values[-1][0]
        self._pending_token_id = pending
        stop = self._stop_reason(pending)
        self._status = Qwen38HybridSessionStatus.FINISHED if stop is not None else Qwen38HybridSessionStatus.ACTIVE
        first_index = self._request_generated - len(values)
        tokens = tuple(
            Qwen38HybridToken(
                request_id=self._request_id,
                token_index=first_index + index,
                token_id=token_id,
                source=source,
                cache_position=(base_position + index + 1 if len(values) == committed_input_count else next_position),
                stop_reason=stop if index == len(values) - 1 else None,
            )
            for index, (token_id, source) in enumerate(values)
        )
        step = Qwen38HybridStep(
            request_id=self._request_id,
            step_index=self._request_step_index,
            base_position=base_position,
            next_position=next_position,
            requested_mode=requested_mode,
            executed_mode=executed_mode,
            tokens=tokens,
            accepted_draft_count=accepted_draft_count,
            committed_input_count=committed_input_count,
            fallback_reason=fallback_reason,
            sampling_parameters=self._sampling_parameters,
            timing=timing,
            sampling_result=sampling_result,
            speculative_round=speculative_round,
        )
        packaged_ns = self._clock_ns()
        final_timing = replace(
            timing,
            end_to_end_ns=_duration(measurement_started_ns, packaged_ns, label=timing.phase.value),
        )
        if final_timing.host_overhead_ns < 0:
            raise RuntimeError(f"{timing.phase.value} timing domains overlap")
        step = replace(step, timing=final_timing)
        self._request_step_index += 1
        self._metrics.append(step)
        return step

    def begin(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        max_new_tokens: int,
        mode: Qwen38HybridMode | str = Qwen38HybridMode.ORDINARY,
        cache_policy: Qwen38HybridCachePolicy | str = Qwen38HybridCachePolicy.MATCH,
        sampling: Qwen38SamplingParameters | None = None,
        generator: torch.Generator | None = None,
        diagnostic_exact_token_budget: bool = False,
    ) -> Qwen38HybridStep:
        """Begin from a full rendered history and emit its first target token.

        On a warm cache, ``MATCH`` consumes only the exact pending-plus-new
        suffix.  ``RESET`` explicitly discards both healthy state trees and
        serially re-prefills the entire supplied history.
        """

        self._assert_usable("begin a request")
        if self._status is Qwen38HybridSessionStatus.ACTIVE:
            raise Qwen38HybridDecodeError("cannot begin a new request while hybrid decode is active")
        full_ids = _normalize_tokens(input_ids, label="rendered input IDs")
        limit = _normalize_max_new_tokens(max_new_tokens)
        if type(diagnostic_exact_token_budget) is not bool:
            raise TypeError("diagnostic_exact_token_budget must be boolean")
        if diagnostic_exact_token_budget and limit != DIAGNOSTIC_EXACT_TOKEN_BUDGET:
            raise ValueError(f"diagnostic_exact_token_budget requires max_new_tokens={DIAGNOSTIC_EXACT_TOKEN_BUDGET}")
        selected_mode = _normalize_mode(mode)
        policy = _normalize_cache_policy(cache_policy)
        parameters = Qwen38SamplingParameters.greedy() if sampling is None else sampling
        if type(parameters) is not Qwen38SamplingParameters:
            raise TypeError("sampling must be the exact Qwen38SamplingParameters")
        if generator is None and parameters.profile is not Qwen38SamplingProfile.GREEDY:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(parameters.seed)
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError("generator must be a torch.Generator")
            if torch.device(generator.device).type != "cpu":
                raise ValueError("sampling generator must be a CPU generator")
            if generator.initial_seed() != parameters.seed:
                raise ValueError(
                    f"sampling generator initial seed {generator.initial_seed()} differs from parameters {parameters.seed}"
                )
        if selected_mode is Qwen38HybridMode.MTP and parameters.profile is not Qwen38SamplingProfile.GREEDY:
            raise ValueError("MTP mode supports exact greedy target verification only; stochastic sampling is rejected")
        if policy is Qwen38HybridCachePolicy.RESET and self._consumed_token_ids:
            self.reset()
        suffix = self._cache_suffix(full_ids, cache_policy=Qwen38HybridCachePolicy.MATCH)
        initial = not self._consumed_token_ids
        planned_position = self.state_position
        if planned_position + len(suffix) > self.allocated_context:
            raise ValueError("rendered prompt suffix exceeds allocated cache capacity")
        old_consumed = self._consumed_token_ids
        old_position = planned_position
        rng_snapshot = None
        try:
            if self._controller is not None:
                self._detach_controller()
            base_position = self._require_target_state().position
            if initial != (self._mtp_seed is None):
                raise RuntimeError("target cache and MTP seed bootstrap states disagree")
            self._request_id += 1
            self._request_generated = 0
            self._request_max_new_tokens = limit
            self._request_diagnostic_exact_token_budget = diagnostic_exact_token_budget
            self._request_step_index = 0
            self._mode = selected_mode
            self._sampling_parameters = parameters
            self._request_generator = generator
            rng_snapshot = self._snapshot_request_generator()
            pending, timing, sampling_result, measurement_started_ns = self._bootstrap_or_extend(
                suffix,
                initial=initial,
                requested_mode=selected_mode,
            )
            if not initial and (old_position != base_position or full_ids[: len(old_consumed)] != old_consumed):
                raise RuntimeError("warm-cache prefix changed during begin")
            self._consumed_token_ids = full_ids
            return self._record_step(
                base_position=base_position,
                next_position=base_position + len(suffix),
                requested_mode=selected_mode,
                executed_mode=Qwen38HybridMode.ORDINARY,
                emissions=(
                    (
                        pending,
                        (
                            Qwen38HybridTokenSource.TARGET_GREEDY
                            if parameters.profile is Qwen38SamplingProfile.GREEDY
                            else Qwen38HybridTokenSource.TARGET_SAMPLED
                        ),
                    ),
                ),
                accepted_draft_count=0,
                committed_input_count=len(suffix),
                fallback_reason=None,
                timing=timing,
                measurement_started_ns=measurement_started_ns,
                sampling_result=sampling_result,
            )
        except Qwen38HybridDecodePoisonedError as error:
            self._restore_request_generator(rng_snapshot, error)
            raise
        except BaseException as error:
            self._poison(
                "streamed prompt bootstrap/continuation",
                self._restore_request_generator(rng_snapshot, error),
            )

    def _mtp_fallback_reason(self) -> Qwen38HybridFallbackReason | None:
        remaining = self._request_max_new_tokens - self._request_generated
        if remaining < VERIFY_POSITIONS:
            return Qwen38HybridFallbackReason.OUTPUT_BUDGET
        if self.state_position + VERIFY_POSITIONS > self.allocated_context:
            return Qwen38HybridFallbackReason.CONTEXT_TAIL
        return None

    def _ordinary_step(self, *, fallback_reason: Qwen38HybridFallbackReason | None) -> Qwen38HybridStep:
        requested_mode = self._mode
        started = self._clock_ns()
        if self._controller is not None:
            self._detach_controller()
        base_position = self._require_target_state().position
        pending_input = self._pending_token_id
        if pending_input is None or base_position >= self.allocated_context:
            raise RuntimeError("ordinary decode has no consumable pending token")
        progress: dict[str, Any] = {
            "target_state": self._require_target_state(),
            "target_rows": 0,
            "target_ns": 0,
            "sampling_ns": 0,
            "sampling_result": None,
            "pending": None,
        }
        pre_sync_ns = self._timed_sync()
        operation_started = self._clock_ns()
        streamed = self._stream_shifted_rows(
            (pending_input,), initial_state=self._require_target_state(), progress=progress
        )
        self._require_no_pending_seed("advance ordinary MTP alignment")
        self._pending_mtp_seed = self.mtp_engine.advance_seed_rows(self._require_seed(), streamed)
        self._mtp_seed = None
        self._unreleased_resources.discard("MTP ReadySeed")
        self._unreleased_resources.add("pending MTP ReadySeed")
        operation_ended = self._clock_ns()
        post_sync_ns = self._timed_sync()
        if progress["target_rows"] != 1 or progress["pending"] is None:
            raise RuntimeError("ordinary target/MTP alignment did not execute exactly one row")
        next_pending = int(progress["pending"])
        next_position = base_position + 1
        self._validate_target_state(progress["target_state"], expected=next_position)
        seed = self._publish_pending_seed(expected_position=next_position, expected_pending=next_pending)
        outer_ns = _duration(operation_started, operation_ended, label="ordinary target/MTP alignment")
        target_ns = int(progress["target_ns"])
        sampling_ns = int(progress["sampling_ns"])
        serialized_target_ns = target_ns - sampling_ns
        mtp_ns = outer_ns - target_ns
        if serialized_target_ns < 0 or mtp_ns < 0:
            raise RuntimeError("ordinary target timing exceeds its enclosing MTP alignment")
        self._target_state = progress["target_state"]
        self._mtp_seed = seed
        self._consumed_token_ids = (*self._consumed_token_ids, pending_input)
        if fallback_reason is not None:
            self._fallback_count += 1
        timing = self._make_timing(
            phase=Qwen38HybridTimingPhase.ORDINARY_DECODE,
            requested_mode=requested_mode,
            executed_mode=Qwen38HybridMode.ORDINARY,
            target_rows=1,
            mtp_extension_calls=0,
            mtp_alignment_rows=1,
            fallback_reason=fallback_reason,
            pre_sync_ns=pre_sync_ns,
            serialized_target_ns=serialized_target_ns,
            mtp_authoritative_ns=mtp_ns,
            speculative_transaction_ns=0,
            sampling_ns=sampling_ns,
            post_sync_ns=post_sync_ns,
        )
        return self._record_step(
            base_position=base_position,
            next_position=next_position,
            requested_mode=requested_mode,
            executed_mode=Qwen38HybridMode.ORDINARY,
            emissions=(
                (
                    next_pending,
                    (
                        Qwen38HybridTokenSource.ORDINARY_GREEDY
                        if self._sampling_parameters.profile is Qwen38SamplingProfile.GREEDY
                        else Qwen38HybridTokenSource.ORDINARY_SAMPLED
                    ),
                ),
            ),
            accepted_draft_count=0,
            committed_input_count=1,
            fallback_reason=fallback_reason,
            timing=timing,
            measurement_started_ns=started,
            sampling_result=progress["sampling_result"],
        )

    def _speculative_step(self) -> Qwen38HybridStep:
        started = self._clock_ns()
        controller = self._ensure_controller()
        base_position = controller.position
        pre_sync_ns = self._timed_sync()
        transaction_started = self._clock_ns()
        round_result = controller.step()
        transaction_ended = self._clock_ns()
        post_sync_ns = self._timed_sync()
        if type(round_result) is not Qwen38SpeculativeRound:
            raise TypeError("speculative controller returned the wrong round owner type")
        if round_result.base_position != base_position or round_result.next_position != controller.position:
            raise RuntimeError("speculative round changed base/next position metadata")
        if round_result.mtp_extension_count != DRAFT_EXTENSION_CALLS:
            raise RuntimeError(
                f"speculative round ran {round_result.mtp_extension_count} proposal calls, "
                f"expected exactly {DRAFT_EXTENSION_CALLS}"
            )
        if round_result.mtp_alignment_count != round_result.committed_input_count:
            raise RuntimeError("speculative MTP alignment rows differ from committed target rows")
        if round_result.pending_token_id != controller.pending_token_id:
            raise RuntimeError("speculative controller pending token differs from its round")
        committed = round_result.input_token_ids[: round_result.committed_input_count]
        if len(committed) != round_result.next_position - round_result.base_position:
            raise RuntimeError("speculative committed input prefix has the wrong length")
        self._consumed_token_ids = (*self._consumed_token_ids, *committed)
        emissions = tuple((item.token_id, _source(item.source)) for item in round_result.emissions)
        timing = self._make_timing(
            phase=Qwen38HybridTimingPhase.SPECULATIVE_DECODE,
            requested_mode=Qwen38HybridMode.MTP,
            executed_mode=Qwen38HybridMode.MTP,
            target_rows=VERIFY_POSITIONS,
            mtp_extension_calls=round_result.mtp_extension_count,
            mtp_alignment_rows=round_result.mtp_alignment_count,
            fallback_reason=None,
            pre_sync_ns=pre_sync_ns,
            serialized_target_ns=0,
            mtp_authoritative_ns=0,
            speculative_transaction_ns=_duration(
                transaction_started,
                transaction_ended,
                label="fixed-five speculative transaction",
            ),
            sampling_ns=0,
            post_sync_ns=post_sync_ns,
        )
        return self._record_step(
            base_position=round_result.base_position,
            next_position=round_result.next_position,
            requested_mode=Qwen38HybridMode.MTP,
            executed_mode=Qwen38HybridMode.MTP,
            emissions=emissions,
            accepted_draft_count=round_result.accepted_draft_count,
            committed_input_count=round_result.committed_input_count,
            fallback_reason=None,
            timing=timing,
            measurement_started_ns=started,
            speculative_round=round_result,
        )

    def step(self) -> Qwen38HybridStep:
        """Consume the pending token using the selected exact-greedy path."""

        self._assert_usable("decode a step")
        if self._status is not Qwen38HybridSessionStatus.ACTIVE:
            raise Qwen38HybridDecodeError(f"cannot decode while session status is {self._status.value}")
        rng_snapshot = self._snapshot_request_generator()
        try:
            if self._mode is Qwen38HybridMode.MTP:
                fallback = self._mtp_fallback_reason()
                if fallback is None:
                    return self._speculative_step()
                return self._ordinary_step(fallback_reason=fallback)
            return self._ordinary_step(fallback_reason=None)
        except Qwen38HybridDecodePoisonedError as error:
            self._restore_request_generator(rng_snapshot, error)
            raise
        except BaseException as error:
            self._poison(
                "ordinary/speculative decode step",
                self._restore_request_generator(rng_snapshot, error),
            )

    def generate(
        self,
        input_ids: torch.Tensor | Sequence[int],
        *,
        max_new_tokens: int,
        mode: Qwen38HybridMode | str = Qwen38HybridMode.ORDINARY,
        cache_policy: Qwen38HybridCachePolicy | str = Qwen38HybridCachePolicy.MATCH,
        sampling: Qwen38SamplingParameters | None = None,
        generator: torch.Generator | None = None,
        diagnostic_exact_token_budget: bool = False,
    ) -> Iterator[Qwen38HybridToken]:
        """Stream tokens; callbacks remain an explicit caller concern.

        ``diagnostic_exact_token_budget`` treats EOS as an ordinary token so a
        bounded mechanical diagnostic executes the requested token count.
        Output budget and native-context termination remain mandatory.  The
        default retains normal EOS semantics.
        """

        first = self.begin(
            input_ids,
            max_new_tokens=max_new_tokens,
            mode=mode,
            cache_policy=cache_policy,
            sampling=sampling,
            generator=generator,
            diagnostic_exact_token_budget=diagnostic_exact_token_budget,
        )
        yield from first.tokens
        while self._status is Qwen38HybridSessionStatus.ACTIVE:
            yield from self.step().tokens

    def _attempt_cleanup(
        self,
        failures: list[Qwen38HybridCleanupFailure],
        *,
        resource: str,
        action: Callable[[], None],
        clear: Callable[[], None],
    ) -> bool:
        """Attempt one ledger-owned release once; never infer success on raise."""

        try:
            action()
        except BaseException as error:
            failures.append(Qwen38HybridCleanupFailure(resource, error))
            self._unreleased_resources.add(resource)
            return False
        clear()
        self._unreleased_resources.discard(resource)
        return True

    def _release_mtp_cleanup_slots(self, failures: list[Qwen38HybridCleanupFailure]) -> None:
        seen: set[int] = set()
        slots = (
            ("_pending_mtp_seed", "pending MTP ReadySeed", self.mtp_engine.release_seed),
            ("_mtp_seed", "MTP ReadySeed", self.mtp_engine.release_seed),
            ("_raw_mtp_state", "fresh MTP state", self.mtp_engine.release_state),
        )
        for attribute, resource, releaser in slots:
            value = getattr(self, attribute)
            if value is None:
                continue
            if id(value) in seen:
                failures.append(
                    Qwen38HybridCleanupFailure(
                        resource,
                        RuntimeError("MTP cleanup slots alias; refusing a possible double release"),
                    )
                )
                self._unreleased_resources.add(resource)
                continue
            seen.add(id(value))
            self._attempt_cleanup(
                failures,
                resource=resource,
                action=lambda value=value, releaser=releaser: releaser(value),
                clear=lambda attribute=attribute: setattr(self, attribute, None),
            )

    def _close_owned_resident_experts_and_fence(self) -> None:
        """Close factory-owned static weights and prove the release boundary."""

        if not self._owns_built_graph or self._resident_expert_owner is None:
            raise RuntimeError("hybrid session does not own a resident built graph")
        self.built_target.components.close_resident_experts()
        self._timed_sync()
        backend_error = self._backend_poison_cause()
        if backend_error is not None:
            raise backend_error

    def reset(self) -> None:
        """Healthy target+MTP reset used for explicit full-history re-prefill."""

        self._assert_usable("reset target and MTP caches")
        failures: list[Qwen38HybridCleanupFailure] = []
        try:
            self._detach_controller()
        except BaseException as error:
            failures.append(Qwen38HybridCleanupFailure("speculative controller handoff", error))
            self._unreleased_resources.add("speculative controller")

        if self._controller is None:
            try:
                replacement_target = self.model.reset_state(self._require_target_state())
                # reset_state consumes the prior cache tree on successful
                # return; publish its replacement before validation.
                self._target_state = replacement_target
                self._validate_target_state(replacement_target, expected=0)
            except BaseException as error:
                failures.append(Qwen38HybridCleanupFailure("target state", error))
                self._unreleased_resources.add("target state")
            self._release_mtp_cleanup_slots(failures)

        if not failures and all(
            value is None for value in (self._pending_mtp_seed, self._mtp_seed, self._raw_mtp_state)
        ):
            try:
                self._raw_mtp_state = self.mtp_engine.allocate_state()
                self._unreleased_resources.add("fresh MTP state")
                if self.mtp_engine.state_position(self._raw_mtp_state) != 0:
                    raise RuntimeError("replacement MTP state did not start at position zero")
            except BaseException as error:
                failures.append(Qwen38HybridCleanupFailure("fresh MTP state", error))
                self._unreleased_resources.add("fresh MTP state")

        try:
            self._timed_sync()
        except BaseException as error:
            failures.append(Qwen38HybridCleanupFailure("mesh synchronization status", error))
            self._unreleased_resources.add("mesh synchronization status")

        if failures:
            self._cleanup_failures.extend(failures)
            self._poison("reset target and MTP caches", Qwen38HybridCleanupError("reset", failures))

        self._pending_token_id = None
        self._consumed_token_ids = ()
        self._request_generated = 0
        self._request_max_new_tokens = 0
        self._request_diagnostic_exact_token_budget = False
        self._request_step_index = 0
        self._sampling_parameters = Qwen38SamplingParameters.greedy()
        self._request_generator = None
        self._status = Qwen38HybridSessionStatus.IDLE
        self._reset_count += 1

    def close(self) -> None:
        """Best-effort release healthy resources exactly once.

        Any failed release is terminal because a raised device deallocation
        does not prove whether it took effect.  The poisoned error retains this
        session, and ``cleanup_failures``/``unreleased_resources`` remain an
        inspectable process-teardown report; computation and per-resource
        cleanup retry are deliberately forbidden.
        """

        if self._status is Qwen38HybridSessionStatus.CLOSED:
            return
        self._assert_usable("close hybrid session")
        failures: list[Qwen38HybridCleanupFailure] = []
        try:
            self._detach_controller()
        except BaseException as error:
            failures.append(Qwen38HybridCleanupFailure("speculative controller handoff", error))
            self._unreleased_resources.add("speculative controller")

        if self._controller is None:
            if self._target_state is not None:
                target_state = self._target_state
                self._attempt_cleanup(
                    failures,
                    resource="target state",
                    action=lambda: self.fixed_target.release_state(target_state),
                    clear=lambda: setattr(self, "_target_state", None),
                )
            self._release_mtp_cleanup_slots(failures)

        if self._sampler_open:
            self._attempt_cleanup(
                failures,
                resource="host sampler",
                action=self.host_sampler.close,
                clear=lambda: setattr(self, "_sampler_open", False),
            )
        if (
            self._mtp_claimed
            and self._controller is None
            and self._pending_mtp_seed is None
            and self._mtp_seed is None
            and self._raw_mtp_state is None
        ):
            self._attempt_cleanup(
                failures,
                resource="MTP runtime claim",
                action=lambda: self.mtp_engine.release_runtime_owner(self._runtime_owner),
                clear=lambda: setattr(self, "_mtp_claimed", False),
            )
        if self._fixed_claimed and self._controller is None and self._target_state is None:
            self._attempt_cleanup(
                failures,
                resource="target runtime claim",
                action=lambda: self.fixed_target.release_runtime_owner(self._runtime_owner),
                clear=lambda: setattr(self, "_fixed_claimed", False),
            )
        if self._fixed_open and self._controller is None and not self._fixed_claimed:
            self._attempt_cleanup(
                failures,
                resource="fixed-five private buffers",
                action=self.fixed_target.close,
                clear=lambda: setattr(self, "_fixed_open", False),
            )
        backend_error = self._backend_poison_cause()
        if backend_error is not None:
            failures.append(Qwen38HybridCleanupFailure("backend health before resident close", backend_error))
        if self._resident_expert_owner_open and not failures:
            self._attempt_cleanup(
                failures,
                resource="resident BF4 experts",
                action=self._close_owned_resident_experts_and_fence,
                clear=lambda: setattr(self, "_resident_expert_owner_open", False),
            )
        try:
            self._timed_sync()
        except BaseException as error:
            failures.append(Qwen38HybridCleanupFailure("mesh synchronization status", error))
            self._unreleased_resources.add("mesh synchronization status")

        backend_error = self._backend_poison_cause()
        if backend_error is not None and not any(failure.error is backend_error for failure in failures):
            failures.append(Qwen38HybridCleanupFailure("backend health after close", backend_error))

        if failures:
            self._cleanup_failures.extend(failures)
            self._poison("close hybrid session", Qwen38HybridCleanupError("close", failures))

        self._pending_token_id = None
        self._request_generator = None
        self._status = Qwen38HybridSessionStatus.CLOSED

    def __enter__(self) -> "Qwen38TTNNHybridDecodeSession":
        self._assert_usable("enter hybrid session")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        del exc_type, exc_value, traceback
        if self._status is not Qwen38HybridSessionStatus.POISONED:
            self.close()
        return False


def validate_hybrid_decode_static_contract() -> None:
    if (
        TP_SIZE,
        MESH_SHAPE,
        BACKBONE_LAYERS,
        DRAFT_EXTENSION_CALLS,
        VERIFY_POSITIONS,
        VOCAB_SIZE,
        MAX_CONTEXT,
    ) != (4, (1, 4), 48, 3, 5, 248_320, 262_144):
        raise RuntimeError("Qwen3.8 hybrid TTNN geometry or shifted-MTP contract drifted")


validate_hybrid_decode_static_contract()


__all__ = [
    "DIAGNOSTIC_EXACT_TOKEN_BUDGET",
    "Qwen38HybridCacheMismatchError",
    "Qwen38HybridCachePolicy",
    "Qwen38HybridCleanupError",
    "Qwen38HybridCleanupFailure",
    "Qwen38HybridConstructionError",
    "Qwen38HybridDecodeError",
    "Qwen38HybridDecodePoisonedError",
    "Qwen38HybridFactoryError",
    "Qwen38HybridFallbackReason",
    "Qwen38HybridMode",
    "Qwen38HybridModeCapabilities",
    "Qwen38HybridSessionIdentity",
    "Qwen38HybridSessionStatus",
    "Qwen38HybridStep",
    "Qwen38HybridStopReason",
    "Qwen38HybridTiming",
    "Qwen38HybridTimingPhase",
    "Qwen38HybridToken",
    "Qwen38HybridTokenSource",
    "Qwen38TTNNHybridDecodeSession",
    "validate_hybrid_decode_static_contract",
]
