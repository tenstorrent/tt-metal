# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmup coverage planning and trace-activation coordination."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator

import torch
from loguru import logger

from models.common.llm_runtime.config import PageTableLayout, TraceConfig, WarmupConfig
from models.common.llm_runtime.decode import DecodeRuntimeConfig
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.program_compiler import CompiledProgram
from models.common.sampling import SamplingParams


@dataclass(frozen=True)
class WarmupCase:
    operation: str
    batch_size: int
    sequence_length: int | None
    sampling_path: str
    cached_tokens: int = 0


@dataclass(frozen=True)
class WarmupPlan:
    prefill: tuple[WarmupCase, ...]
    decode: tuple[WarmupCase, ...]


@dataclass(frozen=True)
class CoverageAlias:
    """One exact compiled-program association with a configured trace."""

    program_signature: Any
    trace_signature: Any


@dataclass(frozen=True)
class CoverageManifest:
    """Registry-authoritative operation identities sealed at activation."""

    eager_program_signatures: tuple[Any, ...]
    traced_source_program_signatures: tuple[Any, ...]
    trace_signatures: tuple[Any, ...]
    aliases: tuple[CoverageAlias, ...]


@dataclass(frozen=True)
class WarmupCoordinatorConfig:
    """Fully resolved immutable warmup policy and coverage."""

    warmup: WarmupConfig
    model: Any
    page_table_layout: PageTableLayout  # Current geometry used to build coverage plans.
    page_table_layout_ceiling: PageTableLayout  # Construction-time upper bound retained across replacement.
    prefill_sequence_lengths: tuple[int, ...]
    lane_batch_size: int
    device_sampling_enabled: bool
    allow_force_argmax: bool
    prime_q128_tile_ends: bool
    prefill_trace_enabled: bool
    decode_trace_enabled: bool
    eager_plan: WarmupPlan
    sampled_plan: WarmupPlan

    def __post_init__(self) -> None:
        if not isinstance(self.warmup, WarmupConfig):
            raise TypeError("warmup must be a WarmupConfig")
        if self.model is None:
            raise ValueError("model is required")
        if not isinstance(self.page_table_layout, PageTableLayout):
            raise TypeError("page_table_layout must be a PageTableLayout")
        _validate_prefill_sequence_lengths(self.prefill_sequence_lengths)
        _require_positive_int("lane_batch_size", self.lane_batch_size)
        for name in (
            "device_sampling_enabled",
            "allow_force_argmax",
            "prime_q128_tile_ends",
            "prefill_trace_enabled",
            "decode_trace_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not self.device_sampling_enabled and self.allow_force_argmax:
            raise ValueError("force-argmax capability requires device sampling")
        if self.prime_q128_tile_ends is not (self.device_sampling_enabled and self.lane_batch_size >= 32):
            raise ValueError("prime_q128_tile_ends must match resolved sampling and lane capabilities")
        if not isinstance(self.page_table_layout_ceiling, PageTableLayout):
            raise TypeError("page_table_layout_ceiling must be a PageTableLayout")
        if self.page_table_layout.block_size != self.page_table_layout_ceiling.block_size:
            raise ValueError("page_table_layout_ceiling cannot change block_size")
        if self.page_table_layout.raw_capacity_width > self.page_table_layout_ceiling.raw_capacity_width:
            raise ValueError("page_table_layout_ceiling must cover page_table_layout capacity")
        if (
            self.page_table_layout.prefill_width > self.page_table_layout_ceiling.prefill_width
            or self.page_table_layout.decode_width > self.page_table_layout_ceiling.decode_width
        ):
            raise ValueError("page_table_layout_ceiling must cover canonical page-table geometry")
        expected_eager = _build_plan(
            warmup=self.warmup,
            layout=self.page_table_layout,
            prefill_sequence_lengths=self.prefill_sequence_lengths,
            lane_batch_size=self.lane_batch_size,
            allow_force_argmax=self.allow_force_argmax,
            can_sample_on_device=False,
        )
        expected_sampled = _build_plan(
            warmup=self.warmup,
            layout=self.page_table_layout,
            prefill_sequence_lengths=self.prefill_sequence_lengths,
            lane_batch_size=self.lane_batch_size,
            allow_force_argmax=self.allow_force_argmax,
            can_sample_on_device=True,
        )
        if self.eager_plan != expected_eager or self.sampled_plan != expected_sampled:
            raise ValueError("warmup plans must match resolved policy and geometry")

    @classmethod
    def resolve(
        cls,
        *,
        warmup: WarmupConfig,
        trace: TraceConfig,
        prefill: PrefillRuntimeConfig,
        decode: DecodeRuntimeConfig,
        prefill_sequence_lengths: tuple[int, ...],
    ) -> "WarmupCoordinatorConfig":
        """Validate resolved runtimes and derive both static coverage plans."""

        if not isinstance(warmup, WarmupConfig):
            raise TypeError("warmup must be a WarmupConfig")
        if not isinstance(trace, TraceConfig):
            raise TypeError("trace must be a TraceConfig")
        if not isinstance(prefill, PrefillRuntimeConfig):
            raise TypeError("prefill must be a PrefillRuntimeConfig")
        if not isinstance(decode, DecodeRuntimeConfig):
            raise TypeError("decode must be a DecodeRuntimeConfig")
        if decode.model is not prefill.model:
            raise ValueError("prefill and decode configs must share one model")
        if decode.page_table_layout is not prefill.page_table_layout:
            raise ValueError("prefill and decode configs must share one page-table layout")
        if decode.lane_capacity != prefill.max_batch_size:
            raise ValueError("prefill and decode configs must share one lane capacity")
        if decode.device_sampling_enabled is not prefill.device_sampling_enabled:
            raise ValueError("prefill and decode configs must share device-sampling policy")
        if decode.allow_force_argmax is not prefill.allow_force_argmax:
            raise ValueError("prefill and decode configs must share force-argmax capability")
        if decode.page_table_layout_ceiling != prefill.page_table_layout_ceiling:
            raise ValueError("prefill and decode configs must share one page-table layout ceiling")

        source_lengths = warmup.prefill_seq_lens
        if source_lengths is None:
            source_lengths = prefill_sequence_lengths
        _validate_prefill_sequence_lengths(source_lengths)

        lane_batch_size = prefill.max_batch_size
        device_sampling_enabled = prefill.device_sampling_enabled
        allow_force_argmax = prefill.allow_force_argmax
        prime_q128_tile_ends = device_sampling_enabled and lane_batch_size >= 32
        eager_plan = _build_plan(
            warmup=warmup,
            layout=prefill.page_table_layout,
            prefill_sequence_lengths=source_lengths,
            lane_batch_size=lane_batch_size,
            allow_force_argmax=allow_force_argmax,
            can_sample_on_device=False,
        )
        sampled_plan = _build_plan(
            warmup=warmup,
            layout=prefill.page_table_layout,
            prefill_sequence_lengths=source_lengths,
            lane_batch_size=lane_batch_size,
            allow_force_argmax=allow_force_argmax,
            can_sample_on_device=True,
        )
        return cls(
            warmup=warmup,
            model=prefill.model,
            page_table_layout=prefill.page_table_layout,
            prefill_sequence_lengths=source_lengths,
            lane_batch_size=lane_batch_size,
            device_sampling_enabled=device_sampling_enabled,
            allow_force_argmax=allow_force_argmax,
            prime_q128_tile_ends=prime_q128_tile_ends,
            prefill_trace_enabled=trace.prefill_enabled,
            decode_trace_enabled=trace.decode_enabled,
            eager_plan=eager_plan,
            sampled_plan=sampled_plan,
            page_table_layout_ceiling=prefill.page_table_layout_ceiling,
        )

    def with_page_table_layout(self, layout: PageTableLayout) -> "WarmupCoordinatorConfig":
        """Return the same policy with final geometry within original ceilings."""

        if not isinstance(layout, PageTableLayout):
            raise TypeError("layout must be a PageTableLayout")
        if layout.block_size != self.page_table_layout.block_size:
            raise ValueError("page-table layout replacement cannot change block_size")
        if layout.raw_capacity_width > self.page_table_layout_ceiling.raw_capacity_width:
            raise ValueError("page-table layout replacement cannot exceed the construction-time capacity ceiling")
        if (
            layout.prefill_width > self.page_table_layout_ceiling.prefill_width
            or layout.decode_width > self.page_table_layout_ceiling.decode_width
        ):
            raise ValueError("page-table layout replacement cannot expand canonical geometry")
        return replace(
            self,
            page_table_layout=layout,
            eager_plan=_build_plan(
                warmup=self.warmup,
                layout=layout,
                prefill_sequence_lengths=self.prefill_sequence_lengths,
                lane_batch_size=self.lane_batch_size,
                allow_force_argmax=self.allow_force_argmax,
                can_sample_on_device=False,
            ),
            sampled_plan=_build_plan(
                warmup=self.warmup,
                layout=layout,
                prefill_sequence_lengths=self.prefill_sequence_lengths,
                lane_batch_size=self.lane_batch_size,
                allow_force_argmax=self.allow_force_argmax,
                can_sample_on_device=True,
            ),
        )


class WarmupCoordinator:
    """Compile configured coverage and activate traces at one shared barrier.

    ``Llama3Executor.warmup_model_prefill`` and ``warmup_model_decode`` call
    `warmup_prefill` and `warmup_decode` in either order. Each
    method compiles its required eager programs and registers trace plans.
    Capture begins only after both configured operation sets are complete.
    """

    def __init__(
        self,
        *,
        config: WarmupCoordinatorConfig,
        execution: Any,
        ensure_sampling_buffers: Callable[[], None],
        validate_bound_cache: Callable[[Any], None],
    ) -> None:
        if not isinstance(config, WarmupCoordinatorConfig):
            raise TypeError("config must be a WarmupCoordinatorConfig")
        eager = getattr(execution, "eager_executor", execution)
        trace_compiler = getattr(execution, "trace_compiler", None)
        prefill_config = getattr(getattr(eager, "prefill", None), "config", None)
        decode_config = getattr(getattr(eager, "decode", None), "config", None)
        if not isinstance(prefill_config, PrefillRuntimeConfig) or not isinstance(decode_config, DecodeRuntimeConfig):
            raise TypeError("execution must compose configured prefill and decode runtimes")
        if prefill_config.model is not config.model or decode_config.model is not config.model:
            raise ValueError("execution runtimes must use the warmup config model")
        if (
            prefill_config.page_table_layout is not config.page_table_layout
            or decode_config.page_table_layout is not config.page_table_layout
        ):
            raise ValueError("execution runtimes must use the warmup config page-table layout")
        if (
            prefill_config.max_batch_size != config.lane_batch_size
            or decode_config.lane_capacity != config.lane_batch_size
        ):
            raise ValueError("execution runtimes must use the warmup config lane capacity")
        if (
            prefill_config.device_sampling_enabled is not config.device_sampling_enabled
            or decode_config.device_sampling_enabled is not config.device_sampling_enabled
        ):
            raise ValueError("execution runtimes must use the warmup config sampling policy")
        if not callable(ensure_sampling_buffers):
            raise TypeError("ensure_sampling_buffers must be callable")
        if not callable(validate_bound_cache):
            raise TypeError("validate_bound_cache must be callable")

        self.config = config
        self.execution = execution
        self.eager = eager
        self.trace_compiler = trace_compiler
        self._ensure_sampling_buffers = ensure_sampling_buffers
        self._validate_bound_cache = validate_bound_cache
        self._eager: set[WarmupCase] = set()
        self._trace_registered: set[WarmupCase] = set()
        self._trace_decisions: dict[str, bool] = {}
        self._sampling_decisions: dict[str, bool] = {}
        self._captured = False
        self._coverage_manifest: CoverageManifest | None = None
        self._required_program_keys: set[Any] = set()
        self._required_trace_program_keys: set[Any] = set()
        self._capture_deferred = False
        self._capture_pending = False
        self._pending_manifest: CoverageManifest | None = None
        self._prefill_trace_postprocess_primed = False
        self._configuration_sealed = False

    # Public API

    @property
    def already_warmed_up_prefill(self) -> bool:
        """Whether all configured prefill programs and traces are ready."""

        can_sample_on_device = self._sampling_decisions.get("prefill", self.config.device_sampling_enabled)
        required = set(self._plan(can_sample_on_device=can_sample_on_device).prefill)
        if not required.issubset(self._eager):
            return False
        if not self.config.prefill_trace_enabled or self._trace_decisions.get("prefill") is False:
            return True
        return required.issubset(self._trace_registered) and self._captured

    @property
    def coverage_manifest(self) -> CoverageManifest | None:
        """Return the immutable identities verified by successful activation."""

        return self._coverage_manifest

    @property
    def capture_pending(self) -> bool:
        """Whether complete validated coverage is staged for activation."""

        return self._capture_pending

    @property
    def trace_activated(self) -> bool:
        """Whether this coordinator has completed trace capture and activation."""

        return self._captured

    @contextmanager
    def defer_capture(self) -> Iterator["WarmupCoordinator"]:
        """Stage readiness without capturing until a multi-lane barrier commits."""

        if self._capture_deferred:
            raise RuntimeError("trace capture deferral is already active")
        if self._capture_pending:
            raise RuntimeError("trace capture is already pending")
        self._capture_deferred = True
        try:
            yield self
        finally:
            self._capture_deferred = False
            self._capture_pending = False
            self._pending_manifest = None

    def activate_pending_capture(self) -> None:
        """Commit one validated capture while its deferral context is active."""

        if not self._capture_deferred:
            raise RuntimeError("pending trace capture can only activate inside its deferral context")
        if not self._capture_pending:
            raise RuntimeError("no trace capture is pending")
        self._capture_now(self._pending_manifest)
        self._capture_pending = False
        self._pending_manifest = None

    def configure_page_table_layout(self, layout: PageTableLayout) -> None:
        """Install final paged-KV geometry before warmup compiles any program."""

        if self._configuration_sealed:
            raise RuntimeError("page-table layout cannot change after warmup configuration is sealed")
        self.config = self.config.with_page_table_layout(layout)

    def seal_configuration(self) -> None:
        """Forbid geometry replacement before physical KV allocation begins."""

        self._configuration_sealed = True

    def warmup_prefill(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        """Compile prefill coverage and capture once decode coverage is ready."""

        self._validate_hints("prefill", enable_trace, can_sample_on_device)
        self._validate_bound_cache(kv_cache)
        self._sampling_decisions["prefill"] = bool(can_sample_on_device)
        self._trace_decisions["prefill"] = bool(enable_trace)
        if enable_trace and self._trace_decisions.get("decode") is False and self.config.decode_trace_enabled:
            del self._trace_decisions["decode"]
        self._configuration_sealed = True
        if can_sample_on_device:
            self._ensure_sampling_buffers()
        plan = self._plan(can_sample_on_device=can_sample_on_device)
        destination = self._trace_registered if enable_trace else self._eager
        cases = plan.prefill
        if enable_trace and can_sample_on_device:
            # The hidden-body trace is sampling-independent, but its retained
            # post-trace inputs must support both aliases. Register the forced
            # top-k variant first so the shared artifact owns a K/P/T buffer.
            cases = tuple(sorted(cases, key=lambda case: case.sampling_path != "topk"))
        for case in cases:
            if case in destination:
                continue
            sampling = None
            if case.sampling_path == "argmax":
                sampling = _greedy_sampling_params(case.batch_size)
            elif case.sampling_path == "topk":
                sampling = _topk_sampling_params(case.batch_size)
            actual_uncached_lengths = (int(case.sequence_length),)
            if (
                case.batch_size == 1
                and case.sequence_length == 128
                and case.cached_tokens == 0
                and (
                    case.sampling_path == "argmax"
                    or (case.sampling_path == "topk" and self.config.prime_q128_tile_ends)
                )
            ):
                # Q128 single-user sampled postprocessing has one TT slice
                # program per tile start. Prime all four without expanding the
                # public warmup coverage model.
                actual_uncached_lengths = (32, 64, 96, 128)
            for actual_uncached_length in actual_uncached_lengths:
                prompt_length = case.cached_tokens + actual_uncached_length
                tokens = torch.zeros((case.batch_size, prompt_length), dtype=torch.long)
                prompt_lens = torch.full((case.batch_size,), prompt_length, dtype=torch.long)
                width = _ceil_div(prompt_length, self.config.page_table_layout.block_size)
                page_table = torch.zeros((case.batch_size, width), dtype=torch.int32)
                start_pos = (
                    torch.full((case.batch_size,), case.cached_tokens, dtype=torch.long) if case.cached_tokens else None
                )
                compile_target = self.execution if enable_trace else self.eager
                programs = compile_target.compile_prefill(
                    tokens=tokens,
                    page_table=page_table,
                    prompt_lens=prompt_lens,
                    start_pos=start_pos,
                    empty_slots=list(range(case.batch_size)),
                    sampling_params=sampling,
                )
                self._record_required_programs(programs, traced=enable_trace)
            destination.add(case)
        self._maybe_capture()

    def warmup_decode(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        max_batch_size: int,  # ↓ Coverage dimensions
        num_blocks: int,
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        """Compile decode coverage and capture once prefill coverage is ready."""

        self._validate_hints("decode", enable_trace, can_sample_on_device)
        self._validate_bound_cache(kv_cache)
        lane_batch = self.config.lane_batch_size
        if int(max_batch_size) != lane_batch:
            raise ValueError(f"decode warmup batch {max_batch_size} does not match lane capacity {lane_batch}")
        if int(num_blocks) <= 0:
            raise ValueError("decode warmup num_blocks must be positive")
        self._sampling_decisions["decode"] = bool(can_sample_on_device)
        self._trace_decisions["decode"] = bool(enable_trace)
        self._configuration_sealed = True
        if can_sample_on_device:
            self._ensure_sampling_buffers()
        plan = self._plan(can_sample_on_device=can_sample_on_device)
        destination = self._trace_registered if enable_trace else self._eager
        for case in plan.decode:
            if case in destination:
                continue
            sampling = None
            if case.sampling_path == "argmax":
                sampling = _greedy_sampling_params(lane_batch)
            elif case.sampling_path == "topk":
                sampling = _topk_sampling_params(lane_batch)
            compile_target = self.execution if enable_trace else self.eager
            program = compile_target.compile_decode(
                tokens=torch.zeros(lane_batch, dtype=torch.long),
                start_pos=torch.zeros(lane_batch, dtype=torch.long),
                page_table=torch.zeros((lane_batch, int(num_blocks)), dtype=torch.int32),
                sampling_params=sampling,
            )
            self._record_required_programs(program, traced=enable_trace)
            if not enable_trace:
                logger.info("Compiled decode")
                if sampling is not None:
                    logger.info("Compiled on-device sampling")
            destination.add(case)
        self._maybe_capture()

    # Private implementation

    def _plan(self, *, can_sample_on_device: bool) -> WarmupPlan:
        return self.config.sampled_plan if can_sample_on_device else self.config.eager_plan

    def _maybe_capture(self) -> None:
        if self.trace_compiler is None or self._captured:
            return
        required_trace: set[WarmupCase] = set()
        if self.config.prefill_trace_enabled:
            prefill_decision = self._trace_decisions.get("prefill")
            if prefill_decision is None:
                return
            if prefill_decision:
                prefill_plan = self._plan(can_sample_on_device=self._sampling_decisions["prefill"])
                required_trace.update(prefill_plan.prefill)
        if self.config.decode_trace_enabled:
            decode_decision = self._trace_decisions.get("decode")
            if decode_decision is None:
                return
            if decode_decision:
                decode_plan = self._plan(can_sample_on_device=self._sampling_decisions["decode"])
                required_trace.update(decode_plan.decode)
        if not required_trace:
            return
        if not required_trace.issubset(self._trace_registered):
            return
        manifest = self._prepare_capture_manifest()
        if self._capture_deferred:
            self._pending_manifest = manifest
            self._capture_pending = True
            return
        self._capture_now(manifest)

    def _prepare_capture_manifest(self) -> CoverageManifest | None:
        # WarmupCase is only an idempotency key for public warmup calls. The
        # compiler registries own identity coverage: validate their exact state
        # before a single-lane capture or a multi-lane barrier reports ready.
        manifest = _resolve_coverage_manifest(
            self.eager,
            self.trace_compiler,
            required_program_keys=self._required_program_keys,
            required_trace_program_keys=self._required_trace_program_keys,
        )
        if manifest is not None and not manifest.aliases:
            raise RuntimeError("Configured trace warmup registered no program-to-trace aliases")
        return manifest

    def _capture_now(self, manifest: CoverageManifest | None) -> None:
        self.trace_compiler.capture_all()
        self._captured = True
        self._coverage_manifest = manifest
        self._prime_prefill_trace_postprocess()

    def _record_required_programs(self, programs: Any, *, traced: bool) -> None:
        if programs is None:
            return
        if isinstance(programs, CompiledProgram):
            programs = (programs,)
        if not isinstance(programs, tuple) or any(not isinstance(program, CompiledProgram) for program in programs):
            raise TypeError("compile targets must return CompiledProgram values")
        keys = {program.key for program in programs}
        self._required_program_keys.update(keys)
        if traced:
            self._required_trace_program_keys.update(keys)

    def _prime_prefill_trace_postprocess(self) -> None:
        if (
            self._prefill_trace_postprocess_primed
            or self._trace_decisions.get("prefill") is False
            or not self.config.prefill_trace_enabled
        ):
            return
        prefill_can_sample = self._sampling_decisions.get("prefill", self.config.device_sampling_enabled)
        if not prefill_can_sample or not self.config.allow_force_argmax:
            self._prefill_trace_postprocess_primed = True
            return
        sequence_length = (
            128 if 128 in self.config.prefill_sequence_lengths else int(self.config.prefill_sequence_lengths[0])
        )
        width = _ceil_div(sequence_length, self.config.page_table_layout.block_size)
        self.execution.prefill_forward(
            tokens=torch.zeros((1, sequence_length), dtype=torch.long),
            page_table=torch.zeros((1, width), dtype=torch.int32),
            prompt_lens=torch.full((1,), sequence_length, dtype=torch.long),
            empty_slots=[0],
            start_pos=None,
            sampling_params=_greedy_sampling_params(1),
        )
        self._prefill_trace_postprocess_primed = True

    def _validate_hints(self, operation: str, enable_trace: bool, can_sample_on_device: bool) -> None:
        trace_enabled = (
            self.config.prefill_trace_enabled if operation == "prefill" else self.config.decode_trace_enabled
        )
        if enable_trace and not trace_enabled:
            raise ValueError(f"{operation} trace warmup exceeds the configured trace policy")
        if can_sample_on_device and not self.config.device_sampling_enabled:
            raise ValueError("warmup cannot enable device sampling when it is statically disabled")


def _validate_prefill_sequence_lengths(values: Any) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError("prefill sequence lengths must be a non-empty tuple")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("prefill sequence lengths must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("prefill sequence lengths must be unique")


def _resolve_coverage_manifest(
    eager: Any,
    trace_compiler: Any,
    *,
    required_program_keys: set[Any] | None = None,
    required_trace_program_keys: set[Any] | None = None,
) -> CoverageManifest | None:
    """Resolve actual registered identities when concrete registries are available."""

    program_compiler = getattr(eager, "program_compiler", None)
    programs = getattr(program_compiler, "compiled_programs", None)
    if programs is None or not callable(getattr(trace_compiler, "trace_key_for_program", None)):
        # Lightweight host-contract doubles intentionally need not reproduce
        # compiler internals; production executors always expose both registries.
        return None

    required_program_keys = set() if required_program_keys is None else set(required_program_keys)
    required_trace_program_keys = set() if required_trace_program_keys is None else set(required_trace_program_keys)
    programs_by_key = {program.key: program for program in programs}
    missing_programs = required_program_keys.difference(programs_by_key)
    if missing_programs:
        digests = sorted(key.digest for key in missing_programs)
        raise RuntimeError(f"Coverage manifest is missing required compiled programs: {digests}")
    missing_aliases = {key for key in required_trace_program_keys if trace_compiler.trace_key_for_program(key) is None}
    if missing_aliases:
        digests = sorted(key.digest for key in missing_aliases)
        raise RuntimeError(f"Coverage manifest is missing required trace aliases: {digests}")

    eager_signatures = []
    traced_signatures = []
    aliases = []
    trace_signatures_by_key = {}
    for program in programs:
        if not isinstance(program, CompiledProgram):
            raise TypeError("program compiler snapshots must contain CompiledProgram values")
        trace_key = trace_compiler.trace_key_for_program(program.key)
        if trace_key is None:
            eager_signatures.append(program.signature)
            continue
        record = trace_compiler.get(trace_key)
        if record is None:
            raise RuntimeError(f"Trace association {trace_key.digest} has no registered trace record")
        traced_signatures.append(program.signature)
        aliases.append(CoverageAlias(program.signature, record.signature))
        trace_signatures_by_key.setdefault(trace_key, record.signature)

    return CoverageManifest(
        eager_program_signatures=tuple(eager_signatures),
        traced_source_program_signatures=tuple(traced_signatures),
        trace_signatures=tuple(trace_signatures_by_key.values()),
        aliases=tuple(aliases),
    )


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _build_plan(
    *,
    warmup: WarmupConfig,
    layout: PageTableLayout,
    prefill_sequence_lengths: tuple[int, ...],
    lane_batch_size: int,
    allow_force_argmax: bool,
    can_sample_on_device: bool,
) -> WarmupPlan:
    sampling_paths = ["logits"]
    if can_sample_on_device:
        sampling_paths.append("topk")
    prefill = []
    for sequence_length in prefill_sequence_lengths:
        batches = warmup.prefill_batch_sizes if sequence_length == 128 else (1,)
        for batch_size in batches:
            if batch_size <= lane_batch_size:
                batch_sampling_paths = sampling_paths + (
                    ["argmax"] if can_sample_on_device and allow_force_argmax and batch_size == 1 else []
                )
                prefill.extend(
                    WarmupCase("prefill", batch_size, sequence_length, sampling_path)
                    for sampling_path in batch_sampling_paths
                )
        cached_prompt_length = layout.block_size + sequence_length
        if cached_prompt_length <= layout.raw_capacity_width * layout.block_size:
            prefill.extend(
                WarmupCase(
                    "prefill",
                    1,
                    sequence_length,
                    sampling_path,
                    cached_tokens=layout.block_size,
                )
                for sampling_path in sampling_paths
                + (["argmax"] if can_sample_on_device and allow_force_argmax else [])
            )

    decode_paths = ["logits"]
    if can_sample_on_device:
        if allow_force_argmax:
            decode_paths.append("argmax")
        if not allow_force_argmax or warmup.include_decode_top_k:
            decode_paths.append("topk")
    decode = tuple(WarmupCase("decode", lane_batch_size, None, sampling_path) for sampling_path in decode_paths)
    return WarmupPlan(tuple(prefill), decode)


def _greedy_sampling_params(batch_size: int) -> SamplingParams:
    return SamplingParams(
        temperature=torch.zeros(batch_size),
        top_k=torch.ones(batch_size, dtype=torch.int32),
        top_p=torch.ones(batch_size),
    )


def _topk_sampling_params(batch_size: int) -> SamplingParams:
    return SamplingParams(
        temperature=torch.ones(batch_size),
        top_k=torch.full((batch_size,), 32, dtype=torch.int32),
        top_p=torch.full((batch_size,), 0.08),
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
