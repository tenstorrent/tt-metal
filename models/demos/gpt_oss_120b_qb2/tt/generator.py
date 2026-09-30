# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the optimized GPT-OSS 120B full model.

The default token-out path is canonical split sampling: the shared
``tt_transformers`` generator captures the model graph and the repository
``SamplingGenerator`` captures sampling over the vocab-sharded LM-head output.
The sampled token is written into the persistent decode token input, and the
model advances position and RoPE indices inside its trace.  The optimized
steady-state path submits model and sampling replays without reading a token or
refreshing fixed greedy parameters between steps.  A caller may collect the
device output synchronously or asynchronously at an explicit boundary; full
logits never cross that boundary.

``sampling_mode="host"`` is an explicit compatibility path for accuracy tools
that require logits.  It is not the optimized or measured token-out path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch

import ttnn
from models.common.sampling.generator import SamplingParams
from models.demos.gpt_oss_120b_qb2.tt.model import HF_CONTEXT_LENGTH, MODEL_LAYERS, PAGE_SIZE, Model, build_model
from models.demos.gpt_oss_120b_qb2.tt.sliding_ring import PREFILL_CHUNK_ALIGN
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import Generator as _TTGenerator

GREEDY = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

SAMPLER_DECISION = {
    "selected": "models.common.sampling.generator.SamplingGenerator",
    "selected_reason": (
        "It is already integrated with tt_transformers decode capture, persistent tt_out_tok feedback, "
        "per-slot sampling state, penalties, seeds, and log-prob handling. Its split greedy path avoids "
        "the slower full-vocabulary all-gather used by generic force-argmax."
    ),
    "rejected": "models.common.modules.sampling.sampling_1d.Sampling1D",
    "rejected_reason": (
        "Sampling1D is a useful stateless kernel wrapper but has no matching trace/token-feedback, seed, "
        "penalty, or request-state owner; selecting it would require a redundant custom sampling runtime."
    ),
    "greedy_semantics": "temperature=0 is canonically normalized to temperature=1, top_k=1, top_p=0",
    "force_argmax": False,
}


@dataclass
class TraceEvidence:
    decode_calls: int = 0
    trace_replays: int = 0
    model_execute_submissions: int = 0
    sampling_execute_submissions: int = 0
    unclassified_execute_submissions: int = 0
    full_input_refreshes: int = 0
    page_table_only_refreshes: int = 0
    page_table_reuses: int = 0
    sampled_token_readbacks: int = 0
    full_logits_readbacks: int = 0
    forced_token_refreshes: int = 0
    host_argmax_calls: int = 0
    token_input_host_refreshes: int = 0
    position_rope_host_refreshes: int = 0
    page_table_host_refreshes: int = 0
    steady_token_input_host_refreshes: int = 0
    steady_position_rope_host_refreshes: int = 0
    steady_page_table_host_refreshes: int = 0
    caller_visible_token_synchronizations: int = 0
    validation_full_logit_synchronizations: int = 0
    device_token_out_submissions: int = 0
    fixed_sampling_state_replays: int = 0
    fixed_greedy_sampling_replays: int = 0
    sampling_state_host_refreshes: int = 0
    decode_output_collections: int = 0
    asynchronous_decode_output_collections: int = 0
    minimal_token_readbacks: int = 0
    token_read_device_shards: int = 0
    token_read_replica_shards_skipped: int = 0
    new_prefill_variants_compiled: int = 0
    decode_trace_releases_for_prefill_compile: int = 0
    decode_trace_releases_for_host_sampling: int = 0

    def to_dict(self):
        return dict(vars(self))


@dataclass(frozen=True)
class _MinimalTokenHostOutput:
    """Host-resident token shards selected before crossing the async boundary."""

    shards: tuple[Any, ...]


def _to_torch_page_table(page_table) -> torch.Tensor:
    if isinstance(page_table, torch.Tensor):
        return page_table.to(dtype=torch.int32, device="cpu")
    if isinstance(page_table, ttnn.Tensor):
        tensors = ttnn.get_device_tensors(page_table)
        source = tensors[0] if tensors else page_table
        return ttnn.to_torch(source).to(dtype=torch.int32, device="cpu")
    raise TypeError(f"page_table must be torch.Tensor or ttnn.Tensor, got {type(page_table).__name__}")


class Generator:
    """Two-level readiness/serving generator with explicit cache ownership."""

    tokenizer: Any

    def __init__(self, model: Model, model_args, *, kv_cache=None, cache_owner: str = "model"):
        if cache_owner not in {"model", "vllm"}:
            raise ValueError(f"cache_owner must be 'model' or 'vllm', got {cache_owner!r}")
        self.model = model
        self.model_args = model_args
        self.mesh_device = model.mesh_device
        self.tokenizer = model_args.tokenizer
        self._kv_cache = model.kv_cache if kv_cache is None else kv_cache
        self.cache_owner = cache_owner
        self._inner = _TTGenerator(
            model=[model],
            model_args=[model_args],
            mesh_device=self.mesh_device,
            tokenizer=self.tokenizer,
        )
        # Resumes must satisfy both SDPA chunks and the sliding ring's retained
        # history. Keep this instance's contract separate from shared defaults.
        self._inner.model_capabilities = {
            **self._inner.model_capabilities,
            "resumed_prefill_token_alignment": PREFILL_CHUNK_ALIGN,
        }
        # A standalone generator owns fixed per-slot cache ranges.  vLLM owns
        # one shared block pool and supplies the scheduler's current table on
        # every call, so creating a private full-context table here would be a
        # latent and physically invalid fallback for the serving path.
        self._page_table = self.allocate_page_table() if cache_owner == "model" else None
        self._dirty_cache = False
        self._last_page_table: torch.Tensor | None = None
        self._last_sampling_mode: str | None = None
        self._decode_started = False
        self._prepared_device_sampling_params: SamplingParams | None = None
        self._compiled_prefill_variants: set[tuple[int, int, int, str]] = set()
        self._lifetime_prefill_variant_compilations = 0
        self._lifetime_decode_trace_releases_for_prefill_compile = 0
        self._lifetime_decode_trace_releases_for_host_sampling = 0
        self._torn_down = False
        self.runtime_evidence_path: Path | None = None
        self.last_generation_metrics: dict[str, Any] = {}
        self.trace_evidence = TraceEvidence()
        self._warmed_before_measurement = False
        self._trace_handles_before_measurement: dict[str, Any] = {}

    @property
    def kv_cache(self):
        return self._kv_cache

    @property
    def already_warmed_up_prefill(self) -> bool:
        """Expose the canonical generator's warmup state to serving wrappers."""

        return bool(self._inner.already_warmed_up_prefill)

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        self._inner.already_warmed_up_prefill = bool(value)

    @property
    def page_table(self):
        return self._page_table

    @property
    def _greedy_sampling_prepared(self) -> bool:
        """Compatibility view for callers that predate generic fixed sampling state."""

        return getattr(self, "_prepared_device_sampling_params", None) == GREEDY

    @_greedy_sampling_prepared.setter
    def _greedy_sampling_prepared(self, value: bool) -> None:
        self._prepared_device_sampling_params = GREEDY if value else None

    def allocate_page_table(self) -> torch.Tensor:
        """Allocate fixed private page ranges for all configured request slots."""

        blocks_per_slot = (self.model_args.physical_kv_context_len + PAGE_SIZE - 1) // PAGE_SIZE
        return torch.arange(
            self.model_args.max_batch_size * blocks_per_slot,
            dtype=torch.int32,
        ).reshape(self.model_args.max_batch_size, blocks_per_slot)

    def _require_private_page_table(self) -> torch.Tensor:
        if self._page_table is None:
            raise RuntimeError("vLLM owns the serving cache; an explicit scheduler-owned page table is required")
        return self._page_table

    def _layer_cache(self, kv_cache):
        cache = self._kv_cache if kv_cache is None else kv_cache
        if len(cache) == 1 and isinstance(cache[0], (list, tuple)) and len(cache[0]) == self.model.n_layers:
            return cache[0]
        return cache

    def _outer_cache(self, kv_cache):
        cache = self._layer_cache(kv_cache)
        return [cache]

    def _validate_prefill(self, tokens, prompt_lens, page_table):
        if tokens.ndim != 2:
            raise ValueError(f"prefill tokens must be [batch, sequence], got {tuple(tokens.shape)}")
        batch, padded_len = tokens.shape
        if len(prompt_lens) != batch:
            raise ValueError(f"prompt_lens has {len(prompt_lens)} entries for batch {batch}")
        if batch > self.model_args.max_batch_size:
            raise ValueError(f"batch {batch} exceeds configured maximum {self.model_args.max_batch_size}")
        for prompt_len in prompt_lens:
            if not 1 <= int(prompt_len) <= min(padded_len, self.model_args.max_context_len):
                raise ValueError(
                    f"prompt length {prompt_len} is outside [1, {min(padded_len, self.model_args.max_context_len)}]"
                )
        if page_table.shape[0] < batch:
            raise ValueError(f"page_table has {page_table.shape[0]} rows for batch {batch}")
        needed_blocks = max((int(length) + PAGE_SIZE - 1) // PAGE_SIZE for length in prompt_lens)
        if page_table.shape[1] < needed_blocks:
            raise ValueError(f"page_table has {page_table.shape[1]} blocks; prompts require {needed_blocks}")

    def _gather_prefill_logits(self, logits) -> torch.Tensor:
        host = logits.cpu(blocking=True)
        shards = ttnn.get_device_tensors(host)
        tp = int(self.mesh_device.shape[1])
        gathered = torch.cat([ttnn.to_torch(shards[index]) for index in range(tp)], dim=-1)
        return gathered[..., : self.model.vocab_size]

    def _decode_trace_stores(self):
        """Return every shared-generator decode trace store without duplicating aliases."""

        stores = []
        for bucket_store in getattr(self._inner, "_bucket_trace_store", {}).values():
            if bucket_store is not None and bucket_store[0] is not None:
                stores.append(bucket_store[0])
        stores.append(self._inner.trace_ids_decode)
        unique = []
        seen = set()
        for store in stores:
            if id(store) not in seen:
                unique.append(store)
                seen.add(id(store))
        return unique

    def _has_live_decode_trace(self) -> bool:
        for store in self._decode_trace_stores():
            for trace_ids in store.values():
                if trace_ids is not None and any(trace_id is not None for trace_id in trace_ids.values()):
                    return True
        return False

    def _release_decode_traces_for_prefill_compile(self, *, host_sampling: bool = False) -> None:
        """Release live decode/sampling captures before a new prefill compile.

        TTNN traces bind allocator addresses that are opaque to subsequent
        allocations.  Compiling a new padded prefill shape while a decode trace
        is live can therefore allocate into trace scratch and be overwritten by
        a later replay.  This synchronized release is a one-time request-boundary
        cost for each prefill variant; warmed token-out replay never enters it.
        """

        ttnn.synchronize_device(self.mesh_device)
        for model in self._inner.model:
            sampling = getattr(model, "sampling", None)
            trace_states = getattr(sampling, "_trace_states", None)
            if trace_states is not None:
                released_sampling_ids = set()
                for slot in trace_states.values():
                    trace_id = slot.get("id")
                    if trace_id is not None and trace_id not in released_sampling_ids:
                        # SamplingGenerator.reset_trace intentionally swallows
                        # release failures.  A lifecycle transition cannot: if
                        # this release fails, clearing its metadata would let a
                        # new prefill compile while the old capture is live.
                        ttnn.release_trace(sampling.mesh_device, trace_id)
                        released_sampling_ids.add(trace_id)
                trace_states.clear()
                sampling._active_trace_bucket = None
            elif sampling is not None and hasattr(sampling, "reset_trace"):
                sampling.reset_trace()

        released = set()
        for store in self._decode_trace_stores():
            for trace_ids in store.values():
                if trace_ids is None:
                    continue
                for model_id, trace_id in trace_ids.items():
                    key = (int(model_id), trace_id)
                    if trace_id is not None and key not in released:
                        ttnn.release_trace(self._inner.model_args[model_id].mesh_device, trace_id)
                        released.add(key)

        bucket_store = getattr(self._inner, "_bucket_trace_store", None)
        if isinstance(bucket_store, dict):
            # Preserve width -> trace-dictionary identity.  The vLLM adapter
            # keeps its active width across a first-time prefill compilation;
            # dropping this map would let the recaptured active trace become
            # orphaned when a later width switch restores a new empty tuple.
            for trace_ids, trace_inputs, trace_outputs in bucket_store.values():
                trace_ids.clear()
                trace_inputs.clear()
                trace_outputs.clear()
        self._inner.trace_ids_decode.clear()
        self._inner.trace_inputs_decode.clear()
        self._inner.trace_output_decode.clear()
        self._inner.mode = None
        self._inner.prev_page_table = None
        self._inner._prev_on_device_sampling = None
        self._inner._slots_prefilled_since_decode = set()
        self._inner._defer_trace_recording = False
        self._inner._pending_decode_trace = None
        self._inner._prepared_decode_traces.clear()
        self._prepared_device_sampling_params = None
        if host_sampling:
            self.trace_evidence.decode_trace_releases_for_host_sampling += 1
            self._lifetime_decode_trace_releases_for_host_sampling += 1
        else:
            self.trace_evidence.decode_trace_releases_for_prefill_compile += 1
            self._lifetime_decode_trace_releases_for_prefill_compile += 1

    def release_decode_traces_for_host_sampling(self) -> bool:
        """Quiesce traces before the optional eager full-logits compatibility path."""

        if not self._has_live_decode_trace():
            return False
        self._release_decode_traces_for_prefill_compile(host_sampling=True)
        return True

    @staticmethod
    def _prefill_program_signature(
        prompt_len: int, path: str, start_pos: int = 0, fill_start: int = 0, ring_cold: bool = False
    ) -> tuple[int, int, int, str]:
        """Return every prompt-dependent prefill program bucket.

        Padding alone is insufficient: sequential serving trims K/V fill to
        the prompt's page-rounded length and slices the last hidden-state tile.
        Both operations have shape/offset-specific programs.  Treating a short
        prompt as the warmed 128-token variant would compile those programs
        while decode traces are live, so a later replay could overwrite their
        outputs.
        """

        prompt_len = int(prompt_len)
        start_pos = int(start_pos or 0)
        if prompt_len < 1:
            raise ValueError(f"prefill prompt length must be positive, got {prompt_len}")
        if not 0 <= start_pos < prompt_len:
            raise ValueError(f"prefill start {start_pos} must be within [0, {prompt_len})")
        chunk_len = prompt_len - start_pos
        padded_length = get_padded_prefill_len(chunk_len)
        rounded_span = prompt_len if start_pos else chunk_len
        page_rounded_length = ((rounded_span + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        last_token_tile_start = ((chunk_len - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if start_pos:
            path = f"{path}:chunk"
            if ring_cold:
                path = f"{path}:cold"
        fill_start = int(fill_start or 0)
        if fill_start > start_pos:
            path = f"{path}:fill{fill_start - start_pos}"
        return padded_length, page_rounded_length, last_token_tile_start, path

    @staticmethod
    def _per_prompt_ints(values, prompt_lens, name: str) -> list[int] | None:
        if values is None:
            return None
        ints = [int(value) for value in torch.as_tensor(values).reshape(-1).tolist()]
        if len(ints) != len(prompt_lens):
            raise ValueError(f"{name} has {len(ints)} entries for {len(prompt_lens)} prompts")
        return ints if any(ints) else None

    def _batched_prefill_shape(self, prompt_lens: List[int]) -> tuple[int, int] | None:
        """Return ``(padded_batch, padded_len)`` when the shared generator will batch these prompts.

        Mirrors ``Generator._prefill_forward_text_impl``: more than one prompt,
        one shared padded length, batched prefill enabled, compact rows, and
        the device batch within the supported sizes and token limit.
        """
        from models.tt_transformers.tt.generator import MAX_BATCHED_PREFILL_SEQ_LEN, SUPPORTED_PREFILL_BATCH_SIZES

        model_args = getattr(self, "model_args", None)
        if len(prompt_lens) < 2 or getattr(model_args, "disable_batched_prefill", True):
            return None
        padded = {get_padded_prefill_len(int(length)) for length in prompt_lens}
        if len(padded) != 1:
            return None
        padded_len = padded.pop()
        if padded_len > self.model_args.max_prefill_chunk_size:
            return None
        cap = getattr(self.model_args, "batched_prefill_max_tokens_per_user", None)
        if cap is not None and padded_len > int(cap):
            return None
        padded_batch = next((b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= len(prompt_lens)), None)
        if padded_batch is None or padded_batch > self.model_args.max_batch_size:
            return None
        if padded_batch * padded_len >= MAX_BATCHED_PREFILL_SEQ_LEN:
            return None
        return padded_batch, padded_len

    def _prepare_prefill_variants(
        self, prompt_lens: List[int], *, path: str, start_pos=None, fill_start_pos=None, ring_cold=None
    ) -> set[tuple]:
        """Make first-time prefill program compilation safe with existing traces."""

        starts = [0] * len(prompt_lens) if start_pos is None else [int(value) for value in start_pos]
        fills = [0] * len(prompt_lens) if fill_start_pos is None else [int(value) for value in fill_start_pos]
        colds = [False] * len(prompt_lens) if ring_cold is None else [bool(value) for value in ring_cold]
        plain = not any(starts) and not any(fills)
        batched = self._batched_prefill_shape(prompt_lens) if path == "device_sampling" and plain else None
        if batched is not None:
            # Same 4-field shape as the per-prompt signature so the variant set
            # stays sortable in the capability report: (padded length, negative
            # device batch as the batched marker, -1, path).
            variants = {(batched[1], -batched[0], -1, path)}
        else:
            variants = {
                self._prefill_program_signature(prompt_len, path, start, fill, cold)
                for prompt_len, start, fill, cold in zip(prompt_lens, starts, fills, colds)
            }
        unseen = variants.difference(self._compiled_prefill_variants)
        if unseen and self._has_live_decode_trace():
            self._release_decode_traces_for_prefill_compile()
        return unseen

    def _record_compiled_prefill_variants(self, variants: set[tuple[int, int, int, str]]) -> None:
        self._compiled_prefill_variants.update(variants)
        self.trace_evidence.new_prefill_variants_compiled += len(variants)
        self._lifetime_prefill_variant_compilations += len(variants)

    def _prefill_one(
        self,
        token_ids,
        page_table_row,
        layer_cache,
        *,
        return_all_logits,
        page_tables_per_layer=None,
    ):
        logical_len = int(token_ids.shape[-1])
        # Match the shared prefill graph used by device sampling and prefix resumes.
        padded_tokens = torch.nn.functional.pad(token_ids, (0, get_padded_prefill_len(logical_len) - logical_len))
        inputs = self.model.prepare_inputs_prefill(
            padded_tokens,
            page_table=page_table_row,
            batch_size=1,
            user_id=0,
        )
        embedded, rope_global, rope_local, tt_page_table, *_ = inputs
        logits = self.model.ttnn_prefill_forward(
            embedded,
            rot_mats_global=rope_global,
            rot_mats_local=rope_local,
            user_id=0,
            page_table=tt_page_table,
            get_last_token=-1 if return_all_logits else ((logical_len - 1) // 32) * 32,
            kv_cache=layer_cache,
            batch_size=1,
            page_tables_per_layer=page_tables_per_layer,
        )
        gathered = self._gather_prefill_logits(logits)
        self.trace_evidence.full_logits_readbacks += 1
        self.trace_evidence.validation_full_logit_synchronizations += 1
        if return_all_logits:
            return gathered[0, 0, :logical_len]
        row = (logical_len - 1) % 32
        return gathered[0, 0, row : row + 1]

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        sampling_params: SamplingParams | None = None,
        empty_slots: List[int] | None = None,
        page_tables_per_layer=None,
        enable_trace: bool = False,
        start_pos=None,
        fill_start_pos=None,
        ring_cold=None,
        **kwargs: Any,
    ):
        """Fill explicit paged cache state for mixed, non-aligned prompts.

        ``start_pos`` (one entry per prompt) is the position the attention
        resumes at: a vLLM chunked-prefill continuation or a cached prefix,
        aligned by the caller. ``fill_start_pos`` is the first position whose
        full-attention K/V is written (earlier positions are cached blocks), and
        ``ring_cold`` says the slot's sliding rings hold nothing for the prompt.
        A resumed prompt sampled on the host returns its last-token logits
        through the shared generator; all-position logits are not served for it.
        """

        del kwargs
        page_table_host = _to_torch_page_table(page_table)
        self._validate_prefill(tokens, prompt_lens, page_table_host)
        starts = self._per_prompt_ints(start_pos, prompt_lens, "start_pos")
        fills = self._per_prompt_ints(fill_start_pos, prompt_lens, "fill_start_pos")
        colds = None if ring_cold is None else [bool(value) for value in ring_cold]
        if colds is not None and len(colds) != len(prompt_lens):
            raise ValueError(f"ring_cold has {len(colds)} entries for {len(prompt_lens)} prompts")
        resumed = starts is not None or fills is not None
        if sampling_params is not None:
            if return_all_logits:
                raise ValueError("device-sampled prefill cannot return all logits")
            slots = list(range(tokens.shape[0])) if empty_slots is None else [int(slot) for slot in empty_slots]
            if len(slots) != tokens.shape[0]:
                raise ValueError(f"empty_slots has {len(slots)} entries for batch {tokens.shape[0]}")
            path = "device_sampling"
            new_variants = self._prepare_prefill_variants(
                prompt_lens, path=path, start_pos=starts, fill_start_pos=fills, ring_cold=colds
            )
            self.model._prefill_row_lengths = [int(length) for length in prompt_lens]
            self.model._prefill_row_resume = None
            if resumed:
                rows = len(prompt_lens)
                self.model._prefill_row_resume = list(
                    zip(starts or [0] * rows, fills or [0] * rows, colds or [False] * rows)
                )
                self.model._prefill_row_cursor = 0
            try:
                result = self._inner.prefill_forward_text(
                    tokens,
                    page_table=page_table_host,
                    kv_cache=self._outer_cache(kv_cache),
                    prompt_lens=[int(length) for length in prompt_lens],
                    empty_slots=slots,
                    enable_trace=enable_trace,
                    sampling_params=sampling_params,
                    warmup_prefill=False,
                    page_tables_per_layer=page_tables_per_layer,
                    **({"start_pos": starts} if starts is not None else {}),
                )
            finally:
                self.model._prefill_row_lengths = None
                self.model._prefill_row_resume = None
            self._dirty_cache = True
            self._inner.mode = None
            self._record_compiled_prefill_variants(new_variants)
            self.trace_evidence.sampled_token_readbacks += 1
            self.trace_evidence.caller_visible_token_synchronizations += 1
            return result
        if resumed:
            if return_all_logits:
                raise ValueError(
                    "all-position logits are not served for resumed prompts (start_pos or fill_start_pos > 0)"
                )
            slots = list(range(tokens.shape[0])) if empty_slots is None else [int(slot) for slot in empty_slots]
            if len(slots) != tokens.shape[0]:
                raise ValueError(f"empty_slots has {len(slots)} entries for batch {tokens.shape[0]}")
            new_variants = self._prepare_prefill_variants(
                prompt_lens, path="host_last_logits", start_pos=starts, fill_start_pos=fills, ring_cold=colds
            )
            rows = len(prompt_lens)
            self.model._prefill_row_lengths = [int(length) for length in prompt_lens]
            self.model._prefill_row_resume = list(
                zip(starts or [0] * rows, fills or [0] * rows, colds or [False] * rows)
            )
            self.model._prefill_row_cursor = 0
            try:
                logits = self._inner.prefill_forward_text(
                    tokens,
                    page_table=page_table_host,
                    kv_cache=self._outer_cache(kv_cache),
                    prompt_lens=[int(length) for length in prompt_lens],
                    empty_slots=slots,
                    enable_trace=False,
                    sampling_params=None,
                    warmup_prefill=False,
                    page_tables_per_layer=page_tables_per_layer,
                    start_pos=starts if starts is not None else [0] * rows,
                )
            finally:
                self.model._prefill_row_lengths = None
                self.model._prefill_row_resume = None
            self._dirty_cache = True
            self._inner.mode = None
            self._record_compiled_prefill_variants(new_variants)
            self.trace_evidence.full_logits_readbacks += 1
            self.trace_evidence.validation_full_logit_synchronizations += 1
            return logits.reshape(rows, 1, -1)
        path = "host_all_logits" if return_all_logits else "host_last_logits"
        new_variants = self._prepare_prefill_variants(prompt_lens, path=path)
        layer_cache = self._layer_cache(kv_cache)
        block_sizes = None if page_tables_per_layer is None else [cache[0].shape[2] for cache in layer_cache]
        rows = []
        for user, prompt_len in enumerate(prompt_lens):
            prompt_len = int(prompt_len)
            user_page_tables = None
            if page_tables_per_layer is not None:
                user_page_tables = [
                    table[user : user + 1, : (prompt_len + block_size - 1) // block_size]
                    for table, block_size in zip(page_tables_per_layer, block_sizes)
                ]
            rows.append(
                self._prefill_one(
                    tokens[user : user + 1, :prompt_len],
                    page_table_host[user : user + 1],
                    layer_cache,
                    return_all_logits=return_all_logits,
                    page_tables_per_layer=user_page_tables,
                )
            )
        self._dirty_cache = True
        self._inner.mode = None
        self._record_compiled_prefill_variants(new_variants)
        if return_all_logits:
            max_len = max(int(length) for length in prompt_lens)
            output = torch.zeros(len(rows), max_len, self.model.vocab_size, dtype=rows[0].dtype)
            for user, row in enumerate(rows):
                output[user, : row.shape[0]] = row
            return output
        return torch.stack(rows, dim=0)

    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        """Run standalone all-position prefill with generator-owned cache state."""

        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one token")
        self.reset()
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        return self.prefill_forward(
            prompt,
            page_table=self._require_private_page_table()[:1],
            kv_cache=self._kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def _record_decode_staging(self, *, page_table, sampling_mode, enable_trace, reset_batch):
        current = page_table.clone()
        changed = self._last_page_table is None or not torch.equal(self._last_page_table, current)
        mode_changed = self._last_sampling_mode is not None and self._last_sampling_mode != sampling_mode
        was_started = self._decode_started
        full_refresh = sampling_mode == "host" or reset_batch or not was_started or mode_changed
        self.trace_evidence.decode_calls += 1
        if full_refresh:
            self.trace_evidence.full_input_refreshes += 1
            self.trace_evidence.token_input_host_refreshes += 1
            self.trace_evidence.position_rope_host_refreshes += 1
            self.trace_evidence.page_table_host_refreshes += 1
            if was_started:
                self.trace_evidence.steady_token_input_host_refreshes += 1
                self.trace_evidence.steady_position_rope_host_refreshes += 1
                self.trace_evidence.steady_page_table_host_refreshes += 1
        elif changed:
            self.trace_evidence.page_table_only_refreshes += 1
            self.trace_evidence.page_table_host_refreshes += 1
            self.trace_evidence.steady_page_table_host_refreshes += 1
        else:
            self.trace_evidence.page_table_reuses += 1
        self._last_page_table = current
        self._last_sampling_mode = sampling_mode
        self._decode_started = True

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table,
        kv_cache,
        enable_trace: bool = True,
        sampling_mode: str = "device",
        sampling_params: SamplingParams | None = None,
        reset_batch: bool = False,
        force_host_tokens: bool = False,
        reload_inputs: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap=None,
        skip_trace_precompile: bool = False,
        read_from_device: bool = True,
        reuse_sampling_state: bool = False,
        reuse_greedy_sampling_state: bool = False,
        **kwargs: Any,
    ):
        """Decode active and inactive rows through the explicit serving state.

        ``read_from_device=False`` is the serving/performance split boundary:
        it returns the sampler's device output without synchronizing.  After
        one canonical device-sampling call has initialized persistent state,
        ``reuse_sampling_state=True`` replays both traces without copying
        unchanged k/p/temperature state from the host. Explicit request seeds
        are deliberately excluded because they require a per-token seed update.
        ``reuse_greedy_sampling_state`` remains as a compatibility alias.
        """

        del kwargs
        if sampling_mode not in {"device", "host"}:
            raise ValueError(f"sampling_mode must be 'device' or 'host', got {sampling_mode!r}")
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError(f"decode tokens must be [batch, 1], got {tuple(tokens.shape)}")
        if start_pos.ndim != 1 or start_pos.shape[0] != tokens.shape[0]:
            raise ValueError(f"start_pos must be [batch], got {tuple(start_pos.shape)}")
        if tokens.shape[0] > self.model_args.max_batch_size:
            raise ValueError(f"batch {tokens.shape[0]} exceeds configured maximum {self.model_args.max_batch_size}")
        active = start_pos >= 0
        if torch.any(start_pos[active] >= self.model_args.max_context_len):
            raise ValueError(f"decode position exceeds supported context {self.model_args.max_context_len}")
        page_table_host = _to_torch_page_table(page_table)
        if page_table_host.shape[0] < tokens.shape[0]:
            raise ValueError("page_table does not cover every decode row")
        reuse_fixed_sampling = reuse_sampling_state or reuse_greedy_sampling_state
        if reuse_fixed_sampling and sampling_mode != "device":
            raise ValueError("fixed sampling-state reuse is valid only for device sampling")
        if reuse_fixed_sampling and not enable_trace:
            raise ValueError("fixed sampling-state reuse requires enable_trace=True")
        if reuse_fixed_sampling and (reset_batch or force_host_tokens):
            raise ValueError("fixed sampling replay cannot reset the batch or force host tokens")
        prepared_sampling_params = getattr(self, "_prepared_device_sampling_params", None)
        if reuse_fixed_sampling and prepared_sampling_params is None:
            raise RuntimeError("initialize unseeded device sampling once before reusing its persistent state")
        if reuse_greedy_sampling_state and prepared_sampling_params != GREEDY:
            raise RuntimeError("reuse_greedy_sampling_state requires canonical greedy persistent state")
        if reuse_fixed_sampling and self._sampling_has_active_request_seed():
            raise RuntimeError("fixed traced sampling-state reuse does not support explicit request seeds")
        self._record_decode_staging(
            page_table=page_table_host,
            sampling_mode=sampling_mode,
            enable_trace=enable_trace,
            reset_batch=reset_batch,
        )
        if force_host_tokens:
            # The shared generator normally preserves an async-ahead token that
            # device sampling wrote into the persistent decode input.  Teacher
            # forcing is the one case where the caller's token is authoritative.
            # Mark these fixed slots as freshly supplied so its reset path copies
            # the host token/position rather than silently continuing free-run.
            if not hasattr(self._inner, "_slots_prefilled_since_decode"):
                self._inner._slots_prefilled_since_decode = set()
            self._inner._slots_prefilled_since_decode.update(range(tokens.shape[0]))
        if reuse_fixed_sampling:
            result = self._inner.decode_forward(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table_host,
                kv_cache=self._outer_cache(kv_cache),
                enable_trace=enable_trace,
                read_from_device=False,
                sampling_params=None,
                defer_device_sampling=True,
                reset_batch=False,
                slot_remap=slot_remap,
                skip_trace_precompile=skip_trace_precompile,
                reload_inputs=reload_inputs,
            )
            if enable_trace:
                self.trace_evidence.model_execute_submissions += 1
                self.trace_evidence.trace_replays += 1
            result = self._replay_prepared_sampling(result, enable_trace=enable_trace)
            if enable_trace:
                self.trace_evidence.sampling_execute_submissions += 1
            self.trace_evidence.fixed_sampling_state_replays += 1
            if prepared_sampling_params == GREEDY:
                self.trace_evidence.fixed_greedy_sampling_replays += 1
        else:
            effective_sampling_params = (sampling_params or GREEDY) if sampling_mode == "device" else None
            result = self._inner.decode_forward(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table_host,
                kv_cache=self._outer_cache(kv_cache),
                enable_trace=enable_trace,
                # The shared generator formats with its configured maximum
                # width. Host logits must use this submission's bucket width.
                read_from_device=read_from_device and sampling_mode == "device",
                sampling_params=effective_sampling_params,
                reset_batch=reset_batch,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                skip_trace_precompile=skip_trace_precompile,
                reload_inputs=reload_inputs,
            )
            if enable_trace:
                self.trace_evidence.model_execute_submissions += 1
                self.trace_evidence.trace_replays += 1
                # Explicit request seeds update the device seed tensor every
                # token and intentionally run the sampler eagerly; only the
                # unseeded fixed-state path replays a sampling trace.
                if sampling_mode == "device" and not self._sampling_has_active_request_seed():
                    self.trace_evidence.sampling_execute_submissions += 1
            if sampling_mode == "device":
                self.trace_evidence.sampling_state_host_refreshes += 1
                self._prepared_device_sampling_params = (
                    None if self._sampling_has_active_request_seed() else effective_sampling_params
                )
        self._dirty_cache = True
        if sampling_mode == "device":
            self.trace_evidence.device_token_out_submissions += 1
            if not read_from_device:
                return result
            self.trace_evidence.sampled_token_readbacks += 1
            self.trace_evidence.caller_visible_token_synchronizations += 1
            if isinstance(result, tuple):
                result = result[0]
            return result.reshape(-1)[: tokens.shape[0]].to(torch.int64)
        if not read_from_device:
            return result
        self.trace_evidence.full_logits_readbacks += 1
        self.trace_evidence.validation_full_logit_synchronizations += 1
        host_output = self._inner.read_decode_output(result)
        logits, _ = self.process_decode_output_host(host_output, is_tokens=False)
        return logits[:, 0, :]

    def _sampling_has_active_request_seed(self) -> bool:
        for inner_model in self._inner.model:
            sampling = getattr(inner_model, "sampling", None)
            seed_manager = getattr(sampling, "seed_manager", None)
            if seed_manager is not None and seed_manager.has_active_request_seed():
                return True
        return False

    def _replay_prepared_sampling(self, tt_logits, *, enable_trace: bool):
        """Replay fixed unseeded sampling without host sampling-state mutation."""

        sampled_outputs = []
        for model_id, logits in enumerate(tt_logits):
            sampling = self._inner.model[model_id].sampling
            if sampling is None:
                raise RuntimeError("fixed sampling replay requires an on-device sampling module")
            if isinstance(logits, tuple):
                logits = logits[0]
            trace_inputs = self._inner.trace_inputs_decode[True][model_id]
            tt_out_tok = self._inner._decode_token_feedback_buffer(self._inner.model[model_id], trace_inputs)
            sampled_outputs.append(
                sampling.sample(
                    logits=logits,
                    tt_out_tok=tt_out_tok,
                    enable_trace=enable_trace,
                )
            )
        return sampled_outputs

    def _replay_prepared_greedy_sampling(self, tt_logits, *, enable_trace: bool):
        """Compatibility wrapper for the original split-greedy helper."""

        if not self._greedy_sampling_prepared:
            raise RuntimeError("canonical greedy sampling state is not prepared")
        return self._replay_prepared_sampling(tt_logits, enable_trace=enable_trace)

    @staticmethod
    def _select_minimal_token_shards(device_shards, *, users_row_sharded: bool, mesh_cols: int):
        """Select one sampled-token replica per distinct data-parallel row."""

        if not device_shards:
            raise RuntimeError("sampled token output has no device shards")
        if users_row_sharded:
            if mesh_cols <= 0 or len(device_shards) % mesh_cols:
                raise RuntimeError(
                    f"cannot select token rows from {len(device_shards)} shards with mesh_cols={mesh_cols}"
                )
            return device_shards[::mesh_cols]
        return device_shards[:1]

    def _minimal_token_output(self, device_output):
        """Return the token tensor when no log-prob payload needs collection."""

        if getattr(self._inner, "data_parallel", 1) != 1 or len(device_output) != 1:
            return None
        output = device_output[0]
        if isinstance(output, tuple):
            if len(output) != 2 or output[1] is not None:
                return None
            output = output[0]
        return output if isinstance(output, ttnn.Tensor) else None

    def _read_minimal_token_output(self, token_output, *, async_read: bool):
        """Read one token replica per mesh row instead of every TP replica."""

        device_shards = ttnn.get_device_tensors(token_output)
        inner_model = self._inner.model[0]
        mesh_cols = int(inner_model.mesh_device.shape[1])
        selected = self._select_minimal_token_shards(
            device_shards,
            users_row_sharded=bool(getattr(inner_model, "users_row_sharded", False)),
            mesh_cols=mesh_cols,
        )
        host_shards = tuple(shard.cpu(blocking=not async_read) for shard in selected)
        self.trace_evidence.minimal_token_readbacks += 1
        self.trace_evidence.token_read_device_shards += len(host_shards)
        self.trace_evidence.token_read_replica_shards_skipped += len(device_shards) - len(host_shards)
        host_output = _MinimalTokenHostOutput(host_shards)
        if async_read:
            return host_output, [ttnn.record_event(inner_model.mesh_device, 0)]
        return host_output

    def read_decode_output(self, device_output, *, async_read: bool = False, is_tokens: bool = True):
        """Collect an explicitly submitted device token output.

        Async collection returns the shared runtime's ``(host_outputs,
        events)`` pair.  Synchronous collection waits and converts the sampled
        token IDs to a flat int64 tensor.
        """

        token_output = self._minimal_token_output(device_output) if is_tokens else None
        if token_output is not None:
            host_output = self._read_minimal_token_output(token_output, async_read=async_read)
        else:
            host_output = self._inner.read_decode_output(device_output, async_read=async_read)
        if async_read:
            self.trace_evidence.asynchronous_decode_output_collections += 1
            return host_output
        self.trace_evidence.decode_output_collections += 1
        self.trace_evidence.sampled_token_readbacks += 1
        self.trace_evidence.caller_visible_token_synchronizations += 1
        result = self.process_decode_output_host(host_output, is_tokens=is_tokens)
        if isinstance(result, tuple):
            result = result[0]
        return result.reshape(-1).to(torch.int64) if is_tokens else result

    def process_decode_output_host(
        self,
        host_output,
        *,
        is_tokens: bool = False,
    ):
        """Format an already-submitted decode read without issuing device work."""

        if isinstance(host_output, _MinimalTokenHostOutput):
            if not is_tokens:
                raise RuntimeError("minimal sampled-token output cannot be processed as logits")
            return torch.cat([ttnn.to_torch(shard).reshape(-1) for shard in host_output.shards], dim=0).to(torch.int64)
        return self._inner.process_decode_output_host(host_output, is_tokens=is_tokens)

    def warmup_model_prefill(self, *, kv_cache, enable_trace: bool, can_sample_on_device: bool):
        """Delegate vLLM's prefill warmup to the canonical generator."""

        result = self._inner.warmup_model_prefill(
            kv_cache=self._outer_cache(kv_cache),
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
        )
        paths = {"host_last_logits"}
        if can_sample_on_device:
            paths.add("device_sampling")
        variants = {
            self._prefill_program_signature(seq_len, path)
            for seq_len in self.model_args.get_warmup_prefill_supported_seq_lens()
            for path in paths
        }
        self._record_compiled_prefill_variants(variants.difference(self._compiled_prefill_variants))
        return result

    def warmup_model_decode(
        self,
        *,
        kv_cache,
        enable_trace: bool,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        skip_trace_precompile: bool = False,
    ):
        """Compile both routes eagerly, then warm only device-sampling traces."""

        outer_cache = self._outer_cache(kv_cache)
        if not enable_trace:
            return self._inner.warmup_model_decode(
                kv_cache=outer_cache,
                enable_trace=False,
                max_batch_size=max_batch_size,
                num_blocks=num_blocks,
                can_sample_on_device=can_sample_on_device,
                read_from_device=False,
                skip_trace_precompile=skip_trace_precompile,
            )
        # Host logits always run eagerly after releasing the resident device
        # traces. Capturing a second full-model host graph would waste its trace
        # region and contradict that lifecycle. Reuse the shared sampling sweep
        # and input construction, but select only the routes this model traces.
        tokens, positions, table = self._inner._create_decode_warmup_inputs(max_batch_size, num_blocks)
        for params in self._inner._create_sampling_params(can_sample_on_device, max_batch_size):
            if params is not None:
                self._inner.decode_forward(
                    tokens=tokens,
                    start_pos=positions,
                    page_table=table,
                    kv_cache=outer_cache,
                    enable_trace=True,
                    read_from_device=False,
                    sampling_params=params,
                    reset_batch=True,
                    prompt_tokens=tokens,
                    skip_trace_precompile=skip_trace_precompile,
                )

    def prepare_model_decode_trace(
        self,
        *,
        kv_cache,
        max_batch_size: int,
        num_blocks: int,
        on_device_sampling: bool = True,
    ):
        """Compile and allocate one decode signature before any trace is live."""

        tokens = torch.zeros(max_batch_size, 1, dtype=torch.int32)
        # Trace capture executes the graph once.  Use the paged-cache inactive
        # sentinel so a later recapture cannot overwrite serving KV (notably
        # logical position zero) while another request cohort is resident.
        # Replay refreshes these persistent positions with the scheduler's
        # authoritative values before executing the captured trace.
        start_pos = torch.full((max_batch_size,), -1, dtype=torch.int32)
        page_table = torch.zeros(max_batch_size, num_blocks, dtype=torch.int32)
        return self._inner._prepare_decode_trace_text(
            torch.chunk(tokens, self._inner.data_parallel, 0),
            torch.chunk(start_pos, self._inner.data_parallel, 0),
            page_table=torch.chunk(page_table, self._inner.data_parallel, 0),
            kv_cache=self._outer_cache(kv_cache),
            on_device_sampling=on_device_sampling,
            skip_precompile=False,
        )

    def capture_prepared_model_decode_trace(self, prepared):
        """Capture a decode bucket whose programs and persistent inputs exist."""

        trace_ids, trace_output, *trace_inputs = self._inner._record_decode_trace_text(prepared)
        on_device_sampling = prepared["on_device_sampling"]
        self._inner.trace_ids_decode[on_device_sampling] = trace_ids
        self._inner.trace_inputs_decode[on_device_sampling] = trace_inputs
        self._inner.trace_output_decode[on_device_sampling] = trace_output
        return trace_output

    def run_device_token_out(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        sampling_params: SamplingParams = GREEDY,
        page_table: torch.Tensor | None = None,
        enable_trace: bool = True,
    ) -> dict[str, Any]:
        """Run an unseeded device-sampling loop with no per-token readback.

        TTFT deliberately includes the one caller-visible prefill token.  The
        decode interval queues every model and sampling replay, then collects
        only the final token.  This path cannot stop early on EOS because doing
        so would require a per-token host boundary.
        """

        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one token")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if len(prompt_token_ids) + max_new_tokens > self.model_args.max_context_len:
            raise ValueError("prompt plus requested generation exceeds the configured context")
        if not enable_trace:
            raise ValueError("token-out replay requires enable_trace=True")
        if sampling_params.seed is not None:
            raise ValueError("split token-out replay does not support explicit request seeds")
        self.reset()
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        pages = self._require_private_page_table()[:1] if page_table is None else _to_torch_page_table(page_table)
        generation_start = time.perf_counter()
        first_token = self._device_prefill_sample(prompt, pages, sampling_params=sampling_params)
        first_token_time = time.perf_counter()

        decode_tokens = max_new_tokens - 1
        final_token = first_token
        decode_start = time.perf_counter()
        if decode_tokens:
            device_output = self.decode_forward(
                torch.tensor([[first_token]], dtype=torch.long),
                torch.tensor([len(prompt_token_ids)], dtype=torch.int64),
                page_table=pages,
                kv_cache=self._kv_cache,
                enable_trace=True,
                sampling_mode="device",
                sampling_params=sampling_params,
                reset_batch=True,
                prompt_tokens=prompt,
                read_from_device=False,
            )
            stale_token = torch.tensor([[first_token]], dtype=torch.long)
            stale_position = torch.tensor([len(prompt_token_ids) + 1], dtype=torch.int64)
            for _ in range(1, decode_tokens):
                device_output = self.decode_forward(
                    stale_token,
                    stale_position,
                    page_table=pages,
                    kv_cache=self._kv_cache,
                    enable_trace=True,
                    sampling_mode="device",
                    reuse_sampling_state=True,
                    read_from_device=False,
                )
                stale_position += 1
            final_token = int(self.read_decode_output(device_output)[0])
        generation_end = time.perf_counter()
        decode_seconds = generation_end - decode_start
        metrics = {
            "sampling_mode": "device_split_unseeded_no_per_token_readback",
            "enable_trace": True,
            "prompt_tokens": len(prompt_token_ids),
            "output_tokens": max_new_tokens,
            "ttft_seconds": first_token_time - generation_start,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second_per_user": (decode_tokens / decode_seconds if decode_seconds else None),
            "total_seconds": generation_end - generation_start,
            "first_token": first_token,
            "final_token": final_token,
        }
        self.last_generation_metrics = metrics
        return dict(metrics)

    def run_greedy_token_out(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        page_table: torch.Tensor | None = None,
        enable_trace: bool = True,
    ) -> dict[str, Any]:
        """Compatibility wrapper for canonical greedy split token-out."""

        metrics = self.run_device_token_out(
            prompt_token_ids,
            max_new_tokens,
            sampling_params=GREEDY,
            page_table=page_table,
            enable_trace=enable_trace,
        )
        metrics["sampling_mode"] = "device_split_greedy_no_per_token_readback"
        self.last_generation_metrics = dict(metrics)
        return metrics

    def _device_prefill_sample(
        self,
        prompt: torch.Tensor,
        page_table: torch.Tensor,
        *,
        sampling_params: SamplingParams,
    ) -> int:
        result = self.prefill_forward(
            prompt,
            page_table=page_table,
            kv_cache=self._kv_cache,
            prompt_lens=[prompt.shape[1]],
            empty_slots=[0],
            enable_trace=False,
            sampling_params=sampling_params,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        return int(torch.as_tensor(tokens).reshape(-1)[0])

    def _eos_token_ids(self) -> set[int]:
        """Return the exact checkpoint generation stop set, with HF fallbacks."""

        values = getattr(getattr(self.model_args, "generation_config", None), "eos_token_id", None)
        if values is None:
            values = getattr(self.model.hf_config, "eos_token_id", None)
        if values is None:
            values = getattr(self.tokenizer, "eos_token_id", None)
        if values is None:
            return set()
        if isinstance(values, (list, tuple, set)):
            return {int(value) for value in values}
        return {int(values)}

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[Callable[[int, int], int]] = None,
        enable_trace: bool = True,
        sampling_mode: str = "device",
        sampling_params: SamplingParams | None = None,
        stop_on_eos: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        """Generate predictions; teacher forcing refreshes only when requested."""

        del kwargs
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one token")
        if len(prompt_token_ids) > self.model_args.max_context_len:
            raise ValueError(
                f"prompt length {len(prompt_token_ids)} exceeds supported context {self.model_args.max_context_len}"
            )
        if max_new_tokens < 0 or len(prompt_token_ids) + max_new_tokens > self.model_args.max_context_len:
            raise ValueError("prompt plus requested generation exceeds the configured context")
        if max_new_tokens == 0:
            return []
        if sampling_mode not in {"device", "host"}:
            raise ValueError(f"sampling_mode must be 'device' or 'host', got {sampling_mode!r}")
        if next_input is None and sampling_mode == "device" and not enable_trace:
            raise ValueError("optimized token-out generation requires enable_trace=True")

        self.reset()
        effective_sampling_params = sampling_params or GREEDY
        generation_start = time.perf_counter()
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        page_table = self._require_private_page_table()[:1]
        if sampling_mode == "device":
            predicted = self._device_prefill_sample(
                prompt,
                page_table,
                sampling_params=effective_sampling_params,
            )
        else:
            logits = self.prefill_forward(
                prompt,
                page_table=page_table,
                kv_cache=self._kv_cache,
                prompt_lens=[prompt.shape[1]],
            )
            predicted = int(torch.argmax(logits[0, 0]).item())
            self.trace_evidence.host_argmax_calls += 1
        first_token_time = time.perf_counter()

        predictions = [predicted]
        eos_ids = self._eos_token_ids()
        if stop_on_eos and predicted in eos_ids:
            return predictions

        feed = int(next_input(0, predicted)) if next_input is not None else predicted
        start_pos = torch.tensor([len(prompt_token_ids)], dtype=torch.int64)
        decode_start = time.perf_counter()
        for step in range(1, max_new_tokens):
            teacher_forced = next_input is not None
            result = self.decode_forward(
                torch.tensor([[feed]], dtype=torch.long),
                start_pos,
                page_table=page_table,
                kv_cache=self._kv_cache,
                enable_trace=enable_trace,
                sampling_mode=sampling_mode,
                sampling_params=effective_sampling_params,
                reset_batch=(step == 1 or teacher_forced),
                force_host_tokens=teacher_forced,
                prompt_tokens=prompt if step == 1 else None,
            )
            if sampling_mode == "device":
                predicted = int(result[0])
            else:
                predicted = int(torch.argmax(result[0]).item())
                self.trace_evidence.host_argmax_calls += 1
            predictions.append(predicted)
            if stop_on_eos and predicted in eos_ids:
                break
            if teacher_forced:
                feed = int(next_input(step, predicted))
                self.trace_evidence.forced_token_refreshes += 1
            else:
                # Deliberately leave ``feed`` stale.  The sampling trace wrote the
                # authoritative token into the persistent device input, and the
                # next replay does not copy this Python value.
                feed = predicted if sampling_mode == "host" else feed
            start_pos += 1
        generation_end = time.perf_counter()
        decode_tokens = max(0, len(predictions) - 1)
        decode_seconds = max(0.0, generation_end - decode_start)
        self.last_generation_metrics = {
            "sampling_mode": sampling_mode,
            "teacher_forcing": next_input is not None,
            "enable_trace": enable_trace,
            "prompt_tokens": len(prompt_token_ids),
            "output_tokens": len(predictions),
            "ttft_seconds": first_token_time - generation_start,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second_per_user": (decode_tokens / decode_seconds if decode_seconds else None),
            "total_seconds": generation_end - generation_start,
        }
        self._write_runtime_evidence()
        return predictions

    def _trace_handle_evidence(self) -> dict[str, Any]:
        model_trace_store = getattr(self._inner, "trace_ids_decode", {})
        model_trace_ids = model_trace_store.get(True, {}) if hasattr(model_trace_store, "get") else {}
        sampling = getattr(self.model, "sampling", None)
        sampling_states = getattr(sampling, "_trace_states", {}) if sampling is not None else {}
        sampling_trace_count = sum(
            1
            for state in sampling_states.values()
            if isinstance(state, dict) and state.get("id") is not None and state.get("output") is not None
        )
        return {
            "model_decode_trace_count": len(model_trace_ids or {}),
            "sampling_trace_count": sampling_trace_count,
            "model_decode_trace_present": bool(model_trace_ids),
            "sampling_trace_present": sampling_trace_count > 0,
        }

    def mark_warmed_measurement(self) -> None:
        """Assert traces already exist and tag the next timed generation as warmed."""

        handles = self._trace_handle_evidence()
        if not handles["model_decode_trace_present"] or not handles["sampling_trace_present"]:
            raise RuntimeError("warmed measurement requires pre-existing model and sampling trace handles")
        self._warmed_before_measurement = True
        self._trace_handles_before_measurement = handles

    def _write_runtime_evidence(self) -> None:
        if self.runtime_evidence_path is None:
            return
        evidence = self.model.precision_runtime_evidence()
        trace = self.trace_evidence.to_dict()
        handles = self._trace_handle_evidence()
        expected_decode_calls = max(int(self.last_generation_metrics.get("output_tokens", 1)) - 1, 0)
        trace_verified = (
            self.last_generation_metrics.get("enable_trace") is True
            and self._warmed_before_measurement
            and expected_decode_calls > 0
            and trace["decode_calls"] == expected_decode_calls
            and trace["model_execute_submissions"] == expected_decode_calls
            and trace["sampling_execute_submissions"] == expected_decode_calls
            and self._trace_handles_before_measurement.get("model_decode_trace_present") is True
            and self._trace_handles_before_measurement.get("sampling_trace_present") is True
            and handles["model_decode_trace_present"]
            and handles["sampling_trace_present"]
        )
        evidence["measurement"] = {
            "schema_version": 2,
            "generation_metrics": dict(self.last_generation_metrics),
            "trace_counters": trace,
            "warmed_before_timing": self._warmed_before_measurement,
            "trace_handles_before_timing": dict(self._trace_handles_before_measurement),
            "trace_handles": handles,
            "expected_decode_calls": expected_decode_calls,
            "trace_verified": trace_verified,
            "unclassified_execute_submissions_evidentiary": False,
            "trace_verification_rule": (
                "enable_trace=true; model and sampling trace handles exist before timing; known successful model "
                "and sampling call-boundary counters each equal output_tokens-1. The legacy unclassified counter "
                "is retained for schema compatibility but is not observable and is not part of verification."
            ),
        }
        self.runtime_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    def reset(self) -> None:
        """Clear cache content and per-request state while retaining warmed traces.

        Decode attention consumes fixed-size physical K chunks.  Clearing the
        resident cache is therefore part of request isolation: logically
        invalidating page ownership alone can expose stale values in the final
        partially populated chunk when a fixed slot is reused.
        """

        if self._dirty_cache and self.cache_owner == "model":
            self.model.clear_kv_caches()
        self._dirty_cache = False
        self._inner.mode = None
        self._inner.prev_page_table = None
        self._inner._prev_on_device_sampling = None
        self._inner._slots_prefilled_since_decode = set()
        self._last_page_table = None
        self._last_sampling_mode = None
        self._decode_started = False
        self._prepared_device_sampling_params = None
        self.trace_evidence = TraceEvidence()

    def capability_report(self) -> dict[str, Any]:
        return {
            "model": "openai/gpt-oss-120b",
            "mesh_shape": [int(v) for v in self.mesh_device.shape],
            "max_batch_size": self.model_args.max_batch_size,
            "max_context_length": self.model_args.max_context_len,
            "physical_kv_context_length": self.model_args.physical_kv_context_len,
            "page_size": PAGE_SIZE,
            "cache_owner": self.cache_owner,
            "inactive_row_sentinel": -1,
            "sampling": dict(SAMPLER_DECISION),
            "trace_evidence": self.trace_evidence.to_dict(),
            "last_generation_metrics": dict(self.last_generation_metrics),
            "host_sampling_compatibility": True,
            "device_greedy": True,
            "device_top_k_top_p": True,
            "split_token_out_collection": "explicit sync or async boundary; no steady per-token readback",
            "compiled_prefill_variants": [
                {
                    "padded_length": padded_length,
                    "page_rounded_length": page_rounded_length,
                    "last_token_tile_start": last_token_tile_start,
                    "path": path,
                }
                for padded_length, page_rounded_length, last_token_tile_start, path in sorted(
                    self._compiled_prefill_variants
                )
            ],
            "lifetime_prefill_variant_compilations": self._lifetime_prefill_variant_compilations,
            "lifetime_decode_trace_releases_for_prefill_compile": (
                self._lifetime_decode_trace_releases_for_prefill_compile
            ),
            "lifetime_decode_trace_releases_for_host_sampling": (
                self._lifetime_decode_trace_releases_for_host_sampling
            ),
        }

    def teardown(self):
        if self._torn_down:
            return
        self._torn_down = True
        self._inner.release_persistent_capture()
        self._inner = None


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator:
    """Readiness discovery entry point; this stage intentionally stops before vLLM."""

    del model_dir
    override_num_layers = kwargs.pop("override_num_layers", None)
    num_layers = int(override_num_layers or kwargs.pop("num_layers", MODEL_LAYERS))
    max_context_length = int(kwargs.pop("max_seq_len", kwargs.pop("max_context_length", HF_CONTEXT_LENGTH)))
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    snapshot_path = kwargs.pop("snapshot_path", None)
    tensor_cache_path = kwargs.pop("tensor_cache_path", None)
    precision_config = kwargs.pop("precision_config", None)
    runtime_evidence_path = kwargs.pop("runtime_evidence_path", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown build_generator arguments: {unknown}")
    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=snapshot_path,
        tensor_cache_path=tensor_cache_path,
        max_batch_size=max_batch_size,
        max_context_length=max_context_length,
        num_layers=num_layers,
        allow_reduced_model=num_layers != MODEL_LAYERS,
        precision_config=precision_config,
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    if runtime_evidence_path is not None:
        evidence_path = Path(runtime_evidence_path)
        generator.runtime_evidence_path = evidence_path
        generator._write_runtime_evidence()
    return generator


__all__ = [
    "GREEDY",
    "Generator",
    "SAMPLER_DECISION",
    "TraceEvidence",
    "build_generator",
]
