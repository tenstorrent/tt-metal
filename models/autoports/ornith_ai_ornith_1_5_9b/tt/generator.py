# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness and explicit-cache generator for Ornith TP4."""

import math
import time
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import SimpleNamespace
from weakref import WeakValueDictionary

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt import cache_contract
from models.autoports.ornith_ai_ornith_1_5_9b.tt.batch_state import BatchState
from models.autoports.ornith_ai_ornith_1_5_9b.tt.cache_contract import CacheOwnershipRegistry, validate_host_page_table
from models.autoports.ornith_ai_ornith_1_5_9b.tt.decode_contract import DecodeCommands, DecodeSubmission
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_contract import Generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_contract import PrefillContract
from models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state import (
    SamplingSlotState,
    _SlotBuffer,
    normalize_histories,
)
from models.common.sampling.generator import (
    SamplingGenerator,
    SamplingParams,
    _hash_request_seed_to_device_seed,
    format_sampling_params,
)

DEVICE_SAMPLING_MAX_TOP_K = 64


class SamplingCCL:
    def __init__(self, model):
        self.model = model

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self.model.collectives.ccl.get_barrier_semaphore()

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self.model.collectives.ccl.get_ag_ping_pong_semaphore()

    def line_all_gather(self, tensor, *, dim, cluster_axis, memory_config, num_links, **kwargs):
        ccl = self.model.collectives.ccl
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=1,
            mesh_device=self.model.device,
            topology=ccl.topology,
            num_links=num_links,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            memory_config=memory_config,
        )


@dataclass
class DecodeState:
    tokens: object
    positions: list
    tables: list
    host_tables: object
    slots: tuple
    cache: object
    logits: object = None
    model_trace: object = None
    sample_trace: object = None
    sample_traces: dict = field(default_factory=dict)
    model_traces: dict = field(default_factory=dict)
    trace_logits: dict = field(default_factory=dict)
    batch_state: object = None
    masked: bool = False
    forward_initialized: bool = False
    sampling_params_initialized: bool = False
    sampling_state_initialized: bool = False
    host_positions: tuple = ()
    mapped_prefix_pages: tuple = ()
    page_tables_valid: bool = True
    position_advances: int = 0
    resident_ready: bool = False
    token_slots: object = None
    token_remap_indices: object = None
    position_snapshots: list = field(default_factory=list)
    table_snapshots: list = field(default_factory=list)


class OrnithGenerator(Generator):
    def __init__(self, model, *, max_batch_size=1, max_seq_len=None, kv_cache=None, page_table=None):
        import torch
        from transformers import AutoTokenizer

        self.model, self.device = model, model.device
        self.tokenizer = AutoTokenizer.from_pretrained(model.checkpoint.path)
        self.max_seq_len = max_seq_len or model.config.max_position_embeddings
        self.max_batch_size = max_batch_size
        if not 1 <= max_batch_size <= 32 or not 0 < self.max_seq_len <= model.config.max_position_embeddings:
            raise ValueError("Invalid configured batch/context")
        if kv_cache is not None and page_table is None:
            raise ValueError("Externally supplied cache requires its explicit page table")
        self.cache_is_external = kv_cache is not None
        self.cache = (
            kv_cache
            if kv_cache is not None
            else model.allocate_cache(batch=max_batch_size, context=self.max_seq_len, global_pages=True)
        )
        pages = (self.max_seq_len + 31) // 32
        page_table = (
            page_table
            if page_table is not None
            else torch.arange(max_batch_size * pages, dtype=torch.int32).reshape(max_batch_size, pages)
        )
        self.cache_ownership = CacheOwnershipRegistry(
            model, tensor_owner=lambda tensor: tensor.device(), allow_reduced=True, strict_tensor_metadata=True
        )
        binding = self.cache_ownership.register_inferred(self.cache, context=self.max_seq_len)
        if binding.slot_capacity != max_batch_size:
            raise ValueError("Cache slot capacity must match configured max_batch_size")
        validate_host_page_table(page_table, binding)
        self.page_table = page_table.clone()
        self.serving_page_table = self.page_table
        self.cache_ownership.bind_serving(self.cache)
        self.sampler = SamplingGenerator(
            args=SimpleNamespace(
                vocab_size=model.vocab_size,
                padded_vocab_size=model.vocab_size,
                max_batch_size=32,
                max_top_k=DEVICE_SAMPLING_MAX_TOP_K,
                cluster_shape=(1, 4),
                sampling_all_gather_axis=1,
            ),
            mesh_device=self.device,
            tt_ccl=SamplingCCL(model),
        )
        self.sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        self.sampler.reset_sampling_params(
            format_sampling_params(self.sampling_params, 32, max_top_k=DEVICE_SAMPLING_MAX_TOP_K)
        )
        self.sampling_state = SamplingSlotState(self.sampler, parameter_owner=self).prepare()
        self.states = {}
        # Host request metadata only: (prefill origin, generated-output count).
        # Counts follow accepted submissions, never token readback or capture.
        self._sampling_progress = {}
        self.batch_states = {}
        self.submissions = {}
        self.output_handles = WeakValueDictionary()
        self.closed = False
        self.next_submission_id = 0
        self.counters = {
            "model_replays": 0,
            "sampling_replays": 0,
            "token_refreshes": 0,
            "position_refreshes": 0,
            "page_table_refreshes": 0,
            "readbacks": 0,
            "captures": 0,
            "releases": 0,
        }
        self.zero_states = []
        for layer, states in zip(model.layers, self.cache):
            if layer.kind == "linear_attention":
                self.zero_states.append((ttnn.zeros_like(states[0].conv), ttnn.zeros_like(states[0].recurrent)))
            else:
                self.zero_states.append(None)
        # Share zero storage only across identical geometry AND dtype; key and
        # value precision are independently owned by the decoder policy.
        self.zero_kv = {}
        for layer, states in zip(model.layers, self.cache):
            if layer.kind == "full_attention":
                for tensor in (states[0].key, states[0].value):
                    key = (tuple(tensor.shape), tensor.dtype)
                    if key not in self.zero_kv:
                        self.zero_kv[key] = ttnn.zeros_like(tensor)

    def _integer(self, value, *, signed=False, device=True):
        return self.model.upload(
            value, dtype=ttnn.int32 if signed else ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    def _copy(self, value, target, counter, *, signed=False):
        ttnn.copy_host_to_device_tensor(self._integer(value, signed=signed, device=False), target)
        self.counters[counter] += 1

    def _validate_sampling(self, params):
        """Validate raw requests before the common formatter clamps/transforms them."""

        def values(name, default=None):
            value = getattr(params, name, default)
            result = value if isinstance(value, list) else [value]
            if not result or len(result) > 32:
                raise ValueError(f"Sampling {name} requires 1..32 entries")
            return result

        temperatures = values("temperature")
        for temperature in temperatures:
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, Real)
                or not math.isfinite(temperature)
                or temperature < 0
            ):
                raise ValueError("Sampling temperature must be finite and nonnegative")
        for probability in values("top_p"):
            if (
                isinstance(probability, bool)
                or not isinstance(probability, Real)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Sampling top_p must be finite and in [0,1]")
        requested_k = values("top_k")
        if any(isinstance(k, bool) or not isinstance(k, Integral) for k in requested_k):
            raise ValueError("Sampling top_k must be an integer")
        # Scalar k broadcasts only when temperature is a per-slot list; a list
        # remains lane-scoped under the common formatter's existing contract.
        expanded_k = requested_k
        if not isinstance(params.top_k, list) and isinstance(params.temperature, list):
            expanded_k = requested_k * len(temperatures)
        expanded_k = expanded_k + [1] * max(0, len(temperatures) - len(expanded_k))
        for temperature, k in zip(temperatures, expanded_k):
            if temperature != 0 and not 1 <= k <= DEVICE_SAMPLING_MAX_TOP_K:
                raise ValueError(f"Non-greedy device sampling supports only top_k in [1,{DEVICE_SAMPLING_MAX_TOP_K}]")
        for name, default in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
            for value in values(name, default):
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"Sampling {name} must be finite")
                if name == "repetition_penalty" and value <= 0:
                    raise ValueError("Sampling repetition_penalty must be positive")
        for seed in values("seed"):
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
                raise ValueError("Sampling seed must be an integer or None")
        formatted = format_sampling_params(params, 32, max_top_k=DEVICE_SAMPLING_MAX_TOP_K)
        if len(formatted.temperature) != 32 or any(not math.isfinite(t) or t <= 0 for t in formatted.temperature):
            raise ValueError("Formatted temperatures must be 32 finite positive device values")
        if any(not 1 <= k <= DEVICE_SAMPLING_MAX_TOP_K for k in formatted.top_k):
            raise ValueError(f"Formatted device top_k must be in [1,{DEVICE_SAMPLING_MAX_TOP_K}]")
        if any(formatted.enable_log_probs):
            raise ValueError("TP4 logprob requests use the explicit host-logits compatibility path")
        if any(
            seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)) for seed in formatted.seed
        ):
            raise ValueError("Formatted seeds must be lane-scoped integers or None")
        return formatted

    def _sampling_histories(self, *, slots, prompt_tokens=None, output_tokens=None):
        """Select and validate both histories without touching device state."""

        def selected(history):
            if history is None:
                return None
            device = getattr(history, "device", None)
            if device is not None and getattr(device, "type", str(device)) != "cpu":
                raise ValueError("Sampling history must be host-resident")
            if len(history) == len(slots):
                return history
            if len(history) >= max(slots) + 1:
                return [
                    history[slot].tolist() if callable(getattr(history[slot], "tolist", None)) else history[slot]
                    for slot in slots
                ]
            raise ValueError("History does not cover the requested sampling-state slots")

        histories = tuple(
            normalize_histories(selected(history), slots=slots, vocab_size=self.model.vocab_size)
            for history in (prompt_tokens, output_tokens)
        )
        if any(len(prompt) + len(output) > self.max_seq_len for prompt, output in zip(*histories)):
            raise ValueError("Sampling history exceeds the supported context")
        return histories

    def configure_sampling(self, params, *, reset_seed=False):
        """Request admission/reload; all sampler math remains the common implementation.

        A request seed initializes a bounded device seed stream. The sampling
        trace advances that stream, avoiding the common helper's eager seeded
        path and per-token host seed uploads. The stream is independent of slot.
        """
        import secrets

        formatted = self._validate_sampling(params)
        self.sampler.reset_sampling_params(formatted)
        self.sampling_params = params
        if reset_seed:
            seeds = [
                (
                    (_hash_request_seed_to_device_seed(seed, 0) % 500_000) + 1
                    if seed is not None
                    else secrets.randbelow(500_000) + 1
                )
                for seed in formatted.seed
            ]
            # Max context is 262144, keeping every request's stream below the
            # common implementation's one-million device-seed bound.
            self.sampler.seed_manager.write_device_seed_values(seeds)

    def _sampling_reload_progress(self, *, cache, positions, commands, histories, outputs_supplied):
        """Plan request RNG offsets before writes; resident host inputs are opaque.

        An ordinary admitted prompt ending at L has generated p-L+1 outputs
        when its next decode input is at p. Explicit output histories override
        that relationship, including empty histories and deliberate rewinds.
        Unknown resumed requests still need authoritative output histories.
        """
        capacity = len(cache[0])
        progress = {slot: self._sampling_progress.get((id(cache), slot)) for slot in range(capacity)}
        if commands.slot_remap is not None:
            progress = {slot: progress[source] for slot, source in enumerate(commands.slot_remap)}
        reset_slots = commands.sampling_state_slots or tuple(range(capacity))
        explicit = (
            dict(zip(reset_slots, map(len, histories[1]))) if commands.reset_sampling_state and outputs_supplied else {}
        )
        if commands.reload_inputs:
            for slot, position in enumerate(positions):
                position = int(position)
                saved = progress[slot]
                origin, count = saved if saved is not None else (None, 0)
                if slot in explicit:
                    count = explicit[slot]
                    if position >= 0:
                        origin = position + 1 - count
                elif position >= 0 and origin is not None:
                    count = position - origin + 1
                elif position >= 0 and commands.reset_sampling_state and slot in reset_slots:
                    raise ValueError("Sampling reset requires admitted prompt metadata or explicit output history")
                if not 0 <= count <= self.max_seq_len:
                    raise ValueError("Sampling output count inferred from authoritative position is outside context")
                if saved is not None or slot in explicit:
                    progress[slot] = (origin, count)
        offsets = tuple(progress[slot][1] if progress[slot] is not None else 0 for slot in reset_slots)
        return progress, offsets

    def _record_prefill_sampling_progress(self, cache, rows, slots, ends, *, sampled, original_prompt_lens=None):
        # Serving supplies the immutable prompt boundary even when preemption
        # replays generated tokens. Standalone calls retain their prompt contract.
        for row, slot in zip(rows, slots):
            origin = int(ends[row] if original_prompt_lens is None else original_prompt_lens[row])
            count = max(0, int(ends[row]) - origin) + int(sampled and ends[row] >= origin)
            self._sampling_progress[(id(cache), slot)] = (origin, count)

    def _reset_sampling_state(self, *, slots, prompt_tokens=None, output_tokens=None, output_counts=None):
        import secrets

        params = format_sampling_params(self.sampling_params, 32, max_top_k=DEVICE_SAMPLING_MAX_TOP_K)
        slots = tuple(slots)

        prompts, outputs = self._sampling_histories(
            slots=slots, prompt_tokens=prompt_tokens, output_tokens=output_tokens
        )
        counts = tuple(map(len, outputs)) if output_counts is None else tuple(output_counts)
        if len(counts) != len(slots) or any(
            isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= self.max_seq_len
            for count in counts
        ):
            raise ValueError("Expected one bounded output count per sampling reset slot")
        values = [
            (
                (_hash_request_seed_to_device_seed(params.seed[slot], 0) % 500_000) + 1
                if params.seed[slot] is not None
                else secrets.randbelow(500_000) + 1
            )
            + count
            for slot, count in zip(slots, counts)
        ]
        if any(value > 1_000_000 for value in values):
            raise ValueError("Reconstructed sampling seed exceeds the device stream bound")
        self.sampling_state.reset_history(
            slots=slots,
            prompt_tokens=prompts,
            output_tokens=outputs,
        )
        self.sampling_state.reset_seed_values(slots=slots, values=values)

    def reset(self, *, clear_kv=False):
        self._invalidate_resident(self.cache)
        # Full-attention valid rows are overwritten by fresh prefill; no stale row
        # is visible when positions restart at zero. Recurrent state must be zero.
        for states, zero in zip(self.cache, self.zero_states):
            if zero is not None:
                ttnn.copy(zero[0], states[0].conv)
                ttnn.copy(zero[1], states[0].recurrent)
            elif clear_kv:
                for tensor in (states[0].key, states[0].value):
                    ttnn.copy(self.zero_kv[(tuple(tensor.shape), tensor.dtype)], tensor)
        if hasattr(self, "prefill_runtime"):
            self.prefill_runtime.reset()
        self._sampling_progress = {
            key: value for key, value in self._sampling_progress.items() if key[0] != id(self.cache)
        }

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        original_prompt_lens=None,
        start_pos=None,
        return_all_logits=False,
        empty_slots=None,
        request_lens=None,
        discard_logits=None,
        allow_missing_history=False,
        sampling_params=None,
        enable_trace=False,
        reset_new_requests=True,
        clear_kv=False,
        **kwargs,
    ):
        import torch

        if self.closed:
            raise RuntimeError("Generator is closed")
        if kwargs:
            raise TypeError("Unknown prefill keyword(s): " + ", ".join(sorted(kwargs)))
        if any(h.completion is None for h, _, _ in self.submissions.values()):
            raise ValueError("Drain decode submissions before prefill admission")
        if sampling_params is not None:
            self._validate_sampling(sampling_params)
        if any(
            type(flag) is not bool
            for flag in (return_all_logits, allow_missing_history, enable_trace, reset_new_requests, clear_kv)
        ):
            raise TypeError("Prefill mode flags must be bool")
        if enable_trace:
            raise NotImplementedError("Prefill uses prepared eager execution; select decode_only trace mode")
        if sampling_params is not None and return_all_logits:
            raise ValueError("All-logit requests use the explicit host compatibility path")
        # Validate caller metadata before preparation can warm programs or touch
        # setup-owned cache/sampler storage. Runtime continuation checks follow
        # below, still before sampling/decode preparation or any admission.
        if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu" or tokens.ndim != 2:
            raise TypeError("tokens must be a two-dimensional host integer tensor")
        if tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("tokens must have integer dtype")
        ends = PrefillContract._integers(prompt_lens, "prompt_lens")
        rows = len(ends)
        if not 1 <= rows <= 32 or tokens.shape[0] != rows:
            raise ValueError("Batch rows must match prompt_lens and be between 1 and 32")
        if discard_logits is None:
            discard_logits = [False] * rows
        if not isinstance(discard_logits, (list, tuple)) or any(type(flag) is not bool for flag in discard_logits):
            raise TypeError("discard_logits must be a sequence of bools")
        if len(discard_logits) != rows:
            raise ValueError("discard_logits must match the logical batch")
        if any(discard_logits) and (sampling_params is not None or return_all_logits):
            raise ValueError("Discarded logits require unsampled last-logit prefill")
        if any(discard and end == 0 for discard, end in zip(discard_logits, ends)):
            raise ValueError("Only active prefill rows can discard logits")
        originals = (
            ends.copy()
            if original_prompt_lens is None
            else PrefillContract._integers(original_prompt_lens, "original_prompt_lens")
        )
        if len(originals) != rows or any(not 0 <= length <= self.max_seq_len for length in originals):
            raise ValueError("Original prompt lengths must match rows and fit the configured context")
        starts = [0] * rows if start_pos is None else PrefillContract._integers(start_pos, "start_pos")
        row_slots = list(range(rows)) if empty_slots is None else PrefillContract._integers(empty_slots, "empty_slots")
        horizons = ends.copy() if request_lens is None else PrefillContract._integers(request_lens, "request_lens")
        if any(len(values) != rows for values in (starts, row_slots, horizons)):
            raise ValueError("Prompt/start/slot/horizon vectors must have equal length")
        capacity = len(kv_cache[0])
        if len(kv_cache) != len(self.model.layers) or any(len(states) != capacity for states in kv_cache):
            raise ValueError("Prefill cache must cover every layer and fixed slot")
        if not isinstance(page_table, torch.Tensor) or page_table.device.type != "cpu" or page_table.ndim != 2:
            raise TypeError("page_table must be a two-dimensional host integer tensor")
        if page_table.dtype not in (torch.int32, torch.int64) or tuple(page_table.shape) != (
            rows,
            (self.max_seq_len + 31) // 32,
        ):
            raise ValueError("Page-table geometry/dtype must match rows and configured context")
        if torch.any(page_table < -1) or torch.any(page_table > 2**31 - 1):
            raise ValueError("Page IDs must be int32 indices or -1 for unallocated pages")
        occupied = [slot for slot in row_slots if slot >= 0]
        if len(set(occupied)) != len(occupied):
            raise ValueError("Fixed prefill slots must be unique")
        for row, (start, end, slot, horizon) in enumerate(zip(starts, ends, row_slots, horizons)):
            if end == 0:
                if start != 0 or not -1 <= slot < capacity or not 0 <= horizon <= self.max_seq_len:
                    raise ValueError("Invalid inactive prefill row")
                continue
            if (
                not 0 <= slot < capacity
                or not 0 <= start < end <= tokens.shape[1]
                or not end <= horizon <= self.max_seq_len
            ):
                raise ValueError("Invalid active prefill slot/start/end/horizon")
            if torch.any(tokens[row, :end] < 0) or torch.any(tokens[row, :end] >= self.model.vocab_size):
                raise ValueError("Invalid token ID in active prefill prefix")
            if torch.any(page_table[row, : (end + 31) // 32] < 0):
                raise ValueError("All active causal prefill pages must be allocated")
            for layer, states in zip(self.model.layers, kv_cache):
                if layer.kind == "full_attention":
                    state = states[slot]
                    limit = (
                        state.key.shape[0]
                        if getattr(state, "global_pages", False)
                        else state.page_count or state.key.shape[0]
                    )
                    if torch.any(page_table[row] >= limit):
                        raise ValueError("Prefill page mapping exceeds physical cache capacity")
        helper = PrefillContract.prepare(
            self,
            bindings=[{"cache": kv_cache, "slots": list(range(capacity)), "context": self.max_seq_len}],
            sampling_state=self.sampling_state,
        )
        forward_args = {
            "page_table": page_table,
            "kv_cache": kv_cache,
            "prompt_lens": ends,
            "start_pos": starts,
            "empty_slots": row_slots,
            "request_lens": request_lens,
            "return_all_logits": return_all_logits,
            "discard_logits": discard_logits,
            "sample_on_device": sampling_params is not None,
            "allow_missing_history": allow_missing_history,
            "reset_new_requests": reset_new_requests,
            "clear_kv": clear_kv,
            # The canonical reset below owns the original prompt/output split.
            # The helper must not upload a second, prompt-only history first.
            "reset_sampling_state": False,
        }
        validated = helper.forward(tokens, **forward_args, validate_only=True)
        self._invalidate_resident(kv_cache)
        state = None
        if sampling_params is not None:
            # Prepare the entire fixed-slot decode binding before sampling can
            # become resident. Submitted prefill rows may have arbitrary order.
            tables = self.page_table[:capacity].clone()
            for row, slot in zip(validated["input_rows"], validated["active_slots"]):
                tables[slot] = page_table[row]
            state = self.prepare_decode(kv_cache=kv_cache, page_table=tables, slots=tuple(range(capacity)))
        result = helper.forward(tokens, **forward_args)
        if sampling_params is None:
            self._record_prefill_sampling_progress(
                kv_cache,
                validated["input_rows"],
                validated["active_slots"],
                ends,
                sampled=False,
                original_prompt_lens=originals,
            )
            return result
        slots = result.active_slots
        if not slots:
            return torch.full((len(prompt_lens), 1), -1, dtype=torch.int64)
        from dataclasses import fields, replace

        from models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state import _raw_parameters

        # Expand at physical32 while retaining RAW temperatures: scoped reset
        # invokes the common formatter once after merging selected logical rows.
        formatted = _raw_parameters(sampling_params, 32, max_top_k=DEVICE_SAMPLING_MAX_TOP_K)

        selected = replace(
            formatted,
            **{
                field.name: [getattr(formatted, field.name)[row] for row in result.input_rows]
                for field in fields(formatted)
            },
        )
        self.sampling_state.reset_parameters(selected, slots=slots)
        self._reset_sampling_state(
            slots=slots,
            prompt_tokens=[tokens[row, : min(originals[row], ends[row])].tolist() for row in result.input_rows],
            output_tokens=[tokens[row, originals[row] : ends[row]].tolist() for row in result.input_rows],
        )
        self.sampling_state.refresh_active_slots(slots)
        self._sample_logits(result.logits, state.tokens)
        self._record_prefill_sampling_progress(
            kv_cache, result.input_rows, slots, ends, sampled=True, original_prompt_lens=originals
        )
        sampled = ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1).long()
        self.counters["readbacks"] += 1
        output = torch.full((len(prompt_lens), 1), -1, dtype=torch.int64)
        for row, slot in zip(result.input_rows, slots):
            output[row, 0] = sampled[slot]
        return output

    def bind_serving_cache(self, kv_cache, *, page_table):
        """Bind setup-owned serving storage for one-argument release callbacks.

        Constructor injection avoids a second model cache allocation. A later
        binding retains standalone storage; it never frees/replaces live tensors.
        All runtime signatures still require preparation before the first trace.
        """
        if self.closed:
            raise RuntimeError("Generator is closed")
        if self.submissions or any(
            state.model_trace is not None or state.sample_trace is not None for state in self.states.values()
        ):
            raise RuntimeError("Bind serving cache before capture and with no pending submissions")
        binding = self.cache_ownership.register_inferred(kv_cache, context=self.max_seq_len)
        if binding.slot_capacity != self.max_batch_size:
            raise ValueError("Serving cache slot capacity must match configured max_batch_size")
        validate_host_page_table(page_table, binding)
        table = page_table.clone()
        self.cache_ownership.bind_serving(kv_cache)
        self.serving_page_table = table
        return binding

    def get_kv_cache_spec(self):
        """Pure-host full-model specs; reduced diagnostic stacks cannot advertise them."""
        return cache_contract.get_kv_cache_spec(self.model, cache=self.cache_ownership.serving.cache)

    @staticmethod
    def get_max_tokens_all_users(**measurement):
        """Require measured, allocation-validated capacity; no guessed token limit."""
        return cache_contract.get_max_tokens_all_users(**measurement)

    def reset_slots(self, slots, *, kv_cache=None, clear_kv=False, page_tables=None):
        if self.closed:
            raise RuntimeError("Generator is closed")
        if self.submissions:
            raise ValueError("Drain decode submissions before resetting slots")
        cache = self.cache_ownership.serving.cache if kv_cache is None else kv_cache
        self.cache_ownership.register_inferred(cache, context=self.max_seq_len)
        helper = PrefillContract.prepare(
            self,
            sampling_state=self.sampling_state,
            bindings=[{"cache": cache, "slots": list(range(len(cache[0]))), "context": self.max_seq_len}],
        )
        helper.reset_slots(cache=cache, slots=slots, clear_kv=clear_kv, page_tables=page_tables)
        for slot in slots:
            self._sampling_progress.pop((id(cache), slot), None)
        self._invalidate_resident(cache)

    def _drain_submissions(self):
        """Complete every outstanding decode readback (host-synchronizing).

        vLLM can preempt a request (KV pool full) while an async-scheduled decode
        is still in flight; the plugin then releases that request's slot before
        it has processed the in-flight output, and reset_slots would refuse
        ("Drain decode submissions before resetting slots"), killing the engine.
        Completing the reads here caches each result on its handle, so the
        plugin's later read/process calls for the same handle return the cached
        output (both are idempotent once ``handle.completion`` is set).
        """
        for handle, _, _ in list(self.submissions.values()):
            self.process_decode_output_host(handle, is_tokens=handle.mode == "tokens")

    def release_request(self, slot, *, kv_cache=None):
        if self.submissions and not self.closed:
            self._drain_submissions()
        if kv_cache is not None:
            return self.reset_slots([slot], kv_cache=kv_cache)
        return self.cache_ownership.release_request(slot, reset_slots=self.reset_slots)

    def prefill_logits(self, prompt_token_ids):
        import torch

        self.reset()
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table[:1],
            kv_cache=self.cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def prefill_logits_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        import torch

        self.reset(clear_kv=not prefill_prefix)
        tokens = torch.tensor([prompt_token_ids])
        if start and prefill_prefix:
            self.prefill_forward(
                tokens,
                page_table=self.page_table[:1],
                kv_cache=self.cache,
                prompt_lens=[start],
                request_lens=[len(prompt_token_ids)],
            )
        result = self.prefill_forward(
            tokens,
            page_table=self.page_table[:1],
            kv_cache=self.cache,
            prompt_lens=[len(prompt_token_ids)],
            start_pos=[start],
            return_all_logits=True,
            request_lens=[len(prompt_token_ids)],
            allow_missing_history=not prefill_prefix,
        )
        if positions is not None:
            result = result[:, [p - start for p in positions], :]
        return result

    def _model_step(self, state):
        logits = self.model.decode(
            state.tokens,
            cache=state.cache,
            slots=state.slots,
            positions=state.positions,
            page_tables=state.tables,
            batch_state=state.batch_state,
            masked=state.masked,
        )
        if logits.shape[2] < 32:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - logits.shape[2]), (0, 0)], 0.0)
        for position in state.positions:
            ttnn.plus_one(position, skip_negative_entries=True)
        return logits

    def _sample_step(self, state):
        return self._sample_logits(state.logits, state.tokens)

    def _sample_logits(self, logits, tokens):
        # Common sampler math, with a generator-owned trace wrapper. The common
        # high-level wrapper exempts the whole capture and has eager seeded paths.
        self.sampling_state.before_sample()
        result = self.sampler._run_sampling(
            logits,
            penalties_on=self.sampler._penalties_active,
            grammar_on=False,
            tt_out_tok=tokens,
            count_tokens=True,
        )
        ttnn.plus_one(self.sampler.tt_sampling.seeds_tt_tensor, skip_negative_entries=True)
        self.sampling_state.after_sample()
        return result

    def prepare_decode(self, *, kv_cache, page_table, slots):
        import torch

        if self.closed:
            raise RuntimeError("Generator is closed")

        state = self.states.get((id(kv_cache), tuple(slots)))
        if not getattr(state, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")

        PrefillContract.prepare(
            self,
            bindings=[{"cache": kv_cache, "slots": slots, "context": self.max_seq_len}],
            sampling_state=self.sampling_state,
        )
        key = (id(kv_cache), tuple(slots))
        if key in self.states:
            return self.states[key]
        if any(state.model_trace is not None for state in self.states.values()):
            raise RuntimeError("New decode bindings must be prepared before capture")
        # Preparation may be requested by an external cache owner after loading
        # state. Restore the actual passed cache, including untouched pooled rows.
        saved, seen = [], set()
        self.sampling_state.snapshot()
        for states in kv_cache:
            for st in states:
                for name in ("conv", "recurrent", "key", "value"):
                    tensor = getattr(st, name)
                    if tensor is not None and id(tensor) not in seen:
                        saved.append((tensor, ttnn.clone(tensor)))
                        seen.add(id(tensor))
        state = DecodeState(
            self._integer(torch.zeros((1, 1, 1, 32), dtype=torch.int32)),
            [self._integer(torch.zeros(1, dtype=torch.int32), signed=True) for _ in slots],
            [self._integer(row[None], signed=True) for row in page_table],
            page_table.clone(),
            tuple(slots),
            kv_cache,
        )
        self._refresh_resident_page_bounds(state)
        # Request-time permutations must retain every trace input address. Warm
        # the same gather and simultaneous-copy signatures before any capture.
        state.token_slots = _SlotBuffer(self.sampling_state, "decode.tokens", state.tokens)
        state.token_remap_indices = self.sampling_state._upload(torch.arange(32).reshape(1, 32), dtype=ttnn.uint32)
        state.position_snapshots = [ttnn.clone(position) for position in state.positions]
        state.table_snapshots = [ttnn.clone(table) for table in state.tables]
        state.token_slots.remap(state.token_remap_indices)
        for originals, snapshots in (
            (state.positions, state.position_snapshots),
            (state.tables, state.table_snapshots),
        ):
            for original, snapshot in zip(originals, snapshots):
                ttnn.copy(original, snapshot)
                ttnn.copy(snapshot, original)
        guard = self.batch_states.get(id(kv_cache))
        if guard is None:
            guard = BatchState(self.model, kv_cache)
            guard.warmup_slot_ops()
            self.batch_states[id(kv_cache)] = guard
        state.batch_state = guard
        state.logits = self._model_step(state)
        # Warm the inactive-row variant on the same complete ordered pool.
        if tuple(slots) == tuple(range(guard.capacity)):
            state.masked = True
            self._model_step(state)
            state.masked = False
        active = self.sampler._penalties_active
        try:
            for penalties in (False, True):
                self.sampler._penalties_active = penalties
                self.sampler.precompile(state.logits, tt_out_tok=state.tokens, compile_token_update=True)
                # Slot-scoped reset is also a request-time program, prepared here.
                self.sampler.reset_output_state(slots=list(range(32)))
        finally:
            self.sampler._penalties_active = active
        self._sample_step(state)
        self.sampler.tt_penalties.reset_output_tokens()
        # Warm reset before recording, and restore request state before prefill.
        for destination, snapshot in saved:
            ttnn.copy(snapshot, destination)
        self.sampling_state.restore()
        for pos in state.positions:
            ttnn.copy_host_to_device_tensor(
                self._integer(torch.zeros(1, dtype=torch.int32), signed=True, device=False), pos
            )
        self.states[key] = state
        return state

    def capture(self, state):
        if not getattr(state, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
        from ttnn.tools import trace_allocation_tracker

        if state.model_trace is not None:
            return
        active, masked = self.sampler._penalties_active, state.masked
        try:
            modes = (False, True) if state.slots == tuple(range(state.batch_state.capacity)) else (False,)
            for mask in modes:
                state.masked = mask
                capture_started = time.perf_counter()
                trace = ttnn.begin_trace_capture(self.device, cq_id=0)
                logits = self._model_step(state)
                ttnn.end_trace_capture(self.device, trace, cq_id=0)
                state.model_traces[mask] = trace
                state.trace_logits[mask] = logits
                self.counters["captures"] += 1
                self._trace_event(state, trace, "model", mask, None, capture_started)
                # The producer always immediately precedes the sampling consumer.
                trace_allocation_tracker.acknowledge_corruptible(logits)
                state.logits = logits
                for penalties in (False, True):
                    self.sampler._penalties_active = penalties
                    capture_started = time.perf_counter()
                    trace = ttnn.begin_trace_capture(self.device, cq_id=0)
                    self._sample_step(state)
                    ttnn.end_trace_capture(self.device, trace, cq_id=0)
                    state.sample_traces[(mask, penalties)] = trace
                    self.counters["captures"] += 1
                    self._trace_event(state, trace, "sampling", mask, penalties, capture_started)
        finally:
            self.sampler._penalties_active, state.masked = active, masked
        state.model_trace = state.model_traces[masked]
        state.logits = state.trace_logits[masked]
        state.sample_trace = state.sample_traces[(masked, active)]

    def _trace_event(self, state, trace, phase, masked, penalties, started):
        observer = getattr(self, "trace_observer", None)
        if observer is not None:
            observer(
                "trace_capture",
                trace_id=str(trace),
                phase=phase,
                signature={
                    "cache_binding": id(state.cache),
                    "slots": state.slots,
                    "masked": masked,
                    "penalties": penalties,
                    "table_shapes": [list(table.shape) for table in state.tables],
                },
                capture_ms=1000 * (time.perf_counter() - started),
                counters=dict(self.counters),
            )

    def _decode_preflight(
        self,
        *,
        commands,
        previous,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        capacity,
        sampling_params,
        prompt_tokens,
        output_tokens,
    ):
        """Validate the entire commanded host transition before any device write."""
        import torch

        if len(kv_cache) != len(self.model.layers) or any(len(states) != capacity for states in kv_cache):
            raise ValueError("Decode cache must cover every layer and fixed slot")
        if (
            commands.slot_remap is not None
            and commands.slot_remap != tuple(range(capacity))
            and any(
                not getattr(state, "global_pages", False)
                for layer, states in zip(self.model.layers, kv_cache)
                if layer.kind == "full_attention"
                for state in states
            )
        ):
            raise ValueError("Slot remapping requires global-page cache bindings; allocate_cache(global_pages=True)")

        def integers(value, name):
            device = getattr(value, "device", None)
            if device is not None and getattr(device, "type", str(device)) != "cpu":
                raise ValueError(f"{name} must be host-resident")
            tensor = torch.as_tensor(value)
            if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError(f"{name} must contain integers without floating-point coercion")
            return tensor.to(torch.int64)

        supplied, positions, tables = None, None, None
        if commands.reload_inputs:
            positions = integers(start_pos, "Positions").reshape(-1)
            if positions.numel() != capacity or torch.any(positions < -1) or torch.any(positions >= self.max_seq_len):
                raise ValueError("Positions must cover every slot and be -1 or within context")
            supplied = integers(tokens, "Tokens").reshape(-1)
            if supplied.numel() != capacity:
                raise ValueError("One input token is required for every fixed slot")
            if (
                torch.any(supplied >= self.model.vocab_size)
                or torch.any(supplied < -1)
                or torch.any((positions >= 0) & (supplied < 0))
            ):
                raise ValueError("Active token IDs must be in vocabulary; inactive rows may additionally use -1")
            supplied = supplied.clone()
            supplied[positions < 0] = 0  # Inactive embedding lookup still requires a valid index.
        elif commands.reload_page_table:
            # Only explicit scheduler transitions reconstruct this small host
            # vector. Steady replay increments one scalar beside its counter.
            resident_positions = previous.host_positions
            if commands.slot_remap is not None:
                resident_positions = tuple(resident_positions[source] for source in commands.slot_remap)
            positions = torch.tensor(
                [position + previous.position_advances if position >= 0 else -1 for position in resident_positions],
                dtype=torch.int64,
            )
            if positions.numel() != capacity:
                raise ValueError("Resident position metadata was not initialized by input reload")
        if commands.reload_inputs or commands.reload_page_table:
            tables = integers(page_table, "Page table")
            expected = (
                tuple(previous.host_tables.shape) if previous is not None else (capacity, (self.max_seq_len + 31) // 32)
            )
            if tables.ndim != 2 or tuple(tables.shape) != expected:
                raise ValueError("Page table geometry differs from the configured fixed-slot binding")
            if torch.any(tables < -1) or torch.any(tables > 2**31 - 1):
                raise ValueError("Page IDs must be int32 physical indices, or -1 for unused pages")
            for index, layer in enumerate(self.model.layers):
                if layer.kind != "full_attention":
                    continue
                for slot, state in enumerate(kv_cache[index]):
                    limit = (
                        state.key.shape[0]
                        if getattr(state, "global_pages", False)
                        else state.page_count or state.key.shape[0]
                    )
                    if torch.any(tables[slot] >= limit):
                        raise ValueError(f"Page mapping exceeds layer {index} slot {slot} cache capacity")
            for slot, position in enumerate(positions.tolist()):
                if position < 0:
                    continue
                needed = position // 32 + 1
                if position >= self.max_seq_len or needed > tables.shape[1] or torch.any(tables[slot, :needed] < 0):
                    raise ValueError("Every active causal page through the current position must be mapped")
            tables = tables.to(torch.int32)
        else:
            # A remap permutes both positions and page rows, so admission of all
            # current slots is equivalent before that transition. Validate here,
            # before remap writes or trace submission can mutate device state.
            self._validate_resident_window(previous, 1)
        if commands.reload_sampling_params:
            self._validate_sampling(sampling_params)
        histories = (None, None)
        if commands.reset_sampling_state:
            histories = self._sampling_histories(
                slots=commands.sampling_state_slots or tuple(range(capacity)),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        if commands.slot_remap is not None and previous is not None:
            bounds = [self.prefill_runtime.bindings.get((id(kv_cache), slot)) for slot in range(capacity)]
            if any(bound is None for bound in bounds) or len({bound["context"] for bound in bounds}) != 1:
                raise ValueError("Remap requires registered fixed slots with matching context capacities")
        return supplied, positions, tables, histories

    def _remap_forward_inputs(self, state, remap):
        """Simultaneously permute persistent forward inputs at a drained transition."""
        import torch

        if not getattr(state, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
        try:
            full_remap = tuple(remap) + tuple(range(len(remap), 32))
            self.sampling_state._copy_host(torch.tensor(full_remap).reshape(1, 32), state.token_remap_indices)
            state.token_slots.remap(state.token_remap_indices)
            for originals, snapshots in (
                (state.positions, state.position_snapshots),
                (state.tables, state.table_snapshots),
            ):
                for original, snapshot in zip(originals, snapshots):
                    ttnn.copy(original, snapshot)
                for destination, source in enumerate(remap):
                    ttnn.copy(snapshots[source], originals[destination])
            state.host_tables = state.host_tables[list(remap)].clone()
            state.mapped_prefix_pages = tuple(state.mapped_prefix_pages[source] for source in remap)
            if state.host_positions:
                state.host_positions = tuple(state.host_positions[source] for source in remap)
                active = tuple(position >= 0 for position in state.host_positions)
                state.batch_state.set_active(active)
                state.masked = not all(active)
        except Exception:
            state.page_tables_valid = False
            raise

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        empty_slots=None,
        enable_trace=True,
        host_sampling=None,
        sampling_params=None,
        read_from_device=True,
        prompt_tokens=None,
        output_tokens=None,
        **kwargs,
    ):
        import torch

        commands = DecodeCommands.from_kwargs(kwargs)
        if self.closed:
            raise RuntimeError("Generator is closed")
        if any(event is None for _, _, event in self.submissions.values()):
            raise ValueError("Queue deferred output reads before submitting another decode")
        if (
            type(enable_trace) is not bool
            or type(read_from_device) is not bool
            or (host_sampling is not None and type(host_sampling) is not bool)
        ):
            raise TypeError("Decode trace/readback/sampling mode flags must be booleans")
        if not enable_trace:
            raise ValueError("Eager decode is not an accepted generation mode")
        host_sampling = sampling_params is None if host_sampling is None else host_sampling
        capacity = len(kv_cache[0])
        slots = tuple(range(capacity)) if empty_slots is None else tuple(empty_slots)
        if any(isinstance(slot, bool) or not isinstance(slot, Integral) for slot in slots) or slots != tuple(
            range(capacity)
        ):
            raise ValueError("Decode retains all configured slots in pool order; mark inactive positions -1")
        key = (id(kv_cache), slots)
        previous = self.states.get(key)
        if previous is not None and not getattr(previous, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
        commands.validate_context(
            mode="host" if host_sampling else "device",
            slot_capacity=capacity,
            batch_size=capacity,
            forward_initialized=previous is not None and previous.forward_initialized,
            sampling_params_initialized=previous is not None and previous.sampling_params_initialized,
            sampling_state_initialized=previous is not None and previous.sampling_state_initialized,
            sampling_params_available=sampling_params is not None,
            pending_submission=any(h.completion is None for h, _, _ in self.submissions.values()),
        )
        supplied, positions, tables, histories = self._decode_preflight(
            commands=commands,
            previous=previous,
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            capacity=capacity,
            sampling_params=sampling_params,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        progress, seed_offsets = None, None
        if commands.reload_inputs or commands.slot_remap is not None:
            progress, seed_offsets = self._sampling_reload_progress(
                cache=kv_cache,
                positions=positions,
                commands=commands,
                histories=histories,
                outputs_supplied=output_tokens is not None,
            )
        state = previous or self.prepare_decode(kv_cache=kv_cache, page_table=tables, slots=slots)
        if commands.slot_remap is not None:
            state.batch_state.permute_slots(commands.slot_remap)
            self.sampling_state.apply_slot_remap(tuple(commands.slot_remap) + tuple(range(capacity, 32)))
            self.prefill_runtime.remap_slots(kv_cache, commands.slot_remap)
            self._remap_forward_inputs(state, commands.slot_remap)
        if commands.reload_inputs:
            values = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
            values.reshape(-1)[:capacity] = supplied
            self._copy(values, state.tokens, "token_refreshes")
            for p, pos in zip(positions, state.positions):
                self._copy(torch.tensor([int(p)], dtype=torch.int32), pos, "position_refreshes", signed=True)
            active = tuple(int(p) >= 0 for p in positions)
            state.batch_state.set_active(active)
            self.sampling_state.refresh_active_slots([i for i, value in enumerate(active) if value])
            state.masked = not all(active)
            state.forward_initialized = True
            state.host_positions = tuple(int(p) for p in positions)
            state.position_advances = 0
        if commands.reload_inputs or commands.reload_page_table:
            if not getattr(state, "page_tables_valid", True):
                raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
            try:
                for row, (source, target) in enumerate(zip(tables, state.tables)):
                    # Reload commands may accompany admission without page changes.
                    # Compare values against our own snapshot, never object identity.
                    if not torch.equal(source, state.host_tables[row]):
                        self._copy(source[None], target, "page_table_refreshes", signed=True)
                state.host_tables = tables.clone()
                self._refresh_resident_page_bounds(state)
            except Exception:
                # Earlier rows (or the failing transfer itself) may have changed.
                # Never retry against an untrusted host/device mapping snapshot.
                state.page_tables_valid = False
                raise
        if commands.reload_sampling_params:
            self.configure_sampling(sampling_params)
            state.sampling_params_initialized = True
        if commands.reset_sampling_state:
            self._reset_sampling_state(
                slots=commands.sampling_state_slots or tuple(range(capacity)),
                prompt_tokens=histories[0],
                output_tokens=histories[1],
                output_counts=seed_offsets,
            )
            state.sampling_state_initialized = True
        if progress is not None:
            for slot, value in progress.items():
                key = (id(kv_cache), slot)
                if value is None:
                    self._sampling_progress.pop(key, None)
                else:
                    self._sampling_progress[key] = value
        self.capture(state)
        self._execute(state, host_sampling=host_sampling)
        handle = self._submit_read(state, host_sampling=host_sampling)
        return self.process_decode_output_host(handle, is_tokens=not host_sampling) if read_from_device else handle

    def _execute(self, state, *, host_sampling=False):
        if not getattr(state, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
        state.model_trace = state.model_traces[state.masked]
        state.logits = state.trace_logits[state.masked]
        ttnn.execute_trace(self.device, state.model_trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        state.position_advances += 1
        if not host_sampling:
            state.sample_trace = state.sample_traces[(state.masked, self.sampler._penalties_active)]
            ttnn.execute_trace(self.device, state.sample_trace, cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
        for slot, position in enumerate(state.host_positions):
            key = (id(state.cache), slot)
            if position >= 0 and key in self._sampling_progress:
                origin, count = self._sampling_progress[key]
                self._sampling_progress[key] = (origin, count + 1)
        state.resident_ready = not host_sampling

    def _invalidate_resident(self, cache):
        for state in self.states.values():
            if state.cache is cache:
                state.resident_ready = False

    def _refresh_resident_page_bounds(self, state):
        """Cache the first unmapped page after preparation or a commanded reload.

        host_tables is generator-owned; scheduler transitions replace it, and
        slot remaps permute this metadata together with the table rows.
        """
        import torch

        width = state.host_tables.shape[1]
        columns = torch.arange(width, dtype=torch.int64)
        state.mapped_prefix_pages = tuple(torch.where(state.host_tables < 0, columns, width).amin(dim=1).tolist())

    def _validate_resident_window(self, state, steps):
        """Admit a window using scheduler-owned bounds; no steady tensor work."""
        if not getattr(state, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
        for slot, position in enumerate(state.host_positions):
            if position < 0:
                continue
            last = position + state.position_advances + steps - 1
            if last >= self.max_seq_len:
                raise ValueError("Resident decode exceeds configured context")
            if last // 32 >= state.mapped_prefix_pages[slot]:
                raise ValueError("Every active causal page through the resident decode window must be mapped")

    def decode_resident(self, steps, *, kv_cache=None):
        """Queue fixed-step sampled decode without host transfers or synchronization.

        Admission, explicit scheduler changes and prefill remain caller-owned.
        Continue the state initialized by generate or decode_forward, retaining
        fixed slots, inactive rows and the current sampling parameters. The
        returned device token tensor is borrowed: its value is the final step's
        token and the next decode overwrites it. This API does not retain the
        intermediate generated sequence. Streaming callers use decode_forward.
        """
        if self.closed:
            raise RuntimeError("Generator is closed")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        cache = self.cache if kv_cache is None else kv_cache
        state = self.states.get((id(cache), tuple(range(len(cache[0])))))
        if not getattr(state, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
        if state is None or not state.model_traces or not state.host_positions or not state.resident_ready:
            raise ValueError("Initialize resident decode inputs before queuing steps")
        if any(h.completion is None for h, _, _ in self.submissions.values()):
            raise ValueError("Drain outstanding submissions before resident decode")
        self._validate_resident_window(state, steps)
        for _ in range(steps):
            self._execute(state, host_sampling=False)
        return state.tokens

    def _submit_read(self, state, *, host_sampling):
        # Return a device-backed submission without a read or synchronization.
        # The caller must enqueue its deferred read before the next replay can
        # overwrite this borrowed persistent output (enforced in decode_forward).
        tensor = state.logits if host_sampling else state.tokens
        handle = DecodeSubmission(
            self.next_submission_id,
            "logits" if host_sampling else "tokens",
            len(state.slots),
            state.slots,
            len(state.cache[0]),
            token_tensor=None if host_sampling else tensor,
            logits_tensor=tensor if host_sampling else None,
            vocab_size=self.model.vocab_size,
            active_rows=state.batch_state.active,
        )
        self.next_submission_id += 1
        self.output_handles[handle.submission_id] = handle
        self.submissions[handle.submission_id] = (handle, None, None)
        return handle

    def read_decode_output(self, handle, async_read=False):
        if self.output_handles.get(handle.submission_id) is not handle:
            raise ValueError("Output handle does not belong to this generator")
        if handle.completion is not None:
            return (handle, []) if async_read else handle
        if self.closed:
            raise RuntimeError("Generator is closed")
        _, hosts, event = self.submissions[handle.submission_id]
        if event is None:
            # CQ0 orders the owning transfer after this output and before a
            # subsequent trace. Device sampling transfers only the first shard's
            # token tile; full-logit reads are explicit host compatibility only.
            tensor = handle.token_tensor if handle.mode == "tokens" else handle.logits_tensor
            shards = ttnn.get_device_tensors(tensor)
            selected = shards[:1] if handle.mode == "tokens" else shards
            hosts = tuple(shard.cpu(blocking=False) for shard in selected)
            event = ttnn.record_event(self.device, 0)
            self.submissions[handle.submission_id] = (handle, hosts, event)
            self.counters["readbacks"] += 1
        if async_read:
            return handle, [event]
        ttnn.event_synchronize(event)
        return handle

    def process_decode_output_host(self, handle, is_tokens=False):
        import torch

        if is_tokens != (handle.mode == "tokens"):
            raise ValueError("Output mode differs from the submitted decode")
        if handle.completion is not None:
            return handle.completion.host_output
        self.read_decode_output(handle)
        _, hosts, _ = self.submissions[handle.submission_id]
        if is_tokens:
            result = ttnn.to_torch(hosts[0]).reshape(-1)[: handle.batch_size].long().reshape(-1, 1)
        else:
            result = torch.cat([ttnn.to_torch(host).float() for host in hosts], dim=-1)
            result = result[0, 0, : handle.batch_size, : self.model.vocab_size].unsqueeze(1).clone()
        # Inactive kernels may leave arbitrary row values. Preserve fixed output
        # geometry with an explicit sentinel using this submission's immutable
        # admission state, even if slots have changed before host processing.
        for row, active in enumerate(handle.active_rows or (True,) * handle.batch_size):
            if not active:
                result[row] = -1 if is_tokens else 0
        handle.complete_read(result, shape=result.shape, row_order=handle.row_order)
        del self.submissions[handle.submission_id]
        return result

    def replay(self, state, *, host_sampling=False):
        if not getattr(state, "page_tables_valid", True):
            raise RuntimeError("Decode page-table state is unusable after a failed refresh; recreate the generator")
        if self.closed:
            raise RuntimeError("Generator is closed")
        if any(event is None for _, _, event in self.submissions.values()):
            raise ValueError("Queue deferred output reads before submitting another replay")
        self._execute(state, host_sampling=host_sampling)
        if host_sampling:
            return self.model.host_logits(state.logits)[:, :, : len(state.slots)].reshape(len(state.slots), -1)
        self.counters["readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1)[: len(state.slots)].long()

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=False,
        sampling_params=None,
        **kwargs,
    ):
        import time

        import torch

        if self.closed:
            raise RuntimeError("Generator is closed")
        if any(h.completion is None for h, _, _ in self.submissions.values()):
            raise ValueError("Drain decode submissions before high-level generation")
        if kwargs:
            # Metal readiness reserves extras and asks generators to ignore
            # unknown ones. Low-level serving admission remains strict.
            known = set(SamplingParams.__dataclass_fields__)
            named_sampling = {key: value for key, value in kwargs.items() if key in known}
            if named_sampling:
                if sampling_params is not None:
                    raise TypeError("Pass sampling_params or named sampler controls, not both")
                sampling_params = SamplingParams(**({"temperature": 0.0, "top_k": 1, "top_p": 1.0} | named_sampling))
        if host_sampling and sampling_params is not None:
            temperatures = (
                sampling_params.temperature
                if isinstance(sampling_params.temperature, list)
                else [sampling_params.temperature]
            )
            if any(float(x) != 0 for x in temperatures):
                raise ValueError("High-level host compatibility is greedy; use low-level full logits for host sampling")
        if not enable_trace:
            raise ValueError("Generation requires traced decode")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if max_new_tokens < 1:
            return []
        if not prompt_token_ids or any(
            isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= token < self.model.vocab_size
            for token in prompt_token_ids
        ):
            raise ValueError("Generation requires a nonempty prompt of valid integer token IDs")
        if len(prompt_token_ids) + max_new_tokens - 1 > self.max_seq_len:
            raise ValueError("Generation exceeds context")
        params = sampling_params or SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        self._validate_sampling(params)
        capacity = len(self.cache[0])
        state = self.prepare_decode(kv_cache=self.cache, page_table=self.page_table, slots=tuple(range(capacity)))
        started = time.perf_counter()
        counters_before = dict(self.counters)
        if capacity == 1:
            # TTFT includes request sampling admission, seed/history reset and
            # prefill, while one-time decode preparation is outside the timer.
            self.configure_sampling(params, reset_seed=True)
            self.reset()
            self.sampling_state.reset_history(slots=[0], prompt_tokens=[prompt_token_ids])
        else:
            from dataclasses import fields, replace

            from models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state import _raw_parameters

            # One high-level request owns slot0. Preserve every peer's cache,
            # parameters, history and seed while retaining the full native pool.
            raw = _raw_parameters(params, 32, max_top_k=DEVICE_SAMPLING_MAX_TOP_K)
            selected = replace(raw, **{field.name: getattr(raw, field.name)[:1] for field in fields(raw)})
            self.reset_slots([0], kv_cache=self.cache)
            self.sampling_state.reset_parameters(selected, slots=[0])
            self._reset_sampling_state(slots=(0,), prompt_tokens=[prompt_token_ids])
        self.sampling_state.refresh_active_slots([0])
        state.batch_state.set_active((True,) + (False,) * (capacity - 1))
        state.masked = capacity > 1
        logits = self.prefill_runtime.prefill(
            prompt_token_ids,
            cache=self.cache,
            slot=0,
            start=0,
            total_length=len(prompt_token_ids),
            page_table=state.host_tables[0],
            last_only=True,
            return_device_logits=True,
        )
        if host_sampling:
            first = int(self.model.host_logits(logits)[0, 0, 0].argmax())
        else:
            self._sample_logits(logits, state.tokens)
            self.counters["readbacks"] += 1
            first = int(ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1)[0])
        del logits
        self._record_prefill_sampling_progress(self.cache, (0,), (0,), (len(prompt_token_ids),), sampled=True)
        first_ready = time.perf_counter()
        output = [first]
        forced = next_input(0, first) if next_input else first
        if max_new_tokens == 1:
            return output
        if next_input is not None or host_sampling:
            values = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
            values.reshape(-1)[0] = forced
            self._copy(values, state.tokens, "token_refreshes")
        self._copy(
            torch.tensor([len(prompt_token_ids)], dtype=torch.int32),
            state.positions[0],
            "position_refreshes",
            signed=True,
        )
        for position in state.positions[1:]:
            self._copy(torch.tensor([-1], dtype=torch.int32), position, "position_refreshes", signed=True)
        state.host_positions = (len(prompt_token_ids),) + (-1,) * (capacity - 1)
        state.position_advances = 0
        self.capture(state)
        predicted = self.replay(state, host_sampling=host_sampling)
        for step in range(1, max_new_tokens):
            token = int(predicted[0].argmax()) if host_sampling else int(predicted[0])
            output.append(token)
            forced = next_input(step, token) if next_input else token
            if step + 1 == max_new_tokens:
                break
            if next_input is not None or host_sampling:
                value = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
                value.reshape(-1)[0] = forced
                self._copy(value, state.tokens, "token_refreshes")
            predicted = self.replay(state, host_sampling=host_sampling)
        completed = time.perf_counter()
        self.last_generation_metrics = {
            "prompt_tokens": len(prompt_token_ids),
            "generated_tokens": len(output),
            "ttft_ms": (first_ready - started) * 1000,
            "decode_ms_per_token": (completed - first_ready) * 1000 / (len(output) - 1),
            "decode_tokens_per_second_per_user": (len(output) - 1) / (completed - first_ready),
            "teacher_forcing": next_input is not None,
            "host_sampling": host_sampling,
            "trace_verified": enable_trace,
            "regime": "serial",
            "chip_ids": [0, 1, 2, 3],
            "counter_delta": {key: value - counters_before[key] for key, value in self.counters.items()},
        }
        return output

    def close(self):
        if self.closed:
            return
        if any(event is None for _, _, event in self.submissions.values()):
            ttnn.synchronize_device(self.device)
        else:
            for _, _, event in self.submissions.values():
                ttnn.event_synchronize(event)
        for state in self.states.values():
            for trace in (*state.model_traces.values(), *state.sample_traces.values()):
                if trace is not None:
                    ttnn.release_trace(self.device, trace)
                    self.counters["releases"] += 1
            state.model_trace = state.sample_trace = None
            state.sample_traces.clear()
            state.model_traces.clear()
            state.trace_logits.clear()
        self.closed = True

    release_persistent_capture = close
    teardown = close


def build_generator(model_dir, mesh_device, **kwargs):
    model = OrnithModel(
        mesh_device,
        checkpoint=kwargs.get("checkpoint"),
        layer_indices=kwargs.get("layer_indices"),
        decode_head=kwargs.get("decode_head", "dram"),
        selective_inactive_restore=kwargs.get("selective_inactive_restore", True),
        precision_config=kwargs.get("precision_config"),
    )
    return OrnithGenerator(
        model,
        max_batch_size=kwargs.get("max_batch_size", 1),
        max_seq_len=kwargs.get("max_seq_len"),
        kv_cache=kwargs.get("kv_cache"),
        page_table=kwargs.get("page_table"),
    )
