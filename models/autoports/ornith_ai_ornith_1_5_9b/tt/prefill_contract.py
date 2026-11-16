# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-request prefill admission and persistent fixed-slot sampler logits.

PrefillRuntime owns layer computation. This helper validates an entire admission
before mutating any row, resets only newly admitted slots, preserves inactive
peers, and assembles each active request's last logits into its sampler slot.
The generator owns sampler parameters, seeds, sampling invocation and token I/O.
Device mode never materializes logits on the host. Host mode is explicit support
for readiness, prompt/logprob diagnostics, and host-sampling compatibility.
"""

from dataclasses import dataclass
from numbers import Integral

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_plan import prefix_profiles_compatible
from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_runtime import PrefillRuntime


@dataclass(frozen=True)
class PrefillBatch:
    """Borrowed persistent logits, valid until the next call on this helper.

    The tensor always has 32 physical sampler rows. Only active_slots are valid;
    rows are slot-indexed, not compacted into input order. input_rows identifies
    the original row for each entry in active_slots. The parent must consume or
    copy logits before another prefill overwrites this persistent output.
    """

    logits: object
    active_slots: tuple[int, ...]
    input_rows: tuple[int, ...]
    prompt_lens: tuple[int, ...]
    start_pos: tuple[int, ...]
    request_lens: tuple[int, ...]
    row_slots: tuple[int, ...]

    @property
    def active_mask(self):
        return tuple(slot in self.active_slots for slot in range(32))


class PrefillContract:
    @classmethod
    def prepare(cls, gen, *, runtime=None, bindings=None, sampling_state=None):
        """Allocate/compile assembly and reset helpers before the first capture."""
        existing = getattr(gen, "prefill_contract", None)
        if existing is not None:
            if runtime is not None and runtime is not existing.runtime:
                raise ValueError("Cannot replace the configured prefill runtime")
            if bindings:
                existing.runtime = PrefillRuntime.prepare(gen, bindings=bindings)
                existing._prepare_bindings()
            return existing
        PrefillRuntime._require_no_traces(gen)
        runtime = runtime or PrefillRuntime.prepare(gen, bindings=bindings)
        if bindings and runtime is not None:
            for binding in bindings:
                runtime.register_cache(**binding)
        if sampling_state is None:
            sampling_state = getattr(gen, "sampling_state", None) or getattr(gen, "sampling_slot_state", None)
        if sampling_state is None:
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state import SamplingSlotState

            sampling_state = SamplingSlotState(gen.sampler, parameter_owner=gen).prepare()
            gen.sampling_state = sampling_state
            gen.sampling_slot_state = sampling_state
        obj = cls(gen, runtime, sampling_state)
        obj._prepare_bindings()
        obj._warm_assembly()
        gen.prefill_contract = obj
        return obj

    def __init__(self, gen, runtime, sampling_state):
        self.gen, self.model, self.device = gen, gen.model, gen.device
        self.runtime, self.sampling_state = runtime, sampling_state
        # head.shape[-1] is the local TP vocabulary width, matching its output.
        self.local_vocab = self.model.head.shape[-1]
        shape = (1, 1, 32, self.local_vocab)
        self.logits_rm = self._zeros(shape, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.zero_rm = self._zeros(shape, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.logits = self._zeros(shape, layout=ttnn.TILE_LAYOUT)
        self.kv_zero_inputs, self.reset_tables = {}, {}
        self.prepared_bindings = set()
        self.events = []
        self.warm_program_counts = []

    def _zeros(self, shape, *, dtype=ttnn.bfloat16, layout):
        return ttnn.zeros(shape, dtype=dtype, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _prepare_bindings(self):
        new = [key for key in self.runtime.bindings if key not in self.prepared_bindings]
        if not new:
            return
        PrefillRuntime._require_no_traces(self.gen)
        saved, seen = [], set()
        before = self.device.num_program_cache_entries()
        for key in new:
            bound = self.runtime.bindings[key]
            for index, (layer, states) in enumerate(zip(self.model.layers, bound["cache"], strict=True)):
                state = states[bound["slot"]]
                for name in ("conv", "recurrent", "key", "value"):
                    tensor = getattr(state, name)
                    if tensor is not None and id(tensor) not in seen:
                        saved.append((tensor, ttnn.clone(tensor)))
                        seen.add(id(tensor))
                if layer.kind == "full_attention":
                    global_pages = bool(getattr(state, "global_pages", False))
                    page_count = (
                        (bound["context"] + 31) // 32 if global_pages else state.page_count or state.key.shape[0]
                    )
                    offset = 0 if global_pages else state.page_offset
                    self.reset_tables[(key, index)] = self.model.upload(
                        torch.arange(offset, offset + page_count, dtype=torch.int32)[None],
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    for tensor in (state.key, state.value):
                        _, heads, block, width = tuple(tensor.shape)
                        zero_key = (page_count, heads, block, width, tensor.dtype)
                        if zero_key not in self.kv_zero_inputs:
                            self.kv_zero_inputs[zero_key] = self._zeros(
                                (1, heads, page_count * block, width), dtype=tensor.dtype, layout=ttnn.TILE_LAYOUT
                            )
        try:
            for key in new:
                self._reset_model_binding(key, clear_kv=True, invalidate=False)
        finally:
            for destination, snapshot in saved:
                ttnn.copy(snapshot, destination)
        self.prepared_bindings.update(new)
        self.warm_program_counts.append(
            {"reset_before": before, "reset_after": self.device.num_program_cache_entries()}
        )

    def _reset_model_binding(self, key, *, clear_kv, invalidate=True, mapping=None):
        bound = self.runtime.bindings[key]
        slot = bound["slot"]
        for index, (layer, states) in enumerate(zip(self.model.layers, bound["cache"], strict=True)):
            state = states[slot]
            if layer.kind == "linear_attention":
                history = bound["layers"][index]["history"]
                for name in ("conv", "recurrent"):
                    layer.write_linear_state(state, name, history[name + "_zero"])
            elif clear_kv:
                global_pages = bool(getattr(state, "global_pages", False))
                page_count = (bound["context"] + 31) // 32 if global_pages else state.page_count or state.key.shape[0]
                if global_pages and mapping is not None:
                    host = self.model.upload(
                        torch.tensor([mapping], dtype=torch.int32),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=False,
                    )
                    ttnn.copy_host_to_device_tensor(host, self.reset_tables[(key, index)])
                for tensor in (state.key, state.value):
                    _, heads, block, width = tuple(tensor.shape)
                    zero = self.kv_zero_inputs[(page_count, heads, block, width, tensor.dtype)]
                    ttnn.experimental.paged_fill_cache(tensor, zero, self.reset_tables[(key, index)])
        if invalidate:
            self.runtime.reset(cache=bound["cache"], slots=(slot,))

    def reset_slots(self, *, cache=None, slots, clear_kv=False, reset_sampling_state=True, page_tables=None):
        """Reset selected request regions; no pool replacement or peer clearing."""
        cache = self.gen.cache if cache is None else cache
        slots = self._integers(slots, "slots")
        if len(set(slots)) != len(slots):
            raise ValueError("Reset slots must be unique")
        keys = [(id(cache), slot) for slot in slots]
        if any(key not in self.prepared_bindings for key in keys):
            raise ValueError("Reset requires preregistered cache slots")
        mappings = {}
        for key in keys:
            bound = self.runtime.bindings[key]
            for layer, states in zip(self.model.layers, cache, strict=True):
                state = states[bound["slot"]]
                if clear_kv and layer.kind == "full_attention" and getattr(state, "global_pages", False):
                    page_count = (bound["context"] + 31) // 32
                    populated = ((bound["end"] or 0) + 31) // 32
                    if populated == 0:
                        # An unknown/fresh request owns no populated pages.
                        # Future aliases are attention-safe addresses, not
                        # permission to clear their physical backing.
                        mappings[key] = [-1] * page_count
                        continue
                    if page_tables is None or bound["slot"] not in page_tables:
                        raise ValueError("Clearing a global-page request requires its explicit owned page mapping")
                    mapping = self._integers(page_tables[bound["slot"]], "reset page mapping")
                    if len(mapping) != page_count or any(page < -1 or page >= state.key.shape[0] for page in mapping):
                        raise ValueError("Invalid global-page reset mapping")
                    causal = mapping[:populated]
                    if any(page < 0 for page in causal) or len(set(causal)) != len(causal):
                        raise ValueError("Populated global pages must be allocated and unique before clearing")
                    mappings[key] = causal + [-1] * (page_count - populated)
        for key in keys:
            self._reset_model_binding(key, clear_kv=clear_kv, mapping=mappings.get(key))
        if reset_sampling_state and slots:
            self.sampling_state.reset_history(slots=slots, prompt_tokens=None, output_tokens=None)

    def _put_row(self, logits, slot):
        if tuple(logits.shape) != (1, 1, 32, self.local_vocab):
            raise ValueError("Prefill head must expose the prepared 32-row sampler shape")
        row_major = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        # The read offset/shape is constant. Only slice_write's runtime outer
        # offset varies across slots; its backing output is already RM.
        first = ttnn.slice(row_major, (0, 0, 0, 0), (1, 1, 1, self.local_vocab))
        ttnn.experimental.slice_write(
            first,
            self.logits_rm,
            (0, 0, slot, 0),
            (1, 1, slot + 1, self.local_vocab),
            (1, 1, 1, 1),
        )

    def _finish_assembly(self):
        ttnn.copy(ttnn.to_layout(self.logits_rm, ttnn.TILE_LAYOUT), self.logits)
        return self.logits

    def _warm_assembly(self):
        before = self.device.num_program_cache_entries()
        ttnn.copy(self.zero_rm, self.logits_rm)
        self._put_row(self.logits, 0)
        self._finish_assembly()
        ttnn.copy(self.zero_rm, self.logits_rm)
        self._finish_assembly()
        self.warm_program_counts.append(
            {"assembly_before": before, "assembly_after": self.device.num_program_cache_entries()}
        )

    @staticmethod
    def _integers(values, name):
        if isinstance(values, torch.Tensor):
            if (
                values.device.type != "cpu"
                or values.ndim != 1
                or values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            ):
                raise TypeError(f"{name} must be a one-dimensional host integer sequence")
            values = values.tolist()
        result = list(values)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
            raise TypeError(f"{name} must contain integers")
        return [int(value) for value in result]

    def forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        empty_slots=None,
        request_lens=None,
        discard_logits=None,
        return_all_logits=False,
        sample_on_device=False,
        allow_missing_history=False,
        reset_new_requests=True,
        clear_kv=False,
        reset_sampling_state=True,
        validate_only=False,
        **unknown,
    ):
        """tokens[i,start_pos[i]:prompt_lens[i]] at absolute cache positions.

        Inactive rows have prompt_lens=0 and start_pos=0, with slot=-1 or an
        otherwise valid fixed slot. They never reset/mutate model or sampler
        state. Device output is PrefillBatch; host output keeps input row order,
        returning a tensor for equal lengths and a list for mixed lengths.
        All prefill rows are processed serially by the same optimized runtime;
        this is mixed-batch semantics, not a concurrent prefill kernel claim.
        Explicit discard_logits rows still advance model/cache state, but their
        terminal head is skipped and the suppressed output is replaced by host zeros.
        The caller must guarantee those rows' outputs are suppressed.
        """
        if unknown:
            raise TypeError(f"Unsupported prefill options: {', '.join(sorted(unknown))}")
        if any(
            type(flag) is not bool
            for flag in (
                return_all_logits,
                sample_on_device,
                allow_missing_history,
                reset_new_requests,
                clear_kv,
                reset_sampling_state,
                validate_only,
            )
        ):
            raise TypeError("Prefill mode flags must be bool")
        if return_all_logits and sample_on_device:
            raise ValueError("All-logit requests use the explicit host compatibility path")
        if not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu" or tokens.ndim != 2:
            raise TypeError("tokens must be a two-dimensional host integer tensor")
        if tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("tokens must have integer dtype")
        ends = self._integers(prompt_lens, "prompt_lens")
        rows = len(ends)
        if not 1 <= rows <= 32 or tokens.shape[0] != rows:
            raise ValueError("Batch rows must match prompt_lens and be between 1 and 32")
        if discard_logits is None:
            discard_logits = [False] * rows
        if not isinstance(discard_logits, (list, tuple)) or any(type(flag) is not bool for flag in discard_logits):
            raise TypeError("discard_logits must be a sequence of bools")
        if len(discard_logits) != rows:
            raise ValueError("discard_logits must match the logical batch")
        if any(discard_logits) and (sample_on_device or return_all_logits):
            raise ValueError("Discarded logits require unsampled last-logit prefill")
        if any(discard and end == 0 for discard, end in zip(discard_logits, ends)):
            raise ValueError("Only active prefill rows can discard logits")
        starts = [0] * rows if start_pos is None else self._integers(start_pos, "start_pos")
        slots = list(range(rows)) if empty_slots is None else self._integers(empty_slots, "empty_slots")
        horizons = ends.copy() if request_lens is None else self._integers(request_lens, "request_lens")
        if any(len(values) != rows for values in (starts, slots, horizons)):
            raise ValueError("Prompt/start/slot/horizon vectors must have equal length")
        if not isinstance(page_table, torch.Tensor) or page_table.device.type != "cpu" or page_table.ndim != 2:
            raise TypeError("page_table must be a two-dimensional host integer tensor")
        if page_table.shape[0] != rows or page_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("Page-table rows/dtype must match the logical batch")
        nonnegative_slots = [slot for slot in slots if slot >= 0]
        if len(set(nonnegative_slots)) != len(nonnegative_slots):
            raise ValueError("Fixed slots must be unique, including inactive placeholders")
        active, admissions, rebuilds = [], [], set()
        for row, (start, end, slot, horizon) in enumerate(zip(starts, ends, slots, horizons, strict=True)):
            if end == 0:
                if start != 0 or slot < -1 or slot >= 32 or not 0 <= horizon <= self.gen.max_seq_len:
                    raise ValueError("Invalid inactive row")
                continue
            if not 0 <= slot < 32 or not 0 <= start < end <= tokens.shape[1]:
                raise ValueError("Invalid active slot/start/end")
            if bool(((tokens[row, :end] < 0) | (tokens[row, :end] >= self.model.vocab_size)).any()):
                raise ValueError("Invalid token ID in the active causal prefix")
            key = (id(kv_cache), slot)
            if key not in self.prepared_bindings:
                raise ValueError("External cache/slot must be prepared before admission")
            ids = tokens[row, start:end].tolist()
            bound = self.runtime.bindings[key]
            if start and not allow_missing_history and bound["end"] == start:
                incompatible = any(
                    not prefix_profiles_compatible(
                        cached_total_length=bound["total_length"],
                        requested_total_length=horizon,
                        prefix_end=start,
                        kind=layer.kind,
                    )
                    for layer in self.model.layers
                )
                no_boundary = start % 32 and any(
                    item["history"]["boundary"] is None or item["history"]["boundary"].tail_end != start
                    for item in bound["layers"]
                )
                if incompatible or no_boundary:
                    # A previously unknown horizon can become known, or grow
                    # into a different precision/geometry profile. Recompute its
                    # prefix through the same device path; do not reuse cache
                    # produced under an incompatible shape-dependent policy.
                    self.runtime.prefill(
                        tokens[row, :start].tolist(),
                        slot=slot,
                        start=0,
                        total_length=horizon,
                        page_table=page_table[row],
                        cache=kv_cache,
                        last_only=True,
                        discard_logits=True,
                        validate_only=True,
                    )
                    rebuilds.add(row)
            # Validate the whole batch before resetting or computing any row.
            self.runtime.prefill(
                ids,
                slot=slot,
                start=start,
                total_length=horizon,
                page_table=page_table[row],
                cache=kv_cache,
                last_only=not return_all_logits,
                allow_missing_history=allow_missing_history or row in rebuilds,
                discard_logits=discard_logits[row],
                validate_only=True,
            )
            active.append(row)
            if (start == 0 and reset_new_requests) or row in rebuilds:
                admissions.append(slot)
        if validate_only:
            return {
                "active_slots": tuple(slots[row] for row in active),
                "input_rows": tuple(active),
                "row_slots": tuple(slots),
            }
        if admissions:
            self.reset_slots(
                cache=kv_cache,
                slots=admissions,
                clear_kv=clear_kv,
                reset_sampling_state=False,
                page_tables={slots[row]: page_table[row] for row in active},
            )
        if reset_sampling_state and active:
            self.sampling_state.reset_history(
                slots=[slots[row] for row in active],
                prompt_tokens=[tokens[row, : ends[row]].tolist() for row in active],
                output_tokens=None,
            )
        if sample_on_device:
            ttnn.copy(self.zero_rm, self.logits_rm)
        results = [torch.empty((1, 0, self.model.vocab_size), dtype=torch.float32) for _ in range(rows)]
        for row in active:
            if row in rebuilds:
                discarded = self.runtime.prefill(
                    tokens[row, : starts[row]].tolist(),
                    slot=slots[row],
                    start=0,
                    total_length=horizons[row],
                    page_table=page_table[row],
                    cache=kv_cache,
                    last_only=True,
                    return_device_logits=True,
                    discard_logits=True,
                )
                del discarded
            result = self.runtime.prefill(
                tokens[row, starts[row] : ends[row]].tolist(),
                slot=slots[row],
                start=starts[row],
                total_length=horizons[row],
                page_table=page_table[row],
                cache=kv_cache,
                last_only=not return_all_logits,
                allow_missing_history=allow_missing_history,
                return_device_logits=sample_on_device or discard_logits[row],
                discard_logits=discard_logits[row],
            )
            if discard_logits[row]:
                # The serving scheduler suppresses this intermediate row. Keep
                # its required host shape without transferring device logits or
                # advancing any sampler state, including in a mixed admission.
                del result
                results[row] = torch.zeros((1, 1, self.model.vocab_size), dtype=torch.float32)
            elif sample_on_device:
                self._put_row(result, slots[row])
                del result
            else:
                results[row] = result
        self.events.append(
            {
                "rows": rows,
                "active_slots": [slots[row] for row in active],
                "starts": starts,
                "ends": ends,
                "horizons": horizons,
                "device_logits": sample_on_device,
                "discard_logits": list(discard_logits),
                "rebuilt_prefix_rows": sorted(rebuilds),
            }
        )
        if sample_on_device:
            return PrefillBatch(
                self._finish_assembly(),
                tuple(slots[row] for row in active),
                tuple(active),
                tuple(ends),
                tuple(starts),
                tuple(horizons),
                tuple(slots),
            )
        if not return_all_logits:
            inactive = set(range(rows)) - set(active)
            for row in inactive:
                results[row] = torch.full((1, 1, self.model.vocab_size), float("-inf"))
        return torch.cat(results, dim=0) if len({result.shape[1] for result in results}) == 1 else results
