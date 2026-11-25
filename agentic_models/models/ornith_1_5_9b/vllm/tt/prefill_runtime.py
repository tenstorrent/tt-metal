# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent canonical-profile prefill staging and aligned-boundary history.

The generator prepares every configured cache/slot/profile before capture. Calls
are serialized through shared staging; each cache/slot retains independent
per-layer input tails and recurrent boundary snapshots. No device state is
allocated lazily at request admission. Host logits are an explicit diagnostic
boundary; last_only + return_device_logits returns the physical 32-row terminal
output for the generator's common sampler, with the requested token in row zero.

RM slice hashes absolute starts/ends in this checkout. Runtime row reads instead
use embedding over BF16 RM staging and fixed-shape index tensors. RM slice_write
excludes outer offsets from its hash; both its arguments stay RM to preserve the
staging allocation. Original 32768/16384 layer chunk caps remain unchanged.
Large canonical profiles may execute at the already-prepared 8192-row extent
when all consumed rows fit. Both extents retain the large-prefill numerical
branches; request horizon, boundary snapshots and logical page ownership remain
canonical. This policy has full-stack exact-comparison and serving gates.
"""

import os
from dataclasses import asdict

import torch

import ttnn
from agentic_models.models.ornith_1_5_9b.vllm.tt.prefill_ops import (
    NormalizedAttentionPrefill,
    RequestProfileLinear,
    make_attention_inputs,
)
from agentic_models.models.ornith_1_5_9b.vllm.tt.prefill_plan import (
    CHUNK_CAPS,
    FULL_PREFILL_BUCKETS,
    LINEAR_BUCKETS,
    physical_bucket,
    plan_prefill,
    prefix_profiles_compatible,
)

# Compact continuation buckets (see PrefillRuntime._execution_height); "0" restores canonical whole-chunk buckets.
COMPACT = os.environ.get("ORNITH_COMPACT_PREFILL", "1") == "1"


class PrefillRuntime:
    @classmethod
    def prepare(cls, gen, warmup_buckets=None, *, bindings=None):
        """Prepare owned slots plus optional {cache, slots, context} bindings.

        Omit warmup_buckets to cover every profile selectable by configured
        context. A supplied subset is a diagnostic selection, not a smaller
        advertised context. A request requiring an unprepared profile fails
        before touching state. Register every external binding before capture.
        """
        existing = getattr(gen, "prefill_runtime", None)
        if existing is not None:
            if warmup_buckets is not None and existing.profile_spec != existing._normalize_spec(warmup_buckets):
                raise ValueError("Profile configuration changed; reconfigure the model before serving")
            for binding in bindings or ():
                existing.register_cache(**binding)
            return existing
        cls._require_no_traces(gen)
        all_bindings = [{"cache": gen.cache, "slots": range(gen.max_batch_size), "context": gen.max_seq_len}]
        all_bindings.extend(bindings or ())
        context = max(binding.get("context", gen.max_seq_len) for binding in all_bindings)
        if not 0 < context <= gen.model.config.max_position_embeddings:
            raise ValueError("Registered context exceeds the model's RoPE/capability contract")
        if warmup_buckets is None:
            warmup_buckets = {}
            for kind, key, choices in (
                ("linear_attention", "linear", LINEAR_BUCKETS),
                ("full_attention", "full", FULL_PREFILL_BUCKETS),
            ):
                largest = physical_bucket(kind, min(context, CHUNK_CAPS[kind]))
                warmup_buckets[key] = [height for height in choices if height <= largest]
        obj = cls(gen, context, cls._normalize_spec(warmup_buckets))
        for binding in all_bindings:
            obj.register_cache(**binding, warm=False)
        obj._warm(list(obj.bindings.values()), common=True)
        obj.prepared = True
        gen.prefill_runtime = obj
        return obj

    @staticmethod
    def _normalize_spec(spec):
        return {key: tuple(sorted(set(spec.get(key, ())))) for key in ("linear", "full")}

    @staticmethod
    def _require_no_traces(gen):
        if any(state.model_trace is not None or state.sample_trace is not None for state in gen.states.values()):
            raise RuntimeError("Prefill bindings and profiles must be prepared before the first capture")

    def __init__(self, gen, context, profile_spec):
        self.gen, self.model, self.device = gen, gen.model, gen.device
        self.context, self.capacity, self.profile_spec = context, context + 32768, profile_spec
        if any(layer.hidden_width != 4096 or layer.residual_layout != "replicated" for layer in self.model.layers):
            raise ValueError("Prefill requires the reviewed BF16 replicated residual layout")
        for key, allowed in (("linear", LINEAR_BUCKETS), ("full", FULL_PREFILL_BUCKETS)):
            if any(height not in allowed for height in profile_spec[key]):
                raise ValueError(f"Unknown physical {key} profile")
        for layer in self.model.layers:
            key = "linear" if layer.kind == "linear_attention" else "full"
            if not profile_spec[key]:
                raise ValueError(f"At least one {key} physical profile must be prepared")
        self.staging = [self._zeros((1, 1, self.capacity, 4096)) for _ in range(2)]
        self.tokens = self._integer(torch.zeros(1, 1024, dtype=torch.int32))
        heights = {32, *profile_spec["linear"], *profile_spec["full"]}
        self.row_indices = {height: self._integer(torch.arange(height, dtype=torch.int32)[None]) for height in heights}
        # Shared per-height index tensors are initialized to arange(height).
        self._row_indices_absolute = {height: 0 for height in heights}
        self.tail_zero = self._zeros((1, 1, 32, 4096))
        self.overlap_mask = self.model.upload(torch.zeros(1, 1, 32, 1))
        self.overlap_inverse = self.model.upload(torch.ones(1, 1, 32, 1))
        self.linear = {
            index: RequestProfileLinear(layer, buckets=profile_spec["linear"])
            for index, layer in enumerate(self.model.layers)
            if layer.kind == "linear_attention"
        }
        self.bindings, self.events, self.warm_events = {}, [], []
        self.prepared = False
        self.metadata = {
            "staging_bytes_per_chip": 2 * self.capacity * 4096 * 2,
            "profiles": profile_spec,
            "physical_token_rows": 1024,
            "physical_head_rows": 32,
            "context": context,
            "history_rows_per_layer_slot": 32,
        }

    def _zeros(self, shape, *, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.zeros(shape, dtype=dtype, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _integer(self, value, *, signed=False, device=True):
        return self.model.upload(
            value,
            dtype=ttnn.int32 if signed else ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def _copy_integer(self, value, destination, *, signed=False):
        ttnn.copy_host_to_device_tensor(self._integer(value, signed=signed, device=False), destination)

    @staticmethod
    def _fingerprint(cache, slot):
        return tuple(
            (
                id(states[slot]),
                *(id(getattr(states[slot], name)) for name in ("conv", "recurrent", "key", "value")),
                states[slot].page_offset,
                states[slot].page_count,
                states[slot].state_row,
                bool(getattr(states[slot], "global_pages", False)),
            )
            for states in cache
        )

    def register_cache(self, cache, *, slots=None, context=None, warm=True):
        """Register external backing/slot bindings before capture; preserve content."""
        context = context or self.gen.max_seq_len
        slots = tuple(range(len(cache[0]))) if slots is None else tuple(slots)
        if len(cache) != len(self.model.layers) or not 0 < context <= self.context:
            raise ValueError("External cache layer count/context exceeds prepared staging configuration")
        additions = []
        for slot in slots:
            if not all(0 <= slot < len(states) for states in cache):
                raise ValueError("Invalid cache slot")
            key = (id(cache), slot)
            fingerprint = self._fingerprint(cache, slot)
            if key in self.bindings:
                bound = self.bindings[key]
                if bound["fingerprint"] != fingerprint or bound["context"] != context:
                    raise ValueError("Registered backing storage or context changed")
                continue
            self._require_no_traces(self.gen)
            layers = []
            for index, (layer, states) in enumerate(zip(self.model.layers, cache, strict=True)):
                if layer.chunk_size != CHUNK_CAPS[layer.kind]:
                    raise ValueError("Do not replace the reviewed per-layer chunk caps")
                state = states[slot]
                history = {"tail": self._zeros((1, 1, 32, 4096)), "boundary": None}
                if layer.kind == "linear_attention":
                    for name in ("conv", "recurrent"):
                        local = layer.read_linear_state(state, name)
                        history[name] = ttnn.zeros_like(local)
                        history[name + "_zero"] = ttnn.zeros_like(local)
                    adapters = self.linear[index]
                else:
                    adapters = {
                        height: NormalizedAttentionPrefill(
                            layer, state=state, physical_bucket=height, supported_context=context
                        )
                        for height in self.profile_spec["full"]
                    }
                layers.append({"history": history, "adapters": adapters})
            bound = {
                "cache": cache,
                "slot": slot,
                "context": context,
                "fingerprint": fingerprint,
                "layers": layers,
                "end": None,
                "total_length": None,
            }
            self.bindings[key] = bound
            additions.append(bound)
        if warm and additions:
            self._warm(additions, common=False)
        return self

    def reset(self, *, cache=None, slots=None):
        """Invalidate request history; cache content reset remains caller-owned."""
        cache = self.gen.cache if cache is None else cache
        for (cache_id, slot), bound in self.bindings.items():
            if cache_id == id(cache) and (slots is None or slot in slots):
                bound["end"] = bound["total_length"] = None
                for item in bound["layers"]:
                    item["history"]["boundary"] = None

    def remap_slots(self, cache, remap):
        """Simultaneous history ownership permutation: destination <- remap[destination].

        The caller separately moves recurrent cache state and supplies the new
        physical page tables. These persistent input tails/boundary snapshots
        are consumed only by eager prefill, never captured decode, so rebinding
        ownership preserves every allocation without device copies. Slot-bound
        adapters and their cache/page-offset bindings do not move.
        """
        remap = tuple(int(source) for source in remap)
        if sorted(remap) != list(range(len(remap))):
            raise ValueError("History remap must be a complete permutation")
        bounds = [self.bindings.get((id(cache), slot)) for slot in range(len(remap))]
        if any(bound is None for bound in bounds):
            raise ValueError("Every remapped slot must have been registered before capture")
        if len({bound["context"] for bound in bounds}) != 1:
            raise ValueError("History remap requires matching slot context capacities")
        old = [(bound["end"], bound["total_length"], [item["history"] for item in bound["layers"]]) for bound in bounds]
        for destination, source in enumerate(remap):
            bound = bounds[destination]
            bound["end"], bound["total_length"], histories = old[source]
            for item, history in zip(bound["layers"], histories, strict=True):
                item["history"] = history

    def _read(self, source, absolute, height):
        if absolute < 0 or absolute + height > self.capacity:
            raise ValueError("Physical read exceeds configured staging slack")
        # Layers/slots share this runtime index tensor. A different source does
        # not change its row addresses; _read is its only post-construction writer.
        if self._row_indices_absolute.get(height) != absolute:
            self._row_indices_absolute.pop(height, None)
            self._copy_integer(
                torch.arange(absolute, absolute + height, dtype=torch.int32)[None], self.row_indices[height]
            )
            self._row_indices_absolute[height] = absolute
        rows = ttnn.embedding(self.row_indices[height], source, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(rows, (1, 1, height, 4096))

    @staticmethod
    def _write(value, destination, absolute):
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(
            value, destination, (0, 0, absolute, 0), (1, 1, absolute + value.shape[2], 4096), (1, 1, 1, 1)
        )

    def _restore_inputs(self, source, history, boundary):
        overlap = boundary.tail_rows
        mask = (torch.arange(32) < overlap).float().reshape(1, 1, 32, 1)
        for value, target in ((mask, self.overlap_mask), (1 - mask, self.overlap_inverse)):
            ttnn.copy_host_to_device_tensor(self.model.upload(value, device=False), target)
        live = self._read(source, boundary.boundary, 32)
        saved = ttnn.to_layout(history["tail"], ttnn.TILE_LAYOUT)
        merged = ttnn.add(ttnn.multiply(saved, self.overlap_mask), ttnn.multiply(live, self.overlap_inverse))
        self._write(merged, source, boundary.boundary)

    def _save_boundary(self, index, bound, source, boundary):
        history = bound["layers"][index]["history"]
        tail = ttnn.to_layout(self._read(source, boundary.boundary, 32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(tail, history["tail"])
        layer, state = self.model.layers[index], bound["cache"][index][bound["slot"]]
        if layer.kind == "linear_attention":
            for name in ("conv", "recurrent"):
                ttnn.copy(layer.read_linear_state(state, name), history[name])
        history["boundary"] = boundary

    def _restore_boundary(self, index, bound, source, boundary, *, missing=False):
        history = bound["layers"][index]["history"]
        if missing:
            # Explicit negative control: never borrow tail/state from a previous request.
            ttnn.copy(self.tail_zero, history["tail"])
            for name in ("conv", "recurrent"):
                if name in history:
                    ttnn.copy(history[name + "_zero"], history[name])
        elif history["boundary"] != boundary:
            raise ValueError("Missing/mismatched per-layer canonical boundary history")
        self._restore_inputs(source, history, boundary)
        layer, state = self.model.layers[index], bound["cache"][index][bound["slot"]]
        if layer.kind == "linear_attention":
            for name in ("conv", "recurrent"):
                layer.write_linear_state(state, name, history[name])

    def _execution_height(self, kind, height, valid):
        # Never fall into the <=4096 precision or <=1024 matmul branches.
        # Custom preparation catalogues may omit 8192; retain their old extent.
        key = "linear" if kind == "linear_attention" else "full"
        if COMPACT:
            # Compact continuations (prefix-cache resumes, chunked and split prefills): execute a segment at the
            # smallest prepared bucket holding its own rows instead of its canonical whole-chunk bucket. A 300-token
            # continuation of a 1.6K prompt otherwise runs 4096 rows. The smaller bucket selects the same precision
            # and matmul branches a prompt of that length uses; results differ from the canonical profile only in
            # rounding (validated by prefill-window KL vs HF fp32), so bit-equality with an unsplit prefill is
            # traded for the resume cost.
            need = physical_bucket(kind, min(valid, CHUNK_CAPS[kind]))
            compact = min((h for h in self.profile_spec[key] if h >= need), default=height)
            if compact < height:
                return compact
        if height in (16384, 32768) and valid <= 8192 and 8192 in self.profile_spec[key]:
            return 8192
        return height

    def _execute(self, index, bound, source, destination, *, absolute, valid, height, mapping):
        layer, state = self.model.layers[index], bound["cache"][index][bound["slot"]]
        adapter = bound["layers"][index]["adapters"]
        height = self._execution_height(layer.kind, height, valid)
        hidden = self._read(source, absolute, height)
        if layer.kind == "linear_attention":
            adapter.refresh_profile(valid_count=valid, bucket=height)
            result = adapter.forward(hidden, state=state)
        else:
            adapter[height].refresh(start=absolute, valid_count=valid, page_mapping=mapping)
            result = adapter[height].forward(hidden)
        self._write(result, destination, absolute)

    def _embed(self, ids, start):
        for offset in range(0, len(ids), 1024):
            physical = torch.zeros(1, 1024, dtype=torch.int32)
            chunk = ids[offset : offset + 1024]
            physical[0, : len(chunk)] = torch.as_tensor(chunk, dtype=torch.int32)
            self._copy_integer(physical, self.tokens)
            self._write(self.model.embed(self.tokens), self.staging[0], start + offset)

    def _head(self, source, start, count, *, last_only, return_device_logits=False):
        if return_device_logits and not last_only:
            raise ValueError("Device logits require last_only=True and retain fixed 32-row shape")
        positions = [start + count - 1] if last_only else range(start, start + count, 32)
        results = []
        for absolute in positions:
            logits = self.model.logits(self._read(source, absolute, 32))
            if return_device_logits:
                return logits
            host = self.model.host_logits(logits).reshape(1, 32, self.model.vocab_size)
            real = 1 if last_only else min(32, start + count - absolute)
            results.append(host[:, :real].clone())
        return torch.cat(results, dim=1)

    def _warm(self, bindings, *, common):
        from agentic_models.models.ornith_1_5_9b.vllm.tt.prefill_plan import PROFILE_VERSION, BoundaryState, ProfileTag

        saved, seen = [], set()
        for bound in bindings:
            for states in bound["cache"]:
                for name in ("conv", "recurrent", "key", "value"):
                    tensor = getattr(states[bound["slot"]], name)
                    if tensor is not None and id(tensor) not in seen:
                        saved.append((tensor, ttnn.clone(tensor)))
                        seen.add(id(tensor))
        before = self.device.num_program_cache_entries()
        try:
            if common:
                self._embed([0], 0)
            for bound in bindings:
                mapping = list(range((bound["context"] + 31) // 32))
                for index, layer in enumerate(self.model.layers):
                    item = bound["layers"][index]
                    if layer.kind == "linear_attention":
                        for name in ("conv", "recurrent"):
                            layer.write_linear_state(
                                bound["cache"][index][bound["slot"]], name, item["history"][name + "_zero"]
                            )
                    key = "linear" if layer.kind == "linear_attention" else "full"
                    for height in self.profile_spec[key]:
                        print(
                            f"PREFILL_WARM layer={index} kind={layer.kind} slot={bound['slot']} "
                            f"bucket={height} valid={min(height, bound['context'])} context={bound['context']}",
                            flush=True,
                        )
                        self._execute(
                            index,
                            bound,
                            self.staging[0],
                            self.staging[1],
                            absolute=0,
                            valid=min(height, bound["context"]),
                            height=height,
                            mapping=mapping,
                        )
                    tag = ProfileTag(PROFILE_VERSION, layer.kind, 0, self.profile_spec[key][0])
                    boundary = BoundaryState(0, 1, tag, layer.kind == "linear_attention")
                    self._save_boundary(index, bound, self.staging[0], boundary)
                    self._restore_boundary(index, bound, self.staging[0], boundary)
                    self._restore_boundary(index, bound, self.staging[0], boundary, missing=True)
                    item["history"]["boundary"] = None
            if common:
                self._head(self.staging[1], 0, 1, last_only=True)
        finally:
            for destination, snapshot in saved:
                ttnn.copy(snapshot, destination)
        ttnn.synchronize_device(self.device)
        self.warm_events.append(
            {"before": before, "after": self.device.num_program_cache_entries(), "bindings": len(bindings)}
        )

    def prefill(
        self,
        ids,
        *,
        slot=0,
        start=0,
        total_length=None,
        page_table=None,
        cache=None,
        last_only=False,
        allow_missing_history=False,
        return_device_logits=False,
        discard_logits=False,
        validate_only=False,
    ):
        """Run one logical segment; total_length fixes profiles for its whole request.

        page_table is a host local-page mapping, not a device tensor. For pooled
        caches each layer adds its owned region offset before inserting -1 fill
        sentinels. Request scheduling and reset remain generator-owned.
        discard_logits returns None after all cache/history writes; callers must
        suppress this intermediate or rebuilt-prefix output. Terminal preparation
        is unchanged, so later final logits use already-prepared head programs.
        """
        if type(discard_logits) is not bool:
            raise TypeError("discard_logits must be bool")
        if discard_logits and not last_only:
            raise ValueError("Discarded logits require last_only=True")
        if not self.prepared:
            raise RuntimeError("Global prefill preparation has not completed")
        if return_device_logits and not last_only:
            raise ValueError("Device logits require last_only=True")
        cache = self.gen.cache if cache is None else cache
        bound = self.bindings.get((id(cache), slot))
        if bound is None or bound["fingerprint"] != self._fingerprint(cache, slot):
            raise ValueError("Unregistered or replaced cache/slot binding")
        ids = [int(token) for token in ids]
        total_length = start + len(ids) if total_length is None else int(total_length)
        end = start + len(ids)
        if not ids or not 0 <= start < end <= total_length <= bound["context"]:
            raise ValueError("Invalid logical prefill interval or horizon")
        if any(token < 0 or token >= self.model.vocab_size for token in ids):
            raise ValueError("Invalid token ID")
        if page_table is None:
            if cache is not self.gen.cache:
                raise ValueError("External cache calls require an explicit host page table")
            page_table = self.gen.page_table[slot]
        if isinstance(page_table, ttnn.Tensor):
            raise TypeError("Pass the scheduler's host page mapping; implicit device-table readback is forbidden")
        mapping = torch.as_tensor(page_table, dtype=torch.int64).reshape(-1).tolist()
        if len(mapping) != (bound["context"] + 31) // 32:
            raise ValueError("Page table capacity differs from registered binding")
        plans = [
            plan_prefill(total_length=total_length, start=start, end=end, kind=layer.kind)
            for layer in self.model.layers
        ]
        if start and not allow_missing_history:
            if bound["end"] != start:
                raise ValueError("Continuation must follow this binding's exact prior end")
            for layer in self.model.layers:
                if not prefix_profiles_compatible(
                    cached_total_length=bound["total_length"],
                    requested_total_length=total_length,
                    prefix_end=start,
                    kind=layer.kind,
                ):
                    raise ValueError("Canonical profiles differ from the cached prefix")
        # Complete all host validation before touching request state.
        for index, (layer, plan) in enumerate(zip(self.model.layers, plans, strict=True)):
            key = "linear" if layer.kind == "linear_attention" else "full"
            if any(segment.physical_bucket not in self.profile_spec[key] for segment in plan.segments):
                raise ValueError("Unprepared physical profile; prepare all configured variants before capture")
            for segment in plan.segments:
                if layer.kind == "full_attention":
                    adapter = bound["layers"][index]["adapters"][segment.physical_bucket]
                    make_attention_inputs(
                        physical_bucket=segment.physical_bucket,
                        supported_context=bound["context"],
                        local_page_count=adapter.page_count,
                        start=segment.absolute_aligned_start,
                        valid_count=segment.valid_count,
                        page_mapping=mapping,
                        page_offset=adapter.page_offset,
                    )
                needed = segment.required_cached_boundary
                if (
                    needed is not None
                    and not allow_missing_history
                    and bound["layers"][index]["history"]["boundary"] != needed
                ):
                    raise ValueError("Missing per-layer tail/state for unaligned continuation")
        if validate_only:
            return {"slot": slot, "start": start, "end": end, "total_length": total_length}
        if start == 0:
            self.reset(cache=cache, slots=(slot,))
        self._embed(ids, start)
        source, destination = self.staging
        for index, plan in enumerate(plans):
            for segment in plan.segments:
                if segment.required_cached_boundary is not None:
                    self._restore_boundary(
                        index, bound, source, segment.required_cached_boundary, missing=allow_missing_history
                    )
                if segment.save_boundary_before is not None:
                    self._save_boundary(index, bound, source, segment.save_boundary_before)
                self._execute(
                    index,
                    bound,
                    source,
                    destination,
                    absolute=segment.absolute_aligned_start,
                    valid=segment.valid_count,
                    height=segment.physical_bucket,
                    mapping=mapping,
                )
            if plan.segments[-1].save_boundary_before is None:
                bound["layers"][index]["history"]["boundary"] = None
            source, destination = destination, source
        bound["end"], bound["total_length"] = end, total_length
        hook = getattr(self, "segment_hook", None)
        if hook is not None:  # MTP drafter prompt KV (tt/mtp_spec.py): final hidden rows are in `source`
            hook(source=source, slot=slot, ids=ids, start=start, end=end, mapping=mapping)
        self.events.append(
            {
                "slot": slot,
                "start": start,
                "end": end,
                "total_length": total_length,
                "missing_history_control": allow_missing_history,
                "discard_logits": discard_logits,
                "plans": [asdict(plan) for plan in plans],
            }
        )
        if discard_logits:
            return None
        return self._head(source, start, len(ids), last_only=last_only, return_device_logits=return_device_logits)
