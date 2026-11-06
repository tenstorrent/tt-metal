# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for the selected full-model generator.

Attention KV belongs to vLLM. Linear-attention state remains slot-owned here.
The ordinary serving path samples exclusively through Generator's split trace.
"""

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn

from .benchmark_phases import PhaseRecorder
from .functional_decoder import PagedState
from .generator import Generator
from .model import Model


@dataclass
class DecodeOutput:
    tensor: object
    is_tokens: bool
    rows: int
    phase: object = None


class Qwen38ForCausalLM:
    decode_input_update_contract = 1
    model_capabilities = {
        "supports_sample_on_device": True,
        "supports_async_decode": True,  # adapter_b1/b8: deferred read, resident input and control parity.
        "supports_chunked_prefill": True,
        "supports_device_prefill_sampling_mask": True,
        "supports_prefix_caching": False,  # recurrent_state exclusion
        "supports_device_penalties": False,
        "max_device_top_k": 32,
    }

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 32:
            raise ValueError("This TP4 adapter supports one lane with 1..32 slots")
        checkpoint = os.environ.get("QWEN38_CHECKPOINT", getattr(hf_config, "_name_or_path", ".cache/qwen38-full"))
        indices = os.environ.get("QWEN38_SERVING_LAYERS")
        indices = [int(i) for i in indices.split(",")] if indices else None
        obj = cls()
        obj.mesh = mesh_device
        obj.checkpoint = checkpoint
        obj.batch = max_batch_size
        obj.context = max_seq_len
        ttnn.CONFIG.throw_exception_on_fallback = True
        obj.model = Model(checkpoint, mesh_device, layer_indices=indices)
        if max_seq_len != obj.model.config.max_position_embeddings:
            raise ValueError("Serving must preserve the supported context contract")
        obj.decode_positions = None
        obj.gen = None
        obj.cache = None
        obj.private_prefix_cache = None
        obj.model_capabilities = dict(cls.model_capabilities)
        obj.tt_supported_decode_batch_sizes = [max_batch_size]
        obj.serial_prefill = os.environ.get("QWEN38_SERIAL_PREFILL", "0") == "1"
        obj.allow_host_sampling = os.environ.get("QWEN38_HOST_SAMPLING_COMPAT", "0") == "1"
        obj.pages = torch.zeros(max_batch_size, ((max_seq_len + 255) // 256) * 8, dtype=torch.int32)
        obj.live_slots = set()
        obj.slot_ends = [0] * max_batch_size
        obj.prefill_provenance = [None] * max_batch_size
        obj.state_epochs = [0] * max_batch_size
        obj.seed_bases = [secrets.randbelow(1 << 30) for _ in range(max_batch_size)]
        phase_path = os.environ.get("QWEN38_PHASE_FILE")
        obj.phases = PhaseRecorder(phase_path, max_batch_size) if phase_path else None
        obj.event(
            "initialize",
            adapter=f"{cls.__module__}:{cls.__name__}",
            adapter_file=str(Path(__file__).resolve()),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            layers=obj.model.layer_indices,
            max_model_len=max_seq_len,
            max_num_seqs=max_batch_size,
            serial_prefill=obj.serial_prefill,
            throw_exception_on_fallback=ttnn.CONFIG.throw_exception_on_fallback,
            precision=obj.model.precision_config,
            memory=obj.memory(),
        )
        return obj

    @staticmethod
    def get_max_tokens_all_users(model_name, num_devices, tt_data_parallel, max_model_len, max_num_seqs):
        # Stage08 measured full64 B32 external-pool capacity supports this working
        # set. The plugin calls this before loading weights, so no live allocator
        # reading is available at this hook. Allocation/preparation logs reconcile it.
        return max(262144, min(max_num_seqs, 32) * 32768)

    def memory(self):
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        return dict(
            allocated_bytes_per_device=view.total_bytes_allocated_per_bank * view.num_banks,
            free_bytes_per_device=view.total_bytes_free_per_bank * view.num_banks,
        )

    def event(self, event, **fields):
        if self.gen is not None:
            fields["trace_lifetime"] = {
                key: self.gen.counters[key]
                for key in ("captures", "trace_retirements", "trace_evictions", "trace_invalidations")
            }
        print("QWEN38_SERVING " + json.dumps({"event": event, **fields}), flush=True)

    def allocate_kv_cache(self, shape, dtype, num_layers):
        if self.cache is not None:
            raise RuntimeError("KV cache allocation is setup-only")
        if tuple(shape[1:]) != (1, 32, 256):
            raise ValueError(f"Expected TP4 KV shape [pages,1,32,256], got {shape}")
        kv_dtype = getattr(ttnn, self.model.precision_config["kv_cache_dtype"])
        self.cache = []
        for layer in self.model.layers:
            if layer.layer_type == "linear_attention":
                state = layer.allocate_state(batch_size=self.batch)
            else:
                state = PagedState(
                    *[
                        ttnn.zeros(
                            shape,
                            dtype=kv_dtype,
                            layout=ttnn.TILE_LAYOUT,
                            device=self.mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        for _ in range(2)
                    ]
                )
            self.cache.append(state)
        self.gen = Generator(
            self.model,
            self.checkpoint,
            batch_size=self.batch,
            context_length=self.context,
            kv_cache=self.cache,
            page_table=self.pages,
        )
        self.gen.event_sink = self.event
        private_bytes = int(os.environ.get("QWEN38_PRIVATE_PREFIX_CACHE_BYTES", "0"))
        if private_bytes:
            if not 0 < private_bytes <= 10 << 30:
                raise ValueError("Private cache is bounded to at most10GiB/device")
            from .private_prefix_cache import PrivatePrefixCache

            self.private_prefix_cache = PrivatePrefixCache(self, private_bytes)
            self.event(
                "private_prefix_configuration", max_bytes=private_bytes, max_nodes=128, alignment=4096, native_apc=False
            )
        self.event(
            "allocate_cache",
            shape=list(shape),
            kv_dtype=str(kv_dtype),
            owns_cache=self.gen.owns_cache,
            pool_tokens=int(shape[0]) * 32,
            memory=self.memory(),
        )
        return self.cache

    def _check_cache(self, kv_cache):
        if kv_cache is not self.cache or self.gen is None:
            raise ValueError("Use the exact vLLM allocated cache")

    def warmup_model_prefill(self, *, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        # The generator owns whole-domain eager preparation followed by capture;
        # repeated plugin callbacks must neither rebuild nor retire those traces.
        self.gen.prepare_decode()
        self.event(
            "prepared",
            captures=self.gen.counters["captures"],
            decode_trace_ids={str(k): str(v) for k, v in self.gen.decode_traces.items()},
            sampling_trace=str(self.gen.sample_trace),
            trace_events=self.gen.trace_events,
            preparation_memory=self.gen.last_preparation_memory,
            prefill_buckets=sorted(self.gen.prepared_buckets),
            programs=self.mesh.num_program_cache_entries(),
        )

    def warmup_model_decode(self, *, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        self.gen.prepare_decode()

    def _pages_for(self, page_table, slots=None):
        if slots is None:
            slots = range(page_table.shape[0])
        updated = self.pages.clone()
        for row, slot in enumerate(slots):
            updated[slot].zero_()
            width = min(page_table.shape[1], updated.shape[1])
            updated[slot, :width] = page_table[row, :width].to(torch.int32)
        self.pages = updated
        return updated

    def _sampling(self, params, slots, *, reset_state, positions=None, active_slots=None):
        k = [1] * self.batch
        p = [0.0] * self.batch
        temp = [1.0] * self.batch
        seed = list(self.seed_bases)
        for row, slot in enumerate(slots):
            greedy = params.temperature[row] == 0
            k[slot] = 1 if greedy else params.top_k[row]
            p[slot] = 0.0 if greedy else params.top_p[row]
            temp[slot] = 1.0 if greedy else params.temperature[row]
            base = self.seed_bases[slot] if params.seed[row] is None else params.seed[row]
            seed[slot] = base + (max(0, int(positions[row])) if positions is not None else 0)
        self.gen.set_sampling_params(
            top_k=k, top_p=p, temperature=temp, seed=seed, reset_sampling_state=reset_state, active_slots=active_slots
        )

    def _invalidate_prefill_provenance(self, slots):
        for slot in slots:
            self.state_epochs[slot] += 1
            self.prefill_provenance[slot] = None

    def _remap_prefill_provenance(self, mapping, *, move_sessions=False):
        mapping = [int(source) for source in mapping]
        if getattr(self, "phases", None) is not None:
            self.phases.remap(mapping)
        self.prefill_provenance = [self.prefill_provenance[source] for source in mapping]
        self.state_epochs = [self.state_epochs[source] for source in mapping]
        self.seed_bases = [self.seed_bases[source] for source in mapping]
        if move_sessions:
            previous_ends = self.slot_ends
            self.slot_ends = [previous_ends[source] for source in mapping]
            self.live_slots = {dest for dest, source in enumerate(mapping) if source in self.live_slots}

    def _plan_prefill_state(self, tokens, page_table, slots, starts):
        """Prove history identity before reusing or moving any recurrent state.

        Prefill slots are destinations, not stable request IDs. Resolve every
        source before a fresh row can overwrite it. Ambiguous/missing provenance
        reconstructs from the supplied full history instead of guessing.
        """
        sources = []
        for row, start in enumerate(starts):
            matches = []
            if start and start % 128 == 0:
                page_count = (start + 31) // 32
                for source in sorted(self.live_slots):
                    proof = self.prefill_provenance[source]
                    if (
                        proof is not None
                        and proof["epoch"] == self.state_epochs[source]
                        and proof["end"] == start
                        and page_table.shape[1] >= page_count
                        and torch.equal(proof["tokens"], tokens[row, :start])
                        and torch.equal(proof["pages"], page_table[row, :page_count])
                    ):
                        matches.append(source)
            sources.append(matches[0] if len(matches) == 1 else None)
        # One old owner cannot authenticate two new requests, even if their
        # histories/page mappings happen to coincide. Rebuild ambiguous rows.
        duplicate_sources = {source for source in sources if source is not None and sources.count(source) > 1}
        sources = [None if source in duplicate_sources else source for source in sources]
        movable = set(slots) | {source for source in sources if source is not None}
        mapping = list(range(self.batch))
        chosen = {slot: source for slot, source in zip(slots, sources) if source is not None}
        remaining = sorted(movable - set(chosen.values()))
        for dest in sorted(movable):
            mapping[dest] = chosen[dest] if dest in chosen else remaining.pop(0)
        if sorted(mapping) != list(range(self.batch)):
            raise RuntimeError("Prefill ownership move must be a full permutation")
        # Outside this set every off-batch slot is an identity mapping. Gather
        # all chosen sources before reset/prefill can clobber overlapping slots.
        if mapping != list(range(self.batch)):
            self.gen.remap_state_slots(mapping)
            self._remap_prefill_provenance(mapping, move_sessions=True)
        vacated = movable - set(slots)
        self._invalidate_prefill_provenance(vacated)
        self.live_slots.difference_update(vacated)
        for slot in vacated:
            self.slot_ends[slot] = 0
            if getattr(self, "phases", None) is not None:
                self.phases.release(slot)
        if any(start and source is None for start, source in zip(starts, sources)):
            # Without a proven source, an old ownership alias may remain in a
            # different slot. Retire unclaimed proofs conservatively, without
            # touching any unrelated device state. Subsequent uncertain resumes
            # reconstruct rather than authenticating stale recycled pages.
            self._invalidate_prefill_provenance(slot for slot in range(self.batch) if slot not in chosen)
        effective = [start if source is not None else 0 for start, source in zip(starts, sources)]
        self.event(
            "prefill_ownership",
            slots=slots,
            sources=sources,
            remap=mapping,
            reused_slots=[slot for slot, source in zip(slots, sources) if source is not None],
            rebuilt_slots=[slot for slot, source in zip(slots, sources) if source is None],
        )
        return effective

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        prefill_sample_mask=None,
        **kwargs,
    ):
        phases = getattr(self, "phases", None)
        phase_start = time.perf_counter() if phases is not None else None
        self._check_cache(kv_cache)
        rows = tokens.shape[0]
        slots = list(range(rows)) if empty_slots is None else list(empty_slots)
        if len(set(slots)) != rows or any(s < 0 or s >= self.batch for s in slots):
            raise ValueError("Invalid prefill destination slots")
        final_rows = (
            [True] * rows
            if prefill_sample_mask is None
            else torch.as_tensor(prefill_sample_mask, dtype=torch.bool).reshape(-1).tolist()
        )
        if len(final_rows) != rows:
            raise ValueError("Prefill sampling mask requires one value per input row")
        final_slots = [slot for slot, final in zip(slots, final_rows) if final]
        ends = [int(n) for n in prompt_lens]
        starts = [0] * rows if start_pos is None else [int(n) for n in start_pos]
        if any(s < 0 or e <= s or e > self.context for s, e in zip(starts, ends)):
            raise ValueError("Invalid scheduled prompt range")
        effective = self._plan_prefill_state(tokens, page_table, slots, starts)
        private_cache = getattr(self, "private_prefix_cache", None)
        canonical_rows = [
            start % 4096 == 0
            and end % 4096 == 0
            and (start == 0 or bool((self.prefill_provenance[slot] or {}).get("canonical_4k")))
            for start, end, slot in zip(effective, ends, slots)
        ]
        private_hits = {}
        if private_cache is not None:
            for row, end in enumerate(ends):
                node = private_cache.lookup(tokens[row], end) if canonical_rows[row] else None
                if node is not None:
                    private_hits[row] = node
        if phases is not None:
            phases.prefill_sessions(slots, effective, [tokens[row, :end].tolist() for row, end in enumerate(ends)])
            phase = phases.begin(
                "prefill",
                phase_start,
                slots,
                positions=starts,
                ends=ends,
                executed_starts=effective,
                sample_mask=final_rows,
            )
        self._invalidate_prefill_provenance(slots)
        reset_slots = [
            slot for row, (slot, start) in enumerate(zip(slots, effective)) if start == 0 and row not in private_hits
        ]
        if reset_slots:
            self.gen.reset_state_slots(reset_slots)
        pages = self._pages_for(page_table, slots)
        if private_hits:
            self.gen.refresh_page_table(pages)
        device = sampling_params is not None
        if not device and not self.allow_host_sampling:
            raise RuntimeError("Host sampling requested; enable explicit QWEN38_HOST_SAMPLING_COMPAT")
        if device:
            for row, slot in enumerate(slots):
                if starts[row] == 0 and sampling_params.seed[row] is None:
                    self.seed_bases[slot] = secrets.randbelow(1 << 30)
            self._sampling(
                sampling_params, slots, reset_state=True, positions=[e - 1 for e in ends], active_slots=final_slots
            )
        self.event(
            "prefill_submission",
            slots=slots,
            starts=starts,
            ends=ends,
            sample_mask=final_rows if device else None,
            sampling_mode="device" if device else "host",
        )
        outputs = [None] * rows
        groups = []
        for start in sorted(set(effective)):
            selected = [row for row, value in enumerate(effective) if value == start and row not in private_hits]
            if not selected:
                continue
            if self.serial_prefill:
                groups.extend((start, [row]) for row in selected)
            else:
                groups.append((start, selected))
        groups.extend((None, [row]) for row in private_hits)
        for start, selected in groups:
            group_slots = [slots[row] for row in selected]
            if start is None:
                row = selected[0]
                logits = private_cache.restore(private_hits[row], slots[row], page_table[row], effective[row])
                lengths = [0] * self.batch
            else:
                lengths = [0] * self.batch
                scattered = torch.zeros(self.batch, max(ends[row] - start for row in selected), dtype=torch.int64)
                for row in selected:
                    slot = slots[row]
                    lengths[slot] = ends[row] - start
                    scattered[slot, : lengths[slot]] = tokens[row, start : ends[row]]
                logits = self.gen.prefill_forward(
                    scattered,
                    page_table=pages,
                    kv_cache=kv_cache,
                    prompt_lens=lengths,
                    start_position=start,
                    return_device=True,
                )
                if private_cache is not None:
                    for row in selected:
                        if canonical_rows[row]:
                            private_cache.capture(tokens[row], ends[row], slots[row], page_table[row])
            self.event(
                "prefill",
                starts=[starts[row] for row in selected],
                ends=[ends[row] for row in selected],
                executed_start=ends[selected[0]] if start is None else start,
                private_cache_hit=start is None,
                executed_lengths=[lengths[slot] for slot in group_slots],
                slots=group_slots,
                other_live_slots=sorted(self.live_slots - set(group_slots)),
                memory=self.memory(),
                programs=self.mesh.num_program_cache_entries(),
                captures=self.gen.counters["captures"],
            )
            self.event("sampling_mode", phase="prefill", mode="device" if device else "host")
            if device:
                sample_slots = [slots[row] for row in selected if final_rows[row]]
                self.event("prefill_sampling_mask", slots=group_slots, sampled_slots=sample_slots)
                host = None
                if sample_slots:
                    output = self.gen.sample_prefill(active_slots=sample_slots, return_device=True)
                    host = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape(-1).long()
                    self._check_tokens(host[sample_slots])
                for row in selected:
                    outputs[row] = (
                        host[slots[row]].reshape(1, 1) if final_rows[row] else torch.zeros(1, 1, dtype=torch.int64)
                    )
            else:
                host = self.gen.host_logits(logits)
                for row in selected:
                    outputs[row] = host[slots[row] : slots[row] + 1].clone()
        private_fenced = private_cache.finish() if private_cache is not None else False
        self.live_slots.update(slots)
        for row, slot in enumerate(slots):
            self.slot_ends[slot] = ends[row]
            self.prefill_provenance[slot] = dict(
                epoch=self.state_epochs[slot],
                end=ends[row],
                canonical_4k=canonical_rows[row],
                tokens=tokens[row, : ends[row]].detach().clone(),
                pages=page_table[row, : (ends[row] + 31) // 32].detach().clone(),
            )
        # Qwen3.5 text-only MRoPE has identical axes and zero per-request
        # delta; the plugin still negotiates the MRoPE tuple contract.
        result = torch.cat(outputs, dim=0), torch.zeros(rows, dtype=torch.int64)
        if phases is not None:
            # Intermediate-only ordinary groups may remain enqueued. The private
            # cache prototype explicitly fences transfers; record that real boundary.
            phases.finish(
                phase,
                output_tokens=(
                    [int(value.item()) if final else None for value, final in zip(outputs, final_rows)]
                    if device
                    else None
                ),
                fenced=private_fenced or all(final_rows) or not device,
                mode="device" if device else "host",
            )
        return result

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        **kwargs,
    ):
        phases = getattr(self, "phases", None)
        phase_start = time.perf_counter() if phases is not None else None
        self._check_cache(kv_cache)
        if slot_remap is not None:
            self.gen.remap_state_slots(slot_remap)
            # Move proof metadata alongside the actual gather, before decoded
            # destinations are invalidated. The later callback moves only the
            # existing live-slot/slot_ends session mirrors, never this proof twice.
            self._remap_prefill_provenance(slot_remap)
        if reload_inputs or reload_page_table:
            self._pages_for(page_table)
        if sampling_params is not None and (reload_sampling_params or reset_sampling_state):
            self._sampling(
                sampling_params,
                range(tokens.shape[0]),
                reset_state=reset_sampling_state,
                positions=start_pos.tolist() if reload_inputs else None,
            )
        device = sampling_params is not None
        if not device and not self.allow_host_sampling:
            raise RuntimeError("Host sampling requires explicit compatibility mode")
        if reload_inputs:
            self.decode_positions = [int(p) for p in start_pos]
        submitted_positions = list(self.decode_positions)
        active_slots = [i for i, position in enumerate(submitted_positions) if position >= 0]
        phase = (
            phases.begin("decode", phase_start, active_slots, positions=[submitted_positions[i] for i in active_slots])
            if phases is not None
            else None
        )
        output = self.gen.decode_forward(
            tokens,
            start_pos,
            page_table=self.pages if reload_inputs or reload_page_table else None,
            kv_cache=kv_cache,
            return_logits=not device,
            return_device=True,
            reload_inputs=reload_inputs,
        )
        self._invalidate_prefill_provenance(i for i, position in enumerate(submitted_positions) if position >= 0)
        self.event(
            "sampling_mode",
            phase="decode",
            mode="device" if device else "host",
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            positions=submitted_positions,
            active_slots=[i for i, p in enumerate(submitted_positions) if p >= 0],
            model_trace=str(self.gen.model_trace),
            sampling_trace=str(self.gen.sample_trace),
            captures=self.gen.counters["captures"],
        )
        self.decode_positions = [p + 1 if p >= 0 else p for p in submitted_positions]
        result = DecodeOutput(output, device, tokens.shape[0], phase)
        if read_from_device:
            result = self.process_decode_output_host(self.read_decode_output(result), is_tokens=device)
        return result

    def read_decode_output(self, output, async_read=False):
        # Token IDs are TP-replicated; read only one replica. Host compatibility
        # requires every vocabulary shard and is excluded from performance claims.
        tensors = ttnn.get_device_tensors(output.tensor)
        if output.is_tokens:
            tensors = tensors[:1]
        host = DecodeOutput(
            [t.cpu(blocking=not async_read) for t in tensors], output.is_tokens, output.rows, output.phase
        )
        if async_read:
            return host, [ttnn.record_event(self.mesh, 0)]
        return host

    def process_decode_output_host(self, output, is_tokens=False):
        values = [ttnn.to_torch(t) for t in output.tensor]
        if output.is_tokens:
            result = values[0].reshape(-1)[: output.rows].long().clone()
            self._check_tokens(result)
            if output.phase is not None:
                self.phases.finish(output.phase, output_tokens=[int(result[slot]) for slot in output.phase["slots"]])
            return result[:, None]
        result = torch.cat(values, dim=-1).reshape(self.batch, 1, -1)[: output.rows].float().clone()
        if output.phase is not None:
            self.phases.finish(output.phase, mode="host")
        return result

    def _check_tokens(self, tokens):
        if bool(((tokens < 0) | (tokens >= self.model.config.vocab_size)).any()):
            raise RuntimeError("Sampled token outside vocabulary")

    def release_request(self, slot):
        if getattr(self, "phases", None) is not None:
            self.phases.release(slot)
        self.live_slots.discard(slot)
        self.slot_ends[slot] = 0
        self._invalidate_prefill_provenance([slot])

    def note_state_slots_moved(self, moves):
        self.live_slots = {moves.get(s, s) for s in self.live_slots}
        previous = list(self.slot_ends)
        for old, new in moves.items():
            self.slot_ends[new] = previous[old]

    def release_persistent_capture(self):
        if self.gen is None:
            return
        traces = set(self.gen.decode_traces.values()) | {self.gen.sample_trace, self.gen.buffered_sample_trace}
        for trace in traces - {None}:
            ttnn.release_trace(self.mesh, trace)
        self.event(
            "shutdown",
            released=len(traces - {None}),
            counters=self.gen.counters,
            programs=self.mesh.num_program_cache_entries(),
            memory=self.memory(),
        )
