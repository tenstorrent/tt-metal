# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness generator and explicit paged prefill/decode entrypoints."""

import secrets
import time
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace

import torch
from readiness_check.contract import Generator
from transformers import AutoTokenizer
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.model import HF_MODEL, HF_REVISION, K2HorizonModel
from models.common.sampling.generator import (
    SamplingGenerator,
    SamplingParams,
    _hash_request_seed_to_device_seed,
    format_sampling_params,
)


@dataclass
class DecodeOutput:
    values: object
    logprobs: object = None
    is_tokens: bool = False
    submission: int | None = None


class K2HorizonGenerator(Generator):
    aggregate_context_budget = 524288
    decode_input_update_contract = 1
    serving_capabilities = {
        "supports_chunked_prefill": True,
        "supports_intermediate_prefill_mask": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_device_penalties": True,
        "supports_async_decode": True,
        "output_tokens_per_step": 1,
        "supports_prefix_caching": False,
        "supports_device_grammar": False,
        "supports_spec_decode": False,
        "supports_async_spec_decode": False,
        "supports_compact_host_logits": False,
        "supports_selective_host_readback": False,
        "supports_stage_inputs": False,
        "supports_stage_outputs": False,
        "tt_adaptive_block_output": False,
        "folds_dp_into_lanes": False,
    }

    def __init__(
        self,
        model,
        *,
        max_batch_size=1,
        max_seq_len=524288,
        page_size=32,
        sampling_mode="device",
        prepare_lengths=(32, 128),
        kv_cache=None,
        num_cache_blocks=None,
        allocate_cache=True,
    ):
        self.model, self.mesh_device = model, model.mesh_device
        self.max_batch_size, self.max_seq_len, self.page_size = max_batch_size, max_seq_len, page_size
        if not 1 <= max_batch_size <= 32 or sampling_mode not in ("device", "host"):
            raise ValueError("Invalid batch or sampling mode")
        self.sampling_mode = sampling_mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL, revision=HF_REVISION, trust_remote_code=True, local_files_only=True
        )
        # Cover the rounded SDPA read window, not just the logical page count.
        self.capacity = (max_seq_len + 1023) // 1024 * 1024
        self.pages_per_slot = self.capacity // page_size
        self.num_cache_blocks = num_cache_blocks or min(max_batch_size * self.capacity, 524288) // page_size
        if self.num_cache_blocks * page_size > self.physical_cache_token_budget:
            raise ValueError("Cache exceeds aggregate context plus one serving reservation page per slot")
        self.kv_cache = None
        if kv_cache is not None:
            self.bind_cache(kv_cache)
        if self.kv_cache is None and allocate_cache:
            self.kv_cache = model.allocate_kv_cache(self.num_cache_blocks, page_size)
        # Standalone defaults partition the aggregate pool; external schedulers
        # may assign the entire pool to any one slot through explicit page tables.
        self.default_pages_per_slot = min(self.pages_per_slot, self.num_cache_blocks // max_batch_size)
        self.page_table_host = torch.zeros(max_batch_size, self.pages_per_slot, dtype=torch.int32)
        self.page_table_host[:, : self.default_pages_per_slot] = torch.arange(
            max_batch_size * self.default_pages_per_slot, dtype=torch.int32
        ).reshape(max_batch_size, -1)
        if not allocate_cache:
            self.page_table_host.zero_()
        self.page_table = model.put(self.page_table_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.tokens = model.put(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._token_readback_view = ttnn.get_device_tensors(self.tokens)[0]
        self.active_host = torch.zeros(32, dtype=torch.int32)
        self.active_host[:max_batch_size] = 1
        self.active = model.put(self.active_host.reshape(1, 1, 1, 32), ttnn.uint32)
        self.positions = model.put(torch.zeros(max_batch_size, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.rope_indices = model.put(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        args = SimpleNamespace(
            vocab_size=model.vocab_size,
            padded_vocab_size=model.padded_vocab_size,
            max_batch_size=max_batch_size,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            use_topk_logprobs=True,
            model_config={
                "GALAXY_NUM_LINKS": 2,
                "SAMPLING_AG_CONFIG": {"allow_force_argmax": True, "num_links": 2, "topology": ttnn.Topology.Ring},
            },
        )
        self.sampler = SamplingGenerator(args=args, mesh_device=self.mesh_device, tt_ccl=model.ccl)
        self.sampler.seed_manager.salt_duplicate_seeds = False
        self.sampler.tt_sampling._allow_force_argmax_sampling = False
        self.sampler.tt_sampling._force_argmax_sampling = False
        self.sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        self.formatted_sampling_params = format_sampling_params(self.sampling_params, 32)
        self.sampler.reset_sampling_params(self.formatted_sampling_params)
        self.prepare_lengths = tuple(prepare_lengths)
        self.model_trace = None
        self.prefill_head_trace = None
        self.sampling_traces = {}
        self.prefill_traces = {}
        self.prefill_chunk_hidden = model.put(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16))
        self._pending_output = None
        self._submission = 0
        self.counters = Counter()
        self.logits = model.put(
            torch.zeros(1, 1, 32, model.padded_vocab_size, dtype=torch.bfloat16),
            dtype=getattr(ttnn, model.precision["logits_dtype"]),
            shard=-1,
        )
        self.prefill_hidden = model.put(torch.zeros(1, 1, 32, 2560, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.prefill_slot = model.put(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.seed_reset_values = model.put(torch.zeros(32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.slot_indices = model.put(torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32), ttnn.uint32)
        self._prepared = False
        self.prefill_buffers = {}
        buckets = [1]
        size = page_size
        while size <= min(model.chunk_size, self.capacity):
            buckets.append(size)
            size *= 2
        for size in buckets:
            physical = max(32, size)
            self.prefill_buffers[size] = dict(
                tokens=model.put(torch.zeros(1, 1, 1, physical, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                position_ids=model.put(
                    torch.zeros(1, 1, 1, physical, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
                current_pos=model.put(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                page_table=model.put(self.page_table_host[:1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                fill_page_table=model.put(
                    torch.zeros(1, max(1, size // page_size), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                ),
            )

    def _copy(self, value, dest, name):
        host = ttnn.from_torch(
            value.contiguous(),
            dtype=dest.dtype,
            layout=dest.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, dest)
        self.counters[name] += 1

    def _event(self, event, **data):
        sink = getattr(self, "event_sink", None)
        if sink is not None:
            sink(event, **data)

    def refresh_page_table(self, page_table, *, force=False):
        if tuple(page_table.shape) != tuple(self.page_table_host.shape):
            raise ValueError("Page table must use the configured fixed-slot shape")
        self.validate_page_table(page_table)
        if force or not torch.equal(page_table, self.page_table_host):
            self._copy(page_table, self.page_table, "page_table_refreshes")
            self.page_table_host = page_table.clone()

    def refresh_decode_inputs(self, tokens, start_pos):
        b = self.max_batch_size
        if tokens.numel() != b or start_pos.numel() != b:
            raise ValueError("Decode inputs must cover every configured slot")
        if torch.any(start_pos < -1) or torch.any(start_pos >= self.max_seq_len):
            raise ValueError("Decode position is outside the supported context")
        active = start_pos.reshape(-1) >= 0
        if torch.any(tokens.reshape(-1)[active] < 0) or torch.any(tokens.reshape(-1)[active] >= self.model.vocab_size):
            raise ValueError("Active decode token is outside the vocabulary")
        padded = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        valid = (tokens.reshape(-1) >= 0) & (tokens.reshape(-1) < self.model.vocab_size)
        padded.reshape(-1)[:b] = torch.where(valid, tokens.reshape(-1), 0).int()
        self._copy(padded, self.tokens, "token_refreshes")
        self._copy(start_pos.int().reshape(b), self.positions, "position_refreshes")
        padded.zero_()
        padded.reshape(-1)[:b] = start_pos.clamp_min(0).int()
        self._copy(padded, self.rope_indices, "rope_refreshes")
        self.set_active_slots(torch.nonzero(start_pos.reshape(-1) >= 0).reshape(-1).tolist())

    def set_active_slots(self, slots):
        active = torch.zeros(32, dtype=torch.int32)
        active[slots] = 1
        self._copy(active.reshape(1, 1, 1, 32), self.active, "active_refreshes")
        self._copy(active.reshape(32, 1), self.sampler.tt_penalties.decode_src, "penalty_active_refreshes")
        self.active_host = active

    def _decode(self):
        logits = self.model.decode_forward(
            self.tokens,
            current_pos=self.positions,
            rope_indices=self.rope_indices,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            batch_size=self.max_batch_size,
        )
        ttnn.copy(logits, self.logits)
        return self.logits

    def _stage_prefill_hidden(self, hidden):
        row = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        updated = ttnn.indexed_fill(self.prefill_slot, self.prefill_hidden, row, dim=2)
        ttnn.copy(updated, self.prefill_hidden)

    def _prefill_head(self):
        logits = self.model.terminal(ttnn.to_layout(self.prefill_hidden, ttnn.TILE_LAYOUT))
        ttnn.copy(logits, self.logits)

    def _capture_prefill_head(self):
        self.prepare()
        if self.prefill_head_trace is not None:
            return
        started = time.monotonic()
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._prefill_head()
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        # Both endpoints were allocated before every trace. CQ0 staging writes
        # hidden before replay; sampling consumes logits before any next writer.
        # No output allocation or allocation-tracker exemption is needed.
        self.prefill_head_trace = tid
        self.counters["prefill_head_captures"] += 1
        self._event(
            "capture",
            kind="prefill_head",
            key=[32, self.model.padded_vocab_size],
            trace_id=str(tid),
            seconds=time.monotonic() - started,
            counters=dict(self.counters),
        )

    def _replay_prefill_head(self):
        self._capture_prefill_head()
        ttnn.execute_trace(self.mesh_device, self.prefill_head_trace, cq_id=0, blocking=False)
        self.counters["prefill_head_replays"] += 1

    def _sample(self, logits):
        if logits.dtype != getattr(ttnn, self.model.precision["sampling_dtype"]):
            raise ValueError("Sampler input does not match selected precision contract")
        old_tokens = ttnn.to_layout(self.tokens, ttnn.TILE_LAYOUT)
        seeds = self.sampler.tt_sampling.seeds_tt_tensor
        old_seeds = ttnn.to_layout(ttnn.reshape(seeds, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
        result = self.sampler._run_sampling(
            logits,
            penalties_on=self.sampler._penalties_active,
            grammar_on=False,
            tt_out_tok=self.tokens,
            count_tokens=False,
        )
        selected = ttnn.where(self.active, ttnn.to_layout(self.tokens, ttnn.TILE_LAYOUT), old_tokens)
        ttnn.copy(ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT), self.tokens)
        if self.sampler._penalties_active:
            self.sampler.tt_penalties.update_output_tokens(self.tokens)
        # Seed advancement is device-side, including sampled fixed-step loops.
        ttnn.plus_one(seeds, skip_negative_entries=True)
        advanced = ttnn.to_layout(ttnn.reshape(seeds, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
        selected_seeds = ttnn.where(self.active, advanced, old_seeds)
        ttnn.copy(ttnn.reshape(ttnn.to_layout(selected_seeds, ttnn.ROW_MAJOR_LAYOUT), (32,)), seeds)
        return result

    def prepare(self):
        if self._prepared:
            return
        started = time.monotonic()
        if self.model_trace is not None:
            raise RuntimeError("Preparation must precede every capture")
        if self.kv_cache is None:
            raise ValueError("Bind the externally allocated cache before preparation")
        for size, buffers in self.prefill_buffers.items():
            for long_context in (False, True) if self.capacity >= 65536 else (False,):
                for output in ("all", "last", "none", "hidden"):
                    out = self.model.prefill_chunk(
                        **buffers, kv_cache=self.kv_cache, long_context=long_context, tokenwise=size == 1, output=output
                    )
                    if output == "hidden":
                        self._stage_prefill_hidden(out)
                    del out
                self._prefill_trace_forward(size, long_context)
        self._prefill_head()
        self.refresh_decode_inputs(torch.zeros(self.max_batch_size, 1), torch.zeros(self.max_batch_size))
        logits = self._decode()
        self.sampler.tt_sampling._allow_force_argmax_sampling = True
        self.sampler.precompile(logits, tt_out_tok=self.tokens, all_configs=True, compile_token_update=True)
        self.sampler.tt_sampling._allow_force_argmax_sampling = False
        self.sampler.reset_sampling_params(self.formatted_sampling_params)
        self.sampler.tt_penalties.reset_output_tokens(torch.full((32, 1), -1), slots=[0])
        self._remap_seed_buffer()
        self._sample(logits)
        del logits
        self.reset()
        self._prepared = True
        self._event(
            "prepared",
            seconds=time.monotonic() - started,
            counters=dict(self.counters),
            prefill_buckets=list(self.prefill_buffers),
            batch=self.max_batch_size,
            capacity=self.capacity,
            page_size=self.page_size,
        )

    def capture(self):
        if self.model_trace is not None:
            return
        self.prepare()
        self.refresh_decode_inputs(torch.zeros(self.max_batch_size, 1), torch.zeros(self.max_batch_size))
        started = time.monotonic()
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.logits = self._decode()
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        self.model_trace = tid
        self.counters["model_captures"] += 1
        self._event(
            "capture",
            kind="decode",
            key=[self.max_batch_size, self.capacity],
            trace_id=str(tid),
            seconds=time.monotonic() - started,
            counters=dict(self.counters),
        )
        self._capture_prefill_head()
        self._capture_sampler()

    def _prefill_trace_forward(self, size, long_context):
        hidden = self.model.prefill_chunk(
            **self.prefill_buffers[size],
            kv_cache=self.kv_cache,
            long_context=long_context,
            tokenwise=size == 1,
            output="hidden",
        )
        ttnn.copy(hidden, self.prefill_chunk_hidden)
        return self.prefill_chunk_hidden

    def _replay_prefill(self, size, long_context):
        # Physical bucket, stable buffer bindings and attention grid define the
        # signature. Logical start/length/page mapping never retire a trace.
        self.prepare()
        key = (size, long_context if size != 1 else False)
        if key not in self.prefill_traces:
            started = time.monotonic()
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self._prefill_trace_forward(*key)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            self.prefill_traces[key] = tid
            self.counters["prefill_captures"] += 1
            self._event(
                "capture",
                kind="prefill",
                key=key,
                trace_id=str(tid),
                seconds=time.monotonic() - started,
                counters=dict(self.counters),
            )
        ttnn.execute_trace(self.mesh_device, self.prefill_traces[key], cq_id=0, blocking=False)
        self.counters["prefill_replays"] += 1
        return self.prefill_chunk_hidden

    def _sampling_key(self):
        return (
            self.sampler._penalties_active,
            self.sampler.tt_sampling.log_probs_calculator.enable_log_probs,
            self.sampler.tt_sampling.force_argmax_sampling,
        )

    def _capture_sampler(self):
        key = self._sampling_key()
        if key not in self.sampling_traces:
            started = time.monotonic()
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            output = self._sample(self.logits)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)

            # Only outputs are exempted. They are read immediately after their
            # producing sampling trace and before the next model replay.
            def acknowledge(value):
                if isinstance(value, ttnn.Tensor) and value is not self.tokens:
                    trace_allocation_tracker.acknowledge_corruptible(value)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        acknowledge(item)
                elif is_dataclass(value):
                    for field in fields(value):
                        acknowledge(getattr(value, field.name))

            acknowledge(output)
            self.sampling_traces[key] = (tid, output)
            self.counters["sampling_captures"] += 1
            self._event(
                "capture",
                kind="sampling",
                key=key,
                trace_id=str(tid),
                seconds=time.monotonic() - started,
                counters=dict(self.counters),
            )
        return self.sampling_traces[key]

    def replay(self, *, sample=True):
        self.capture()
        ttnn.execute_trace(self.mesh_device, self.model_trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        if sample:
            tid, output = self._capture_sampler()
            ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
            return output
        return self.logits

    def read_tokens(self):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1)[: self.max_batch_size].long()

    def read_logits(self, logits):
        self.counters["logit_readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))[
            ..., : self.model.vocab_size
        ].float()

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        start_pos=None,
        slots=None,
        empty_slots=None,
        sampling_mode=None,
        sampling_params=None,
        enable_trace=True,
        intermediate_prefill_mask=None,
        **kwargs,
    ):
        if self._pending_output is not None:
            raise RuntimeError("Queue the outstanding decode readback before prefill changes request state")
        b = len(prompt_lens)
        if intermediate_prefill_mask is None:
            intermediate = [False] * b
        else:
            if (
                not isinstance(intermediate_prefill_mask, torch.Tensor)
                or intermediate_prefill_mask.device.type != "cpu"
                or intermediate_prefill_mask.dtype != torch.bool
                or tuple(intermediate_prefill_mask.shape) != (b,)
            ):
                raise ValueError("intermediate_prefill_mask must be a CPU bool tensor with one entry per row")
            intermediate = intermediate_prefill_mask.tolist()
        if return_all_logits and any(intermediate):
            raise ValueError("KV-only intermediate rows cannot return all logits")
        final_rows = [row for row in range(b) if not intermediate[row]]
        starts = [0] * b if start_pos is None else list(start_pos)
        self._event(
            "prefill_request",
            starts=[int(x) for x in starts],
            ends=[int(x) for x in prompt_lens],
            empty_slots=empty_slots,
            device_sampling=sampling_params is not None,
            intermediate=intermediate,
        )
        if b == 0 or len(starts) != b or tokens.ndim != 2 or tokens.shape[0] != b:
            raise ValueError("Prefill tokens, starts and prompt lengths must have the same nonempty batch")
        if empty_slots is not None:
            if slots is not None:
                raise ValueError("Specify only empty_slots")
            slots = empty_slots
        slots = list(range(b)) if slots is None else list(slots)
        if len(slots) != b or len(set(slots)) != b or any(s < 0 or s >= self.max_batch_size for s in slots):
            raise ValueError("Invalid prefill destination slots")
        self.bind_cache(kv_cache)
        if tuple(page_table.shape) != (b, self.pages_per_slot):
            raise ValueError("Prefill page table must contain one full-width row per submitted request")
        self.validate_page_table(page_table)
        for row, (end, start) in enumerate(zip(prompt_lens, starts)):
            if not 0 <= int(start) < int(end) <= min(tokens.shape[-1], self.max_seq_len):
                raise ValueError("Invalid logical prompt extent")
            ids = tokens[row, int(start) : int(end)]
            if torch.any(ids < 0) or torch.any(ids >= self.model.vocab_size):
                raise ValueError("Prefill token is outside the vocabulary")
        self.prepare()
        mode = sampling_mode or ("device" if sampling_params is not None else "host")
        if mode not in ("device", "host"):
            raise ValueError("Unknown sampling mode")
        device_sample = mode == "device" and not return_all_logits
        if device_sample and final_rows:
            if enable_trace:
                self.capture()
            self.prepare_prefill_sampling(tokens, prompt_lens, slots, sampling_params, final_rows=final_rows)
        outputs = {}
        for row, (end, start) in enumerate(zip(prompt_lens, starts)):
            end, start = int(end), int(start)
            length = end - start
            if length < 1 or start < 0 or end > self.max_seq_len:
                raise ValueError("Invalid logical prompt extent")
            chunks = []
            offset = 0
            while offset < length:
                absolute = start + offset
                count = max(
                    n
                    for n in self.prefill_buffers
                    if n <= length - offset and (n == 1 or absolute % self.page_size == 0)
                )
                self._event(
                    "prefill_chunk",
                    row=row,
                    slot=slots[row],
                    start=absolute,
                    length=count,
                    end=end,
                    key=[count, absolute + count >= 65536 if count != 1 else False],
                )
                buffers = self.prefill_buffers[count]
                ids = torch.zeros(1, 1, 1, max(32, count), dtype=torch.int32)
                ids.reshape(-1)[:count] = tokens[row, absolute : absolute + count].int()
                positions = torch.zeros_like(ids)
                positions.reshape(-1)[:count] = torch.arange(absolute, absolute + count, dtype=torch.int32)
                self._copy(ids, buffers["tokens"], "prefill_token_refreshes")
                self._copy(positions, buffers["position_ids"], "prefill_rope_refreshes")
                self._copy(
                    torch.tensor([absolute], dtype=torch.int32), buffers["current_pos"], "prefill_position_refreshes"
                )
                self._copy(page_table[row : row + 1].int(), buffers["page_table"], "prefill_page_refreshes")
                first_page = absolute // self.page_size
                self._copy(
                    page_table[row : row + 1, first_page : first_page + max(1, count // self.page_size)].int(),
                    buffers["fill_page_table"],
                    "prefill_fill_page_refreshes",
                )
                output = "all" if return_all_logits else ("last" if offset + count == length else "none")
                if intermediate[row]:
                    output = "none"
                if device_sample and output == "last":
                    output = "hidden"
                if enable_trace and not return_all_logits:
                    logits = self._replay_prefill(count, absolute + count >= 65536)
                    if output == "last":
                        logits = self.model.terminal(logits)
                else:
                    logits = self.model.prefill_chunk(
                        **buffers,
                        kv_cache=kv_cache,
                        long_context=absolute + count >= 65536,
                        tokenwise=count == 1,
                        output=output,
                    )
                if output == "hidden":
                    self._copy(
                        torch.tensor([slots[row]], dtype=torch.int32), self.prefill_slot, "prefill_slot_refreshes"
                    )
                    self._stage_prefill_hidden(logits)
                elif output in ("all", "last"):
                    chunks.append(self.read_logits(logits).reshape(1, -1, self.model.vocab_size))
                del logits
                offset += count
            if chunks:
                outputs[row] = torch.cat(chunks, dim=1)
        if not final_rows:
            # The scheduler discards every row; no terminal head, sampler, or
            # device read is necessary, including for host-only controls.
            return torch.zeros(
                (b, 1) if device_sample else (b, 1, self.model.vocab_size),
                dtype=torch.int64 if device_sample else torch.float32,
            )
        if device_sample:
            if enable_trace:
                self._replay_prefill_head()
                tid, sampled = self._capture_sampler()
                ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=False)
                self.counters["sampling_replays"] += 1
            else:
                self._prefill_head()
                sampled = self._sample(self.logits)
            result = self.process_decode_output_host(
                self.read_decode_output(DecodeOutput(self.tokens, sampled[1], True))
            )
            if isinstance(result, tuple):
                values, lp = result
                return values[slots], tuple(v[slots] for v in lp) if isinstance(lp, tuple) else lp[slots]
            return result[slots]
        if return_all_logits:
            lengths = [int(end) - int(start) for end, start in zip(prompt_lens, starts)]
            result = torch.zeros(b, max(lengths), self.model.vocab_size)
            for row, output in outputs.items():
                result[row, : lengths[row]] = output[0]
            return result
        result = torch.zeros(b, 1, self.model.vocab_size)
        for row, output in outputs.items():
            result[row] = output[0]
        return result

    def prepare_prefill_sampling(self, tokens, prompt_lens, slots, params, final_rows=None):
        final_rows = range(len(slots)) if final_rows is None else final_rows
        final_slots = [slots[row] for row in final_rows]
        if params is not None:
            self.validate_sampling_params(params)
        current = self.formatted_sampling_params
        requested = format_sampling_params(params or SamplingParams(0.0, 1, 1.0), 32)
        merged = {}
        for field in fields(current):
            values = list(getattr(current, field.name))
            incoming = getattr(requested, field.name)
            for row in final_rows:
                slot = slots[row]
                values[slot] = incoming[row]
            merged[field.name] = values
        self.formatted_sampling_params = SamplingParams(**merged)
        self.sampler.reset_sampling_params(self.formatted_sampling_params)
        self.set_active_slots(final_slots)
        if self.sampler._penalties_active:
            history = torch.full((32, tokens.shape[-1]), -1, dtype=torch.int64)
            for row in final_rows:
                slot = slots[row]
                history[slot, : int(prompt_lens[row])] = tokens[row, : int(prompt_lens[row])]
            self.sampler.reset_prompt_tokens(history, slots=final_slots)
            self.sampler.reset_output_state(slots=final_slots)
        seeds = torch.zeros(32, dtype=torch.int64)
        for row in final_rows:
            slot = slots[row]
            seed = requested.seed[row]
            base = secrets.randbelow(1_000_000) + 1 if seed is None else _hash_request_seed_to_device_seed(seed, 0)
            seeds[slot] = base + int(prompt_lens[row]) - 1
        self._copy(seeds, self.seed_reset_values, "prefill_seed_resets")
        new = ttnn.to_layout(ttnn.reshape(self.seed_reset_values, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
        target = self.sampler.tt_sampling.seeds_tt_tensor
        old = ttnn.to_layout(ttnn.reshape(target, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
        selected = ttnn.where(self.active, new, old)
        ttnn.copy(ttnn.reshape(ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT), (32,)), target)

    def bind_cache(self, kv_cache):
        if kv_cache is None:
            raise ValueError("A bound K/V cache is required")
        if kv_cache is not self.kv_cache:
            if getattr(self, "_prepared", False) or getattr(self, "model_trace", None) is not None:
                raise ValueError("External caches must be bound before preparation/capture")
            if len(kv_cache) != len(self.model.layers):
                raise ValueError("One K/V pair is required per attention layer")
            for pair in kv_cache:
                if len(pair) != 2:
                    raise ValueError("Cache entries must be K/V pairs")
                for tensor in pair:
                    if tensor.dtype != self.model.kv_cache_dtype or tuple(tensor.shape)[1:] != (2, self.page_size, 128):
                        raise ValueError(
                            "Cache must preserve selected storage dtype, two local heads, configured pages and head128"
                        )
            blocks = kv_cache[0][0].shape[0]
            if any(tensor.shape[0] != blocks for pair in kv_cache for tensor in pair):
                raise ValueError("All layers share one uniform block allocation")
            if blocks * self.page_size > self.physical_cache_token_budget:
                raise ValueError("Cache exceeds aggregate context plus serving output reservations")
            self.num_cache_blocks = blocks
            self.kv_cache = kv_cache

    def validate_page_table(self, page_table):
        if torch.any(page_table < 0) or torch.any(page_table >= self.num_cache_blocks):
            raise ValueError("Page table references blocks outside the bound aggregate cache")

    def allocate_kv_cache(self, shape, dtype=None, num_layers=None):
        dtype = self.model.kv_cache_dtype if dtype is None else dtype
        if self._prepared or self.kv_cache is not None:
            raise ValueError("Use allocate_cache=False at construction for caller-managed allocation")
        if tuple(shape)[1:] != (2, self.page_size, 128) or dtype != self.model.kv_cache_dtype:
            raise ValueError("Allocation must preserve the selected local-head/page/cache-dtype contract")
        if num_layers is not None and num_layers != len(self.model.layers):
            raise ValueError("Cache layer count does not match the model")
        cache = self.model.allocate_kv_cache(shape[0], self.page_size, dtype)
        self.bind_cache(cache)
        return cache

    def apply_slot_remap(self, remap):
        indices = torch.as_tensor(remap, dtype=torch.int32).reshape(-1)
        if (
            indices.numel() != self.max_batch_size
            or torch.any(indices < 0)
            or torch.any(indices >= self.max_batch_size)
        ):
            raise ValueError("Invalid state-slot remap")
        padded = torch.arange(32, dtype=torch.int32)
        padded[: self.max_batch_size] = indices
        self.sampler.apply_slot_remap(padded)
        self._copy(padded.reshape(1, 1, 1, 32), self.slot_indices, "slot_index_refreshes")
        self._remap_seed_buffer()
        formatted = self.formatted_sampling_params
        self.formatted_sampling_params = SamplingParams(
            **{field.name: [getattr(formatted, field.name)[int(i)] for i in padded] for field in fields(formatted)}
        )
        self.counters["slot_remaps"] += 1

    def _remap_seed_buffer(self):
        target = self.sampler.tt_sampling.seeds_tt_tensor
        values = ttnn.to_layout(ttnn.reshape(target, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
        moved = ttnn.gather(values, 3, self.slot_indices)
        ttnn.copy(ttnn.reshape(ttnn.to_layout(moved, ttnn.ROW_MAJOR_LAYOUT), (32,)), target)

    def update_sampling_state(
        self, params, *, start_pos, reload_params, reset_state, prompt_tokens=None, output_tokens=None
    ):
        if reload_params:
            if params is None:
                raise ValueError("Sampling parameter reload requires parameters")
            self.validate_sampling_params(params)
            self.sampling_params = params
            self.formatted_sampling_params = format_sampling_params(params, 32)
        self.sampler.apply_decode_state(
            [self.sampling_params],
            reload_sampling_params=reload_params,
            reset_sampling_state=reset_state,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        if reset_state:
            formatted = self.formatted_sampling_params
            positions = torch.as_tensor(start_pos).reshape(-1)
            seeds = torch.full((32,), 2**32 - 1, dtype=torch.int64)
            for slot in range(self.max_batch_size):
                if positions[slot] >= 0:
                    request_seed = formatted.seed[slot]
                    base = (
                        secrets.randbelow(1_000_000) + 1
                        if request_seed is None
                        else _hash_request_seed_to_device_seed(request_seed, 0)
                    )
                    seeds[slot] = base + int(positions[slot])
            self._copy(seeds, self.sampler.tt_sampling.seeds_tt_tensor, "seed_resets")
        self.counters["sampling_param_reloads"] += int(reload_params)
        self.counters["sampling_state_resets"] += int(reset_state)

    @staticmethod
    def validate_sampling_params(params):
        def values(name):
            value = getattr(params, name)
            return value if isinstance(value, (list, tuple)) else [value]

        temperatures, top_k = values("temperature"), values("top_k")
        for index, temperature in enumerate(temperatures):
            k = top_k[min(index, len(top_k) - 1)]
            if temperature < 0 or (temperature != 0 and not 1 <= k <= 32):
                raise ValueError(
                    "Device sampling requires temperature>=0 and stochastic top_k in1..32; use explicit host mode otherwise"
                )
        if any(not 0 <= p <= 1 for p in values("top_p")):
            raise ValueError("top_p must be in0..1")
        if any(n is not None and not 0 <= n <= 20 for n in values("num_logprobs")):
            raise ValueError("Device logprobs support0..20; all-vocabulary output uses explicit host mode")

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        enable_trace=True,
        sampling_mode=None,
        sampling_params=None,
        read_from_device=True,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        **kwargs,
    ):
        if self._pending_output is not None:
            raise RuntimeError("Queue read_decode_output for the previous submission before another forward")
        if "reset_batch" in kwargs:
            raise TypeError("Use the four explicit decode reload commands")
        if reload_inputs and reload_page_table:
            raise ValueError("Full-input and page-only reload commands are mutually exclusive")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Sampling reset requires authoritative forward inputs")
        mode = sampling_mode or ("device" if sampling_params is not None else "host")
        if mode not in ("device", "host"):
            raise ValueError("Unknown sampling mode")
        if (mode == "host" or not enable_trace) and not reload_inputs:
            raise ValueError("Host sampling and eager decode require authoritative inputs")
        if slot_remap is not None and not reload_inputs:
            raise ValueError("Slot transitions require authoritative forward inputs")
        self.bind_cache(kv_cache)
        if enable_trace:
            self.capture()
        if slot_remap is not None:
            self.apply_slot_remap(slot_remap)
        if mode == "device":
            self.sampler.validate_decode_state_commands(
                reload_sampling_params=reload_sampling_params, reset_sampling_state=reset_sampling_state
            )
        if reload_sampling_params or reset_sampling_state:
            self.update_sampling_state(
                sampling_params,
                start_pos=start_pos,
                reload_params=reload_sampling_params,
                reset_state=reset_sampling_state,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        if reload_inputs:
            self.refresh_page_table(page_table, force=True)
            self.refresh_decode_inputs(tokens, start_pos)
        elif reload_page_table:
            self.refresh_page_table(page_table, force=True)
        sampled = None
        if enable_trace:
            sampled = self.replay(sample=mode == "device")
            logits = self.logits
        else:
            logits = self._decode()
            if mode == "device":
                sampled = self._sample(logits)
        output = DecodeOutput(
            self.tokens if mode == "device" else logits,
            sampled[1] if mode == "device" else None,
            mode == "device",
            self._submission,
        )
        self._submission += 1
        if getattr(self, "event_sink", None) is not None:
            self._event(
                "decode_submit",
                submission=output.submission,
                positions=start_pos.tolist(),
                reload_inputs=reload_inputs,
                reload_page_table=reload_page_table,
                reload_sampling_params=reload_sampling_params,
                reset_sampling_state=reset_sampling_state,
                slot_remap=None if slot_remap is None else torch.as_tensor(slot_remap).tolist(),
                device_sampling=mode == "device",
                counters=dict(self.counters),
            )
        self._pending_output = output
        if not read_from_device:
            self.counters["device_output_submissions"] += 1
            return output
        return self.process_decode_output_host(self.read_decode_output(output))

    def read_decode_output(self, output, async_read=False):
        """Queue observational copies before the next replay can overwrite output."""
        if output.submission is not None and self._pending_output is not output:
            raise ValueError("Decode output is stale or its transfer was already queued")
        # TP sampling replicates token IDs. Read one replica, rather than
        # transferring four copies and discarding three in host formatting.
        # Queue this observation before allowing another trace to write tokens.
        if output.is_tokens and output.values is not self.tokens:
            raise ValueError("Device sampling must return the persistent feedback tensor")
        values = self._token_readback_view if output.is_tokens else output.values
        host = DecodeOutput(
            values.cpu(blocking=not async_read),
            None if output.logprobs is None else output.logprobs.cpu(blocking=not async_read),
            output.is_tokens,
        )
        if output.submission is not None:
            self._pending_output = None
        self.counters["token_readbacks" if output.is_tokens else "logit_readbacks"] += 1
        self.counters["async_reads" if async_read else "sync_reads"] += 1
        if output.is_tokens:
            self.counters["token_readback_replicas"] += 1
        if async_read:
            return host, [ttnn.record_event(self.mesh_device, 0)]
        return host

    def process_decode_output_host(self, output, is_tokens=None):
        if is_tokens is not None and is_tokens != output.is_tokens:
            raise ValueError("Output kind does not match its submission")
        if output.is_tokens:
            values = ttnn.to_torch(output.values).reshape(-1)[: self.max_batch_size].long().reshape(-1, 1)
        else:
            values = (
                ttnn.to_torch(output.values, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))[
                    0, 0, : self.max_batch_size, : self.model.vocab_size
                ]
                .float()
                .clone()
            )
        lp = output.logprobs
        if lp is not None:
            if is_dataclass(lp):
                lp = tuple(
                    ttnn.to_torch(ttnn.get_device_tensors(v)[0])[0, 0, : self.max_batch_size].clone()
                    for v in (lp.topk_logprobs_host, lp.topk_indices_host)
                )
            else:
                lp = ttnn.to_torch(ttnn.get_device_tensors(lp)[0]).reshape(-1)[: self.max_batch_size].float()
            return values, lp
        return values

    def prefill_logits(self, prompt_token_ids):
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table_host[:1],
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        sampling_mode=None,
        on_token=None,
        stop_on_eos=True,
        **kwargs,
    ):
        if self.max_batch_size != 1:
            raise ValueError(
                "High-level generate is single-request; use explicit fixed-slot prefill/decode for batches"
            )
        if not prompt_token_ids or max_new_tokens < 0:
            raise ValueError("Generation requires a nonempty prompt and nonnegative output length")
        if len(prompt_token_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError("Request exceeds configured context")
        if max_new_tokens == 0:
            return []
        if enable_trace:
            self.capture()
        mode = sampling_mode or self.sampling_mode
        logits = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table_host[:1],
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sampling_mode=mode,
            enable_trace=enable_trace,
        )
        predicted = int(logits[0, 0]) if mode == "device" else int(logits[0, -1].argmax())
        result = [predicted]
        if on_token is not None:
            on_token(0, predicted)
        eos = self.model.config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if next_input is None and stop_on_eos and predicted in eos:
            return result
        feedback = next_input(0, predicted) if next_input else predicted
        if next_input is not None or mode == "host":
            self.refresh_decode_inputs(torch.tensor([[feedback]]), torch.tensor([len(prompt_token_ids)]))
        else:
            self._copy(torch.tensor([len(prompt_token_ids)], dtype=torch.int32), self.positions, "position_refreshes")
            positions = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            positions[0, 0, 0, 0] = len(prompt_token_ids)
            self._copy(positions, self.rope_indices, "rope_refreshes")
        for step in range(1, max_new_tokens):
            if enable_trace:
                self.replay(sample=mode == "device")
                logits = self.logits
            else:
                logits = self._decode()
                if mode == "device":
                    self._sample(logits)
            predicted = (
                int(self.read_tokens()[0]) if mode == "device" else int(self.read_logits(logits)[0, 0, 0].argmax())
            )
            result.append(predicted)
            if on_token is not None:
                on_token(step, predicted)
            if next_input is None and stop_on_eos and predicted in eos:
                break
            if next_input is not None or mode == "host":
                feedback = next_input(step, predicted) if next_input else predicted
                padded = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                padded[0, 0, 0, 0] = feedback
                self._copy(padded, self.tokens, "token_refreshes")
        return result

    def reset(self):
        # Old cache values are unreachable after a new prompt fills its prefix.
        # Keep allocations and traces alive across requests.
        self.sampler.reset_penalty_counts()
        self.sampling_params = SamplingParams(0.0, 1, 1.0)
        self.formatted_sampling_params = format_sampling_params(self.sampling_params, 32)
        self.sampler.reset_sampling_params(self.formatted_sampling_params)
        self._copy(
            torch.tensor([secrets.randbelow(1_000_000) + 1 for _ in range(32)]),
            self.sampler.tt_sampling.seeds_tt_tensor,
            "seed_resets",
        )

    def close(self):
        self._event(
            "shutdown_release",
            prefill={str(k): str(v) for k, v in self.prefill_traces.items()},
            sampling={str(k): str(v[0]) for k, v in self.sampling_traces.items()},
            decode=str(self.model_trace),
            prefill_head=str(self.prefill_head_trace),
            counters=dict(self.counters),
        )
        for tid in self.prefill_traces.values():
            ttnn.release_trace(self.mesh_device, tid)
        self.prefill_traces.clear()
        for tid, _ in self.sampling_traces.values():
            ttnn.release_trace(self.mesh_device, tid)
        if self.model_trace is not None:
            ttnn.release_trace(self.mesh_device, self.model_trace)
        if self.prefill_head_trace is not None:
            ttnn.release_trace(self.mesh_device, self.prefill_head_trace)
        self.sampling_traces.clear()
        self.model_trace = None
        self.prefill_head_trace = None

    teardown = close
    release_persistent_capture = close

    def get_max_tokens_all_users(self):
        return min(self.num_cache_blocks * self.page_size, self.aggregate_context_budget)

    @property
    def physical_cache_token_budget(self):
        # Pinned plugin adds one output/null-block reservation per fixed slot
        # to the advertised aggregate context. This is allocation headroom,
        # not a reduction of the 524288-token logical context contract.
        return self.aggregate_context_budget + self.max_batch_size * self.page_size


def build_generator(model_dir, mesh_device, **kwargs):
    ttnn.CONFIG.throw_exception_on_fallback = True
    layers = kwargs.pop("layer_indices", None)
    override = kwargs.pop("override_num_layers", None)
    if layers is None and override is not None:
        layers = [0, 3] if override == 2 else list(range(override))
    context = kwargs.get("max_seq_len", 524288)
    precision = kwargs.pop("precision_config", None)
    model = K2HorizonModel(mesh_device, layer_indices=layers, max_seq_len=context, precision_config=precision)
    return K2HorizonGenerator(model, **kwargs)
