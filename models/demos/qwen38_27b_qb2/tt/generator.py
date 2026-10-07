# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving state and canonical split sampling for the TP4 Qwen text model."""

import os
import time
from collections import Counter
from types import SimpleNamespace

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.sampling.tt_sampling import TTSampling
from models.demos.qwen38_27b_qb2.tt.model import Qwen38Model


def configure_fabric(*, payload_bytes=8192):
    """Configure the measured TP4 ring before the caller opens its mesh."""
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload_bytes
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)


class Qwen38Generator:
    def __init__(self, model, *, host_sampling=False, sampling_strategy="split"):
        self.model, self.mesh = model, model.mesh
        self.tokenizer = AutoTokenizer.from_pretrained(model.snapshot, local_files_only=True)
        self.host_sampling = host_sampling
        self.batched_prefill = os.getenv("QWEN_BATCHED_PREFILL", "0") == "1"
        self.skip_intermediate_prefill_head = os.getenv("QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD", "0") == "1"
        self.seed = 0
        args = SimpleNamespace(
            vocab_size=model.config.vocab_size,
            padded_vocab_size=model.config.vocab_size,
            cluster_shape=(1, 4),
            max_batch_size=32,
            max_top_k=32,
            pad_logits_to_power_of_2=False,
        )
        if sampling_strategy not in ("split", "argmax"):
            raise ValueError("Unknown common sampling strategy")
        args.model_config = {
            "SAMPLING_AG_CONFIG": dict(
                allow_force_argmax=sampling_strategy == "argmax", num_links=2, topology=ttnn.Topology.Ring
            )
        }
        self.sampling_strategy = sampling_strategy
        self.active_slots = None
        self.reset_active_slots = False
        self.rope_indices = None
        self.sampler = TTSampling(self.mesh, model.ccl, args)
        self.remaining_steps = None
        self.owns_cache = True
        self.cache = None
        self.trace = self.sample_trace = None
        self.prefill_trace = None
        self.prefill_sample_trace = None
        self.prefill_sample_input = None
        self.prefill_prepared = None
        self.trace_records_history = None
        self.token_history = self.history_cursor = None
        self.history_capacity = self.history_count = 0
        self.tokens = model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.positions = self.page_table = self.logits = None
        self.prefill_signatures = set()
        self.page_host = None
        self.counters = Counter()
        self.last_perf = {}

    def _copy(self, host, target, counter):
        source = ttnn.from_torch(host.contiguous(), dtype=target.dtype, layout=target.layout)
        ttnn.copy_host_to_device_tensor(source, target)
        self.counters[counter] += 1

    def _read_tokens(self):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).reshape(-1).long()

    def _host_logits(self, logits):
        self.counters["full_logits_readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()

    def _ensure_history(self, capacity):
        """Reserve output rows before capture; each row retains all 32 physical slots."""
        if not 1 <= capacity <= self.model.context:
            raise ValueError("Token history capacity must lie within the model context")
        if self.token_history is not None and self.history_capacity >= capacity:
            return
        self._release_traces()
        self.token_history = self.model.upload(
            torch.zeros(capacity, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.history_cursor = self.model.upload(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.history_capacity, self.history_count = capacity, 0
        self.counters["history_allocations"] += 1

    def _reset_history(self):
        if self.history_cursor is not None:
            self._copy(torch.zeros(1, dtype=torch.int32), self.history_cursor, "history_cursor_refreshes")
        self.history_count = 0

    def _append_history(self):
        updated = ttnn.indexed_fill(self.history_cursor, self.token_history, self.tokens, dim=0)
        ttnn.copy(updated, self.token_history)
        ttnn.plus_one(self.history_cursor)

    def _read_history(self):
        """Transfer one history buffer and return its valid prefix in physical slot order."""
        if self.history_count == 0:
            return torch.empty(0, 32, dtype=torch.int64)
        self.counters["history_readbacks"] += 1
        history = ttnn.to_torch(ttnn.get_device_tensors(self.token_history)[0])
        return history[: self.history_count, 0, 0, :].long()

    def _release_traces(self, *, keep_prefill=False):
        for trace in (self.trace, self.sample_trace, self.prefill_trace, self.prefill_sample_trace):
            if trace is not None:
                ttnn.release_trace(self.mesh, trace)
        self.trace = self.sample_trace = self.prefill_trace = self.prefill_sample_trace = None
        if getattr(self.model, "_resident_decode_bucket", None) is not None:
            self.model.flush_decode_bucket()
        self.trace_records_history = None
        if not keep_prefill:
            self.prefill_prepared = None

    def _prefill_trace_logits(self):
        """Device-only prefill over generator-owned inputs; output remains transient."""
        prepared = self.prefill_prepared
        resident = getattr(self.model, "_resident_decode_bucket", None)
        cache = self.cache
        if resident is not None and resident[1] == (0,) and self.model._resident_decode_valid:
            cache = resident[2]
        logits = self.model.prefill(
            prepared["tokens"],
            cache=cache,
            page_table=self.page_table,
            length=prepared["length"],
            positions=prepared["positions"],
        )
        return ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)

    def _prefill_trace_step(self):
        # The copy target predates every trace. No capture-created output may
        # remain alive when an earlier decode trace is replayed.
        ttnn.copy(self._prefill_trace_logits(), self.prefill_prepared["output"])

    def _prefill_for_generate(self, tokens, *, trace_sampling=False):
        """Own one prefill result until first-token sampling consumes it."""
        resident = getattr(self.model, "_resident_decode_bucket", None)
        if resident is not None and not (resident[1] == (0,) and self.model._resident_decode_valid):
            self.model.suspend_decode_bucket()
        length = tokens.shape[-1]
        if (tokens < 0).any() or (tokens >= self.model.config.vocab_size).any():
            raise ValueError("Token IDs lie outside the vocabulary")
        key = (
            id(self.cache),
            self.cache.batch_size,
            self.cache.capacity,
            self.cache.num_pages,
            id(self.page_table),
            tuple(self.page_table.shape),
            0,  # Physical slot.
            0,  # Prefix position.
            length,
            False,  # Last-token logits only; public all-logits prefill stays eager.
            trace_sampling,
        )
        if self.prefill_prepared is None or self.prefill_prepared["key"] != key:
            # Fresh persistent buffers must not overlap any live trace's scratch.
            self._release_traces()
            self.prefill_prepared = dict(
                key=key,
                length=length,
                trace_sampling=trace_sampling,
                tokens=self.model.upload(tokens.int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                positions=self.model.upload(
                    torch.arange(length, dtype=torch.int32).reshape(1, length),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            )
            # This is the real prefill, so it both warms the graph and advances
            # request state exactly once. Allocate and warm the output copy too.
            logits = self._prefill_trace_logits()
            self.prefill_prepared["output"] = ttnn.clone(logits)
            ttnn.copy(logits, self.prefill_prepared["output"])
            self.counters["prefill_eager_calls"] += 1
        else:
            self._copy(tokens.int(), self.prefill_prepared["tokens"], "prefill_token_refreshes")
            if self.prefill_trace is None:
                # G=1 need not create a decode pair just to capture prefill.
                self._prefill_trace_step()
                self.counters["prefill_eager_calls"] += 1
            else:
                ttnn.execute_trace(self.mesh, self.prefill_trace, cq_id=0, blocking=False)
                self.counters["prefill_replays"] += 1
        self.prefill_signatures.add((self.cache.batch_size, tuple(self.page_table.shape), 0, 0, length, False))
        return self.prefill_prepared["output"]

    def serving_prefill_tokens(self, tokens, *, page_table, kv_cache, prompt_lens, start_pos, slots):
        """Sample scheduler rows; prompt_lens are absolute exclusive prompt ends."""
        if kv_cache is not self.cache:
            raise ValueError("Serving prefill requires the exact bound cache")
        tokens = torch.as_tensor(tokens)
        ends = torch.as_tensor(prompt_lens).reshape(-1).tolist()
        starts = torch.as_tensor(start_pos).reshape(-1).tolist()
        slots = list(slots)
        if (
            tokens.ndim != 2
            or not ends
            or tokens.shape[0] != len(ends)
            or len(starts) != len(ends)
            or len(slots) != len(ends)
            or len(set(slots)) != len(slots)
        ):
            raise ValueError("Serving prefill needs one token row, start, end and distinct slot per prompt")
        for start, end, slot in zip(starts, ends, slots):
            if not 0 <= slot < kv_cache.batch_size or not 0 <= start < end <= min(tokens.shape[1], kv_cache.capacity):
                raise ValueError("Serving prompt positions or slots exceed the bound cache")
        if len(ends) == 1 and slots == [0] and starts == [0] and ends[0] <= 4096:
            self._refresh_table(page_table)
            output = self._prefill_for_generate(tokens[:, : ends[0]], trace_sampling=True)
            if self.prefill_sample_trace is None:
                # The real first sample warms the graph without advancing RNG twice.
                self._sampling_step(output)
                self.counters["prefill_sampling_eager_calls"] += 1
            else:
                ttnn.execute_trace(self.mesh, self.prefill_sample_trace, cq_id=0, blocking=False)
                self.counters["prefill_sampling_replays"] += 1
            return self.tokens
        signature = self._prepare_serving_prefill_sampling(
            tokens, page_table=page_table, kv_cache=kv_cache, ends=ends, starts=starts, slots=slots
        )
        # Preparation owns every public logit and packed temporary. They must
        # die before replay can reuse the sampling trace's scratch addresses.
        if self.prefill_sample_trace is None:
            self._sampling_step(self.prefill_sample_input)
            self.counters["prefill_sampling_eager_calls"] += 1
        else:
            ttnn.execute_trace(self.mesh, self.prefill_sample_trace, cq_id=0, blocking=False)
            self.counters["prefill_sampling_replays"] += 1
        self._prefill_sampling_signatures.add(signature)
        return self.tokens

    def _prepare_serving_prefill_sampling(self, tokens, *, page_table, kv_cache, ends, starts, slots):
        """Copy packed logits into cache-bound storage; return no device temporaries."""
        outputs = []
        grouped = (
            getattr(self, "batched_prefill", False)
            and len(ends) > 1
            and len(set(starts)) == 1
            and len(set(ends)) == 1
            and starts[0] % 32 == 0
            and slots == list(range(slots[0], slots[0] + len(slots)))
        )
        if grouped:
            outputs = self.prefill_forward(
                tokens[:, starts[0] : ends[0]],
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=[ends[0] - starts[0]] * len(ends),
                start_pos=starts,
                slots=slots,
            )
        for row, (start, end, slot) in enumerate(zip(starts, ends, slots)) if not grouped else []:
            outputs.extend(
                self.prefill_forward(
                    tokens[row : row + 1, start:end],
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=[end - start],
                    start_pos=[start],
                    slots=[slot],
                )
            )
        if getattr(self, "_prefill_sampling_cache", None) is not self.cache:
            self._prefill_sampling_signatures = set()
            self._prefill_sampling_cache = self.cache
        signature = tuple((tuple(x.shape), x.dtype, x.layout) for x in outputs)
        if signature not in self._prefill_sampling_signatures or self.prefill_sample_input is None:
            # New packing programs and the persistent destination must predate
            # capture, including a previously warmed shape's first staged use.
            self._release_traces(keep_prefill=True)
        packed = ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
        packed = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - len(outputs)), (0, 0)], value=0.0)
        if self.prefill_sample_input is None:
            self.prefill_sample_input = ttnn.clone(packed)
            self.counters["prefill_sampling_input_allocations"] += 1
        ttnn.copy(packed, self.prefill_sample_input)
        self.counters["prefill_sampling_input_copies"] += 1
        return signature

    def _ensure_cache(self, batch, capacity):
        if (
            self.owns_cache
            and self.cache is not None
            and self.cache.batch_size == batch
            and self.cache.capacity >= capacity
        ):
            return self.cache
        self._release_traces()
        self.prefill_sample_input = None
        self.prefill_signatures.clear()
        self.remaining_steps = None
        self.owns_cache = True
        self.cache = self.model.allocate_cache(
            batch_size=batch, capacity=min(self.model.context, ((capacity + 31) // 32) * 32)
        )
        self.positions = self.model.upload(
            torch.zeros(batch, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        pages = self.cache.num_pages // batch
        self.page_host = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
        self.page_table = self.model.upload(self.page_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.rope_indices = self.model.upload(
            torch.zeros(batch, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.active_slots = None
        self._reset_history()
        self.counters["page_table_allocations"] += 1
        return self.cache

    def _refresh_table(self, table):
        if isinstance(table, ttnn.Tensor):
            if table is not self.page_table:
                raise ValueError("Device page tables must retain their bound tensor identity; update in place")
            self.page_host = None  # Caller may have mutated the device table in place.
            return
        table = torch.as_tensor(table, dtype=torch.int32)
        if (table < 0).any() or (table >= self.cache.num_pages).any():
            raise ValueError("Page IDs lie outside the bound physical cache")
        if self.page_host is not None and torch.equal(table, self.page_host):
            return
        if tuple(table.shape) != tuple(self.page_table.shape):
            raise ValueError("Page-table geometry changes require a new cache binding")
        self._copy(table, self.page_table, "page_table_refreshes")
        self.page_host = table.clone()

    def reset(self):
        if getattr(self.model, "_resident_decode_bucket", None) is not None:
            self.model.suspend_decode_bucket()
        if self.cache is not None:
            self.model.reset_cache(self.cache)
        if self.positions is not None:
            self._copy(torch.zeros(self.cache.batch_size, dtype=torch.int32), self.positions, "position_refreshes")
        self._copy(torch.zeros(1, 1, 1, 32, dtype=torch.int32), self.tokens, "token_refreshes")
        if self.rope_indices is not None:
            self._copy(torch.zeros(self.cache.batch_size, dtype=torch.int32), self.rope_indices, "rope_refreshes")
        self._copy(torch.arange(32, dtype=torch.int32) + self.seed + 1, self.sampler.seeds_tt_tensor, "seed_refreshes")
        self._reset_history()
        self.remaining_steps = None
        self.reset_active_slots = True
        self.counters["resets"] += 1

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        start_pos=None,
        **kwargs,
    ):
        if getattr(self.model, "_resident_decode_bucket", None) is not None:
            self.model.suspend_decode_bucket()
        # Public outputs retain their original independent ownership. They must
        # not be allocated while a future owned-prefill replay reserves scratch.
        if self.prefill_prepared is not None:
            self._release_traces()
        if kv_cache is not self.cache:
            self.bind_cache(kv_cache, page_table)
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        if tokens.ndim != 2 or tokens.shape[0] != len(prompt_lens):
            raise ValueError("Prefill needs one token row per prompt")
        if (tokens < 0).any() or (tokens >= self.model.config.vocab_size).any():
            raise ValueError("Token IDs lie outside the vocabulary")
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        starts = [0] * len(slots) if start_pos is None else list(start_pos)
        if len(starts) != len(slots) or len(slots) != len(prompt_lens) or len(set(slots)) != len(slots):
            raise ValueError("Each prompt must name one distinct fixed slot")
        signatures = {
            (kv_cache.batch_size, tuple(page_table.shape), slot, start, length, return_all_logits)
            for slot, start, length in zip(slots, starts, prompt_lens)
        }
        if getattr(self, "batched_prefill", False) and len(slots) > 1:
            signatures.add(("batched", tuple(slots), tuple(starts), tuple(prompt_lens), return_all_logits))
        # New prefill programs own persistent buffers. Compile them before a
        # decode trace reserves scratch addresses; known shapes reuse the trace.
        if not signatures.issubset(self.prefill_signatures):
            self._release_traces()
        self._refresh_table(page_table)
        table = self.page_table
        if (
            getattr(self, "batched_prefill", False)
            and len(slots) > 1
            and not return_all_logits
            and len(set(starts)) == 1
            and len(set(prompt_lens)) == 1
            and starts[0] % 32 == 0
            and slots == list(range(slots[0], slots[0] + len(slots)))
        ):
            length, start = prompt_lens[0], starts[0]
            if not 1 <= length <= tokens.shape[-1] or start < 0 or start + length > kv_cache.capacity:
                raise ValueError("Invalid batched prompt length or prefix")
            for offset in range(0, length, 4096):
                count = min(4096, length - offset)
                ids = self.model.upload(
                    tokens[:, offset : offset + count].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                results = self.model.prefill_batch(
                    ids,
                    cache=kv_cache,
                    page_table=table,
                    length=count,
                    start_pos=start + offset,
                    slots=slots,
                    **(
                        {"return_logits": False}
                        if getattr(self, "skip_intermediate_prefill_head", False) and offset + count < length
                        else {}
                    ),
                )
            self.prefill_signatures.update(signatures)
            self.counters["batched_prefill_calls"] += 1
            return results
        results = []
        for row, (slot, length, start) in enumerate(zip(slots, prompt_lens, starts)):
            if not 0 <= slot < kv_cache.batch_size or not 1 <= length <= tokens.shape[-1]:
                raise ValueError("Invalid slot or logical prompt length")
            if start < 0 or start + length > kv_cache.capacity:
                raise ValueError("Prompt exceeds cache capacity")
            # Chunk across the whole stack, retaining only one chunk of activations.
            chunks = []
            for offset in range(0, length, 4096):
                count = min(4096, length - offset)
                ids = self.model.upload(
                    tokens[row : row + 1, offset : offset + count].int(),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                logits = self.model.prefill(
                    ids,
                    cache=kv_cache,
                    page_table=table,
                    length=count,
                    start_pos=start + offset,
                    slot=slot,
                    all_logits=return_all_logits,
                )
                if return_all_logits:
                    chunks.append(self._host_logits(logits))
            results.append(torch.cat(chunks, dim=1) if return_all_logits else logits)
        self.prefill_signatures.update(signatures)
        if return_all_logits:
            maximum = max(prompt_lens)
            return torch.cat([torch.nn.functional.pad(x, (0, 0, 0, maximum - x.shape[1])) for x in results], dim=0)
        return results

    def prefill_logits(self, prompt_token_ids):
        length = len(prompt_token_ids)
        cache = self._ensure_cache(1, length)
        self.reset()
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.page_table,
            kv_cache=cache,
            prompt_lens=[length],
            return_all_logits=True,
        )

    def _model_step(self):
        logits = self.model.decode(
            self.tokens,
            self.positions,
            cache=self.cache,
            page_table=self.page_table,
            rope_indices=self.rope_indices,
            active_slots=self.active_slots,
        )
        b = self.cache.batch_size
        if (
            self.active_slots is not None
            and len(self.active_slots) != b
            and not getattr(self.model, "decode_buckets", False)
        ):
            # Inactive SDPA rows are unwritten; never sample their undefined logits.
            zero = ttnn.zeros_like(logits[:, :, :1, :])
            rows = [logits[:, :, slot : slot + 1, :] if slot in self.active_slots else zero for slot in range(b)]
            logits = ttnn.concat(rows, dim=2) if b > 1 else rows[0]
        if b != 32:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - b), (0, 0)], value=0.0)
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        ttnn.plus_one(self.rope_indices)
        return logits

    def _sampling_step(self, logits):
        self.sampler(logits, tt_out_tok=self.tokens)
        ttnn.plus_one(self.sampler.seeds_tt_tensor)

    def sample_prefill(self, logits):
        """Sample packed prompt results with the canonical token-feedback sampler."""
        if getattr(self, "_prefill_sampling_cache", None) is not self.cache:
            self._prefill_sampling_signatures = set()
            self._prefill_sampling_cache = self.cache
        signature = tuple((tuple(x.shape), x.dtype, x.layout) for x in logits)
        warmed = getattr(self, "_prefill_sampling_signatures", set())
        if signature not in warmed:
            # Concat/padding can allocate persistent program buffers. Warm each
            # packed shape before capturing decode scratch addresses again.
            self._release_traces(keep_prefill=True)
        packed = ttnn.concat(logits, dim=2) if len(logits) > 1 else logits[0]
        packed = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - len(logits)), (0, 0)], value=0.0)
        self._sampling_step(packed)
        warmed.add(signature)
        self._prefill_sampling_signatures = warmed
        return self.tokens

    def reset_recurrent_slots(self, slots):
        """Start new requests without clearing any scheduler-owned attention pages."""
        resident = getattr(self.model, "_resident_decode_bucket", None)
        if resident is not None and resident[1] == (0,) and list(slots) == [0] and self.model._resident_decode_valid:
            # The owned slot-zero prefill trace uses this same B1 state. Reset
            # it directly; publishing/resetting/regathering all 16 rows would
            # add latency without preserving any live request in slot zero.
            for state in resident[2].layers:
                for name in ("conv", "recurrent"):
                    tensor = getattr(state, name)
                    if tensor is not None:
                        ttnn.copy(ttnn.zeros_like(tensor), tensor)
            return
        if getattr(self.model, "_resident_decode_bucket", None) is not None:
            self.model.suspend_decode_bucket(discard_slots=slots)
        if getattr(self, "_recurrent_reset_warmed", None) is not self.cache:
            self._release_traces(keep_prefill=True)
        for state in self.cache.layers:
            for name in ("conv", "recurrent"):
                tensor = getattr(state, name)
                if tensor is None:
                    continue
                parts = [tensor[i : i + 1] for i in range(self.cache.batch_size)]
                for slot in slots:
                    parts[slot] = ttnn.zeros_like(parts[slot])
                ttnn.copy(ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0], tensor)
        # All row slices and the same-shaped zero/concat/copy programs are now
        # warm. Their transient outputs die here, before any decode replay.
        self._recurrent_reset_warmed = self.cache

    def remap_recurrent_slots(self, remap):
        """Move constant-size request state on device when the scheduler compacts rows."""
        order = torch.as_tensor(remap).reshape(-1).tolist()
        if sorted(order) != list(range(self.cache.batch_size)):
            raise ValueError("Recurrent slot remap must be a complete permutation")
        if order == list(range(self.cache.batch_size)):
            return
        self._release_traces(keep_prefill=True)
        for state in self.cache.layers:
            for name in ("conv", "recurrent"):
                tensor = getattr(state, name)
                if tensor is not None:
                    ttnn.copy(ttnn.concat([tensor[i : i + 1] for i in order], dim=0), tensor)

    def set_batch_sampling_params(self, *, top_k, top_p, temperature, seed=None):
        """Bind scheduler sampling rows; steady replay retains device RNG state."""
        if not all(len(values) == 32 for values in (top_k, top_p, temperature)) or (
            seed is not None and len(seed) != 32
        ):
            raise ValueError("Sampling parameters must cover all 32 physical rows")
        if any(not 1 <= k <= 32 for k in top_k) or any(t <= 0 for t in temperature):
            raise ValueError("Device sampling requires k1..32 and positive temperature")
        self.sampler.reset_params(top_k, top_p, [1.0 / t for t in temperature])
        if seed is not None:
            self._copy(torch.tensor(seed, dtype=torch.int32), self.sampler.seeds_tt_tensor, "seed_refreshes")

    def _capture(self, *, record_history=False):
        """Warm both graphs before capture; restore only mutable request state."""
        decode_cache = self.cache
        if getattr(self.model, "decode_buckets", False):
            decode_cache = self.model.prepare_decode_bucket(self.cache, self.active_slots)
        backups = [
            (state, name, ttnn.clone(getattr(state, name)))
            for state in decode_cache.layers
            for name in ("conv", "recurrent")
            if getattr(state, name) is not None
        ]
        token_backup, pos_backup = ttnn.clone(self.tokens), ttnn.clone(self.positions)
        rope_backup = ttnn.clone(self.rope_indices)
        seed_backup = ttnn.clone(self.sampler.seeds_tt_tensor)
        cursor_backup = ttnn.clone(self.history_cursor) if record_history else None
        warm = self._model_step()
        self._sampling_step(warm)
        if record_history:
            self._append_history()
        for state, name, tensor in backups:
            ttnn.copy(tensor, getattr(state, name))
        for source, target in (
            (token_backup, self.tokens),
            (pos_backup, self.positions),
            (rope_backup, self.rope_indices),
            (seed_backup, self.sampler.seeds_tt_tensor),
        ):
            ttnn.copy(source, target)
        if record_history:
            # Warmup touched only the next unused row. Real replay overwrites it;
            # restoring the cursor preserves every already-recorded output.
            ttnn.copy(cursor_backup, self.history_cursor)
        ttnn.synchronize_device(self.mesh)
        captured = []
        open_trace = None
        prefill_trace = prefill_sample_trace = None
        try:
            open_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            captured.append(open_trace)
            logits = self._model_step()
            ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
            open_trace = None
            open_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            captured.append(open_trace)
            self._sampling_step(logits)
            if record_history:
                self._append_history()
            ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
            open_trace = None
            if self.prefill_prepared is not None:
                # Record only after decode logits have acquired a stable address.
                # The real prefill already warmed this exact graph. Recording it
                # does not execute or mutate the current decode cache state.
                open_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                captured.append(open_trace)
                prefill_trace = open_trace
                self._prefill_trace_step()
                ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
                open_trace = None
                if self.prefill_prepared["trace_sampling"]:
                    open_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                    captured.append(open_trace)
                    prefill_sample_trace = open_trace
                    self._sampling_step(self.prefill_prepared["output"])
                    ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
                    open_trace = None
            elif self.prefill_sample_input is not None:
                open_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                captured.append(open_trace)
                prefill_sample_trace = open_trace
                self._sampling_step(self.prefill_sample_input)
                ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
                open_trace = None
        except BaseException:
            if open_trace is not None:
                ttnn.end_trace_capture(self.mesh, open_trace, cq_id=0)
            for trace in captured:
                ttnn.release_trace(self.mesh, trace)
            raise
        self.trace, self.sample_trace = captured[:2]
        self.prefill_trace = prefill_trace
        self.prefill_sample_trace = prefill_sample_trace
        self.trace_records_history = record_history
        self.logits = logits
        self.counters["trace_captures"] += 2
        self.counters["prefill_trace_captures"] += int(self.prefill_trace is not None)
        self.counters["prefill_sampling_trace_captures"] += int(self.prefill_sample_trace is not None)

    def decode_forward(
        self,
        tokens=None,
        start_pos=None,
        *,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        record_history=False,
        host_sampling=None,
        active_slots=None,
        **kwargs,
    ):
        if kv_cache is not self.cache:
            raise ValueError("Bind external cache with bind_cache before traced decode")
        compat = self.host_sampling if host_sampling is None else host_sampling
        if record_history:
            if compat:
                raise ValueError("Token history requires device sampling")
            if self.token_history is None:
                raise ValueError("Reserve token history before recording decode outputs")
            if self.history_count >= self.history_capacity:
                raise ValueError("Token history capacity is exhausted")
        if self.reset_active_slots:
            if active_slots is None and self.active_slots is not None:
                self._release_traces(keep_prefill=True)
                self.active_slots = None
            self.reset_active_slots = False
        if active_slots is not None and tuple(active_slots) != self.active_slots:
            if start_pos is None:
                raise ValueError("Changing active slots requires authoritative positions")
            proposed = tuple(active_slots)
            if len(set(proposed)) != len(proposed) or any(i < 0 or i >= kv_cache.batch_size for i in proposed):
                raise ValueError("Invalid active-slot set")
            supplied_positions = torch.as_tensor(start_pos).reshape(-1)
            if supplied_positions.numel() != kv_cache.batch_size or any(
                supplied_positions[i] < 0 or supplied_positions[i] >= kv_cache.capacity for i in proposed
            ):
                raise ValueError("Active slots require positions inside the cache")
            self._release_traces(keep_prefill=True)
            self.active_slots = proposed
            if getattr(self.model, "decode_buckets", False) and proposed:
                bucket = next((size for size in (1, 8, 16) if size >= len(proposed)), None)
                if bucket is None:
                    raise ValueError("Bucketed decode supports at most 16 active requests")
                logger.debug(
                    "Qwen3.8 decode bucket: active={} shape={} capacity={}",
                    len(proposed),
                    bucket,
                    kv_cache.batch_size,
                )
        if not enable_trace:
            raise ValueError("Optimized decode requires tracing")
        self._refresh_table(page_table)
        if tokens is not None:
            supplied = torch.as_tensor(tokens).reshape(-1)
            if (
                supplied.numel() != kv_cache.batch_size
                or (supplied < 0).any()
                or (supplied >= self.model.config.vocab_size).any()
            ):
                raise ValueError("Decode requires one valid token per fixed slot")
            ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            ids.reshape(-1)[: kv_cache.batch_size] = supplied
            self._copy(ids, self.tokens, "token_refreshes")
        if start_pos is not None:
            positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
            if self.active_slots is not None:
                positions = positions.clone()
                positions[[i for i in range(kv_cache.batch_size) if i not in self.active_slots]] = -1
            if (
                positions.numel() != kv_cache.batch_size
                or (positions < -1).any()
                or (positions >= kv_cache.capacity).any()
            ):
                raise ValueError("Decode positions lie outside the cache")
            if self.active_slots is None and (positions < 0).any():
                raise ValueError("Negative positions require an explicit active-slot set")
            active = list(range(kv_cache.batch_size)) if self.active_slots is None else list(self.active_slots)
            if active and (positions[active] < 0).any():
                raise ValueError("Active slots require nonnegative positions")
            self.remaining_steps = int((kv_cache.capacity - positions[active]).min()) if active else 0
            self._copy(positions, self.positions, "position_refreshes")
            self._copy(positions.clamp_min(0), self.rope_indices, "rope_refreshes")
        if self.remaining_steps is None or self.remaining_steps <= 0:
            raise ValueError("No decode positions are bound, or cache capacity is exhausted")
        if self.trace is not None and self.trace_records_history != record_history:
            self._release_traces(keep_prefill=True)
        if self.trace is None:
            self._capture(record_history=record_history)
        elif getattr(self.model, "_resident_decode_bucket", None) is not None:
            self.model.resume_decode_bucket()
        ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        self.remaining_steps -= 1
        if compat:
            # Explicit test-only compatibility boundary. Never used in token-out timings.
            logits = self._host_logits(self.logits)[0, 0, : kv_cache.batch_size]
            return logits
        ttnn.execute_trace(self.mesh, self.sample_trace, cq_id=0, blocking=False)
        self.counters["sampling_replays"] += 1
        if record_history:
            self.history_count += 1
            self.counters["history_appends"] += 1
        return self._read_tokens()[: kv_cache.batch_size] if read_from_device else self.tokens

    def bind_cache(self, cache, page_table):
        if not 1 <= cache.batch_size <= 32 or not 1 <= cache.capacity <= self.model.context:
            raise ValueError("External cache dimensions exceed the model contract")
        shape = (
            tuple(page_table.shape) if isinstance(page_table, ttnn.Tensor) else tuple(torch.as_tensor(page_table).shape)
        )
        if len(shape) != 2 or shape[0] != cache.batch_size or shape[1] < (cache.capacity + 31) // 32:
            raise ValueError("Page table must cover the advertised cache capacity for every slot")
        if len(cache.layers) != len(self.model.layers):
            raise ValueError("Cache must cover every model layer")
        for layer, state in zip(self.model.layers, cache.layers):
            if layer.kind == "full_attention":
                if any(
                    t is None
                    or t.dtype != getattr(ttnn, layer.policy["kv_dtype"])
                    or t.layout != ttnn.TILE_LAYOUT
                    or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    or tuple(t.shape) != (cache.num_pages, 1, 32, 256)
                    for t in (state.key, state.value)
                ):
                    raise ValueError("Full-attention cache violates the TP4 selected-dtype page contract")
            elif (
                state.recurrent is None
                or tuple(state.recurrent.shape) != (cache.batch_size, 12, 128, 128)
                or state.recurrent.dtype != ttnn.float32
                or state.recurrent.layout != ttnn.TILE_LAYOUT
                or state.recurrent.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                or state.conv is None
                or tuple(state.conv.shape) != (cache.batch_size, 3, 2560)
                or state.conv.dtype != ttnn.bfloat16
                or state.conv.layout != ttnn.ROW_MAJOR_LAYOUT
                or state.conv.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError("Linear state violates the FP32 recurrence / BF16 row-major convolution contract")
        if isinstance(page_table, ttnn.Tensor):
            if (
                page_table.dtype != ttnn.int32
                or page_table.layout != ttnn.ROW_MAJOR_LAYOUT
                or page_table.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            ):
                raise ValueError("Device page table must be INT32 row major in interleaved DRAM")
            page_host = None
        else:
            page_host = torch.as_tensor(page_table, dtype=torch.int32).clone()
            if (page_host < 0).any() or (page_host >= cache.num_pages).any():
                raise ValueError("Invalid external page table")
        self._release_traces()
        self.prefill_sample_input = None
        self.prefill_signatures.clear()
        self.remaining_steps = None
        self.cache = cache
        self.owns_cache = False
        self.positions = self.model.upload(
            torch.zeros(cache.batch_size, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.rope_indices = self.model.upload(
            torch.zeros(cache.batch_size, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.active_slots = None
        self.page_host = page_host
        if isinstance(page_table, ttnn.Tensor):
            self.page_table = page_table
        else:
            self.page_table = self.model.upload(self.page_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._reset_history()

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        defer_token_readback=True,
        trace_prefill=True,
        host_sampling=None,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Readiness and generation require traced decode")
        if (
            max_new_tokens < 0
            or not prompt_token_ids
            or len(prompt_token_ids) + max(0, max_new_tokens - 1) > self.model.context
        ):
            raise ValueError("Invalid generation length or context")
        if max_new_tokens == 0:
            return []
        compat = self.host_sampling if host_sampling is None else host_sampling
        deferred = bool(defer_token_readback and not compat and next_input is None and max_new_tokens > 1)
        if compat and top_k != 1:
            raise ValueError("Host compatibility implements greedy sampling; use device sampling for top-k/top-p")
        start = time.perf_counter()
        before_request = self.counters.copy()
        self.set_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature, seed=seed)
        cache = self._ensure_cache(1, len(prompt_token_ids) + max_new_tokens - 1)
        if deferred:
            self._ensure_history(max_new_tokens - 1)
        self.reset()
        # Bound trace storage to one existing prefill chunk. Longer prompts
        # retain the full-context, chunked eager path without limiting inputs.
        use_prefill_trace = bool(trace_prefill and len(prompt_token_ids) <= 4096)
        if use_prefill_trace:
            padded = self._prefill_for_generate(torch.tensor([prompt_token_ids]))
        else:
            logits = self.prefill_forward(
                torch.tensor([prompt_token_ids]),
                page_table=self.page_table,
                kv_cache=cache,
                prompt_lens=[len(prompt_token_ids)],
            )[0]
            padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
            del logits
            self.counters["prefill_eager_calls"] += 1
        if compat:
            first = int(self._host_logits(padded)[0, 0, 0].argmax())
        else:
            self._sampling_step(padded)
            first = int(self._read_tokens()[0])
        del padded
        ttft = time.perf_counter() - start
        outputs = [first]
        forced = next_input(0, first) if next_input is not None else first
        self._copy(torch.tensor([len(prompt_token_ids)], dtype=torch.int32), self.positions, "position_refreshes")
        self._copy(torch.tensor([len(prompt_token_ids)], dtype=torch.int32), self.rope_indices, "rope_refreshes")
        if next_input is not None or compat:
            ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            ids.reshape(-1)[0] = forced
            self._copy(ids, self.tokens, "token_refreshes")
        self.remaining_steps = cache.capacity - len(prompt_token_ids)
        before_steady = self.counters.copy()
        begin = time.perf_counter()
        for step in range(1, max_new_tokens):
            out = self.decode_forward(
                page_table=self.page_table,
                kv_cache=cache,
                host_sampling=compat,
                read_from_device=not deferred,
                record_history=deferred,
            )
            if deferred:
                continue
            predicted = int(out[0].argmax()) if compat else int(out[0])
            outputs.append(predicted)
            forced = next_input(step, predicted) if next_input is not None else predicted
            if (next_input is not None or compat) and step + 1 < max_new_tokens:
                ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ids.reshape(-1)[0] = forced
                self._copy(ids, self.tokens, "token_refreshes")
        after_steady = self.counters.copy()
        if deferred:
            # The final blocking read drains queued decode and belongs in its
            # elapsed time, while delivery counters remain outside the loop.
            outputs.extend(self._read_history()[:, 0].tolist())
        elapsed = time.perf_counter() - begin
        self.last_perf = dict(
            ttft_s=ttft,
            decode_s=elapsed,
            decode_tokens=max_new_tokens - 1,
            tokens_per_second=(max_new_tokens - 1) / elapsed,
            counters=dict(self.counters - before_request),
            steady_state_counters=dict(after_steady - before_steady),
            delivery_counters=dict(self.counters - after_steady),
            deferred_token_readback=deferred,
            prefill_trace_eligible=use_prefill_trace,
            history_capacity=self.history_capacity if deferred else 0,
            host_sampling=compat,
            teacher_forcing=next_input is not None,
        )
        return outputs

    def set_sampling_params(self, *, top_k=1, top_p=0.0, temperature=1.0, seed=0):
        if not 1 <= top_k <= 32 or not 0 <= top_p <= 1 or temperature <= 0:
            raise ValueError("Sampling requires k1..32, p0..1 and positive temperature")
        self.seed = seed
        previous = self.sampler.force_argmax_sampling
        # The common native sampler multiplies by inverse temperature.
        self.sampler.reset_params([top_k] * 32, [top_p] * 32, [1.0 / temperature] * 32)
        self._copy(torch.arange(32, dtype=torch.int32) + seed + 1, self.sampler.seeds_tt_tensor, "seed_refreshes")
        if previous != self.sampler.force_argmax_sampling:
            self._release_traces()

    def teardown(self):
        self.close()

    def close(self):
        self._release_traces()


def build_generator(model_dir, mesh_device, **kwargs):
    indices = kwargs.pop("layer_indices", None)
    model = Qwen38Model(
        mesh_device,
        snapshot=kwargs.pop("snapshot", None),
        layer_indices=indices,
        head_strategy=kwargs.pop("head_strategy", "dram"),
        precision_config=kwargs.pop("precision_config", None),
    )
    return Qwen38Generator(
        model,
        host_sampling=kwargs.pop("host_sampling", False),
        sampling_strategy=kwargs.pop("sampling_strategy", "split"),
    )
