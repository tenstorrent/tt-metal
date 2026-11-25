# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ornith readiness generator: stable request state and split token-out traces."""
import secrets
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import ttnn
from models.common.sampling.tt_penalties import TTPenalties
from models.common.sampling.tt_sampling import TTSampling

from .model import OrnithModel, measured_kv_capacity
from .readiness_contract import Generator

DRAM = ttnn.DRAM_MEMORY_CONFIG
_UNSET = object()


@dataclass
class DecodeOutput:
    """One submission; enqueue its read before submitting another decode."""

    owner: object
    serial: int
    is_tokens: bool
    active: tuple
    source: object
    host: object = None
    events: object = None


class SamplingCCL:
    """Use the decoder's serialized collective workspace for terminal sampling."""

    def __init__(self, workspace):
        self.ccl = workspace.ccl

    def get_num_links(self, cluster_axis=None):
        return 2

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self.ccl.get_ag_ping_pong_semaphore()

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self.ccl.get_barrier_semaphore()

    def line_all_gather(self, tensor, *, dim, cluster_axis=None, memory_config=DRAM, num_links=1, buffer_key=None):
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            memory_config=memory_config,
        )


class OrnithGenerator(Generator):
    decode_input_update_contract = 1
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefill_sampling_mask": True,
        "supports_prefix_caching": False,  # recurrent_state: KV pages alone are insufficient.
        "output_tokens_per_step": 1,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_device_penalties": True,
        "supports_device_grammar": False,
        "supports_async_decode": True,
        "supports_compact_host_logits": False,
        "supports_selective_host_readback": False,
        "supports_spec_decode": False,
        "supports_async_spec_decode": False,
        "supports_stage_inputs": False,
        "supports_stage_outputs": False,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D},
    }

    @classmethod
    def get_max_tokens_all_users(cls, model_name, num_devices, tt_data_parallel, max_model_len, max_num_seqs):
        """Preconstruction budget from measured full-stack acceptance evidence.

        Refuse a missing measurement instead of inheriting the plugin's 131072
        fallback or mistaking a byte-arithmetic plan for device capacity.
        The saved budget excludes the plugin's additional output reservation.
        """
        if num_devices != 4 or tt_data_parallel != 1:
            raise ValueError("The selected Ornith model requires one TP4/EP4 1x4 mesh")
        record = measured_kv_capacity()
        if not 1 <= int(max_model_len) <= 262144 or not 1 <= int(max_num_seqs) <= 32:
            raise ValueError("Requested context/batch exceeds the checkpoint contract")
        return min(int(record["max_tokens_all_users"]), int(max_model_len) * int(max_num_seqs))

    def __init__(
        self, model, *, batch_size=1, max_seq_len=262144, cache_state=None, prefill_buckets=None, num_blocks=None
    ):
        self.model, self.device = model, model.device
        self.batch_size = int(batch_size)
        if not 1 <= self.batch_size <= 32:
            raise ValueError("batch_size must be 1..32")
        self.context = int(max_seq_len)
        if not 1 <= self.context <= model.config.max_position_embeddings:
            raise ValueError("Context exceeds checkpoint capability")
        self.capacity = (self.context + 255) // 256 * 256
        self.state = (
            model.create_state(context=self.capacity, batch_size=self.batch_size, num_blocks=num_blocks)
            if cache_state is None
            else cache_state
        )
        self.owns_cache = cache_state is None
        model.validate_state(self.state)
        if self.state.batch_size != self.batch_size or self.state.context < self.capacity:
            raise ValueError("External state must cover the configured batch and rounded context")
        full_pools = [
            rows[0].key for layer, rows in zip(model.layers, self.state.layers) if layer.kind == "full_attention"
        ]
        if not full_pools or len({int(pool.shape[0]) for pool in full_pools}) != 1:
            raise ValueError("All full-attention layers require the same physical page-pool geometry")
        self.num_blocks = int(full_pools[0].shape[0])
        if num_blocks is not None and self.num_blocks != num_blocks:
            raise ValueError("External cache num_blocks differs from the requested physical pool budget")
        initial_per_slot = min(
            self.capacity // 32, self.num_blocks // self.batch_size if self.state.shared_cache else self.num_blocks
        )
        if initial_per_slot < 1:
            raise ValueError("KV pool needs at least one page per initial warmup slot")
        self.tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
        self.counters = Counter()
        self.traces = {}
        self.prepared = False
        self._warm_finished = False
        self.closed = False
        self._pending_output = None
        self._serial = 0
        self._active = [True] * self.batch_size
        self._positions = [0] * self.batch_size
        self._decode_steps = 0
        self._slot_ends = [0] * self.batch_size
        self._sampling_clock = 0
        self._seed_epochs = [0] * self.batch_size
        self._seed_draws = [0] * self.batch_size
        self._sampling_spec = None
        self._sampler_param_cache = {}
        self._request_epochs = [0] * self.batch_size
        self.tokens = self.upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._prefill_token_backup = self.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.logits = ttnn.empty(
            (1, 1, 32, model.padded_vocab_size // 4),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=DRAM,
        )
        self._zero_logits_row = ttnn.zeros(
            (1, 1, 1, model.padded_vocab_size // 4),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=DRAM,
        )
        # Concat keys include input aliasing. Give each serving slot a distinct
        # zero row so partial prefill has the same alias pattern as warmup.
        self._slot_zero_logits_rows = [ttnn.clone(self._zero_logits_row) for _ in range(self.batch_size)]
        self.pages_host = torch.zeros(self.batch_size, self.capacity // 32, dtype=torch.int32)
        for slot in range(self.batch_size):
            offset = slot * initial_per_slot if self.state.shared_cache else 0
            self.pages_host[slot, :initial_per_slot] = torch.arange(
                offset, offset + initial_per_slot, dtype=torch.int32
            )
        if self.state.shared_cache:
            model.bind_page_tables(self.state, self.pages_host, valid_page_counts=[initial_per_slot] * self.batch_size)
        self.operands = []
        for slot in range(self.batch_size):
            self.operands.append(
                dict(
                    page_table=self.upload(self.pages_host[slot : slot + 1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    current_pos=self.upload(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    rope_index=self.upload(torch.zeros(1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                    active_mask=self.upload(torch.ones(1, 1, 1, 1)),
                    rope_increment=self.upload(torch.ones(1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                )
            )
        if self.state.shared_cache:
            for slot, op in enumerate(self.operands):
                op["page_table"] = self.state.row_page_tables[slot]
        args = SimpleNamespace(
            vocab_size=model.config.vocab_size,
            padded_vocab_size=model.padded_vocab_size,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            model_config={
                "SAMPLING_AG_CONFIG": dict(allow_force_argmax=True, num_links=2, topology=ttnn.Topology.Linear)
            },
        )
        self.sampler = TTSampling(mesh_device=self.device, tt_ccl=SamplingCCL(model.workspace), args=args)
        greedy_args = SimpleNamespace(**vars(args))
        greedy_args.max_top_k = 32
        greedy_args.model_config = {
            "SAMPLING_AG_CONFIG": dict(args.model_config["SAMPLING_AG_CONFIG"], allow_force_argmax=False)
        }
        self.greedy_sampler = TTSampling(mesh_device=self.device, tt_ccl=SamplingCCL(model.workspace), args=greedy_args)
        # Both traces consume the same mutable parameters and advance one RNG
        # stream. Width-specific immutable offsets/masks remain sampler-owned.
        # Drop unused constructor inputs before any warmup or live trace.
        for name in ("k_tensor", "p_tensor", "temp_tensor", "_greedy_col", "seeds_tt_tensor"):
            setattr(self.greedy_sampler, name, getattr(self.sampler, name))
        self.penalties = TTPenalties(mesh_device=self.device, args=args)
        self._penalty_dirty = [False] * self.batch_size
        self.seed_sentinel = self.upload(
            torch.full((32,), 0xFFFFFFFF, dtype=torch.uint32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.seed_increments = self.upload(
            torch.tensor([1] * self.batch_size + [0] * (32 - self.batch_size), dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        # One reusable snapshot per state shape and slot, not per layer. Slot
        # transitions execute serially and never replace captured state storage.
        self._move_scratch = {}
        if self.batch_size > 1:
            for name, value in model._state_tensors(self.state, [0]):
                if name in ("key", "value") and self.state.shared_cache:
                    continue
                key = model._reset_signature(value)
                if key not in self._move_scratch:
                    self._move_scratch[key] = tuple(
                        ttnn.empty(
                            value.shape,
                            dtype=value.dtype,
                            layout=value.layout,
                            device=self.device,
                            memory_config=value.memory_config(),
                        )
                        for _ in range(self.batch_size)
                    )
        self.buckets = tuple((*range(1, 129), 512, 2048) if prefill_buckets is None else prefill_buckets)
        if (
            not self.buckets
            or self.buckets[0] != 1
            or any(type(n) is not int or n < 1 for n in self.buckets)
            or tuple(sorted(set(self.buckets))) != self.buckets
        ):
            raise ValueError("Prefill lengths must be positive, sorted and unique, starting at 1")
        # Recurrent layers consume exactly the logical token count. Only full
        # attention pads to a tile, then each layer crops back to logical rows.
        physical = {n: 1 if n == 1 else (n + 31) // 32 * 32 for n in self.buckets}
        self.buckets = tuple(n for n in self.buckets if physical[n] <= min(self.capacity, initial_per_slot * 32))
        self._prefill_physical = {n: physical[n] for n in self.buckets}
        self._prefill_alignment = {
            n: max(
                (
                    min(layer.PREFILL_SDPA_Q_CHUNK, physical[n] & -physical[n])
                    for layer in model.layers
                    if layer.kind == "full_attention"
                ),
                default=1,
            )
            for n in self.buckets
        }
        self.prefill_buffers = {}
        for n in self.buckets:
            self.model.workspace.prepare(n)
            self.model.workspace.prepare(physical[n])
            self.prefill_buffers[n] = dict(
                tokens=self.upload(torch.zeros(1, 1, 1, n, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                cos=self.upload(torch.ones(1, 2, 32, 256) if n == 1 else torch.ones(1, 1, physical[n], 256)),
                sin=self.upload(torch.zeros(1, 2, 32, 256) if n == 1 else torch.zeros(1, 1, physical[n], 256)),
                position=self.upload(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                chunk_page_table=self.upload(
                    torch.zeros(1, max(1, physical[n] // 32), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                ),
            )
        # Final hidden rows survive later scalar replays. Packing the rows
        # avoids per-allocation DRAM-bank padding of individual output rows.
        self.prefill_hidden_rows = ttnn.empty(
            (1, 1, 32, model.layers[0].hidden_width),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=DRAM,
        )
        # B1's common chunk retains every output row for selected/all-logit
        # callers. Allocate before any trace so replay cannot clobber its owner.
        self.prefill_128_hidden = (
            ttnn.empty(
                (1, 1, 128, model.layers[0].hidden_width),
                # Match eager hidden dtype so all-logit slices use warmed keys.
                dtype=getattr(ttnn, model.precision["residual_dtype"]),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
            if self.batch_size == 1 and 128 in self.buckets
            else None
        )
        self._sampling_mode = "split"
        self.set_sampling()
        self._upload_penalty_emitters()

    def upload(self, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous(),
            device=self.device,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def copy(self, host, target, counter=None):
        donor = ttnn.from_torch(host.contiguous(), dtype=target.dtype, layout=target.layout)
        ttnn.copy_host_to_device_tensor(donor, target)
        if counter:
            self.counters[counter] += 1

    def _decode(self):
        operands = (
            self.operands
            if self.batch_size > 1
            else [{k: v for k, v in self.operands[0].items() if k not in ("active_mask", "rope_increment")}]
        )
        # Generator-owned increments agree with active masks at every setter.
        # UINT32 subtraction yields0 (active) orUINT_MAX (inactive), so the
        # existing indexed state-copy kernels can skip inactive destinations.
        indices = [ttnn.subtract(operand["rope_increment"], 1) for operand in operands] if self.batch_size > 1 else None
        value = self.model.decode_forward(self.tokens, state=self.state, operands=operands, state_write_indices=indices)
        if self.batch_size < 32:
            value = ttnn.pad(value, ((0, 0), (0, 0), (0, 32 - self.batch_size), (0, 0)), value=0)
        ttnn.copy(value, self.logits)
        for operand in self.operands:
            ttnn.plus_one(operand["current_pos"], skip_negative_entries=True)
            if self.batch_size == 1:
                ttnn.plus_one(operand["rope_index"])
            else:
                advanced = ttnn.add(operand["rope_index"], operand["rope_increment"])
                ttnn.copy(advanced, operand["rope_index"])

    def _sample(self, mode="split", *, penalties=False):
        sampler = self.greedy_sampler if mode == "greedy32" else self.sampler
        sampler._force_argmax_sampling = mode == "argmax"
        logits = self.penalties.apply(self.logits) if penalties else self.logits
        sampler(logits, tt_out_tok=self.tokens)
        if penalties:
            # Counts describe prior outputs when penalties are applied. The
            # new token is counted exactly once, inside the same traced step.
            self.penalties.update_output_tokens(self.tokens)
        updated_seeds = ttnn.add(sampler.seeds_tt_tensor, self.seed_increments)
        # manual_seed reserves UINT32_MAX as "retain hardware RNG state".
        # Advance in Z/(2**32-1), keeping every real draw explicitly seeded.
        updated_seeds = ttnn.where(ttnn.eq(updated_seeds, self.seed_sentinel), 0, updated_seeds)
        ttnn.copy(updated_seeds, sampler.seeds_tt_tensor)

    def _clear_penalty_slots(self, slots):
        slots = [slot for slot in slots if self._penalty_dirty[slot]]
        if not slots:
            return
        self.penalties.reset_prompt_tokens(torch.full((32, 1), -1, dtype=torch.int64), slots=slots)
        self.penalties.reset_output_tokens(slots=slots)
        for slot in slots:
            self._penalty_dirty[slot] = False

    def _set_penalty_prompts(self, tokens, ends, slots):
        prompts = torch.full((32, max(ends)), -1, dtype=torch.int64)
        for row, (end, slot) in enumerate(zip(ends, slots, strict=True)):
            prompts[slot, :end] = tokens[row, :end]
        self.penalties.reset_prompt_tokens(prompts, slots=slots)
        for slot in slots:
            self._penalty_dirty[slot] = True

    def _remap_penalties(self, remap, *, warm=False):
        if not warm and not any(self._penalty_dirty):
            return
        # Preserve captured destinations and never read counts back. Fresh
        # row allocations keep concat's alias pattern fixed for duplicate
        # sources as well as permutations. Every source offset is warmed.
        sources = remap + list(range(self.batch_size, 32))
        for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered"):
            target = getattr(self.penalties, name)
            source = ttnn.to_layout(target, ttnn.ROW_MAJOR_LAYOUT)
            # A replicated 262144-int row is a 1MiB row-major page: clone's
            # double-buffered page CB exceeds L1. Keep each logical request
            # as 32 narrow rows while slicing/cloning/concatenating, then
            # restore its exact original shape before tilizing into storage.
            source = ttnn.reshape(source, (32, 32, target.shape[-1] // 32))
            rows = [ttnn.clone(source[i : i + 1, :, :]) for i in sources]
            moved = ttnn.reshape(ttnn.concat(rows, dim=0), target.shape)
            moved = ttnn.to_layout(moved, ttnn.TILE_LAYOUT)
            ttnn.copy(moved, target)
        shadow = self.penalties._prompt_tokens_host
        if shadow is not None:
            self.penalties._prompt_tokens_host = shadow[sources].clone()
        self._penalty_dirty = [self._penalty_dirty[source] for source in remap]

    def _validated_penalty_history(self, prompt_tokens, output_tokens):
        if prompt_tokens is None or output_tokens is None:
            raise ValueError("Penalty sampling-state reset requires prompt and committed output histories")
        result = []
        for tokens in (prompt_tokens, output_tokens):
            tokens = torch.as_tensor(tokens, dtype=torch.int64)
            if tokens.ndim != 2 or tokens.shape[0] != self.batch_size:
                raise ValueError("Penalty histories need one row per configured slot")
            if ((tokens < -1) | (tokens >= self.model.config.vocab_size)).any():
                raise ValueError("Penalty history token outside vocabulary; padding must be -1")
            result.append(tokens)
        return result

    def _prefill(self, n, slot):
        if n == 128 and self.prepared and "prefill_128" in self.traces:
            ttnn.execute_trace(self.device, self.traces["prefill_128"], cq_id=0, blocking=False)
            self.counters["prefill_128_replays"] += 1
            return self.prefill_128_hidden
        if n == 1 and self.prepared and f"prefill_1_{slot}" in self.traces:
            ttnn.execute_trace(self.device, self.traces[f"prefill_1_{slot}"], cq_id=0, blocking=False)
            self.counters["scalar_prefill_replays"] += 1
            return self._prefill_hidden(slot)
        b = self.prefill_buffers[n]
        kw = dict(cos=b["cos"], sin=b["sin"], page_table=self.operands[slot]["page_table"])
        if n == 1:
            kw["current_pos"] = b["position"]
        else:
            kw.update(chunk_start=b["position"], chunk_page_table=b["chunk_page_table"])
        plans = {
            "full_attention": dict(seq_len=n, start_pos=0, chunks=[(0, n, self._prefill_physical[n], n == 1, kw)]),
            "linear_attention": dict(seq_len=n, start_pos=0, chunks=[(0, n, n, False, {})]),
        }
        return self.model.prefill_chunk(
            b["tokens"], state=self.state, slot=slot, plan_by_kind=plans, return_hidden=True
        )

    def _terminal_last(self, hidden, n):
        return self.model.terminal(hidden[:, :, n - 1 : n, :])

    def _retain_prefill_hidden(self, hidden, n, slot):
        value = ttnn.to_layout(hidden[:, :, n - 1 : n, :], ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(
            value,
            self.prefill_hidden_rows,
            [0, 0, slot, 0],
            [1, 1, slot + 1, self.model.layers[0].hidden_width],
            [1, 1, 1, 1],
        )

    def _prefill_hidden(self, slot):
        return ttnn.to_layout(self.prefill_hidden_rows[:, :, slot : slot + 1, :], ttnn.TILE_LAYOUT)

    def _capture_trace(self, key, call):
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        call()
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        self.traces[key] = tid
        self.counters["captures"] += 1
        print(f"ORNITH_CAPTURE key={key} trace_id={tid}", flush=True)

    def prepare(self, *, capture=True):
        """All program variants precede the first live trace, including eager helpers."""
        if self.prepared:
            return
        if self.closed:
            raise RuntimeError("Generator has been closed")
        if not self._warm_finished:
            self._warmup()
            self._warm_finished = True
        if not capture:
            return
        for key, call in (
            ("decode", self._decode),
            ("split", self._sample),
            ("greedy32", lambda: self._sample("greedy32")),
            ("argmax", lambda: self._sample("argmax")),
            ("split_penalties", lambda: self._sample(penalties=True)),
            ("greedy32_penalties", lambda: self._sample("greedy32", penalties=True)),
            ("argmax_penalties", lambda: self._sample("argmax", penalties=True)),
        ):
            self._capture_trace(key, call)
        # Keep the configured trace carveout and KV budget. All scalar slots
        # are already compiled; retain as many identical-size bindings as fit.
        # Other slots use the prepared eager prefill path, never recapture.
        self.scalar_prefill_trace_plan = dict(captured_slots=[], eager_slots=[], bytes_per_bank=[])
        scalar_reserve = None
        for slot in range(self.batch_size):
            before = ttnn.get_memory_view(self.device, ttnn.BufferType.TRACE)
            if scalar_reserve is not None and before.largest_contiguous_bytes_free_per_bank < scalar_reserve:
                self.scalar_prefill_trace_plan["eager_slots"].append(slot)
                continue
            self._capture_trace(
                f"prefill_1_{slot}", lambda: self._retain_prefill_hidden(self._prefill(1, slot), 1, slot)
            )
            after = ttnn.get_memory_view(self.device, ttnn.BufferType.TRACE)
            size = int(after.total_bytes_allocated_per_bank - before.total_bytes_allocated_per_bank)
            # Same programs/shapes and tensor argument counts for every slot;
            # leave one extra page per bank for command-buffer alignment.
            scalar_reserve = max(scalar_reserve or 0, size + 4096)
            self.scalar_prefill_trace_plan["captured_slots"].append(slot)
            self.scalar_prefill_trace_plan["bytes_per_bank"].append(size)
        if self.prefill_128_hidden is not None:
            self._refresh_prefill(128, [0] * 128, 0, 0)
            self._capture_trace("prefill_128", lambda: ttnn.copy(self._prefill(128, 0), self.prefill_128_hidden))
        self.prepared = True
        self._penalty_dirty = [True] * self.batch_size  # capture executed counted steps
        self.reset()
        print("PREPARED_SPLIT_TRACES", flush=True)

    def _warmup(self):
        for n in self.buckets:
            self._refresh_prefill(n, [0] * n, 0, 0)
            hidden = self._prefill(n, 0)
            if n == 128 and self.prefill_128_hidden is not None:
                ttnn.copy(hidden, self.prefill_128_hidden)
            last = self._terminal_last(hidden, n)
            if n == 1:
                padded_last = ttnn.pad(last, ((0, 0), (0, 0), (0, 31), (0, 0)), value=0)
                ttnn.copy(padded_last, self.logits)
                del padded_last
                self._assemble_prefill_logits([last] * self.batch_size, list(range(self.batch_size)))
                for slot in range(self.batch_size):
                    self._retain_prefill_hidden(hidden, n, slot)
                    tiled = self._prefill_hidden(slot)
                    # Scalar replay returns retained BF16 rows even with BF8
                    # residuals; warm its untilize/update signature explicitly.
                    self._retain_prefill_hidden(tiled, 1, slot)
                    terminal = self.model.terminal(tiled)
                    del terminal, tiled
            for offset in range(0, n, 32):
                tile = hidden[:, :, offset : min(offset + 32, n), :]
                out = self.model.terminal(tile)
                del out, tile
            if n != 1:
                # Final logical slices can create distinct program keys even
                # when several lengths share one padded full-attention shape.
                self._retain_prefill_hidden(hidden, n, 0)
                retained = self._prefill_hidden(0)
                terminal = self.model.terminal(retained)
                del terminal, retained
            del hidden, last
            print(f"PREPARED_PREFILL {n}", flush=True)
        # Every state-copy and token permutation variant is warmed before capture.
        self._remap_slots(list(range(self.batch_size)), warm=True)
        self._decode()
        for mode in ("split", "greedy32", "argmax"):
            self._sample(mode)
            self._sample(mode, penalties=True)
        # Warm partial clear/history operations and counted sampling before
        # capture; ordinary sampler precompile does not compile count updates.
        for slot in range(self.batch_size):
            self._penalty_dirty[slot] = True
            self._clear_penalty_slots([slot])
        self.penalties.reset_output_tokens(torch.zeros((self.batch_size, 1), dtype=torch.int64))
        self._penalty_dirty = [True] * self.batch_size
        # Compile the output copy and each cache-binding variant before any
        # trace exists. The following reset restores all warmup-mutated state.
        for slot in range(self.batch_size):
            self._retain_prefill_hidden(self._prefill(1, slot), 1, slot)
        self.reset()
        ttnn.synchronize_device(self.device)
        self.counters["synchronizations"] += 1

    def reset(self):
        self._require_read_enqueued()
        self.model.reset_state(self.state)
        self._clear_penalty_slots(list(range(self.batch_size)))
        self._slot_ends = [0] * self.batch_size
        self.bind(torch.zeros(self.batch_size, dtype=torch.int32), torch.zeros(self.batch_size, dtype=torch.int32))
        self.set_sampling()
        self.counters["resets"] += 1

    def _validate_pages(self, pages, *, extents=None):
        pages = torch.as_tensor(pages, dtype=torch.int32)
        if pages.shape != self.pages_host.shape or (pages < 0).any() or (pages >= self.num_blocks).any():
            raise ValueError("Page mapping must match configured per-slot cache capacity")
        # The plugin zero-pads unused rows/columns. Only logically occupied
        # pages are writable; duplicated zero padding is not shared KV.
        if extents is None:
            extents = [
                max(end, self._position(slot)) if self._active[slot] else 0 for slot, end in enumerate(self._slot_ends)
            ]
        occupied = [pages[slot, : (max(0, int(end)) + 31) // 32] for slot, end in enumerate(extents)]
        if self.state.shared_cache:
            used = torch.cat(occupied)
            if torch.unique(used).numel() != used.numel():
                raise ValueError("Writable shared KV pages must be disjoint")
        elif any(torch.unique(row).numel() != row.numel() for row in occupied):
            raise ValueError("A request's occupied logical pages must map to distinct physical pages")
        return pages

    def refresh_page_tables(self, pages, *, extents=None):
        self._require_read_enqueued()
        pages = self._validate_pages(pages, extents=extents)
        # All checks precede every device copy, including row-table copies.
        for slot, op in enumerate(self.operands):
            if not torch.equal(pages[slot], self.pages_host[slot]):
                self.copy(pages[slot : slot + 1], op["page_table"], "page_table_refreshes")
        if self.state.shared_cache and not torch.equal(pages, self.pages_host):
            self.copy(pages, self.state.page_table, "shared_page_table_refreshes")
        self.pages_host = pages.clone()

    def _position(self, slot):
        return self._positions[slot] + self._decode_steps if self._active[slot] else -1

    def bind(self, tokens, positions, page_table=None):
        self._require_read_enqueued()
        tokens = torch.as_tensor(tokens, dtype=torch.int64).flatten()
        positions = torch.as_tensor(positions, dtype=torch.int32).flatten()
        if tokens.numel() != self.batch_size or positions.numel() != self.batch_size:
            raise ValueError("One token and position per configured fixed slot required")
        if (tokens < 0).any() or (tokens >= self.model.config.vocab_size).any():
            raise ValueError("Token outside vocabulary")
        if (positions < -1).any() or (positions >= self.context).any():
            raise ValueError("Position outside context; inactive rows use -1")
        extents = [max(end, int(pos) + 1) if pos >= 0 else 0 for end, pos in zip(self._slot_ends, positions)]
        if page_table is not None:
            self._validate_pages(page_table, extents=extents)
        self._checkpoint_seed_counters()
        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        ids.flatten()[: self.batch_size] = tokens.to(torch.int32)
        self.copy(ids, self.tokens, "token_refreshes")
        for slot, op in enumerate(self.operands):
            pos = positions[slot : slot + 1]
            self.copy(pos, op["current_pos"], "position_refreshes")
            self.copy(pos.clamp_min(0).reshape(1, 1), op["rope_index"], "rope_refreshes")
            self.copy((pos >= 0).reshape(1, 1, 1, 1).bfloat16(), op["active_mask"])
            self.copy((pos >= 0).reshape(1, 1).int(), op["rope_increment"])
        self._active = [int(pos) >= 0 for pos in positions]
        self._positions = positions.tolist()
        self._decode_steps = 0
        self._upload_seed_increments()
        if page_table is not None:
            self.refresh_page_tables(page_table, extents=extents)

    def _upload_seed_increments(self):
        self.copy(
            torch.tensor(self._active + [False] * (32 - self.batch_size), dtype=torch.int32), self.seed_increments
        )
        if self._sampling_mode.endswith("_penalties"):
            self._upload_penalty_emitters()

    def _upload_penalty_emitters(self):
        self.copy(
            torch.tensor(self._active + [False] * (32 - self.batch_size), dtype=torch.int32).reshape(32, 1),
            self.penalties.decode_src,
        )

    def _checkpoint_seed_counters(self):
        for slot in range(self.batch_size):
            if self._active[slot]:
                self._seed_draws[slot] += self._sampling_clock - self._seed_epochs[slot]
            self._seed_epochs[slot] = self._sampling_clock

    def _upload_seeds(self):
        seeds = [(base + count) % 0xFFFFFFFF for base, count in zip(self._sampling_spec["seed"], self._seed_draws)]
        self.copy(torch.tensor(seeds + [0] * (32 - self.batch_size), dtype=torch.uint32), self.sampler.seeds_tt_tensor)

    def _sampling_values(
        self,
        *,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        mode="split",
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    ):
        def rows(x, dtype, default):
            a = torch.as_tensor(x, dtype=dtype).flatten()
            if a.numel() == 1:
                a = a.repeat(self.batch_size)
            if a.numel() != self.batch_size:
                raise ValueError("Sampling parameters need one value per fixed slot")
            return torch.cat((a, torch.full((32 - self.batch_size,), default, dtype=dtype)))

        k = rows(top_k, torch.int32, 1)
        p = rows(top_p, torch.float32, 0)
        temp = rows(temperature, torch.float32, 1)
        presence = rows(presence_penalty, torch.float32, 0)
        frequency = rows(frequency_penalty, torch.float32, 0)
        repetition = rows(repetition_penalty, torch.float32, 1)
        if (
            not torch.isfinite(presence).all()
            or (presence.abs() > 2).any()
            or not torch.isfinite(frequency).all()
            or (frequency.abs() > 2).any()
            or not torch.isfinite(repetition).all()
            or (repetition <= 0).any()
        ):
            raise ValueError("Invalid presence/frequency/repetition penalty")
        if (
            not torch.isfinite(temp).all()
            or not torch.isfinite(p).all()
            or (temp < 0).any()
            or (p < 0).any()
            or (p > 1).any()
        ):
            raise ValueError("Invalid temperature/top_p")
        k = torch.where(temp == 0, 1, k)
        temp = torch.where(temp == 0, 1, temp)
        if (k < 1).any() or (k > 32).any():
            raise ValueError("Device sampler supports top_k 1..32; host compatibility supports full vocabulary")
        if mode not in ("split", "argmax") or (mode == "argmax" and (k != 1).any()):
            raise ValueError("Argmax mode requires greedy parameters")
        raw_seeds = [seed] * self.batch_size if seed is None or isinstance(seed, int) else list(seed)
        if len(raw_seeds) != self.batch_size:
            raise ValueError("Sampling seed needs one value per fixed slot")
        # Accept all 32 seed bits; effective seeds use modulus2**32-1 because
        # manual_seed reserves0xffffffff. Explicit0xffffffff therefore aliases0.
        seeds = [secrets.randbits(32) if value is None else int(value) & 0xFFFFFFFF for value in raw_seeds]
        return dict(
            top_k=k[: self.batch_size].tolist(),
            top_p=p[: self.batch_size].tolist(),
            temperature=temp[: self.batch_size].tolist(),
            seed=seeds,
            mode=mode,
            presence_penalty=presence[: self.batch_size].tolist(),
            frequency_penalty=frequency[: self.batch_size].tolist(),
            repetition_penalty=repetition[: self.batch_size].tolist(),
        )

    def set_sampling(
        self,
        *,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=0,
        mode="split",
        reset_state=True,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    ):
        self._require_read_enqueued()
        if not reset_state and self._sampling_spec is not None:
            raw_seeds = [seed] * self.batch_size if seed is None or isinstance(seed, int) else list(seed)
            if len(raw_seeds) != self.batch_size:
                raise ValueError("Sampling seed needs one value per fixed slot")
            seed = [
                self._sampling_spec["seed"][slot] if value is None else value for slot, value in enumerate(raw_seeds)
            ]
        spec = self._sampling_values(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            mode=mode,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )
        self._apply_sampling_values(spec, reset_state=reset_state)

    def _invalidate_sampler_param_cache(self):
        """Required before direct sampler.reset_params or external parameter writes."""
        self._sampler_param_cache.clear()

    def _copy_sampler_param(self, field, key, target, make_host):
        previous = self._sampler_param_cache.get(field)
        if previous is not None and previous[0] is target and previous[1] == key:
            self.counters[f"sampler_{field}_refresh_skips"] += 1
            return
        try:
            self.copy(make_host(), target)
        except Exception:
            self._invalidate_sampler_param_cache()
            raise
        self._sampler_param_cache[field] = (target, key)
        self.counters[f"sampler_{field}_refreshes"] += 1

    @staticmethod
    def _sampling_trace_mode(spec, penalties):
        mode = spec["mode"]
        # Include inactive configured slots: a stale stochastic slot safely
        # retains the wide path instead of admitting k>32 to narrow sampling.
        if mode == "split" and all(k == 1 for k in spec["top_k"]):
            mode = "greedy32"
        return mode + ("_penalties" if penalties else "")

    def _apply_sampling_values(self, spec, *, reset_state=False, reset_slots=None, seed_positions=None):
        self._checkpoint_seed_counters()
        self._sampling_spec = spec
        # Immutable source-value keys; conversion stays identical to the original
        # payload and happens only when a value or destination changed.
        padding = 32 - self.batch_size
        k = tuple(spec["top_k"]) + (1,) * padding
        p = tuple(spec["top_p"]) + (0.0,) * padding
        temp = tuple(spec["temperature"]) + (1.0,) * padding
        greedy = tuple(value == 1 for value in k)
        for field, key, target, make_host in (
            ("k", k, self.sampler.k_tensor, lambda: torch.tensor(k, dtype=torch.int32)),
            ("p", p, self.sampler.p_tensor, lambda: torch.tensor(p).bfloat16()),
            ("temp", temp, self.sampler.temp_tensor, lambda: torch.tensor(temp).reciprocal().bfloat16()),
            (
                "greedy",
                greedy,
                self.sampler._greedy_col,
                lambda: torch.tensor(greedy).reshape(1, 1, 32, 1).bfloat16(),
            ),
        ):
            self._copy_sampler_param(field, key, target, make_host)
        if reset_state:
            for slot in range(self.batch_size) if reset_slots is None else reset_slots:
                self._seed_draws[slot] = 0 if seed_positions is None else max(0, int(seed_positions[slot]) + 1)
        self._upload_seeds()
        penalties = any(
            value != neutral
            for name, neutral in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1))
            for value in spec[name]
        )
        self._sampling_mode = self._sampling_trace_mode(spec, penalties)
        if penalties or not getattr(self, "_penalty_params_initialized", False):
            self.penalties.reset_params(
                spec["presence_penalty"] + [0.0] * (32 - self.batch_size),
                spec["frequency_penalty"] + [0.0] * (32 - self.batch_size),
                spec["repetition_penalty"] + [1.0] * (32 - self.batch_size),
            )
            self._penalty_params_initialized = True
        if penalties:
            self._upload_penalty_emitters()
            # Any active row can be counted by a mixed penalties-on trace.
            # Conservative dirty ownership avoids work on neutral-only runs.
            self._penalty_dirty = [True] * self.batch_size

    def _params_values(self, params, *, slots=None, new_request=False, seed_sources=None, param_rows=None):
        slots = list(range(self.batch_size)) if slots is None else list(slots)

        def field(name, default):
            value = params.get(name, default) if isinstance(params, dict) else getattr(params, name, default)
            if value is None or isinstance(value, (int, float, bool)):
                return [value] * len(slots)
            values = list(value)
            return values if param_rows is None else [values[row] for row in param_rows]

        if any(field("enable_log_probs", False)) or any(x is not None and x > 0 for x in field("num_logprobs", None)):
            raise ValueError("Logprob requests on TP4 require explicit host sampling")
        merged = {key: list(value) if isinstance(value, list) else value for key, value in self._sampling_spec.items()}
        merged["mode"] = "split"
        for name, default in (
            ("top_k", 1),
            ("top_p", 0.0),
            ("temperature", 1.0),
            ("seed", None),
            ("presence_penalty", 0.0),
            ("frequency_penalty", 0.0),
            ("repetition_penalty", 1.0),
        ):
            values = field(name, default)
            if len(values) != len(slots):
                raise ValueError(f"{name} must match submitted row count")
            for slot, value in zip(slots, values):
                if name == "seed" and value is None and not new_request:
                    value = self._sampling_spec["seed"][slot if seed_sources is None else seed_sources[slot]]
                merged[name][slot] = value
        # Semantic k=1 uses the same tile-shaped local candidate / gather
        # sampler as standalone generation, with persistent token feedback.
        return self._sampling_values(**merged)

    def _assemble_prefill_logits(self, values, slots):
        rows = self._slot_zero_logits_rows + [self._zero_logits_row] * (32 - self.batch_size)
        for slot, value in zip(slots, values, strict=True):
            rows[slot] = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        combined = ttnn.concat(rows, dim=2)
        ttnn.copy(ttnn.to_layout(combined, ttnn.TILE_LAYOUT), self.logits)

    def _restore_unsubmitted_tokens(self, slots):
        if self.batch_size > 1:
            rows = [
                (self.tokens if row in slots else self._prefill_token_backup)[:, :, :, row : row + 1]
                for row in range(32)
            ]
            ttnn.copy(ttnn.concat(rows, dim=3), self.tokens)

    def replay(self, *, sample=True):
        self._require_read_enqueued()
        if not self.prepared:
            raise RuntimeError("prepare() must finish before trace replay")
        if self.batch_size == 1 and not self._active[0]:
            self.counters["inactive_noops"] += 1
            return
        if any(self._active[slot] and self._position(slot) >= self.context for slot in range(self.batch_size)):
            raise ValueError("Decode would exceed the supported context")
        for key in ("decode", self._sampling_mode) if sample else ("decode",):
            ttnn.execute_trace(self.device, self.traces[key], cq_id=0, blocking=False)
            self.counters[key + "_replays"] += 1
        self._decode_steps += 1
        if sample:
            self._sampling_clock += 1

    def read_tokens(self):
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).flatten()[: self.batch_size].long()

    def read_logits(self, value=None):
        self.counters["logit_readbacks"] += 1
        value = self.logits if value is None else value
        return torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(value)], dim=-1)[
            ..., : self.model.config.vocab_size
        ]

    def _require_read_enqueued(self):
        if self.closed:
            raise RuntimeError("Generator has been closed")
        if self._pending_output is not None and self._pending_output.host is None:
            raise RuntimeError("Enqueue read_decode_output for the previous submission before mutating decode buffers")

    def read_decode_output(self, output, async_read=False, sample_rows=None):
        """Observational CQ0 reads; no sampling, position update, or state reset."""
        if sample_rows is not None:
            raise ValueError("Selective readback is not declared; request full-slot output")
        if not isinstance(output, DecodeOutput) or output.owner is not self:
            raise ValueError("Expected this generator's decode submission handle")
        if output.host is None:
            source = ttnn.get_device_tensors(output.source)[0] if output.is_tokens else output.source
            output.host = source.cpu(blocking=not async_read)
            output.events = [ttnn.record_event(self.device, 0)] if async_read else []
            self.counters["token_readbacks" if output.is_tokens else "logit_readbacks"] += 1
        elif not async_read:
            for event in output.events:
                ttnn.event_synchronize(event)
        return (output, output.events) if async_read else output

    def process_decode_output_host(self, output, is_tokens=False, sample_rows=None):
        if sample_rows is not None:
            raise ValueError("Compact host logits are not declared; request full-slot output")
        if not isinstance(output, DecodeOutput) or output.owner is not self or output.host is None:
            raise ValueError("read_decode_output must precede host processing")
        if output.is_tokens != bool(is_tokens):
            raise ValueError("Output sampling mode does not match its submission")
        # The async caller waits on returned events before this conversion.
        if is_tokens:
            result = ttnn.to_torch(output.host).reshape(-1)[: self.batch_size].long().reshape(self.batch_size, 1)
        else:
            result = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(output.host)], -1)
            result = result[0, 0, : self.batch_size, : self.model.config.vocab_size].unsqueeze(1)
        result = result.clone()  # Caller may apply logits processors in place.
        result[torch.tensor([not value for value in output.active])] = 0
        return result

    def _remap_slots(self, remap, *, warm=False):
        remap = [int(x) for x in remap]
        if len(remap) != self.batch_size or any(x < 0 or x >= self.batch_size for x in remap):
            raise ValueError("slot_remap must cover every configured state slot")
        if remap == list(range(self.batch_size)) and not warm:
            return
        self._checkpoint_seed_counters()
        self._remap_penalties(remap, warm=warm)
        for rows in self.state.layers:
            for name in ("conv", "recurrent", "key", "value"):
                if rows[0].__getattribute__(name) is None or (name in ("key", "value") and self.state.shared_cache):
                    continue
                if self.batch_size == 1:
                    continue
                scratch = self._move_scratch[self.model._reset_signature(getattr(rows[0], name))]
                for slot in range(self.batch_size):
                    ttnn.copy(getattr(rows[slot], name), scratch[slot])
                for slot, source in enumerate(remap):
                    ttnn.copy(scratch[source], getattr(rows[slot], name))
        # Token movement stays on device. All scalar slice offsets and the
        # fixed concat signature are warmed before capture, even for identity.
        if self.batch_size > 1:
            ttnn.copy(self.tokens, self._prefill_token_backup)
            parts = [self.tokens[:, :, :, slot : slot + 1] for slot in range(32)]
            # Concat hashes input aliasing. Duplicate sources must retain the
            # distinct-input signature warmed by the identity permutation.
            selected = [ttnn.clone(part) for part in [parts[source] for source in remap] + parts[self.batch_size :]]
            moved = ttnn.concat(selected, dim=3)
            ttnn.copy(moved, self.tokens)
            if warm:
                self._restore_unsubmitted_tokens(list(range(self.batch_size)))
        positions = [self._position(source) for source in remap]
        for key in ("_active", "_slot_ends", "_seed_draws", "_seed_epochs", "_request_epochs"):
            previous = getattr(self, key)
            setattr(self, key, [previous[source] for source in remap])
        self._positions, self._decode_steps = positions, 0
        self._sampling_spec = {
            key: [value[source] for source in remap] if isinstance(value, list) else value
            for key, value in self._sampling_spec.items()
        }
        for slot, op in enumerate(self.operands):
            pos = positions[slot]
            self.copy(torch.tensor([pos], dtype=torch.int32), op["current_pos"])
            self.copy(torch.tensor([[max(0, pos)]], dtype=torch.int32), op["rope_index"])
            self.copy(torch.tensor([[[[pos >= 0]]]], dtype=torch.bfloat16), op["active_mask"])
            self.copy(torch.tensor([[pos >= 0]], dtype=torch.int32), op["rope_increment"])
        self._apply_sampling_values(self._sampling_spec)
        self._upload_seed_increments()
        self.counters["slot_remaps"] += int(not warm)

    def note_state_slots_moved(self, moves):
        """Notification only: decode_forward already moved every owned state."""
        if any(not 0 <= int(slot) < self.batch_size for pair in moves.items() for slot in pair):
            raise ValueError("Slot movement notification exceeds configured capacity")
        self.counters["slot_move_notifications"] += 1

    def release_request(self, slot):
        self._require_read_enqueued()
        slot = int(slot)
        if not 0 <= slot < self.batch_size:
            raise ValueError("Request slot outside configured capacity")
        self._checkpoint_seed_counters()
        self.model.reset_state(self.state, slots=[slot], clear_kv=False)
        self._clear_penalty_slots([slot])
        for name, neutral in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
            self._sampling_spec[name][slot] = neutral
        positions = [self._position(i) for i in range(self.batch_size)]
        positions[slot] = -1
        self._positions, self._decode_steps = positions, 0
        self._active[slot] = False
        self._slot_ends[slot] = 0
        self._seed_draws[slot] = 0
        self._request_epochs[slot] += 1
        self._sampling_spec["seed"][slot] = secrets.randbits(32)
        self._apply_sampling_values(self._sampling_spec)
        self._upload_seed_increments()
        op = self.operands[slot]
        for value, key in (
            (torch.tensor([-1], dtype=torch.int32), "current_pos"),
            (torch.zeros(1, 1, dtype=torch.int32), "rope_index"),
            (torch.zeros(1, 1, dtype=torch.int32), "rope_increment"),
            (torch.zeros(1, 1, 1, 1), "active_mask"),
        ):
            self.copy(value, op[key])
        self.counters["request_releases"] += 1

    def cache_spec(self):
        """Authoritative TT storage metadata for a later serving adapter.

        vLLM's AttentionSpec is intentionally not imported in the model stage.
        Its adapter translates these rows to AttentionSpec without reallocating
        or changing the selected paged-cache dtype and activation split.
        """
        result = {}
        for original, layer, rows in zip(self.model.layer_indices, self.model.layers, self.state.layers):
            if layer.kind == "full_attention":
                result[f"model.layers.{original}.self_attn"] = dict(
                    layer_index=original,
                    shape=tuple(rows[0].key.shape),
                    dtype=rows[0].key.dtype,
                    num_kv_heads=2,
                    heads_per_device=1,
                    head_size=256,
                    block_size=32,
                    shared_request_pool=self.state.shared_cache,
                    storage=(rows[0].key, rows[0].value),
                )
        return result

    def get_kv_cache_spec(self, vllm_config=None):
        """Return model-native specs; adapter translates types at integration."""
        return self.cache_spec()

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Validate and return the stable pool allocated at construction.

        Cache ownership is selected by constructor ``cache_state``. A serving
        adapter must construct that state for its profiled pool before this
        binding call, warmup, or capture; late storage replacement is forbidden.
        """
        expected = list(self.cache_spec().values())
        if len(per_layer_specs) != len(expected):
            raise ValueError("Cache specs must cover each loaded full-attention layer")
        indices = []
        for (shape, dtype, tensor_idx), actual in zip(per_layer_specs, expected):
            if tuple(shape) != actual["shape"] or dtype != actual["dtype"]:
                raise ValueError("Cache shape/dtype must match construction-time selected TT storage")
            indices.append(tensor_idx)
        if len(set(indices)) != len(indices):
            raise ValueError("Different Ornith attention layers cannot alias their K/V storage")
        return self.state

    def warmup_model_prefill(self, *, kv_cache, can_sample_on_device=True, enable_trace=True, **kwargs):
        if kwargs or kv_cache is not self.state:
            raise ValueError("Warmup must use the configured cache and supported ordinary text options")
        self.prepare(capture=enable_trace)

    def warmup_model_decode(
        self, *, kv_cache, max_batch_size=None, num_blocks=None, can_sample_on_device=True, enable_trace=True, **kwargs
    ):
        if max_batch_size is not None and max_batch_size != self.batch_size:
            raise ValueError("Warmup batch must match fixed state capacity")
        if num_blocks is not None and num_blocks != self.pages_host.shape[1]:
            raise ValueError("Warmup page-table width must match construction")
        self.warmup_model_prefill(
            kv_cache=kv_cache, can_sample_on_device=can_sample_on_device, enable_trace=enable_trace, **kwargs
        )

    def _page_mapping(self, page_table, page_tables_per_layer):
        if page_tables_per_layer is not None:
            expected = len(self.cache_spec())
            if len(page_tables_per_layer) != expected:
                raise ValueError("Per-layer mappings must cover all loaded full-attention layers")
            if any(not torch.equal(torch.as_tensor(page_table), torch.as_tensor(p)) for p in page_tables_per_layer):
                raise ValueError("Ornith full-attention layers share one addressing group; mappings must agree")
        return page_table

    def _refresh_prefill(self, n, ids, pos, slot):
        b = self.prefill_buffers[n]
        self.copy(torch.tensor(ids, dtype=torch.int32).reshape(1, 1, 1, n), b["tokens"])
        self.copy(torch.tensor([pos], dtype=torch.int32), b["position"])
        inv = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2).float() / 64))
        physical = self._prefill_physical[n]
        phase = torch.arange(pos, pos + physical).float()[:, None] * inv[None]
        for name, fn, identity in (("cos", torch.cos, 1.0), ("sin", torch.sin, 0.0)):
            values = torch.cat((fn(phase).repeat_interleave(2, dim=-1), torch.full((physical, 192), identity)), dim=-1)
            values = values.reshape(1, 1, physical, 256).bfloat16()
            if n == 1:
                values = values.repeat(1, 2, 32, 1)
            self.copy(values, b[name])
        if n > 1:
            self.copy(self.pages_host[slot : slot + 1, pos // 32 : (pos + physical) // 32], b["chunk_page_table"])
        self.counters["prefill_chunks"] += 1

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        return_all_logits=False,
        slots=None,
        return_device=False,
        empty_slots=None,
        sampling_params=None,
        prefill_sampling_mask=None,
        page_tables_per_layer=None,
        selected_positions=None,
        enable_trace=True,
        **kwargs,
    ):
        self._require_read_enqueued()
        if kwargs:
            raise TypeError(f"Unsupported prefill arguments: {sorted(kwargs)}")
        if kv_cache is not self.state:
            raise ValueError("External cache must be bound at generator construction and retain its storage")
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        starts = [0] * len(prompt_lens) if start_pos is None else list(map(int, start_pos))
        if empty_slots is not None:
            if slots is not None and list(slots) != list(empty_slots):
                raise ValueError("slots and empty_slots specify different destinations")
            slots = empty_slots
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        ends = list(map(int, prompt_lens))
        if tokens.ndim != 2 or tokens.shape[0] != len(ends) or not ends:
            raise ValueError("Prefill requires one token-history row per nonempty prompt")
        if len(starts) != len(ends) or len(slots) != len(ends) or len(set(slots)) != len(slots):
            raise ValueError("Prefill starts, ends, and unique destination slots must match")
        if prefill_sampling_mask is None:
            emit = [True] * len(ends)
        else:
            mask = torch.as_tensor(prefill_sampling_mask)
            if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != len(ends):
                raise ValueError("prefill_sampling_mask must contain one boolean per submitted row")
            emit = mask.tolist()
            if return_device or return_all_logits or selected_positions is not None:
                raise ValueError("prefill_sampling_mask applies only to final-token serving output")
        emit_rows = [row for row, enabled in enumerate(emit) if enabled]
        emit_slots = [slots[row] for row in emit_rows]
        extents = [
            max(end, self._position(slot)) if self._active[slot] else 0 for slot, end in enumerate(self._slot_ends)
        ]
        for row, (start, end, slot) in enumerate(zip(starts, ends, slots)):
            if not 0 <= slot < self.batch_size or not 0 <= start < end <= self.context or end > tokens.shape[1]:
                raise ValueError("Invalid absolute prompt extent or fixed slot")
            if ((tokens[row, start:end] < 0) | (tokens[row, start:end] >= self.model.config.vocab_size)).any():
                raise ValueError("Prompt token outside vocabulary")
            extents[slot] = end
        if selected_positions is not None:
            if len(ends) != 1 or return_device or sampling_params is not None:
                raise ValueError("Selected logit positions require one host-logit row")
            selected_positions = list(map(int, selected_positions))
            if not selected_positions or any(not starts[0] <= pos < ends[0] for pos in selected_positions):
                raise ValueError("Selected logit positions must lie inside the continuation")
        if return_device and return_all_logits:
            raise ValueError("Device prefill returns final logits; all-position logits use the host API")
        if sampling_params is not None and (return_device or return_all_logits):
            raise ValueError("Sampled prefill returns tokens, not caller-owned device/all-position logits")
        spec = (
            self._params_values(sampling_params, slots=emit_slots, new_request=True, param_rows=emit_rows)
            if sampling_params is not None and emit_slots
            else None
        )
        page_table = self._page_mapping(page_table, page_tables_per_layer)
        self._validate_pages(page_table, extents=extents)
        self.prepare()
        self.refresh_page_tables(page_table, extents=extents)
        rows = []
        for row, (start, end, slot) in enumerate(zip(starts, ends, slots, strict=True)):
            if start == 0:
                self.model.reset_state(self.state, slots=[slot], clear_kv=False)
                self._clear_penalty_slots([slot])
                self._request_epochs[slot] += 1
            pieces = []
            selected = {}
            pos = start
            while pos < end:
                # Paged SDPA expresses chunk_start in units of its selected
                # Q chunk. Align with a smaller existing bucket before taking
                # a longer chunk; recurrence consumes exactly the real tokens.
                n = max(
                    n
                    for n in self.buckets
                    if n <= end - pos and (n == 1 or (pos % 32 == 0 and pos % self._prefill_alignment[n] == 0))
                )
                self._refresh_prefill(n, tokens[row, pos : pos + n].tolist(), pos, slot)
                hidden = self._prefill(n, slot)
                if return_all_logits or selected_positions is not None:
                    for offset in range(0, n, 32):
                        requested = (
                            None
                            if selected_positions is None
                            else [p for p in selected_positions if pos + offset <= p < pos + min(offset + 32, n)]
                        )
                        if requested == []:
                            continue
                        output = self.model.terminal(hidden[:, :, offset : min(offset + 32, n), :])
                        values = self.read_logits(output)[0, 0]
                        if requested is None:
                            pieces.append(values)
                        else:
                            for p in requested:
                                selected[p] = values[p - pos - offset].clone()
                        del output
                elif pos + n == end and emit[row]:
                    if spec is not None or return_device:
                        self._retain_prefill_hidden(hidden, n, slot)
                        pieces.append(slot)
                    else:
                        output = self._terminal_last(hidden, n)
                        pieces.append(self.read_logits(output)[0, 0])
                        del output
                del hidden
                pos += n
            self._slot_ends[slot] = end
            if sampling_params is None:
                rows.append(
                    torch.zeros(1, self.model.config.vocab_size)
                    if not emit[row]
                    else torch.stack([selected[p] for p in selected_positions])
                    if selected_positions is not None
                    else pieces[-1]
                    if return_device
                    else torch.cat(pieces, dim=0)
                )
        if spec is not None:
            if any(spec["repetition_penalty"][slot] != 1 for slot in emit_slots):
                self._set_penalty_prompts(tokens[emit_rows], [ends[row] for row in emit_rows], emit_slots)
            self._apply_sampling_values(
                spec,
                reset_state=True,
                reset_slots=emit_slots,
                seed_positions={slots[row]: ends[row] - 1 for row in emit_rows},
            )
            # Sampling runs once after the final chunk, never on intermediates.
            # Fixed-slot terminal assembly is warmed before trace capture.
            terminal_rows = [self.model.terminal(self._prefill_hidden(slot)) for slot in emit_slots]
            self._assemble_prefill_logits(terminal_rows, emit_slots)
            terminal_rows.clear()
            old_active = list(self._active)
            self._checkpoint_seed_counters()
            self._active = [slot in emit_slots for slot in range(self.batch_size)]
            self._upload_seed_increments()
            if self.batch_size > 1:
                ttnn.copy(self.tokens, self._prefill_token_backup)
            ttnn.execute_trace(self.device, self.traces[self._sampling_mode], cq_id=0, blocking=False)
            self._sampling_clock += 1
            self._restore_unsubmitted_tokens(emit_slots)
            result = torch.zeros(len(slots), 1, dtype=torch.long)
            result[emit_rows, 0] = self.read_tokens()[emit_slots]
            self._checkpoint_seed_counters()
            self._active = old_active
            self._upload_seed_increments()
            return result
        if sampling_params is not None:
            # Every row is intermediate: no terminal, draw, token read or feedback.
            return torch.zeros(len(slots), 1, dtype=torch.long)
        if return_device:
            return [self.model.terminal(self._prefill_hidden(slot)) for slot in rows]
        width = max(x.shape[0] for x in rows)
        return torch.stack([torch.nn.functional.pad(x, (0, 0, 0, width - x.shape[0])) for x in rows])

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        host_sampling=None,
        sampling_params=_UNSET,
        read_from_device=True,
        enable_trace=True,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        slot_remap=None,
        page_tables_per_layer=None,
        decode_input_update_contract=1,
        prompt_tokens=None,
        output_tokens=None,
        **kwargs,
    ):
        self._require_read_enqueued()
        if kwargs:
            raise TypeError(f"Unsupported decode arguments (legacy reset_batch is forbidden): {sorted(kwargs)}")
        if decode_input_update_contract != 1:
            raise ValueError("Ornith implements decode_input_update_contract=1")
        if reload_inputs and reload_page_table:
            raise ValueError("reload_inputs already includes page tables")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("reset_sampling_state requires authoritative forward inputs")
        if not enable_trace:
            raise ValueError("Decode uses retained traces; eager execution is warmup-only")
        if kv_cache is not self.state:
            raise ValueError("Cache storage must match the configured generator")
        # Omitting both controls preserves the standalone device-token API;
        # explicit sampling_params=None is the plugin's host-logit boundary.
        sample = sampling_params is not None if sampling_params is not _UNSET else True
        if host_sampling is not None:
            if host_sampling and sampling_params not in (_UNSET, None):
                raise ValueError("Host sampling cannot also request device sampling")
            sample = not host_sampling
        remap = list(range(self.batch_size)) if slot_remap is None else list(map(int, slot_remap))
        if len(remap) != self.batch_size or any(not 0 <= slot < self.batch_size for slot in remap):
            raise ValueError("slot_remap must cover the full fixed-slot capacity")
        spec = None
        if reload_sampling_params:
            if sampling_params in (_UNSET, None):
                raise ValueError("reload_sampling_params requires sampling_params")
            # Resolve unspecified seeds against continuing request ownership
            # before mutating anything. Parameters are destination-row ordered.
            spec = self._params_values(sampling_params, seed_sources=remap)
        if reload_inputs:
            # Validate every host input before remap or any device mutation.
            checked_tokens = torch.as_tensor(tokens, dtype=torch.int64).flatten()
            checked_pos = torch.as_tensor(start_pos, dtype=torch.int32).flatten()
            if checked_tokens.numel() != self.batch_size or checked_pos.numel() != self.batch_size:
                raise ValueError("Decode requires one token/position per fixed slot")
            if (checked_tokens < 0).any() or (checked_tokens >= self.model.config.vocab_size).any():
                raise ValueError("Token outside vocabulary")
            if (checked_pos < -1).any() or (checked_pos >= self.context).any():
                raise ValueError("Position outside context")
        penalty_history = None
        if sample and reset_sampling_state:
            current_spec = self._sampling_spec if spec is None else spec
            if any(
                value != neutral
                for name, neutral in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1))
                for value in current_spec[name]
            ):
                penalty_history = self._validated_penalty_history(prompt_tokens, output_tokens)
        if reload_inputs or reload_page_table:
            page_table = self._page_mapping(page_table, page_tables_per_layer)
            extents = [
                max(self._slot_ends[src], int(checked_pos[slot]) + 1 if reload_inputs else self._position(src) + 1)
                if (int(checked_pos[slot]) >= 0 if reload_inputs else self._active[src])
                else 0
                for slot, src in enumerate(remap)
            ]
            self._validate_pages(page_table, extents=extents)
        self.prepare()
        self._remap_slots(remap)
        if reload_inputs:
            self.bind(checked_tokens, checked_pos, page_table)
        elif reload_page_table:
            self.refresh_page_tables(page_table, extents=extents)
        # No token/start_pos access is permitted in the resident-input branch.
        if spec is not None:
            self._apply_sampling_values(
                spec, reset_state=reset_sampling_state, seed_positions=checked_pos if reset_sampling_state else None
            )
        elif reset_sampling_state:
            self._apply_sampling_values(self._sampling_spec, reset_state=True, seed_positions=checked_pos)
        if penalty_history is not None:
            self.penalties.reset_prompt_tokens(penalty_history[0])
            self.penalties.reset_output_tokens(penalty_history[1])
        self.replay(sample=sample)
        self._serial += 1
        output = DecodeOutput(self, self._serial, sample, tuple(self._active), self.tokens if sample else self.logits)
        self._pending_output = output
        if not read_from_device:
            return output
        self.read_decode_output(output)
        value = self.process_decode_output_host(output, is_tokens=sample)
        # Preserve the established standalone return shape. Explicit plugin
        # sampling_params selects its [B,1] / [B,1,V] output contract.
        return value[:, 0] if sampling_params is _UNSET else value

    def prefill_logits(self, prompt_token_ids):
        self.prepare()
        self.reset()
        self._prepare_standalone_pages(len(prompt_token_ids))
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.pages_host,
            kv_cache=self.state,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def prefill_logits_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        self.prepare()
        self.reset()
        self._prepare_standalone_pages(len(prompt_token_ids))
        ids = torch.tensor([prompt_token_ids])
        if start and prefill_prefix:
            self.prefill_forward(ids, page_table=self.pages_host, kv_cache=self.state, prompt_lens=[start])
        values = self.prefill_forward(
            ids,
            page_table=self.pages_host,
            kv_cache=self.state,
            prompt_lens=[len(prompt_token_ids)],
            start_pos=[start],
            selected_positions=list(range(start, len(prompt_token_ids))) if positions is None else list(positions),
        )
        return values

    def _prepare_standalone_pages(self, length):
        """After whole-generator reset, let the sole request use the full arena."""
        if not 1 <= length <= self.context:
            raise ValueError("Prompt length must lie inside the configured context")
        count = (length + 31) // 32
        if count > self.num_blocks:
            raise ValueError("The physical KV pool cannot hold this prompt; construct a larger num_blocks pool")
        if torch.unique(self.pages_host[0, :count]).numel() == count:
            return  # Retain a caller's valid permutation and its unchanged-table fast path.
        pages = self.pages_host.clone()
        pages[0, :count] = torch.arange(count, dtype=torch.int32)
        self.refresh_page_tables(pages, extents=[length] + [0] * (self.batch_size - 1))

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        host_sampling=False,
        stop_on_eos=False,
        **kwargs,
    ):
        unknown = set(kwargs) - {
            "top_k",
            "top_p",
            "temperature",
            "seed",
            "mode",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        }
        if unknown:
            raise TypeError(f"Unsupported generation arguments: {sorted(unknown)}")
        parameters = self._sampling_values(**kwargs)
        if host_sampling and any(k != 1 for k in parameters["top_k"]):
            raise ValueError("Standalone host sampling is greedy; use low-level host logits for caller-owned sampling")
        if host_sampling and any(
            value != neutral
            for name, neutral in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1))
            for value in parameters[name]
        ):
            raise ValueError("Standalone penalty generation requires device sampling")
        if not enable_trace:
            raise ValueError("Ornith generation requires traced decode")
        if len(prompt_token_ids) + max_new_tokens > self.context:
            raise ValueError("Prompt plus generation exceeds configured context")
        if max_new_tokens <= 0:
            return []
        self.prepare()
        self.reset()
        self._prepare_standalone_pages(len(prompt_token_ids) + max_new_tokens)
        terminal = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.pages_host,
            kv_cache=self.state,
            prompt_lens=[len(prompt_token_ids)],
            return_device=True,
        )[0]
        padded = ttnn.pad(terminal, ((0, 0), (0, 0), (0, 31), (0, 0)), value=0)
        ttnn.copy(padded, self.logits)
        del terminal, padded
        self._apply_sampling_values(parameters, reset_state=True)
        if parameters["repetition_penalty"][0] != 1:
            self._set_penalty_prompts(torch.tensor([prompt_token_ids]), [len(prompt_token_ids)], [0])
        self.bind(
            torch.zeros(self.batch_size, dtype=torch.int32),
            torch.tensor([len(prompt_token_ids)] + [-1] * (self.batch_size - 1)),
        )
        # Match serving prefill and reset recovery: each draw is addressed by
        # its absolute output-token position, independently of slot lifetime.
        self._checkpoint_seed_counters()
        self._seed_draws[0] = len(prompt_token_ids)
        self._upload_seeds()
        outputs = []
        for step in range(max_new_tokens):
            if step:
                self.replay(sample=not host_sampling)
            elif not host_sampling:
                ttnn.execute_trace(self.device, self.traces[self._sampling_mode], cq_id=0, blocking=False)
                self.counters[self._sampling_mode + "_replays"] += 1
                self._sampling_clock += 1
            if host_sampling:
                # Explicit test compatibility boundary; never the measured token-out path.
                predicted = int(self.read_logits()[0, 0, 0].argmax())
            else:
                predicted = int(self.read_tokens()[0])
            outputs.append(predicted)
            if next_input is not None or host_sampling:
                following = predicted if next_input is None else int(next_input(step, predicted))
                ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ids.flatten()[0] = following
                self.copy(
                    ids, self.tokens, "teacher_forcing_token_refreshes" if next_input else "host_sampling_feedback"
                )
            if stop_on_eos and predicted == self.tokenizer.eos_token_id:
                break
        return outputs

    def teardown(self):
        self.close()

    def release_persistent_capture(self):
        self.close()

    def close(self):
        if self.closed:
            return
        if self._pending_output is not None and self._pending_output.host is None:
            self.read_decode_output(self._pending_output)
        for tid in self.traces.values():
            ttnn.release_trace(self.device, tid)
            self.counters["teardown_releases"] += 1
        self.closed = True


def build_generator(model_dir, mesh_device, **kwargs):
    model = OrnithModel.from_pretrained(
        mesh_device,
        checkpoint=kwargs.pop("checkpoint", None),
        layer_indices=kwargs.pop("layer_indices", None),
        rope_context=kwargs.get("max_seq_len", 262144),
        precision_config=kwargs.pop("precision_config", None),
    )
    return OrnithGenerator(model, **kwargs)
