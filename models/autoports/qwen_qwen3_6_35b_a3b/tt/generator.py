# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Readiness and explicit-cache generator for the Qwen TP4 full model."""

import json
import secrets
import time
from types import SimpleNamespace

import torch
from readiness_check.contract import Generator
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel
from models.autoports.qwen_qwen3_6_35b_a3b.tt.prefill import BoundedPrefill
from models.autoports.qwen_qwen3_6_35b_a3b.tt.serving_capabilities import QwenServingCapabilities
from models.autoports.qwen_qwen3_6_35b_a3b.tt.serving_state import (
    DecodeOutputPool,
    PrefillLogitRows,
    ReloadExecutor,
    SlotState,
)
from models.common.sampling.generator import SamplingGenerator, SamplingParams


class SamplerCCL:
    def __init__(self, model):
        self.model = model

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self.model.workspace.ccl.get_ag_ping_pong_semaphore()

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self.model.workspace.ccl.get_barrier_semaphore()

    def get_num_links(self, cluster_axis=None):
        return 2

    def line_all_gather(self, tensor, *, dim, cluster_axis, memory_config, num_links, buffer_key=None):
        ccl = self.model.workspace.ccl
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=1,
            mesh_device=self.model.mesh,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            memory_config=memory_config,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )


class QwenGenerator(Generator, QwenServingCapabilities):
    def __init__(self, model, *, max_batch_size=1, host_sampling=False, kv_cache=None, decode_buckets=None):
        self.model = model
        self.mesh = model.mesh
        self.tokenizer = AutoTokenizer.from_pretrained(model.checkpoint.path, local_files_only=True)
        self.max_batch_size = max_batch_size
        self.decode_buckets = (max_batch_size,) if decode_buckets is None else tuple(decode_buckets)
        self.solo_slot0_attention = model.specialize_solo_slot0_attention
        self._variants_prepared = False
        self._preparation_seconds = None
        if self.solo_slot0_attention and (max_batch_size < 2 or 2 not in self.decode_buckets):
            raise ValueError("Solo attention specialization requires a retained physical B2 bucket")
        minimum = 1 if max_batch_size == 1 else 2
        if (
            not self.decode_buckets
            or any(type(bucket) is not int or not minimum <= bucket <= max_batch_size for bucket in self.decode_buckets)
            or tuple(sorted(set(self.decode_buckets))) != self.decode_buckets
            or self.decode_buckets[-1] != max_batch_size
        ):
            raise ValueError(
                "Decode buckets must be increasing physical prefixes ending at capacity; multirow uses >=2"
            )
        self.host_sampling = host_sampling
        self.cache = model.allocate_cache(max_batch_size) if kv_cache is None else kv_cache
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.cache_contract import validate_cache

        validate_cache(model, self.cache, batch=max_batch_size)
        self.positions = [
            model.upload(torch.zeros(1, dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(max_batch_size)
        ]
        self.tokens = model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.previous_tokens = ttnn.zeros_like(self.tokens)
        self.incoming_tokens = ttnn.zeros_like(self.tokens)
        self.activity = [
            model.upload(torch.tensor([int(s == 0)], dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for s in range(max_batch_size)
        ]
        initial_mask = torch.zeros(32, dtype=torch.int32)
        initial_mask[0] = 1
        self.active_mask = model.upload(initial_mask, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sampling_update_mask = model.upload(initial_mask, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sampling_seed_updates = model.upload(
            torch.ones(32, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.active_slots = (0,)
        self.sampler_logits = model.upload(
            torch.zeros(1, 1, 32, model.padded_vocab_size), model.boundary_dtypes["sampling_dtype"], shard_dim=3
        )
        args = SimpleNamespace(
            vocab_size=model.vocab_size,
            padded_vocab_size=model.padded_vocab_size,
            max_batch_size=max_batch_size,
            max_top_k=64,
            device_sampling_draw_width=64,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
        )
        self.zero_logits_row = model.upload(
            torch.zeros(1, 1, 1, model.padded_vocab_size), model.boundary_dtypes["sampling_dtype"], shard_dim=3
        )
        self.sampler = SamplingGenerator(args=args, mesh_device=self.mesh, tt_ccl=SamplerCCL(model))
        # These are contractual fixed BF16 boundaries, not ignored policy knobs.
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import device_dtype

        assert self.sampler_logits.dtype == device_dtype(model.precision_config["logits_dtype"])
        assert self.zero_logits_row.dtype == device_dtype(model.precision_config["sampling_dtype"])
        model.observe_boundary(self.sampler_logits, "sampling_dtype", "sampler.logits_buffer")
        model.observe_boundary(self.sampler_logits, "logits_dtype", "sampler.model_logits_compatibility")
        model.observe_boundary(self.zero_logits_row, "sampling_dtype", "sampler.zero_logits_row")
        self.traces = {}
        self.trace_replays = {}
        self.sampling_trace_replays = {}
        self.prepared = False
        self.closed = False
        self.seed_values = torch.ones(32, dtype=torch.int32)
        self.sampling_values = {
            "temperature": [0.0] * 32,
            "top_k": [1] * 32,
            "top_p": [1.0] * 32,
            "seed": [0] * 32,
        }
        self.initial_host_rng_states = [torch.Generator().manual_seed(0).get_state() for _ in range(32)]
        self.host_rng = [torch.Generator().manual_seed(0) for _ in range(32)]
        self.counters = dict(
            model_replays=0,
            sampling_replays=0,
            token_refreshes=0,
            position_refreshes=0,
            page_table_refreshes=0,
            activity_refreshes=0,
            readbacks=0,
            captures=0,
            releases=0,
        )
        # Small row-major history uses stable zero sources. Large tiled KV
        # pools are filled in place, avoiding a second full-context allocation.
        self.zeros = []
        self.kv_pools = {}
        self.initial_pages = []
        seen = {}
        for states in self.cache:
            group = []
            for state in states:
                if "page_table" in state:
                    pages = ttnn.to_torch(ttnn.get_device_tensors(state["page_table"])[0]).to(torch.int32)
                    self.initial_pages.append((state, pages.clone()))
                    state["_host_pages"] = pages.clone()
                saved = {}
                for key, value in state.items():
                    if key in ("k", "v"):
                        self.kv_pools[id(value)] = value
                    elif key in ("conv", "recurrent"):
                        identity = id(value)
                        if identity not in seen:
                            seen[identity] = ttnn.zeros_like(value)
                        saved[key] = seen[identity]
                group.append(saved)
            self.zeros.append(group)
        self.prefill_executor = BoundedPrefill(model)
        self.slot_state = SlotState.from_generator(self)
        self.output_pool = DecodeOutputPool(
            self.mesh,
            token_template=self.tokens,
            logits_template=self.sampler_logits,
            vocab_size=model.vocab_size,
            capacity=max_batch_size,
        )
        self.prefill_logit_rows = PrefillLogitRows(self.mesh)
        self.reload_executor = ReloadExecutor(self.slot_state, require_drained=self.output_pool.require_drained)
        model.census("generator_persistent_state")

    def reset(self):
        if self.closed:
            raise RuntimeError("Generator has been closed")
        self.output_pool.require_drained()
        for value in self.kv_pools.values():
            ttnn.fill(value, 0.0, output_tensor=value)
        for states, zeros in zip(self.cache, self.zeros):
            for state, saved in zip(states, zeros):
                for key, value in saved.items():
                    ttnn.copy(value, state[key])
        for state, pages in self.initial_pages:
            if not torch.equal(state["_host_pages"], pages):
                self._copy_host(pages, state["page_table"])
                state["_host_pages"] = pages.clone()
                self.counters["page_table_refreshes"] += 1
        self._copy_host(torch.zeros(1, 1, 1, 32, dtype=torch.int32), self.tokens)
        self.counters["token_refreshes"] += 1
        for position in self.positions:
            self._copy_host(torch.zeros(1, dtype=torch.int32), position)
            self.counters["position_refreshes"] += 1
        self._copy_host(self.seed_values, self.sampler.tt_sampling.seeds_tt_tensor)
        for rng, state in zip(self.host_rng, self.initial_host_rng_states):
            rng.set_state(state)
        self._set_active_slots((0,))
        self.serving_rows = (0,)
        self.serving_device_sampling = None
        self.serving_output_history = None
        self.sampler.tt_penalties._prompt_tokens_host = None
        for slot in range(self.max_batch_size):
            self.sampler.seed_manager.release_slot(slot)
        self.slot_state.poisoned = False
        self.reload_executor.poisoned = False

    def configure_sampling(self, *, temperature=0.0, top_k=1, top_p=1.0, seed=0, slots=None, reset_seed_state=True):
        # Common sampler parameters live in stable tensors and are read during
        # every sampling trace. Greedy is k=1,p=0,temp=1 within the canonical candidate draw.
        slots = tuple(range(self.max_batch_size)) if slots is None else tuple(slots)
        if len(set(slots)) != len(slots) or any(s < 0 or s >= self.max_batch_size for s in slots):
            raise ValueError("Invalid sampling slots")

        def values(value):
            if isinstance(value, torch.Tensor):
                value = value.flatten().tolist()
            if isinstance(value, (list, tuple)):
                if len(value) != len(slots):
                    raise ValueError("Sampling parameter rows must match slots")
                return list(value)
            return [value] * len(slots)

        temps, ks, ps, seeds = map(values, (temperature, top_k, top_p, seed))
        if any(t < 0 or not 0 <= p <= 1 or k < -1 for t, k, p in zip(temps, ks, ps)):
            raise ValueError("Invalid sampling parameters")
        limit = self.sampler.tt_sampling.max_top_k
        if not self.host_sampling and any(t > 0 and (k <= 0 or k > limit) for t, k in zip(temps, ks)):
            raise ValueError(f"Unbounded or top_k > {limit} sampling requires explicit host-logit compatibility")
        for row, slot in enumerate(slots):
            for name, entries in (("temperature", temps), ("top_k", ks), ("top_p", ps)):
                self.sampling_values[name][slot] = entries[row]
            self.sampling_values["seed"][slot] = seeds[row]
            if reset_seed_state:
                request_seed = secrets.randbits(63) if seeds[row] is None else int(seeds[row])
                self.seed_values[slot] = request_seed % 1000000 + 1
                self.host_rng[slot].manual_seed(request_seed % (2**64))
                self.initial_host_rng_states[slot] = self.host_rng[slot].get_state()
        temperatures = self.sampling_values["temperature"]
        top_ks = self.sampling_values["top_k"]
        greedy = [t == 0 or k == 1 for t, k in zip(temperatures, top_ks)]
        params = SamplingParams(
            temperature=torch.tensor([1.0 if g else 1.0 / t for g, t in zip(greedy, temperatures)]),
            top_k=torch.tensor(
                [1 if g else min(self.sampler.tt_sampling.max_top_k, max(1, k)) for g, k in zip(greedy, top_ks)]
            ),
            top_p=[0.0 if g else p for g, p in zip(greedy, self.sampling_values["top_p"])],
            presence_penalty=[0.0] * 32,
            frequency_penalty=[0.0] * 32,
            repetition_penalty=[1.0] * 32,
        )
        self.sampler.reset_sampling_params(params)
        if not reset_seed_state:
            return
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import merge_rows

        selected = torch.zeros(32, dtype=torch.int32)
        selected[list(slots)] = 1
        self._copy_host(selected, self.sampling_update_mask)
        self._copy_host(self.seed_values, self.sampling_seed_updates)
        merge_rows(
            self.sampling_seed_updates, self.sampler.tt_sampling.seeds_tt_tensor, self.sampling_update_mask, self.mesh
        )

    def _copy_host(self, value, dest):
        host = ttnn.from_torch(
            value.contiguous(), dtype=dest.dtype, layout=dest.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, dest)

    def _refresh(self, tokens, positions, slots):
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import merge_rows

        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        for row, slot in enumerate(slots):
            ids[..., slot] = int(tokens[row])
            self._copy_host(torch.tensor([int(positions[row])], dtype=torch.int32), self.positions[slot])
            self.counters["position_refreshes"] += 1
        self._copy_host(ids, self.incoming_tokens)
        merge_rows(self.incoming_tokens, self.tokens, self.active_mask, self.mesh)
        self.counters["token_refreshes"] += 1

    def _set_active_slots(self, slots):
        selected = tuple(sorted(slots))
        if len(set(selected)) != len(selected) or any(s < 0 or s >= self.max_batch_size for s in selected):
            raise ValueError("Request slots must be unique and within configured capacity")
        if selected == self.active_slots:
            return
        mask = torch.zeros(32, dtype=torch.int32)
        mask[list(selected)] = 1
        self._copy_host(mask, self.active_mask)
        for slot, value in enumerate(self.activity):
            self._copy_host(mask[slot : slot + 1], value)
        self.active_slots = selected
        self.counters["activity_refreshes"] += 1

    def _bind_pages(self, page_table, cache, slots, page_tables_per_layer=None):
        if page_table is None and page_tables_per_layer is None:
            return
        full_states = [states for states in cache if "page_table" in states[0]]
        tables = [page_table] * len(full_states) if page_tables_per_layer is None else list(page_tables_per_layer)
        if len(tables) != len(full_states):
            raise ValueError("Provide one page table per full-attention layer")
        for row, slot in enumerate(slots):
            for states, table in zip(full_states, tables):
                state = states[slot]
                pages = torch.as_tensor(table[row : row + 1], dtype=torch.int32)
                old = state.get("_host_pages")
                if old is None or not torch.equal(old, pages):
                    if tuple(pages.shape) != tuple(state["page_table"].shape):
                        raise ValueError("Page table must match the allocated cache capacity")
                    self._copy_host(pages, state["page_table"])
                    state["_host_pages"] = pages.clone()
                    self.counters["page_table_refreshes"] += 1

    def prefill_forward(
        self,
        tokens,
        *,
        prompt_lens,
        start_pos=None,
        page_table=None,
        page_tables_per_layer=None,
        kv_cache=None,
        slots=None,
        empty_slots=None,
        return_all_logits=False,
        logit_positions=None,
        device_output=False,
        sampling_params=None,
        read_from_device=None,
        enable_trace=False,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported prefill fields: {sorted(kwargs)}")
        if enable_trace:
            raise ValueError("Prefill graph tracing is not prepared; use decode-only tracing")
        arguments = dict(
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            page_table=page_table,
            page_tables_per_layer=page_tables_per_layer,
            kv_cache=kv_cache,
            slots=slots,
            empty_slots=empty_slots,
        )
        if sampling_params is not None or read_from_device is not None:
            if return_all_logits or logit_positions is not None or device_output:
                raise ValueError("Serving prefill output cannot combine compatibility-logit selectors")
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.serving import prefill_with_sampling

            return prefill_with_sampling(
                self,
                tokens,
                low_level=self._prefill_forward,
                sampling_params=sampling_params,
                read_from_device=True if read_from_device is None else read_from_device,
                enable_trace=enable_trace,
                **arguments,
            )
        return self._prefill_forward(
            tokens,
            return_all_logits=return_all_logits,
            logit_positions=logit_positions,
            device_output=device_output,
            **arguments,
        )

    def _prefill_forward(
        self,
        tokens,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens,
        start_pos=None,
        return_all_logits=False,
        logit_positions=None,
        slots=None,
        empty_slots=None,
        page_tables_per_layer=None,
        device_output=False,
        **kwargs,
    ):
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.prefill_validation import validate_prefill

        submission = validate_prefill(
            self,
            tokens,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            slots=slots,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            page_table=page_table,
            page_tables_per_layer=page_tables_per_layer,
            return_all_logits=return_all_logits,
            device_output=device_output,
            logit_positions=logit_positions,
        )
        tokens, slots, starts = submission.tokens, submission.slots, submission.starts
        prompt_lens, logit_positions = submission.ends, submission.requested_indices
        cache = self.cache
        page_table, page_tables_per_layer = None, submission.page_tables_per_layer
        self.output_pool.require_drained()
        try:
            self.prefill_executor.prepare(
                cache=self.cache, reset_state=self.reset, progress=lambda row: print("PREFILL_PREPARE", row, flush=True)
            )
            self._bind_pages(page_table, cache, slots, page_tables_per_layer)
            outputs = []
            for row, (slot, start, end) in enumerate(zip(slots, starts, prompt_lens)):
                if not 0 <= start < end <= self.model.max_seq_len:
                    raise ValueError("Invalid logical prefill interval")
                if return_all_logits and device_output:
                    raise ValueError(
                        "All-logits compatibility output is host-chunked; device output returns the final row"
                    )
                if start == 0:
                    for states, zeros in zip(cache, self.zeros):
                        for key, zero in zeros[slot].items():
                            ttnn.copy(zero, states[slot][key])
                chunks = []
                requested = None if logit_positions is None else list(logit_positions[row])

                def receive_logits(offset, valid_rows, tensor):
                    host = self.model.host_logits(tensor)[:, :valid_rows, :]
                    if requested is None:
                        chunks.append(host)
                    else:
                        chunks.extend(
                            (p, host[:, p - offset : p - offset + 1, :])
                            for p in requested
                            if offset <= p < offset + valid_rows
                        )

                logits = self.prefill_executor.prefill(
                    tokens[row, start:end],
                    cache=cache,
                    slot=slot,
                    start=start,
                    on_logits=receive_logits if return_all_logits else None,
                    logit_positions=requested,
                )
                self._copy_host(torch.tensor([end], dtype=torch.int32), self.positions[slot])
                self.counters["position_refreshes"] += 1
                if return_all_logits:
                    if requested is not None:
                        by_position = dict(chunks)
                        chunks = [by_position[p] for p in requested]
                    outputs.append(torch.cat(chunks, dim=1))
                    ttnn.deallocate(logits)
                else:
                    outputs.append(logits if device_output else self.model.host_logits(logits))
            if device_output:
                return outputs
            if not return_all_logits:
                return torch.cat(outputs, dim=0)
            # Logical continuation rows are right-padded only in the return value.
            return torch.nn.utils.rnn.pad_sequence([x[0] for x in outputs], batch_first=True)
        except BaseException:
            self.slot_state.poisoned = self.reload_executor.poisoned = True
            raise

    def _decode_into_sampler(self, cache, slots, *, masked=False, solo_slot0_attention=False):
        if solo_slot0_attention and (not masked or self.active_slots != (0,) or tuple(slots) != (0, 1)):
            raise ValueError("The solo attention graph requires authoritative slot0-only activity")
        logits = self.model.decode(
            self.tokens,
            cache=cache,
            positions=self.positions,
            slots=slots,
            activity=self.activity if masked else None,
            active_mask=self.active_mask if masked else None,
            solo_slot0_attention=solo_slot0_attention,
        )
        logits = ttnn.reshape(logits, (1, 1, len(slots), logits.shape[-1]))
        if tuple(slots) == tuple(range(len(slots))):
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - len(slots)), (0, 0)], 0)
        else:
            # Sampler row indices are physical request slots. Preserve that
            # mapping for a sparse/reordered active set rather than feeding the
            # next token into row-order indices.
            by_slot = {slot: logits[:, :, row : row + 1, :] for row, slot in enumerate(slots)}
            logits = ttnn.concat([by_slot.get(slot, self.zero_logits_row) for slot in range(32)], dim=2)
        ttnn.copy(logits, self.sampler_logits)

    def _sampling_forward(self):
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import advance_seeds, merge_rows

        self.model.observe_boundary(self.sampler_logits, "sampling_dtype", "sampler.forward_input")
        ttnn.copy(self.tokens, self.previous_tokens)
        self.sampler._run_sampling(self.sampler_logits, penalties_on=False, grammar_on=False, tt_out_tok=self.tokens)
        merge_rows(self.tokens, self.previous_tokens, self.active_mask, self.mesh)
        ttnn.copy(self.previous_tokens, self.tokens)
        advance_seeds(self.sampler.tt_sampling.seeds_tt_tensor, self.active_mask, self.mesh)

    def prepare_without_capture(self, *, prefill_cases=((32, 0),)):
        """Prepare all variants before any generator sharing this mesh captures.

        Ordinary callers use prepare(). This idempotent first phase also lets
        diagnostic control/candidate generators finish all preparation before
        either retains a trace; no new warmup is allowed after capture starts.
        """
        if self.closed:
            raise RuntimeError("Generator has been closed")
        if self._variants_prepared:
            if self.decode_buckets != self._prepared_decode_buckets:
                raise ValueError("Prepared decode variants cannot change before capture")
            return self._preparation_seconds
        if self.traces:
            raise RuntimeError("Cannot prepare new variants while this generator retains traces")
        prepare_started = time.monotonic()
        # Phase 1: all eager and traced programs, buffers, then restore state.
        for length, start in prefill_cases:
            tokens = torch.ones(1, start + length, dtype=torch.int32)
            self.prefill_forward(tokens, prompt_lens=[start + length], start_pos=[start], device_output=True)
        self._set_active_slots((0,))
        self._refresh([1], [1], (0,))
        self._decode_into_sampler(self.cache, (0,))
        for bucket in self.decode_buckets if self.max_batch_size > 1 else ():
            fixed = tuple(range(bucket))
            self._set_active_slots(fixed)
            self._refresh([1] * len(fixed), [1] * len(fixed), fixed)
            self._decode_into_sampler(self.cache, fixed, masked=True)
        if self.solo_slot0_attention:
            self._set_active_slots((0,))
            self._refresh([1], [1], (0,))
            self._decode_into_sampler(self.cache, (0, 1), masked=True, solo_slot0_attention=True)
        for width in (32, 64):
            self.sampler.tt_sampling._sampling_draw_width = width
            self.sampler.precompile(self.sampler_logits, tt_out_tok=self.tokens)
            self._sampling_forward()
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import merge_rows

        merge_rows(
            self.sampling_seed_updates, self.sampler.tt_sampling.seeds_tt_tensor, self.sampling_update_mask, self.mesh
        )
        self.prefill_logit_rows.prepare(source=self.zero_logits_row, destination=self.sampler_logits)
        self.slot_state.prepare()
        self.output_pool.prepare(token_template=self.tokens, logits_template=self.sampler_logits)
        print("GENERATOR_PREPARED", flush=True)
        self.reset()
        self._set_active_slots((0,))
        self._refresh([1], [1], (0,))
        ttnn.synchronize_device(self.mesh)
        self._preparation_seconds = time.monotonic() - prepare_started
        self._prepared_decode_buckets = self.decode_buckets
        self._variants_prepared = True
        return self._preparation_seconds

    def prepare(self, *, prefill_cases=((32, 0),), slots=(0,)):
        if self.closed:
            raise RuntimeError("Generator has been closed")
        if self.prepared:
            return
        preparation_seconds = self.prepare_without_capture(prefill_cases=prefill_cases)
        # Phase 2: capture only over preallocated state. The model writes logits
        # before the sampling trace reads them; sampling writes tokens before
        # the next model replay. Neither output aliases temporary trace storage.
        for bucket in self.decode_buckets:
            capture_started = time.monotonic()
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            key = self._bucket_trace_key(bucket)
            try:
                self._decode_into_sampler(self.cache, tuple(range(bucket)), masked=self.max_batch_size > 1)
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                self.traces[key] = trace
            print("MODEL_CAPTURED", key, flush=True)
            self.counters["captures"] += 1
            self._record_capture(key, trace, time.monotonic() - capture_started, preparation_seconds, bucket)
        if self.solo_slot0_attention:
            capture_started = time.monotonic()
            key = "model_solo0"
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._decode_into_sampler(self.cache, (0, 1), masked=True, solo_slot0_attention=True)
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                self.traces[key] = trace
            print("MODEL_CAPTURED", key, flush=True)
            self.counters["captures"] += 1
            self._record_capture(key, trace, time.monotonic() - capture_started, preparation_seconds, 2)
        # Both native widths use the same canonical candidate pipeline and
        # persistent parameters, tokens and seeds. Compile both before any
        # model trace; capture both once and retain them for the server lifetime.
        for width, key in ((32, "sampling"), (64, "sampling_64")):
            self.sampler.tt_sampling._sampling_draw_width = width
            capture_started = time.monotonic()
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._sampling_forward()
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                self.traces[key] = trace
            print("SAMPLER_CAPTURED", key, flush=True)
            self.counters["captures"] += 1
            self._record_capture(key, trace, time.monotonic() - capture_started, preparation_seconds)
        self.prepared = True
        self.reset()

    def _bucket_trace_key(self, bucket):
        if self.max_batch_size == 1:
            return "model"
        return "model_fixed" if bucket == self.max_batch_size else f"model_{bucket}"

    def model_trace_key(self, active_slots):
        if self.solo_slot0_attention and 2 in self.decode_buckets and tuple(active_slots) == (0,):
            return "model_solo0"
        # Slot identity matters: a lone off-prefix owner still needs the larger graph.
        extent = max(active_slots, default=0) + 1
        for bucket in self.decode_buckets:
            if extent <= bucket:
                return self._bucket_trace_key(bucket)
        raise ValueError("Active state lies outside the prepared decode buckets")

    def sampling_trace_key(self):
        # Native sampling reads every physical row before the active-row merge.
        # Retained dormant parameters must also fit the selected draw width;
        # discarding their tokens cannot make an out-of-range native read safe.
        values = self.sampling_values
        wide = any(temperature > 0 and top_k > 32 for temperature, top_k in zip(values["temperature"], values["top_k"]))
        return "sampling_64" if wide else "sampling"

    def replay_sampling(self):
        """Replay the canonical token-out graph sized for all retained sampler rows."""
        key = self.sampling_trace_key()
        if key not in self.traces:
            raise RuntimeError(f"Sampling execution signature {key} was not prepared")
        self.sampler.tt_sampling._sampling_draw_width = 64 if key == "sampling_64" else 32
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
        self.sampling_trace_replays[key] = self.sampling_trace_replays.get(key, 0) + 1
        self.counters["sampling_replays"] += 1

    def _record_capture(self, key, trace, seconds, preparation_seconds, physical_rows=32):
        """Observe generator-owned capture without changing its lifetime."""
        print(
            "QWEN_TRACE_EVENT "
            + json.dumps(
                dict(
                    event="capture",
                    key=key,
                    trace_id=str(trace),
                    capture_seconds=seconds,
                    preparation_seconds=preparation_seconds,
                    signature={
                        "graph": key,
                        "capacity": self.max_batch_size,
                        "physical_rows": physical_rows,
                        "attention_slots": [0] if key == "model_solo0" else None,
                        "solo_zero_buffers": (
                            [self.model.solo_linear_zero.buffer_address(), self.model.solo_full_zero.buffer_address()]
                            if key == "model_solo0"
                            else None
                        ),
                        "layers": list(self.model.layer_indices),
                        "cache_shapes": [list(group[0]["k"].shape) for group in self.cache if "k" in group[0]],
                        "sampler_shape": list(self.sampler_logits.shape),
                        "sampling_candidate_width": self.sampler.tt_sampling.max_top_k,
                        "native_draw_width": (
                            (64 if key == "sampling_64" else 32) if key.startswith("sampling") else None
                        ),
                        "token_buffer": self.tokens.buffer_address(),
                    },
                    captures=self.counters["captures"],
                    releases=self.counters["releases"],
                )
            ),
            flush=True,
        )

    def _sample(self):
        if self.host_sampling:
            # Explicit test compatibility boundary, excluded from token-out
            # device performance. Never selected as a runtime fallback.
            logits = self.model.host_logits(self.sampler_logits)[0, 0]
            ids = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
            for slot in self.active_slots:
                t = self.sampling_values["temperature"][slot]
                k = self.sampling_values["top_k"][slot]
                p = self.sampling_values["top_p"][slot]
                scores = logits[slot].float()
                if t == 0 or k == 1:
                    token = int(scores.argmax())
                else:
                    scores = scores / t
                    if k > 0:
                        cutoff = scores.topk(min(k, scores.numel())).values[-1]
                        scores = scores.masked_fill(scores < cutoff, -torch.inf)
                    sorted_scores, indices = scores.sort(descending=True)
                    probabilities = sorted_scores.softmax(-1)
                    remove = probabilities.cumsum(-1) - probabilities >= p
                    remove[0] = False
                    probabilities.masked_fill_(remove, 0)
                    chosen = torch.multinomial(probabilities, 1, generator=self.host_rng[slot])
                    token = int(indices[chosen])
                ids[..., slot] = token
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import merge_rows

            self._copy_host(ids, self.incoming_tokens)
            merge_rows(self.incoming_tokens, self.tokens, self.active_mask, self.mesh)
            self.counters["readbacks"] += 1
            self.counters["token_refreshes"] += 1
            return
        self.replay_sampling()

    def _read_tokens(self, slots):
        result = ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).flatten()
        self.counters["readbacks"] += 1
        return [int(result[s]) for s in slots]

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        kv_cache=None,
        slots=None,
        refresh_inputs=True,
        read_tokens=True,
        return_logits=False,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported legacy decode fields; use decode_forward_v1: {sorted(kwargs)}")
        cache = self.cache if kv_cache is None else kv_cache
        slots = (
            (tuple(range(len(start_pos))) if refresh_inputs else self.active_slots) if slots is None else tuple(slots)
        )
        if not self.prepared or cache is not self.cache:
            raise RuntimeError("Prepare this stable cache/slot execution signature before traced decode")
        normalized_pages = None
        if refresh_inputs:
            if len(slots) != len(start_pos):
                raise ValueError("Positions and slots must have the same row count")
            if len(set(slots)) != len(slots) or any(s < 0 or s >= self.max_batch_size for s in slots):
                raise ValueError("Invalid decode slots")
            if any(int(pos) < -1 or int(pos) >= self.model.max_seq_len for pos in start_pos):
                raise ValueError("Decode position exceeds configured context")
            active_rows = [row for row, pos in enumerate(start_pos) if int(pos) >= 0]
            active_slots = tuple(slots[row] for row in active_rows)
            token_rows = tokens.flatten()
            if token_rows.device.type != "cpu" or token_rows.dtype not in (torch.int32, torch.int64):
                raise ValueError("Legacy decode requires CPU INT32/INT64 tokens")
            if token_rows.numel() != len(slots):
                raise ValueError("Tokens and slots must have the same row count")
            if any(not 0 <= int(token_rows[row]) < self.model.vocab_size for row in active_rows):
                raise ValueError("Decode token is outside the vocabulary")
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.serving import _validated_pages

            normalized_pages = _validated_pages(self, page_table, None, slots, active_slots, start_pos)
            self._set_active_slots(active_slots)
        if not self.active_slots:
            return torch.empty(0, dtype=torch.int64) if read_tokens else self.tokens
        self._bind_pages(
            None if normalized_pages is not None else page_table, cache, slots, page_tables_per_layer=normalized_pages
        )
        if refresh_inputs:
            self._refresh(token_rows[active_rows], [start_pos[row] for row in active_rows], active_slots)
        key = self.model_trace_key(self.active_slots)
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
        self.trace_replays[key] = self.trace_replays.get(key, 0) + 1
        self.counters["model_replays"] += 1
        if return_logits:
            self.counters["readbacks"] += 1
            return self.model.host_logits(self.sampler_logits)[0, 0, list(slots), :]
        self._sample()
        return torch.tensor(self._read_tokens(slots)) if read_tokens else self.tokens

    def decode_forward_v1(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        page_tables_per_layer=None,
        kv_cache=None,
        sampling_params=None,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        slots=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        read_from_device=True,
        enable_trace=True,
        **kwargs,
    ):
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.serving import decode_with_commands

        if kwargs:
            raise TypeError(f"Unsupported explicit decode fields: {sorted(kwargs)}")
        return decode_with_commands(
            self,
            tokens,
            start_pos,
            page_table=page_table,
            page_tables_per_layer=page_tables_per_layer,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            slot_remap=slot_remap,
            slots=slots,
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
            read_from_device=read_from_device,
            enable_trace=enable_trace,
        )

    def read_decode_output(self, output, *, async_read=False, sample_rows=None):
        return self.output_pool.read(output, async_read=async_read, sample_rows=sample_rows)

    def process_decode_output_host(self, output, *, is_tokens, sample_rows=None, events_synchronized=False):
        return self.output_pool.process(
            output, is_tokens=is_tokens, sample_rows=sample_rows, events_synchronized=events_synchronized
        )

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]), prompt_lens=[len(prompt_token_ids)], return_all_logits=True
        )

    def prefill_logits_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        self.reset()
        ids = torch.tensor([prompt_token_ids])
        if start and prefill_prefix:
            self.prefill_forward(ids, prompt_lens=[start], device_output=True)
        logits = self.prefill_forward(
            ids,
            prompt_lens=[len(prompt_token_ids)],
            start_pos=[start],
            return_all_logits=True,
            logit_positions=None if positions is None else [[p - start for p in positions]],
        )
        return logits

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        stop_on_eos=False,
        stop_token_ids=(),
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Generation requires traced decode")
        if max_new_tokens < 1:
            return []
        for name, neutral in (
            ("presence_penalty", 0.0),
            ("frequency_penalty", 0.0),
            ("repetition_penalty", 1.0),
            ("min_p", 0.0),
        ):
            if kwargs.get(name, neutral) != neutral:
                raise ValueError(f"Standalone {name} is unsupported; use the explicit caller-owned host-logit path")
        if kwargs.get("stop"):
            raise ValueError("String stopping belongs to the caller; use stop_token_ids for standalone generation")
        # The readiness ABC reserves unknown extras for callers and asks that
        # they be ignored. Recognized request features must never be silently
        # mistaken for implemented standalone sampling controls.
        for name in (
            "min_tokens",
            "bad_words",
            "logit_bias",
            "allowed_token_ids",
            "custom_logits_processors",
            "structured_outputs",
        ):
            if kwargs.get(name) is not None and kwargs[name] not in (0, False, [], {}):
                raise ValueError(f"Standalone {name} requires the explicit caller-owned host-logit path")
        for name in ("logprobs", "prompt_logprobs"):
            if kwargs.get(name) is not None:
                raise ValueError(f"Standalone {name} requires the explicit caller-owned host-logit path")
        stop_ids = set(map(int, stop_token_ids))
        if stop_on_eos:
            eos = self.model.config.eos_token_id
            stop_ids.update(eos if isinstance(eos, (list, tuple)) else (() if eos is None else (eos,)))
        length = len(prompt_token_ids)
        if length + max_new_tokens - 1 > self.model.max_seq_len:
            raise ValueError("Generation exceeds configured context")
        self.configure_sampling(
            temperature=kwargs.get("temperature", 0.0),
            top_k=kwargs.get("top_k", 1),
            top_p=kwargs.get("top_p", 1.0),
            seed=kwargs.get("seed", 0),
        )
        self.prepare(prefill_cases=((length, 0),))
        self.reset()
        logits = self.prefill_forward(torch.tensor([prompt_token_ids]), prompt_lens=[length], device_output=True)[0]
        logits = ttnn.reshape(logits, (1, 1, 1, logits.shape[-1]))
        ttnn.copy(ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], 0), self.sampler_logits)
        ttnn.deallocate(logits)
        self._copy_host(torch.tensor([length], dtype=torch.int32), self.positions[0])
        self.counters["position_refreshes"] += 1
        self._sample()
        predictions = []
        for step in range(max_new_tokens):
            predicted = self._read_tokens([0])[0]
            predictions.append(predicted)
            if step + 1 == max_new_tokens or predicted in stop_ids:
                break
            if next_input is not None:
                forced = int(next_input(step, predicted))
                ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ids[..., 0] = forced
                self._copy_host(ids, self.tokens)
                self.counters["token_refreshes"] += 1
            self.decode_forward(None, [0], refresh_inputs=False, read_tokens=False)
        if next_input is not None:
            next_input(len(predictions) - 1, predictions[-1])
        return predictions

    def teardown(self):
        self.close()

    def release_request(self, slot):
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.cache_contract import release_request

        return release_request(self, slot)

    def release_persistent_capture(self):
        self.close()

    def close(self):
        if self.closed:
            return
        for handle in list(self.output_pool.leases):
            if handle is not None:
                self.output_pool.discard(handle)
        for trace in self.traces.values():
            ttnn.release_trace(self.mesh, trace)
            self.counters["releases"] += 1
        self.traces.clear()
        self.closed = True


def build_generator(model_dir, mesh_device, **kwargs):
    allowed = {
        "checkpoint",
        "layer_indices",
        "max_seq_len",
        "max_batch_size",
        "host_sampling",
        "kv_cache",
        "per_layer_cache_specs",
        "prepare",
        "lm_head_mode",
        "precision_config",
    }
    if kwargs.keys() - allowed:
        raise TypeError(f"Unsupported generator construction fields: {sorted(kwargs.keys() - allowed)}")
    model = QwenModel(
        mesh_device,
        checkpoint=kwargs.get("checkpoint"),
        layer_indices=kwargs.get("layer_indices"),
        max_seq_len=kwargs.get("max_seq_len"),
        lm_head_mode=kwargs.get("lm_head_mode", "dram_sharded"),
        precision_config=kwargs.get("precision_config"),
    )
    cache = kwargs.get("kv_cache")
    specs = kwargs.get("per_layer_cache_specs")
    if cache is not None and specs is not None:
        raise ValueError("Provide kv_cache or per_layer_cache_specs")
    if specs is not None:
        cache = model.allocate_kv_cache_per_layer(specs, batch=kwargs.get("max_batch_size", 1))
    generator = QwenGenerator(
        model,
        max_batch_size=kwargs.get("max_batch_size", 1),
        host_sampling=kwargs.get("host_sampling", False),
        kv_cache=cache,
    )
    if kwargs.get("prepare", False):
        generator.prepare(prefill_cases=((min(32, model.max_seq_len), 0),))
    return generator
