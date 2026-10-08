# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for the selected tensor-parallel Qwen generator."""

import json
import os
import secrets
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import resolve_mesh_tp, supported_device_counts
from models.demos.qwen38_27b_t3k.tt.generator import MAX_PREFIX_SNAPSHOTS, build_generator
from models.demos.qwen38_27b_t3k.tt.model import MAX_SERVING_BATCH, ModelCache


def _shared_pool_ceiling():
    """Largest paged KV pool that fits beside the weights, trace reserve and GDN state.

    Re-derived from the recorded byte budget rather than a transcribed token count, so a
    correction to the capacity contract cannot leave this bound behind.

    Prefix snapshots are the same recurrent bytes held a second time, drawn from this same
    budget, so the store's hard cap is reserved here whether or not prefix caching is on: the
    pool is sized once at startup and cannot give the memory back later.
    """
    contract = json.loads((Path(__file__).parents[1] / "doc/context_contract.json").read_text())
    budget = contract["per_device_bytes"]
    per_request = budget["gdn_recurrent_and_conv_per_request"]
    spare = budget["state_budget_remaining"] - per_request * (1 + MAX_PREFIX_SNAPSHOTS)
    return spare // budget["kv_cache_per_page"] * 32


# Shortest prefix worth stopping a prefill for. Restoring one costs ~55 ms and splitting the
# prefill that saves it ~200 ms, against ~2.5 tokens/ms of prefill avoided on a hit, so a prefix
# repays its own split about fivefold at this length and not at all a block or two above zero.
MIN_SNAPSHOT_TOKENS = 4096


class Qwen38ForCausalLM:
    _MAX_CONTEXT = 262144
    # Leave room for every output position and the final traced increment.
    # No explicit-seed stream reaches int32 overflow or manual_seed's -1 sentinel.
    _SEED_MODULUS = (1 << 31) - _MAX_CONTEXT - 1

    # Half the store's hard cap, so a snapshot taken before the displaced one is freed -- the
    # free travels a step behind -- still finds room rather than being dropped.
    _PREFIX_SNAPSHOTS = MAX_PREFIX_SNAPSHOTS // 2

    model_capabilities = {
        # The 48 GDN layers summarise one slot's tokens sequentially, so a cached prefix is
        # servable only where a snapshot holds that summary. The caller caps hits to those,
        # and asks a prefill to stop on a block boundary so there is something to hold.
        "supports_prefix_caching": True,
        "recurrent_prefix_snapshots": _PREFIX_SNAPSHOTS,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_device_penalties": False,
    }

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        # resolve_mesh_tp is the single authority on qualified hardware; it raises with the
        # arch/cluster detail. Data parallelism is orthogonal and unsupported either way.
        resolve_mesh_tp(mesh_device)
        if tt_data_parallel != 1:
            raise ValueError("Qwen3.8 does not support tt_data_parallel > 1")
        if not 1 <= max_batch_size <= MAX_SERVING_BATCH or not 1 <= max_seq_len <= cls._MAX_CONTEXT:
            raise ValueError(
                f"Serving dimensions exceed the validated model contract: max_num_seqs must be "
                f"1..{MAX_SERVING_BATCH}"
            )
        if os.getenv("QWEN_DECODE_BUCKETS", "0") == "1" and max_batch_size not in (1, 8, 16):
            raise ValueError("Bucketed decode requires max_num_seqs of 1, 8, or 16; capacity above 16 is unsupported")
        root = Path(__file__).parents[1]
        generator = build_generator(
            root,
            mesh_device,
            precision_config=root / "config/precision.json",
        )
        logger.info("Qwen3.8 vLLM precision: {}", generator.model.precision)
        return cls(generator, max_batch_size, max_seq_len)

    def __init__(self, generator, batch_size, context, *, vllm_config=None):
        self.generator = generator
        self.batch_size, self.context = batch_size, context
        self.host_compatibility = os.environ.get("QWEN_VLLM_HOST_COMPATIBILITY") in ("1", "all")
        self.cache = None
        self._sampling_key = None
        self._decode_bound = False
        self._prefix_snapshots = {}
        self._last_device_sampling = None
        self.prefill_startup_warmup = os.getenv("QWEN_PREFILL_STARTUP_WARMUP", "0") == "1"
        # A served step costs more than this entry point does. Timing decode_forward from the
        # inside is the only way to split that: the difference against the harness's reported
        # TPOT is the work the plugin and vLLM do around the call.
        period = os.getenv("QWEN_DECODE_STEP_TIMING", "0")
        self._step_period = int(period) if period.isascii() and period.isdecimal() else 0
        self._step_times = []

    def _record_step(self, seconds):
        self._step_times.append(seconds * 1e3)
        if len(self._step_times) < self._step_period:
            return
        ordered = sorted(self._step_times)
        n = len(ordered)
        logger.info(
            "Qwen3.8 decode_forward: p50={:.2f} ms p10={:.2f} p90={:.2f} n={} batch={}",
            ordered[n // 2],
            ordered[n // 10],
            ordered[(9 * n) // 10],
            n,
            self.batch_size,
        )
        self._step_times.clear()

    # vLLM inspects this protocol before selecting the TT loader. Execution is
    # through the TT plugin's prefill/decode APIs, never the GPU forward API.
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Use the TT plugin prefill_forward/decode_forward interface")

    def forward(self, input_ids, positions):
        raise NotImplementedError("Use the TT plugin prefill_forward/decode_forward interface")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("The canonical TT generator owns the LM head and sampling")

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, **kwargs):
        # Shared paged pool: one maximum-context request or many short requests.
        context = int(max_model_len or cls._MAX_CONTEXT)
        configured = os.environ.get("QWEN_VLLM_KV_POOL_TOKENS")
        if configured is None:
            return context
        if not configured.isascii() or not configured.isdecimal():
            raise ValueError("QWEN_VLLM_KV_POOL_TOKENS must be positive ASCII decimal tokens")
        tokens = int(configured)
        # The pool is shared across concurrent requests, so it is bounded by device memory
        # rather than by the per-request context limit.
        ceiling = _shared_pool_ceiling()
        if tokens % 32 or not cls._MAX_CONTEXT <= tokens <= ceiling:
            raise ValueError(f"Explicit KV pool must be 32-token aligned within {cls._MAX_CONTEXT}..{ceiling}")
        # The bound is per-device, and this tree qualifies exactly one mesh, so an unspecified
        # device count means that mesh. Data parallelism would multiply the requirement.
        (qualified,) = supported_device_counts()
        if kwargs.get("num_devices", qualified) != qualified or kwargs.get("tt_data_parallel", 1) != 1:
            raise ValueError("Explicit KV pool requires a single qualified mesh")
        return tokens

    @classmethod
    def supports_device_sampling(cls, params, *, is_decode):
        compatibility = os.environ.get("QWEN_VLLM_HOST_COMPATIBILITY")
        # Shared reproducibility tests need one RNG backend for every request.
        # Performance runs leave this explicit compatibility mode disabled.
        if compatibility == "all":
            return False
        unsupported = (
            ((params.temperature > 0) & ((params.top_k > 32) | (params.top_k < 1))).any()
            or (params.presence_penalty != 0).any()
            or (params.frequency_penalty != 0).any()
            or (params.repetition_penalty != 1).any()
        )
        if unsupported:
            if compatibility != "1":
                raise ValueError("Requested sampling needs explicit QWEN_VLLM_HOST_COMPATIBILITY=1")
            return False
        return True

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        pages, heads, block, dim = kv_cache_shape
        model = self.generator.model
        attention = next(layer for layer in model.layers if layer.kind == "full_attention")
        expected = attention.state_shapes(batch_size=self.batch_size, num_pages=pages)["key"][1:]
        if (heads, block, dim) != expected:
            raise ValueError(f"Expected cache [pages, {', '.join(map(str, expected))}], got {kv_cache_shape}")
        self.cache = ModelCache(
            [layer.allocate_state(batch_size=self.batch_size, num_pages=pages) for layer in model.layers],
            self.batch_size,
            self.context,
            pages,
        )
        table = torch.zeros(self.batch_size, (self.context + 31) // 32, dtype=torch.int32)
        self.generator.bind_cache(self.cache, table)
        return self.cache

    def _cache(self, kv_cache):
        if kv_cache is not self.cache or self.generator.cache is not kv_cache:
            raise ValueError("Serving must pass the exact vLLM allocated cache")

    def _table(self, page_table, slots=None):
        if self.generator.page_host is None:
            source = torch.as_tensor(page_table, dtype=torch.int32)
            shape = tuple(self.generator.page_table.shape)
            if source.ndim != 2 or len(shape) != 2 or not 1 <= source.shape[1] <= shape[1]:
                raise ValueError("Page table exceeds the serving context or row mapping")
            rows = list(range(source.shape[0])) if slots is None else list(slots)
            if (
                len(rows) != shape[0]
                or len(rows) != source.shape[0]
                or any(type(row) is not int for row in rows)
                or set(rows) != set(range(shape[0]))
            ):
                raise ValueError("A device-table rebind requires every serving row before partial updates")
            # Every physical row is replaced; do not invent unobserved device rows.
            target = torch.zeros(shape, dtype=torch.int32)
            target[rows, : source.shape[1]] = source
            return target
        source = torch.as_tensor(page_table, dtype=torch.int32)
        target = self.generator.page_host.clone()
        rows = list(range(source.shape[0])) if slots is None else slots
        if source.shape[1] > target.shape[1] or len(rows) != source.shape[0]:
            raise ValueError("Page table exceeds the serving context or row mapping")
        target[rows] = 0
        target[rows, : source.shape[1]] = source
        return target

    def _sampling(self, params, *, reset=False, output_positions=None):
        if params is None:
            if not self.host_compatibility:
                raise ValueError("Host sampling requires explicit QWEN_VLLM_HOST_COMPATIBILITY=1")
            return False
        n = len(params.temperature)
        temps = list(params.temperature)
        ks = [1 if t == 0 else int(k) for k, t in zip(params.top_k, temps)]
        ps = [0.0 if t == 0 else float(p) for p, t in zip(params.top_p, temps)]
        ts = [1.0 if t == 0 else float(t) for t in temps]
        seeds = [int(s) if s is not None else None for s in params.seed]
        key = (tuple(ks), tuple(ps), tuple(ts), tuple(seeds))
        if reset or key != self._sampling_key:
            bound_seeds = None
            if reset or self._sampling_key is None:
                positions = [0] * n if output_positions is None else list(output_positions)
                if len(positions) != n or any(p < 0 or p > self._MAX_CONTEXT for p in positions):
                    raise ValueError("Sampling positions must match rows and lie inside the supported context")
                bound_seeds = [
                    (s % self._SEED_MODULUS if s is not None else secrets.randbelow(self._SEED_MODULUS)) + int(p)
                    for s, p in zip(seeds, positions)
                ] + [1] * (32 - n)
            elif key[3] != self._sampling_key[3]:
                raise ValueError("Changing request seeds requires an authoritative batch reset")
            # A parameter-only update can arrive with lagging async host positions.
            # Preserve advancing device seeds; only a real reset reanchors them.
            self.generator.set_batch_sampling_params(
                top_k=ks + [1] * (32 - n),
                top_p=ps + [0.0] * (32 - n),
                temperature=ts + [1.0] * (32 - n),
                seed=bound_seeds,
            )
            self._sampling_key = key
        return True

    def restore_recurrent_prefix(self, slot, handle):
        """Reinstate a saved prefix into ``slot``, so a prefill may continue from it."""
        return self.generator.restore_slot_state(slot, handle)

    def save_recurrent_prefix(self, slot):
        """Keep ``slot``'s current recurrent state, returning a handle or None when full."""
        return self.generator.save_slot_state(slot)

    def free_recurrent_prefix(self, handle):
        self.generator.free_slot_state(handle)

    def take_prefix_snapshots(self):
        """Row -> ``(tokens, handle)`` for snapshots the last prefill took, cleared on read."""
        taken, self._prefix_snapshots = self._prefix_snapshots, {}
        return taken

    def _snapshot_at_boundary(self, tokens, table, kv_cache, starts, ends, slots, rows, block):
        """Stop the named rows on a block boundary, keep the state there, return the new starts.

        A snapshot is only nameable at a block boundary, because that is where the caller's
        content hashes fall, and recurrent state is sequential: once a prefill has run past a
        boundary there is no way back to it. So the boundary becomes a stopping point rather
        than something to look for afterwards, and the prompt's last partial block is left for
        the sampling pass that follows.

        The boundary is strictly below ``end`` so that pass always has tokens to run.

        Splitting costs a second prefill, and a prefill's cost is mostly the traversal of 64
        layers rather than the tokens in it, so the tail pass is nearly as expensive as a short
        whole one: measured at ~165-200 ms against ~15 ms for the snapshot itself. A boundary
        below MIN_SNAPSHOT_TOKENS cannot repay that even if it is hit, so those rows prefill
        whole and keep nothing.
        """
        starts = list(starts)
        for row in rows:
            boundary = (ends[row] - 1) // block * block
            if boundary <= starts[row] or boundary < MIN_SNAPSHOT_TOKENS:
                # Nothing whole beyond what this slot already holds, or not worth the split.
                continue
            self.generator.prefill_forward(
                tokens[row : row + 1, starts[row] : boundary],
                page_table=table,
                kv_cache=kv_cache,
                prompt_lens=[boundary - starts[row]],
                start_pos=[starts[row]],
                slots=[slots[row]],
            )
            handle = self.generator.save_slot_state(slots[row])
            if handle is not None:
                self._prefix_snapshots[row] = (boundary, handle)
            starts[row] = boundary
        return starts

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        empty_slots=None,
        snapshot_rows=None,
        snapshot_block=0,
        **kwargs,
    ):
        self._cache(kv_cache)
        ends = torch.as_tensor(prompt_lens).reshape(-1).tolist()
        starts = [0] * len(ends) if start_pos is None else torch.as_tensor(start_pos).reshape(-1).tolist()
        slots = list(range(len(ends))) if empty_slots is None else list(empty_slots)
        table = self._table(page_table, slots)
        fresh = [slot for slot, start in zip(slots, starts) if start == 0]
        if fresh:
            self.generator.reset_recurrent_slots(fresh)
        # After the reset, so a fresh slot's snapshot summarises this prompt and not the last
        # request to hold the slot.
        if snapshot_rows and snapshot_block:
            starts = self._snapshot_at_boundary(
                tokens, table, kv_cache, starts, ends, slots, snapshot_rows, snapshot_block
            )
        device_sampling = self._sampling(sampling_params, reset=True, output_positions=ends)
        if device_sampling:
            tokens_out = self.generator.serving_prefill_tokens(
                tokens,
                page_table=table,
                kv_cache=kv_cache,
                prompt_lens=ends,
                start_pos=starts,
                slots=slots,
            )
            result = self.process_decode_output_host(self.read_decode_output(tokens_out), is_tokens=True)[: len(ends)]
        else:
            outputs = []
            for row, (start, end, slot) in enumerate(zip(starts, ends, slots)):
                outputs.extend(
                    self.generator.prefill_forward(
                        tokens[row : row + 1, start:end],
                        page_table=table,
                        kv_cache=kv_cache,
                        prompt_lens=[end - start],
                        start_pos=[start],
                        slots=[slot],
                    )
                )
            result = torch.cat([self.generator._host_logits(x).reshape(1, 1, -1) for x in outputs], dim=0)
        self._decode_bound = False
        # HF declares M-RoPE; text-only positions have zero spatial offset.
        return result, torch.zeros(len(ends), dtype=torch.int64)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=True,
        slot_remap=None,
        **kwargs,
    ):
        start = time.perf_counter() if self._step_period else None
        try:
            self._cache(kv_cache)
            if slot_remap is not None:
                self.generator.remap_recurrent_slots(slot_remap)
            refresh = (
                reset_batch
                or not self._decode_bound
                or (sampling_params is not None and self._last_device_sampling is False)
            )
            positions = torch.as_tensor(start_pos).reshape(-1)
            device_sampling = self._sampling(
                sampling_params, reset=refresh, output_positions=(positions + 1).tolist() if refresh else None
            )
            active = (positions >= 0).nonzero().reshape(-1).tolist() if refresh else None
            # Page growth is independent of reset_batch. The generator compares tables
            # and copies only changes, preserving pending device tokens and positions.
            table = self._table(page_table)
            result = self.generator.decode_forward(
                tokens=tokens if refresh or not device_sampling else None,
                start_pos=positions if refresh or not device_sampling else None,
                page_table=table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                read_from_device=False,
                active_slots=active,
                host_sampling=not device_sampling,
            )
            self._decode_bound = True
            self._last_device_sampling = device_sampling
            if not device_sampling:
                return result.reshape(self.batch_size, 1, -1)
            if read_from_device:
                return self.process_decode_output_host(self.read_decode_output(result), is_tokens=True)
            return result
        finally:
            if start is not None:
                self._record_step(time.perf_counter() - start)

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            if not self.host_compatibility:
                raise ValueError("Host logits require explicit QWEN_VLLM_HOST_COMPATIBILITY=1")
            # Explicit compatibility already materialized these logits on host.
            return (tt_out, []) if async_read else tt_out
        # Copy just one replica's 32 uint32 tokens, ordered before the next replay.
        host = ttnn.get_device_tensors(tt_out)[0].cpu(blocking=not async_read)
        self.generator.counters["token_readbacks"] += 1
        return (host, [ttnn.record_event(self.generator.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if isinstance(tt_out, torch.Tensor) and not is_tokens:
            if not self.host_compatibility:
                raise ValueError("Host logits require explicit QWEN_VLLM_HOST_COMPATIBILITY=1")
            return tt_out
        if not is_tokens:
            raise ValueError("Device output contains tokens only")
        # The plugin's synchronous path passes the raw device result here.
        # Its async path has already submitted the same minimal token read.
        if ttnn.is_tensor_storage_on_device(tt_out):
            tt_out = self.read_decode_output(tt_out)
        return ttnn.to_torch(tt_out).reshape(-1)[: self.batch_size].long().reshape(-1, 1)

    def warmup_model_prefill(self, **kwargs):
        # Optional deployment warmup, independent of any benchmark's request list.
        # Compile the model's native stack chunk at every supported occupancy.
        if (
            not getattr(self, "prefill_startup_warmup", False)
            or not self.generator.batched_prefill
            or self.batch_size == 1
            or getattr(self, "_prefill_startup_warmed", False)
        ):
            return
        cache = kwargs["kv_cache"]
        self._cache(cache)
        chunk = min(4096, self.context // 32 * 32, (cache.num_pages // self.batch_size - 1) * 32)
        if chunk < 32:
            return
        begin = time.perf_counter()
        original_table = self.generator.page_host.clone()
        pages = min(chunk // 32 + 1, original_table.shape[1])
        table = torch.zeros_like(original_table)
        table[:, :pages] = torch.arange(self.batch_size * pages, dtype=torch.int32).reshape(self.batch_size, pages)
        self.generator._release_traces()
        try:
            for batch in range(1, self.batch_size + 1):
                slots = list(range(batch))
                self.generator.reset_recurrent_slots(slots)
                outputs = self.generator.prefill_forward(
                    torch.zeros(batch, chunk, dtype=torch.int64),
                    page_table=table,
                    kv_cache=cache,
                    prompt_lens=[chunk] * batch,
                    slots=slots,
                )
                del outputs
            ttnn.synchronize_device(self.generator.mesh)
        finally:
            self.generator._release_traces()
            self.generator.reset()
            self.generator._refresh_table(original_table)
            self._decode_bound = False
        self._prefill_startup_warmed = True
        logger.info(
            "Qwen3.8 prefill startup warmup: {}",
            json.dumps(
                dict(
                    chunk=chunk,
                    occupancies=list(range(1, self.batch_size + 1)),
                    seconds=time.perf_counter() - begin,
                    cache_reset=True,
                )
            ),
        )

    def warmup_model_decode(self, **kwargs):
        # Warmup/capture restores request state; first decode uses that same path.
        pass

    def teardown(self):
        self.close()

    def close(self):
        logger.debug("Qwen3.8 vLLM counters: {}", json.dumps(dict(self.generator.counters), sort_keys=True))
        self.generator.close()
