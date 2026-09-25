# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for the selected TP4 Qwen generator."""

import json
import os
import secrets
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.qwen38_27b_qb2.tt.generator import build_generator
from models.demos.qwen38_27b_qb2.tt.model import ModelCache


class Qwen38ForCausalLM:
    _MAX_CONTEXT = 262144
    # Leave room for every output position and the final traced increment.
    # No explicit-seed stream reaches int32 overflow or manual_seed's -1 sentinel.
    _SEED_MODULUS = (1 << 31) - _MAX_CONTEXT - 1

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        if tuple(mesh_device.shape) != (1, 4) or tt_data_parallel != 1:
            raise ValueError("Qwen3.8 requires a TP4 MeshShape(1,4)")
        if not 1 <= max_batch_size <= 32 or not 1 <= max_seq_len <= cls._MAX_CONTEXT:
            raise ValueError("Serving dimensions exceed the validated model contract")
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
        self._last_device_sampling = None
        self.prefill_startup_warmup = os.getenv("QWEN_PREFILL_STARTUP_WARMUP", "0") == "1"

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
        # Bound the allocation to the measured TP4 BFP8 KV pool, independently
        # of the per-request context limit. Eight 128K prompts need more than
        # the default shared 256K pool even when admission permits eight users.
        if tokens % 32 or not cls._MAX_CONTEXT <= tokens <= 1179648:
            raise ValueError("Explicit KV pool must be 32-token aligned within 262144..1179648")
        if kwargs.get("num_devices", 4) != 4 or kwargs.get("tt_data_parallel", 1) != 1:
            raise ValueError("Explicit KV pool requires TP4")
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
        if (heads, block, dim) != (1, 32, 256):
            raise ValueError(f"Expected TP4 cache [pages,1,32,256], got {kv_cache_shape}")
        model = self.generator.model
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

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        empty_slots=None,
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
