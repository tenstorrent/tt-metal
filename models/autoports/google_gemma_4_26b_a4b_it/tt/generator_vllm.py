# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM interface for the selected Gemma4 autoport generator."""

import os

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator


class AutoportGemma4ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "supports_async_decode": True,
    }
    kv_cache_specs_use_local_heads = True
    needs_sampling_output_counts = True

    def __init__(self, generator, max_batch_size, *, vllm_config=None):
        self.generator = generator
        self.mesh = generator.mesh
        self.max_batch_size = max_batch_size
        self._sampling_signature = None
        self._sampling_on_host = None
        self._decode_batch = None
        self.eager_prefill_decode_reuse = os.environ.get("GEMMA4_EAGER_PREFILL_DECODE_REUSE", "1") == "1"
        self._eager_prefill_key = None
        self._eager_decode_trace_id = None
        self._pending_eager_prefill_key = None
        self.allow_host_sampling = os.environ.get("GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING") == "1"
        if os.environ.get("GEMMA4_AUTOPORT_TTFT_DIAGNOSTICS") == "1":
            from models.autoports.google_gemma_4_26b_a4b_it.tools.ttft_diagnostics import install

            install(self)
        if control_path := os.environ.get("GEMMA4_BENCHMARK_CONTROL"):
            from models.autoports.google_gemma_4_26b_a4b_it.tools.benchmark_runtime import install

            install(self, control_path)

    def embed_input_ids(self, input_ids):
        return self.generator.model.embed(input_ids)

    def compute_logits(self, hidden_states):
        return self.generator.model.logits(hidden_states)

    def forward(self, input_ids, positions):
        raise NotImplementedError("TT serving uses prefill_forward/decode_forward with scheduler-owned KV cache")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
    ):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 32:
            raise ValueError("Gemma4 autoport requires TP4, DP1 and at most 32 sequences")
        probe = os.environ.get("GEMMA4_AUTOPORT_PROBE_LAYERS")
        indices = tuple(int(v) for v in probe.split(",")) if probe else None
        generator = Gemma4Generator(mesh_device, max_seq_len=max_seq_len, layer_indices=indices)
        generator.model.precision_summary()
        return cls(generator, max_batch_size)

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=262144, max_num_seqs=32, **kwargs):
        # Five sliding groups and one full group share each physical pool.
        # Full-prompt prefill must allocate a page in every group before old
        # sliding pages can be retired. Include page rounding for all slots.
        return 6 * (max_model_len + 32 * max_num_seqs)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        config = vllm_config.model_config.hf_config
        config = getattr(config, "text_config", config)
        if vllm_config.cache_config.block_size != 32:
            raise ValueError("The selected precision policy requires 32-token KV pages")
        result = {}
        for index, kind in enumerate(config.layer_types):
            sliding = kind == "sliding_attention"
            heads = (
                config.num_key_value_heads
                if sliding
                else (config.num_global_key_value_heads or config.num_key_value_heads)
            )
            width = config.head_dim if sliding else (config.global_head_dim or config.head_dim)
            common = dict(block_size=32, num_kv_heads=max(1, heads // 4), head_size=width, dtype=torch.bfloat16)
            spec = (
                SlidingWindowSpec(**common, sliding_window=config.sliding_window)
                if sliding
                else FullAttentionSpec(**common)
            )
            result[f"model.layers.{index}.self_attn"] = spec
        return result

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate scheduler-owned pools with tile-preserving shape aliases.

        Each group has the same physical page byte size (2x32x256 or
        1x32x512 BFP8). Their allocator block IDs are disjoint. The experimental
        view reinterprets tile storage without copying; it is not a logical
        reshape of initialized values between different attention geometries.
        """
        pools = {}
        caches = []
        for index, layer in zip(self.generator.model.layer_indices, self.generator.model.layers):
            shape, _, tensor_index = per_layer_specs[index]
            expected = layer.layer.self_attn.config
            if tuple(shape[1:]) != (expected.num_key_value_heads, 32, expected.head_dim):
                raise ValueError(f"Layer {index} KV geometry mismatch: {shape}")
            key = (tensor_index, layer.kv_cache_dtype)
            if key not in pools:
                pools[key] = tuple(
                    self.generator.model.upload(torch.zeros(shape, dtype=torch.bfloat16), layer.kv_cache_dtype)
                    for _ in range(2)
                )
            pair = tuple(
                tensor if tuple(tensor.shape) == tuple(shape) else ttnn.experimental.view(tensor, shape)
                for tensor in pools[key]
            )
            if any(a.buffer_address() != b.buffer_address() for a, b in zip(pair, pools[key])):
                raise RuntimeError("Hybrid cache views must preserve physical buffer ownership")
            caches.append(pair)
        self._serving_cache = caches
        return caches

    def _tables(self, page_table, page_tables_per_layer):
        if page_tables_per_layer is None:
            raise ValueError("Hybrid serving requires scheduler-provided per-layer page tables")
        return tuple(page_tables_per_layer[i] for i in self.generator.model.layer_indices)

    def _select_sampling_mode(self, sampling_params):
        on_host = sampling_params is None
        if on_host and not self.allow_host_sampling:
            raise ValueError("Host sampling requires explicit GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1 compatibility mode")
        if on_host != self._sampling_on_host:
            self.generator._release_trace()
            self._sampling_signature = None
            self._sampling_on_host = on_host
            self._decode_batch = None
        return on_host

    def _read_host_logits(self, logits, batch):
        host = self.generator._read_logits(logits)
        return host.reshape(-1, host.shape[-1])[:batch, None, :]

    @staticmethod
    def _output_counts(output_token_counts, batch):
        counts = torch.as_tensor(output_token_counts)
        if (
            counts.ndim != 1
            or counts.numel() != batch
            or counts.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or (counts < 0).any()
        ):
            raise ValueError("output_token_counts must contain one nonnegative integer per input row")
        return counts.to(dtype=torch.int64)

    def _decode_inputs(self, tokens, start_pos, page_table, page_tables_per_layer, *, reset_batch):
        # The plugin pads its wire tensors to max_num_seqs. Retain interior
        # inactive slots but avoid running the model over unused trailing rows.
        if reset_batch or self._decode_batch is None:
            active = torch.nonzero(start_pos >= 0).flatten()
            if not len(active):
                raise ValueError("Decode requires at least one active request")
            self._decode_batch = int(active[-1]) + 1
        batch = self._decode_batch
        tables = tuple(table[:batch] for table in self._tables(page_table, page_tables_per_layer))
        return None if tokens is None else tokens[:batch], start_pos[:batch], tables

    def _eager_prefill_signature(self, tokens, tables, cache, prompt_lens, sampling_params, counts):
        gen = self.generator
        if (
            not self.eager_prefill_decode_reuse
            or sampling_params is None
            or gen.host_sampling
            or len(prompt_lens) != 1
            or tokens.shape[0] != 1
            or not gen.serving_prefill_eligible(sampling_params)
            or (counts is not None and counts.any())
            or gen._serving_prefill_key(tokens, tables, cache, prompt_lens) is not None
        ):
            return None
        length = int(prompt_lens[0])
        if not 1 <= length <= min(tokens.shape[1], gen.model.max_seq_len):
            return None
        if any(not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[0] < 1 for t in tables):
            return None
        # Cache objects own the addresses baked into decode. Values and page IDs
        # may change, but replacing even one view invalidates the warmed bundle.
        cache_specs = tuple(
            (
                id(t),
                t.buffer_address(),
                tuple(t.shape),
                tuple(t.padded_shape),
                str(t.dtype),
                str(t.layout),
                str(t.memory_config()),
            )
            for pair in cache
            for t in pair
        )
        return (
            id(gen.model),
            id(cache),
            cache_specs,
            length,
            tuple(tokens.shape),
            str(tokens.dtype),
            tuple(tokens.stride()),
            str(tokens.device),
            tuple((tuple(t.shape), str(t.dtype), tuple(t.stride()), str(t.device)) for t in tables),
        )

    def _has_eager_decode_trace(self):
        gen = self.generator
        return (
            self._eager_decode_trace_id is not None
            and self._eager_decode_trace_id == gen.trace_id
            and gen.batch == 1
            and gen.active_slots == (0,)
            and not gen.host_sampling
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        empty_slots=None,
        page_tables_per_layer=None,
        enable_trace=False,
        output_token_counts=None,
    ):
        on_host = self._select_sampling_mode(sampling_params)
        if start_pos is not None and torch.any(torch.as_tensor(start_pos) != 0):
            raise ValueError("Scheduler chunked prefill needs the low-level continuation contract")
        gen = self.generator
        counts = None
        if not on_host:
            prompt_lengths = torch.as_tensor(prompt_lens, device=tokens.device).reshape(-1, 1)
            counts = None if output_token_counts is None else self._output_counts(output_token_counts, tokens.shape[0])
            if counts is not None and (counts > prompt_lengths.flatten()).any():
                raise ValueError("output_token_counts cannot exceed the resumed prefill prefix length")
        tables = self._tables(page_table, page_tables_per_layer)
        trace_prefill = (
            not on_host
            and getattr(gen, "prefill_trace_enabled", False)
            and gen.serving_prefill_eligible(sampling_params)
        )
        if trace_prefill:
            # The scheduler may retain max_num_seqs rows in hybrid tables
            # even for a compact prefill token batch. Only its leading prompt
            # rows are consumed; retain all columns for the context contract.
            tables = tuple(table[: tokens.shape[0]] for table in tables)
        reuse_prefill = trace_prefill and gen.can_reuse_serving_prefill(
            tokens, page_table=tables, kv_cache=kv_cache, prompt_lens=prompt_lens
        )
        eager_key = self._eager_prefill_signature(tokens, tables, kv_cache, prompt_lens, sampling_params, counts)
        reuse_eager = (
            eager_key is not None
            and eager_key == self._eager_prefill_key
            and self._has_eager_decode_trace()
            and gen.cache is kv_cache
            and gen.can_reuse_serving_decode(sampling_params, allow_eager_prefill=True)
        )
        self._pending_eager_prefill_key = None
        if not reuse_eager:
            self._eager_prefill_key = None
            self._eager_decode_trace_id = None
        reuse_request = reuse_prefill or reuse_eager
        if on_host:
            gen._release_trace()
        else:
            positions = torch.arange(tokens.shape[-1], device=tokens.device)
            if counts is not None and counts.any():
                original_prompt_lengths = prompt_lengths - counts[:, None]
                prompt_tokens = tokens.masked_fill(positions >= original_prompt_lengths, -1)
                output_tokens = tokens.masked_fill(
                    (positions < original_prompt_lengths) | (positions >= prompt_lengths), -1
                )
                # Re-prefill rebuilds KV state but consumes the next sampling
                # draw after the retained outputs; prefill does not increment.
                gen.configure_sampling(
                    sampling_params, prompt_tokens=prompt_tokens, seed_offsets=counts, _reuse_trace=reuse_request
                )
                gen.sampler.reset_output_state(output_tokens)
            else:
                prompt_tokens = tokens.masked_fill(positions >= prompt_lengths, -1)
                gen.configure_sampling(sampling_params, prompt_tokens=prompt_tokens, _reuse_trace=reuse_request)
        if trace_prefill:
            output = gen.serving_prefill_tokens(tokens, page_table=tables, kv_cache=kv_cache, prompt_lens=prompt_lens)
            self._sampling_signature = repr(sampling_params) if gen.prefill_prepared is not None else None
            self._decode_batch = None
            host = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape(-1)
            # The blocking token read completes eager work. Request-local device
            # tensors die on return, before the retained decode trace replays.
            self._pending_eager_prefill_key = eager_key
            return host[: len(prompt_lens)].reshape(-1, 1)
        logits = gen.prefill_forward(
            tokens,
            page_table=tables,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            # vLLM attention tables use compact request rows. empty_slots
            # addresses persistent recurrent state, which this model lacks.
        )
        self._sampling_signature = None
        self._decode_batch = None
        if on_host:
            return self._read_host_logits(logits, len(prompt_lens))
        output = gen.sample_prefill(logits)
        host = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape(-1)
        self._pending_eager_prefill_key = eager_key
        return host[: len(prompt_lens)].reshape(-1, 1)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        sampling_params=None,
        page_tables_per_layer=None,
        enable_trace=True,
        read_from_device=True,
        reset_batch=True,
        prompt_tokens=None,
        output_tokens=None,
        output_token_counts=None,
        slot_remap=None,
    ):
        on_host = self._select_sampling_mode(sampling_params)
        gen = self.generator
        if on_host:
            self._sampling_signature = None
            tokens, start_pos, tables = self._decode_inputs(
                tokens, start_pos, page_table, page_tables_per_layer, reset_batch=True
            )
            logits = gen.decode_forward(
                tokens,
                start_pos,
                page_table=tables,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                device_feedback=False,
                return_logits=True,
            )
            return self._read_host_logits(logits, len(start_pos))
        signature = repr(sampling_params)
        if reset_batch or signature != self._sampling_signature:
            if output_token_counts is None and output_tokens is not None:
                output_token_counts = (output_tokens >= 0).sum(dim=-1)
            seed_offsets = None
            if output_token_counts is not None:
                counts = self._output_counts(output_token_counts, tokens.shape[0])
                # Prefill uses the base seed; _forward increments before each
                # decode sample. The last generated token is this step's input.
                seed_offsets = (counts - 1).clamp_min(0)
            if signature != self._sampling_signature:
                eager_decode = self._has_eager_decode_trace()
                reuse_trace = (
                    getattr(gen, "prefill_prepared", None) is not None or eager_decode
                ) and gen.can_reuse_serving_decode(sampling_params, allow_eager_prefill=eager_decode)
                gen.configure_sampling(
                    sampling_params, prompt_tokens=prompt_tokens, seed_offsets=seed_offsets, _reuse_trace=reuse_trace
                )
                if output_tokens is not None:
                    gen.sampler.reset_output_state(output_tokens)
            else:
                gen.restore_sampling_state(
                    sampling_params,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    seed_offsets=seed_offsets,
                )
            self._sampling_signature = signature
            reset_batch = True
        tokens, start_pos, tables = self._decode_inputs(
            tokens, start_pos, page_table, page_tables_per_layer, reset_batch=reset_batch
        )
        output = gen.decode_forward(
            tokens,
            start_pos,
            page_table=tables,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            device_feedback=not reset_batch,
        )
        if self._pending_eager_prefill_key is not None:
            if (
                gen.batch == 1
                and gen.active_slots == (0,)
                and id(gen.cache) == self._pending_eager_prefill_key[1]
                and not gen._trace_returns_logits
                and gen.serving_prefill_eligible(sampling_params)
            ):
                self._eager_prefill_key = self._pending_eager_prefill_key
                self._eager_decode_trace_id = gen.trace_id
            self._pending_eager_prefill_key = None
        if not read_from_device:
            return output
        return self.process_decode_output_host(self.read_decode_output(output), is_tokens=True)

    def read_decode_output(self, output, async_read=False):
        if isinstance(output, torch.Tensor):
            return (output, []) if async_read else output
        self.generator.counters["token_readbacks"] += 1
        host = ttnn.get_device_tensors(output)[0].cpu(blocking=not async_read)
        if async_read:
            return host, [ttnn.record_event(self.mesh, 0)]
        return host

    def process_decode_output_host(self, output, is_tokens=False):
        if isinstance(output, torch.Tensor):
            return output
        if not is_tokens:
            raise ValueError("Only device-sampled token outputs are supported")
        return ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape(-1, 1)
