# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Translate vLLM scheduler updates into the Gemma4 TP4 generator's inputs."""

import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, build_generator


class Gemma4ForCausalLM:
    _router = ttnn.FabricRouterConfig()
    _router.max_packet_payload_size_bytes = 8192
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "supports_chunked_prefill": False,
        "max_device_top_k": 32,
        "fabric_config": {
            "config": ttnn.FabricConfig.FABRIC_1D_RING,
            "router_config": _router,
        },
    }
    decode_input_update_contract = 1

    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None):
        # vLLM inspects this keyword before the TT loader supplies a live mesh.
        if vllm_config is not None:
            raise RuntimeError("Initialize the TT model through initialize_vllm_model")
        self.generator = generator
        self.mesh = generator.mesh
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.entry = None
        self.cache = None

    def embed_input_ids(self, input_ids):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def forward(self, input_ids, positions):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def compute_logits(self, hidden_states):
        raise RuntimeError("The TT generator owns the terminal norm and LM head")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 32:
            raise ValueError("Gemma4 QB2 requires TP4, DP1, and batch 1..32")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        return cls(build_generator(mesh_device, max_seq_len=max_seq_len), max_batch_size, max_seq_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm_tt_plugin.whole_prompt_cache import WholePromptSlidingWindowSpec, validate_whole_prompt_cache

        validate_whole_prompt_cache(vllm_config)
        config = vllm_config.model_config.hf_text_config
        block = vllm_config.cache_config.block_size
        if block != 128:
            raise ValueError("Gemma4 QB2 uses 128-token cache pages")
        # Equal accounting page sizes preserve the logical block size across
        # hybrid groups. Different native geometries still use separate pools.
        # The runner divides these global head counts by the TP degree.
        page_bytes = 2 * block * config.num_key_value_heads * config.head_dim * 2
        result = {}
        for i, kind in enumerate(config.layer_types):
            sliding = kind == "sliding_attention"
            common = dict(
                block_size=block,
                num_kv_heads=config.num_key_value_heads if sliding else config.num_global_key_value_heads,
                head_size=config.head_dim if sliding else config.global_head_dim,
                dtype=torch.bfloat16,
                page_size_padded=page_bytes,
            )
            result[f"model.layers.{i}.self_attn"] = (
                WholePromptSlidingWindowSpec(**common, sliding_window=config.sliding_window)
                if sliding
                else FullAttentionSpec(**common)
            )
        return result

    @classmethod
    def get_max_tokens_all_users(cls, *, max_num_seqs, **kwargs):
        # The shared pool uses one full-attention group and five sliding groups.
        # Each window can straddle nine pages. The plugin adds one output page
        # per request separately; this budget covers full history plus live tails.
        return 262144 + 5 * 9 * 128 * max_num_seqs

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        pools, kv, scratch = {}, [], {}
        for i, (layer, (shape, dtype, pool_id)) in enumerate(zip(self.generator.model.layers, per_layer_specs)):
            if tuple(shape[1:]) != (layer.kv_heads, 128, layer.head_dim) or dtype != torch.bfloat16:
                raise ValueError(f"Unexpected allocator geometry or accounting dtype: {shape}, {dtype}")
            key = (pool_id, tuple(shape[1:]))
            if key not in pools:
                pools[key] = layer.allocate_cache(physical_pages=shape[0] + self.max_batch_size)
            kv.append(pools[key])
            scratch[i] = list(range(shape[0], shape[0] + self.max_batch_size))
        if len(kv) != len(self.generator.model.layers):
            raise ValueError("The allocator must supply one spec per layer")
        tables = {
            i: torch.zeros(self.max_batch_size, self.max_seq_len // 128, dtype=torch.int32) for i in range(len(kv))
        }
        self.cache = CacheState(
            kv, tables, self.max_batch_size, self.max_seq_len, scratch_pages=scratch, vllm_owned=True
        )
        return self.cache

    def _tables(self, cache, page_tables_per_layer, *, slots=None):
        if cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        if page_tables_per_layer is None or len(page_tables_per_layer) != len(cache.kv):
            raise ValueError("Hybrid attention requires a page table for every layer")
        result, translated = {}, {}
        for i, source in enumerate(page_tables_per_layer):
            # Share immutable host rows only within this update: vLLM mutates
            # its source tensors in place between scheduler steps.
            key = id(source)
            if key not in translated:
                dest = torch.zeros_like(cache.page_tables[i])
                rows, cols = source.shape
                if rows > cache.batch or cols > dest.shape[1]:
                    raise ValueError("Scheduler page table exceeds the configured cache")
                indices = list(range(rows)) if slots is None else slots
                if len(indices) > rows:
                    raise ValueError("Page table has fewer rows than prefill requests")
                # Prefill rows are compact, then padded to the decode batch.
                # Scatter only the live prefix into its assigned state slots.
                dest[indices, :cols] = source[: len(indices)].to(torch.int32).clamp_min(0)
                translated[key] = dest
            result[i] = translated[key]
        return result

    def _sampling(self, params, batch, positions, *, reset_seed=True):
        def rows(name):
            value = getattr(params, name)
            value = value.tolist() if isinstance(value, torch.Tensor) else value
            return list(value)[:batch] if isinstance(value, (list, tuple)) else [value] * batch

        for name, default in (
            ("presence_penalty", 0),
            ("frequency_penalty", 0),
            ("repetition_penalty", 1),
            ("enable_log_probs", False),
        ):
            if any(x != default for x in rows(name)):
                raise ValueError(f"The shared host sampler must handle {name}")
        seeds = [0] * batch
        if reset_seed:
            modulus = 2**31 - self.generator.max_seq_len
            seeds = [
                int(torch.randint(0, modulus, ()).item()) if seed is None else int(seed) % modulus
                for seed in rows("seed")
            ]
            seeds = [seed + max(0, int(pos)) for seed, pos in zip(seeds, positions)]
        return dict(
            top_k=rows("top_k"), top_p=rows("top_p"), temperature=rows("temperature"), seed=seeds, reset_seed=reset_seed
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        *,
        sampling_params=None,
        page_tables_per_layer=None,
        start_pos=None,
        empty_slots=None,
        **kwargs,
    ):
        if start_pos is not None and torch.as_tensor(start_pos).ne(0).any():
            raise ValueError("Gemma4 QB2 prefills whole prompts; the generator chunks them internally")
        lengths = [int(x) for x in prompt_lens]
        slots = list(range(len(lengths))) if empty_slots is None else list(empty_slots)
        tables = self._tables(kv_cache, page_tables_per_layer, slots=slots)
        self.entry = None
        outputs = self.generator.prefill_forward(
            tokens,
            page_table=tables,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            slots=slots,
            return_device=sampling_params is not None,
        )
        if sampling_params is None:
            return outputs
        params = self._sampling(sampling_params, len(lengths), [n - 1 for n in lengths])
        return self.generator.sample_prefill_outputs(outputs, kv_cache, params)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires explicit reload flags")
        if not enable_trace:
            raise ValueError("Gemma4 QB2 requires traced decode")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions")
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        sample = sampling_params is not None
        if reload_inputs:
            tables = self._tables(kv_cache, page_tables_per_layer)
            self.entry = self.generator.prepare_decode(tokens, start_pos, page_table=tables, kv_cache=kv_cache)
        elif self.entry is None or self.generator.traces.get(id(kv_cache)) is not self.entry:
            raise ValueError("A retired decode trace requires current scheduler inputs")
        elif reload_page_table:
            tables = self._tables(kv_cache, page_tables_per_layer)
            self.generator._tables(kv_cache, tables)
        if sample and (reload_sampling_params or reset_sampling_state):
            params = self._sampling(sampling_params, self.max_batch_size, start_pos, reset_seed=reset_sampling_state)
            self.generator.configure_sampling(batch=self.max_batch_size, **params)
        self.generator.replay(self.entry, sample=sample)
        if not sample:
            return self.generator._host_logits(self.entry["logits"])[0, 0, : self.max_batch_size].unsqueeze(1)
        output = self.entry["tokens"]
        return self.process_decode_output_host(output, is_tokens=True) if read_from_device else output

    def read_decode_output(self, output, async_read=False):
        if isinstance(output, torch.Tensor):
            return (output, []) if async_read else output
        # Tokens are replicated. Read one rank instead of gathering the vocabulary.
        host = ttnn.get_device_tensors(output)[0].cpu(blocking=not async_read)
        self.generator.counters["token_readbacks"] += 1
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=True):
        if not is_tokens:
            return output.reshape(self.max_batch_size, 1, -1)
        if output.storage_type() != ttnn.StorageType.HOST:
            output = self.read_decode_output(output)
        return ttnn.to_torch(output).reshape(-1)[: self.max_batch_size].long().reshape(-1, 1)

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        # Each unseen logical prefill shape retires traces before compilation.
        pass

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        self.generator._tables(kv_cache, kv_cache.page_tables)
        self.generator.configure_sampling(batch=self.max_batch_size, temperature=0)
        self.entry = self.generator._entry(kv_cache)

    def release_persistent_capture(self):
        self.generator.close()
        self.entry = None
