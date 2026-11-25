# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned standalone TT-plugin translation for the Nemotron-H generator."""

import json
import os
import time
from pathlib import Path

import torch

import ttnn
from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt.generator import NemotronGenerator
from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt.optimized_decoder import layer_kinds
from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt.precision import SELECTED, load_precision


class NemotronForCausalLM:
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefill_no_output": True,
        # agentic-models: vllm-tt-plugin patch P1 keeps a continuing chunked prefill on its own state slot
        # (the Mamba state lives in the model, indexed by slot); gemma/K2 do not declare it.
        "preserve_prefill_state_slots": True,
        # perf/multiuser: NEMOTRON_PREFIX_CACHE=1 enables vLLM attention prefix caching + model-side Mamba snapshots.
        "supports_prefix_caching": os.environ.get("NEMOTRON_PREFIX_CACHE", "0") == "1",
        "prefix_cache_recurrent_state_in_model": True,  # Mamba snapshots live in the model (perf/patches plugin patch)
        "supports_sample_on_device": True,
        "supports_async_decode": True,  # Tracked resident/page-only/deferred-read adapter probe passed.
        "supports_device_penalties": False,
        # perf/multiuser: None = the device sampler accepts unset/oversized top_k (mapped to the
        # top-64 candidates, see generator.UNBOUNDED_TOP_K); 64 restores host routing for them.
        "max_device_top_k": None if os.environ.get("NEMOTRON_UNBOUNDED_TOP_K", "64") != "0" else 64,
    }
    decode_input_update_contract = 1

    # Upstream registry inspection protocol; execution uses the TT methods below.
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Use the TT generator interface")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("Use the TT generator interface")

    def compute_logits(self, hidden_states, **kwargs):
        raise NotImplementedError("Use the TT generator interface")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Nemotron requires the validated TP4 1x4 mesh")
        if not 1 <= max_batch_size <= 32 or not 1 <= max_seq_len <= 262144:
            raise ValueError("Serving geometry exceeds the validated model contract")
        obj = cls()
        obj.mesh = mesh_device
        obj.hf_config = hf_config
        if getattr(hf_config, "layers_block_type", None) is not None:
            # The plugin counts KV-cache layers with vLLM's get_num_layers_by_block_type(..., "attention"), which
            # matches layers_block_type entries literally; transformers 5.16 reports them as full_attention, giving 0.
            # Restore the names config.json holds (mamba / attention / moe) on this process's config.
            hf_config.layers_block_type = layer_kinds(hf_config)
        obj.max_batch_size = max_batch_size
        obj.max_seq_len = max_seq_len
        # perf/multiuser: one decode trace per bucket (must match NemotronGenerator.decode_buckets).
        bucket_env = os.environ.get("NEMOTRON_DECODE_BUCKETS", "1,2,4,8,16,32")
        buckets = {int(b) for b in bucket_env.split(",") if b.strip() and int(b) <= max_batch_size}
        obj.tt_supported_decode_batch_sizes = sorted(buckets | {max_batch_size})
        obj.precision = load_precision(SELECTED)
        obj.generator = None
        # Packed MoE prefill failed exact solo equivalence for real 25-token
        # prompts at mixed 8+2 admission. Keep it opt-in for diagnostics only.
        obj.batched_prefill = os.environ.get("NEMOTRON_VLLM_SERIAL_PREFILL", "1") != "1"
        obj.host_compat = os.environ.get("NEMOTRON_VLLM_HOST_COMPAT", "0") == "1"
        obj.event_path = os.environ.get("NEMOTRON_VLLM_EVENTS")
        # Inner-loop diagnostic only; preserve original HF config and all cache specs.
        layers = os.environ.get("NEMOTRON_VLLM_LAYERS")
        obj.layer_indices = None if layers is None else tuple(json.loads(layers))
        obj.event(
            "initialize",
            adapter=f"{cls.__module__}:{cls.__name__}",
            module_file=__file__,
            max_model_len=max_seq_len,
            max_num_seqs=max_batch_size,
            precision=obj.precision,
            layer_indices=obj.layer_indices,
            host_compat=obj.host_compat,
            batched_prefill=obj.batched_prefill,
        )
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        return NemotronGenerator.get_max_tokens_all_users(**kwargs)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        config = vllm_config.model_config.hf_config
        if vllm_config.cache_config.block_size != 32:
            raise ValueError("Nemotron serving requires --block-size 32")
        # vLLM's torch dtype is conservative allocation accounting, not an IEEE
        # float8 alias for BFP8_B. The allocator below applies the selected TT dtype.
        return {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=32,
                num_kv_heads=config.num_key_value_heads,
                head_size=config.head_dim,
                dtype=torch.bfloat16,
            )
            for i in range(layer_kinds(config).count("attention"))
        }

    def event(self, kind, **fields):
        if self.event_path:
            with open(self.event_path, "a") as stream:
                stream.write(
                    json.dumps(dict(event=kind, time=time.time(), pid=os.getpid(), **fields), default=str) + "\n"
                )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        if self.generator is not None:
            raise RuntimeError("The serving cache is allocated exactly once")
        indices = [i for i, kind in enumerate(layer_kinds(self.hf_config)) if kind == "attention"]
        if len(per_layer_specs) != len(indices):
            raise ValueError("vLLM must supply one cache spec per attention layer")
        specs = []
        for index, (shape, dtype, tensor_idx) in zip(indices, per_layer_specs):
            if dtype != torch.bfloat16 or tuple(shape[1:]) != (1, 32, 128):
                raise ValueError(f"Unexpected plugin cache geometry/dtype: {shape}, {dtype}")
            if self.layer_indices is None or index in self.layer_indices:
                specs.append((shape, getattr(ttnn, self.precision["kv_cache_dtype"]), tensor_idx))
        self.generator = NemotronGenerator(
            self.mesh,
            checkpoint=(self.hf_config._name_or_path if Path(self.hf_config._name_or_path).is_dir() else None),
            layer_indices=self.layer_indices,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            per_layer_specs=specs,
            precision_config=self.precision,
            batched_prefill=self.batched_prefill,
        )
        self.generator.event_sink = self.event
        if os.environ.get("NEMOTRON_PREFIX_CACHE", "0") == "1" and os.environ.get("NEMOTRON_PREFIX_CLIP", "1") == "1":
            try:  # perf/patches plugin patch: the scheduler clips prefix hits to restorable Mamba snapshots
                import vllm_tt_plugin

                vllm_tt_plugin.prefix_hit_clip = self.generator.snap_hit_len
            except ImportError:
                pass
        cache = self.generator.allocate_kv_cache_per_layer(specs)
        self.event(
            "allocate_cache",
            plugin_specs=per_layer_specs,
            actual_specs=specs,
            capacity=self.generator.cache_capacity(),
            precision=self.generator.model.precision_summary(),
        )
        return cache

    def _bound(self, kv_cache):
        if self.generator is None or kv_cache is not self.generator.cache:
            raise ValueError("Pass the exact cache object returned to vLLM")
        return self.generator

    def _prepare(self, kv_cache):
        generator = self._bound(kv_cache)
        if not generator._prepared:
            start = time.monotonic()
            generator.prepare_decode(cache=kv_cache)
            self.event(
                "prepared",
                elapsed_s=time.monotonic() - start,
                decode_traces={str(k): v for k, v in generator.decode_traces.items()},
                sampling_traces=generator.sampling_traces,
                counters=dict(generator.counters),
                warmup=generator.warmup_report,
                cache=generator.cache_capacity(),
            )

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device):
        self._prepare(kv_cache)

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device):
        self._prepare(kv_cache)

    def _mode(self, phase, sampling_params):
        device = sampling_params is not None
        mode = (phase, device)
        if getattr(self, "_last_mode", None) != mode:  # log transitions only (was every step)
            self._last_mode = mode
            print(f"NEMOTRON_SAMPLING phase={phase} sampling_mode={'device' if device else 'host_compat'}", flush=True)
        if not device and not self.host_compat:
            raise ValueError("Host compatibility requires NEMOTRON_VLLM_HOST_COMPAT=1")
        return device

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        empty_slots=None,
        enable_trace=True,
        sampling_params=None,
        start_pos=None,
        page_tables_per_layer=None,
        *,
        return_output=True,
    ):
        if not isinstance(return_output, bool):
            raise ValueError("return_output must be a bool")
        if not return_output and sampling_params is not None:
            raise ValueError("No-output prefill cannot sample")
        generator = self._bound(kv_cache)
        if return_output:
            self._mode("prefill", sampling_params)
        starts = [0] * len(prompt_lens) if start_pos is None else [int(x) for x in start_pos]
        self.event(
            "prefill",
            starts=starts,
            ends=[int(x) for x in prompt_lens],
            lengths=[int(n) - a for a, n in zip(starts, prompt_lens)],
            slots=empty_slots,
            active_decode_slots=torch.nonzero(generator.active_host).flatten().tolist(),
            return_output=return_output,
        )
        result = generator.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=list(range(len(prompt_lens))) if empty_slots is None else empty_slots,
            enable_trace=enable_trace,
            sampling_params=sampling_params,
            start_pos=start_pos,
            page_tables_per_layer=page_tables_per_layer,
            return_output=return_output,
        )
        if sampling_params is not None:
            self._check_tokens(result)
        return result

    def _check_tokens(self, result):
        if torch.any(result < 0) or torch.any(result >= self.hf_config.vocab_size):
            raise RuntimeError("Device sampler emitted an out-of-vocabulary token")
        self.event("token_check", count=result.numel(), out_of_vocabulary=0)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        page_tables_per_layer=None,
        *,
        reload_inputs,
        reload_page_table,
        reload_sampling_params,
        reset_sampling_state,
    ):
        generator = self._bound(kv_cache)
        device = self._mode("decode", sampling_params)
        # The plugin pads decode rows to one of tt_supported_decode_batch_sizes;
        # rows are condensed to 0..n-1, so that bucket's trace covers every live row.
        rows = int(torch.as_tensor(tokens).shape[0]) if tokens is not None else self.max_batch_size
        page_table, page_tables_per_layer = self._pad_pages(generator, page_table, page_tables_per_layer)
        result = generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            on_device_sampling=device,
            slot_remap=slot_remap,
            page_tables_per_layer=page_tables_per_layer,
            slots=tuple(range(rows)),
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
        )
        self.event(
            "decode",
            reload_inputs=reload_inputs,
            reload_page_table=reload_page_table,
            reload_sampling_params=reload_sampling_params,
            reset_sampling_state=reset_sampling_state,
            slot_remap=slot_remap,
            counters=dict(generator.counters),
        )
        if isinstance(result, torch.Tensor):
            return result.reshape(-1, 1) if device else result.reshape(rows, 1, -1)
        return result

    @staticmethod
    def _pad_pages(generator, page_table, per_layer):
        """Bucketed decode: the plugin pads block tables only to the bucket; keep slots >= bucket unchanged."""

        def pad(table, current):
            if table is None:
                return None
            table = torch.as_tensor(table, dtype=torch.int32)
            if table.shape[0] >= current.shape[0]:
                return table
            if table.shape[1] != current.shape[1]:
                return table  # let the generator's shape validation report it
            return torch.cat([table, current[table.shape[0] :].to(table.dtype)], dim=0)

        page_table = pad(page_table, generator.page_host)
        if isinstance(per_layer, dict):
            per_layer = {k: pad(v, generator.layer_page_host[k]) for k, v in per_layer.items()}
        elif isinstance(per_layer, (list, tuple)):
            per_layer = [pad(v, cur) for v, cur in zip(per_layer, generator.layer_page_host.values())]
        return page_table, per_layer

    def read_decode_output(self, output, async_read=False):
        return self.generator.read_decode_output(output, async_read=async_read)

    def process_decode_output_host(self, output, is_tokens=False):
        result = self.generator.process_decode_output_host(output, is_tokens=is_tokens)
        if is_tokens:
            self._check_tokens(result)
        return result

    def release_request(self, slot):
        self.generator.release_request(slot, kv_cache=self.generator.cache)

    def note_state_slots_moved(self, moves):
        self.generator.note_state_slots_moved(moves)

    def release_persistent_capture(self):
        if self.generator is not None:
            self.event("shutdown_release", counters=dict(self.generator.counters))
            self.generator.close()
