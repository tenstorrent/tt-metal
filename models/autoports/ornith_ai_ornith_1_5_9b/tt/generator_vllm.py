# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving translation for the canonical Ornith generator.

The plugin owns attention page allocation. Recurrent state remains slot-owned
inside the opaque allocation returned to the plugin. No standalone cache is
constructed: the generator is created only after the serving allocation.
"""

import json
import os
import time
from dataclasses import fields

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt import cache_contract
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import DEVICE_SAMPLING_MAX_TOP_K, OrnithGenerator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_9b.tt.precision import load_policy
from models.common.sampling.generator import SamplingParams


class OrnithForCausalLM:
    layer_indices = None
    uses_mrope = True
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_original_prompt_lens": True,
        # KV pages cannot reconstruct the 24 recurrent/conv state layers.
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "max_device_top_k": 64,
        "supports_device_penalties": True,
        "supports_async_decode": True,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D},
    }
    decode_input_update_contract = 1

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if cls.model_capabilities["max_device_top_k"] != DEVICE_SAMPLING_MAX_TOP_K:
            raise RuntimeError("Advertised device top-k capacity must match the canonical generator")
        if tt_data_parallel != 1:
            raise ValueError("Ornith serves one TP4 runner")
        if optimizations not in (None, "performance"):
            raise ValueError("Serving requires the selected datatype-sweep policy")
        obj = cls()
        from vllm.transformers_utils.config import uses_mrope

        obj.uses_mrope = uses_mrope(hf_config)
        obj.max_batch_size, obj.max_seq_len = max_batch_size, max_seq_len
        obj.tt_supported_decode_batch_sizes = [max_batch_size]
        obj.model = OrnithModel(mesh_device, layer_indices=cls.layer_indices)
        obj.generator = None
        obj._event("registration", target=f"{cls.__module__}:{cls.__name__}", policy=load_policy())
        return obj

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        if vllm_config.cache_config.block_size != 32:
            raise ValueError("Ornith requires physical pages of 32 tokens")
        # Nominal torch BF16 is conservative scheduler accounting. Allocation
        # translates it to the selected TT BFP8 storage, as other TT adapters do.
        return {
            f"model.layers.{index}.self_attn": FullAttentionSpec(
                block_size=32, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
            )
            for index in cache_contract.FULL_LAYERS
        }

    @staticmethod
    def get_max_tokens_all_users(model_name, num_devices, tt_data_parallel, max_model_len, max_num_seqs):
        if num_devices != 4 or tt_data_parallel != 1 or not 1 <= max_num_seqs <= 32:
            raise ValueError("Unsupported serving mesh or capacity")
        if not 0 < max_model_len <= 262144:
            raise ValueError("Context exceeds the model contract")
        # Stage 8 measured the complete 262144-token pool. The plugin adds one
        # reservation page per slot; serving allocation validates that extension.
        return 262144

    def _event(self, kind, **fields):
        # Retain routing evidence in the measured serving path as well as the
        # allocation audit; these are mode flags, not an adapter profiler.
        print(
            "ORNITH_SERVING " + json.dumps({"event": kind, "pid": os.getpid(), "time": time.time(), **fields}),
            flush=True,
        )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        if self.generator is not None:
            raise RuntimeError("Serving cache may only be allocated once")
        dtype = load_policy()["kv_cache_dtype"]
        translated = []
        for shape, nominal_dtype, identity in per_layer_specs:
            if nominal_dtype != torch.bfloat16:
                raise ValueError("Unexpected scheduler cache dtype")
            translated.append((shape, dtype, identity))
        plan = cache_contract.plan_cache_specs(translated, batch=self.max_batch_size, context=self.max_seq_len)
        cache = self.model.allocate_cache(**plan.model_kwargs())
        table = torch.arange((self.max_seq_len + 31) // 32, dtype=torch.int32).repeat(self.max_batch_size, 1)
        self.generator = OrnithGenerator(
            self.model,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            kv_cache=cache,
            page_table=table,
        )
        self.generator.trace_observer = self._event
        self.benchmark_timing = None
        if os.environ.get("ORNITH_BENCHMARK_TIMING"):
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.benchmark_timing import PhaseRecorder

            self.benchmark_timing = PhaseRecorder(self, os.environ["ORNITH_BENCHMARK_TIMING"])
        self._event("cache", blocks=plan.num_blocks, slots=self.max_batch_size, context=self.max_seq_len)
        return cache

    def _check_cache(self, cache):
        if self.generator is None or cache is not self.generator.cache:
            raise ValueError("Execution must use the exact vLLM allocation")

    @staticmethod
    def _sampling_params(params):
        if params is None or isinstance(params, SamplingParams):
            return params
        return SamplingParams(
            **{
                field.name: getattr(params, field.name)
                for field in fields(SamplingParams)
                if hasattr(params, field.name)
            }
        )

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device):
        self._check_cache(kv_cache)
        if enable_trace:
            raise ValueError("Ornith requires trace_mode=decode_only")
        started = time.perf_counter()
        before = self.generator.device.num_program_cache_entries()
        self.generator.prepare_decode(
            kv_cache=kv_cache,
            page_table=self.generator.page_table,
            slots=tuple(range(self.max_batch_size)),
        )
        self._event(
            "preparation",
            preparation_ms=1000 * (time.perf_counter() - started),
            programs_before=before,
            programs_after=self.generator.device.num_program_cache_entries(),
        )

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device):
        self._check_cache(kv_cache)
        state = self.generator.prepare_decode(
            kv_cache=kv_cache,
            page_table=self.generator.page_table,
            slots=tuple(range(self.max_batch_size)),
        )
        if enable_trace:
            self.generator.capture(state)
            self._event(
                "traces",
                model={str(k): str(v) for k, v in state.model_traces.items()},
                sampling={str(k): str(v) for k, v in state.sample_traces.items()},
                counters=self.generator.counters,
            )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        empty_slots=None,
        enable_trace=False,
        sampling_params=None,
        original_prompt_lens=None,
        **kwargs,
    ):
        timing = getattr(self, "benchmark_timing", None)
        phase = timing.begin("prefill") if timing else None
        self._check_cache(kv_cache)
        if original_prompt_lens is None:
            raise ValueError("Serving prefill requires authoritative original_prompt_lens")
        if not isinstance(original_prompt_lens, list) or any(type(x) is not int for x in original_prompt_lens):
            raise TypeError("original_prompt_lens must be a CPU list of Python integers")
        if len(original_prompt_lens) != len(prompt_lens) or any(
            not 0 <= x <= self.max_seq_len for x in original_prompt_lens
        ):
            raise ValueError("Original prompt lengths must match rows and fit the configured context")
        for name in (
            "image_grid_thw",
            "pixel_values",
            "mm_identifiers",
            "mm_placeholder_lengths",
            "mm_placeholder_offsets",
        ):
            value = kwargs.pop(name, None)
            if value is not None and not (isinstance(value, (list, tuple)) and all(x is None for x in value)):
                raise ValueError(f"Text-only Ornith adapter cannot consume {name}")
        ends = [int(x) for x in prompt_lens]
        starts = [0] * len(ends) if start_pos is None else [int(x) for x in start_pos]
        # Original prompt boundaries prove intermediate rows and fix canonical
        # prefill profiles. Replay can extend into generated tokens: once any
        # row does so, retain the generator's unknown-horizon rebuild contract.
        request_lens = original_prompt_lens if all(e <= o for e, o in zip(ends, original_prompt_lens)) else None
        discard_logits = [0 < e < o for e, o in zip(ends, original_prompt_lens)]
        self._event(
            "prefill",
            ends=ends,
            original_prompt_lens=original_prompt_lens,
            request_lens=request_lens,
            discard_logits=discard_logits,
            starts=starts,
            chunks=[e - s for e, s in zip(ends, starts)],
            slots=empty_slots,
            page_table_head=(
                page_table[:, :8].tolist() if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1" else None
            ),
            sampling="device" if sampling_params is not None else "host",
            counters=dict(self.generator.counters),
        )
        result = self.generator.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=ends,
            original_prompt_lens=original_prompt_lens,
            start_pos=starts,
            request_lens=request_lens,
            discard_logits=discard_logits,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            sampling_params=self._sampling_params(sampling_params),
            **kwargs,
        )
        if timing:
            timing.prefill_end(phase, starts, ends, original_prompt_lens, empty_slots, discard_logits)
        # Canonical prefill already returns host [B,1,V] or sampled [B,1].
        # Text positions coincide on all three M-RoPE channels.
        return (result, torch.zeros((len(ends), 1), dtype=torch.int64)) if self.uses_mrope else result

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, **kwargs):
        timing = getattr(self, "benchmark_timing", None)
        phase = timing.begin("decode") if timing else None
        self._check_cache(kv_cache)
        deltas = kwargs.pop("rope_deltas_all_users", None)
        if deltas is not None and any(delta != 0 for delta in deltas):
            raise ValueError("Text-only Ornith requires zero M-RoPE deltas")
        if "sampling_params" in kwargs:
            kwargs["sampling_params"] = self._sampling_params(kwargs["sampling_params"])
        self._event(
            "decode",
            sampling="device" if kwargs.get("sampling_params") is not None else "host",
            reload_inputs=kwargs.get("reload_inputs"),
            reload_page_table=kwargs.get("reload_page_table"),
            read_from_device=kwargs.get("read_from_device", True),
            enable_trace=kwargs.get("enable_trace", True),
        )
        result = self.generator.decode_forward(tokens, start_pos, page_table=page_table, kv_cache=kv_cache, **kwargs)
        if timing:
            timing.decode_submitted(phase, result, self.generator)
        return result

    def read_decode_output(self, output, async_read=False):
        self._event("decode_read", async_read=async_read)
        return self.generator.read_decode_output(output, async_read=async_read)

    def process_decode_output_host(self, output, is_tokens=False):
        result = self.generator.process_decode_output_host(output, is_tokens=is_tokens)
        if is_tokens and torch.any(result >= self.model.vocab_size):
            raise RuntimeError("Out-of-vocabulary device sample")
        timing = getattr(self, "benchmark_timing", None)
        if timing:
            timing.decode_complete(output)
        return result

    def release_request(self, slot):
        return self.generator.release_request(slot)

    def release_persistent_capture(self):
        if self.generator is not None:
            self._event("shutdown", counters=self.generator.counters)
            self.generator.close()


class OrnithServingDiagnostic(OrnithForCausalLM):
    """Two real layer kinds for integration debugging, never quality/perf evidence."""

    layer_indices = (0, 3)
