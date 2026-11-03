# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pinned TT plugin interface; execution stays in K2HorizonGenerator."""

import json
import os
import time
from dataclasses import fields

import torch

import ttnn
from models.common.sampling.generator import SamplingParams

from .generator import K2HorizonGenerator
from .model import K2HorizonModel
from .precision import ROOT, load_precision


class K2HorizonForCausalLM:
    model_capabilities = dict(K2HorizonGenerator.serving_capabilities)
    decode_input_update_contract = 1

    # vLLM registry inspection protocol. TT execution uses the methods below.
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Use the TT generator interface")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("Use the TT generator interface")

    def compute_logits(self, hidden_states, **kwargs):
        raise NotImplementedError("Use the TT generator interface")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=524288, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or tuple(mesh_device.shape) != (1, 4):
            raise ValueError("K2 Horizon requires the accepted 1x4 TP/EP mesh")
        if optimizations not in (None, "performance"):
            raise ValueError("Serving uses the selected datatype-sweep policy")
        ttnn.CONFIG.throw_exception_on_fallback = True
        obj = cls()
        obj.batch_size, obj.context = max_batch_size, max_seq_len
        obj.host_compat = os.environ.get("K2_VLLM_HOST_COMPAT", "0") == "1"
        indices = os.environ.get("K2_VLLM_LAYERS")
        selected = load_precision(ROOT / "doc/datatype_sweep/selected_precision_config.json")
        obj.model = K2HorizonModel(
            mesh_device,
            max_seq_len=max_seq_len,
            precision_config=selected,
            layer_indices=None if indices is None else json.loads(indices),
        )
        obj.generator = None
        if timing_path := os.environ.get("K2_BENCHMARK_TIMING"):
            from ..tests.benchmark_timing import install

            install(obj, timing_path)
        obj.event(
            "initialize",
            adapter=f"{cls.__module__}:{cls.__name__}",
            module_file=__file__,
            precision=selected,
            layers=obj.model.layer_indices,
            max_model_len=max_seq_len,
            max_num_seqs=max_batch_size,
            host_compat=obj.host_compat,
        )
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        return K2HorizonGenerator.aggregate_context_budget

    def event(self, event, **data):
        path = os.environ.get("K2_VLLM_EVENTS")
        if path:
            with open(path, "a") as stream:
                stream.write(
                    json.dumps(dict(time=time.time(), pid=os.getpid(), event=event, **data), default=str) + "\n"
                )

    def allocate_kv_cache(self, shape, dtype, num_layers):
        if self.generator is not None:
            raise ValueError("Serving cache is allocated exactly once")
        if num_layers != 48 or tuple(shape)[1:2] != (2,) or shape[-1] != 128:
            raise ValueError(f"Unexpected vLLM cache specification: {shape}, {dtype}, {num_layers}")
        # vLLM's torch dtype describes logical attention; physical TT cache
        # storage follows the selected BFP8 policy, including its tile metadata.
        self.generator = K2HorizonGenerator(
            self.model,
            max_batch_size=self.batch_size,
            max_seq_len=self.context,
            page_size=shape[2],
            allocate_cache=False,
        )
        if os.environ.get("K2_VLLM_EVENTS"):
            self.generator.event_sink = self.event
        cache = self.generator.allocate_kv_cache(shape, self.model.kv_cache_dtype, len(self.model.layers))
        self.event("allocate_cache", shape=shape, logical_dtype=dtype, physical_dtype=self.model.kv_cache_dtype)
        return cache

    def _prepare(self, kv_cache):
        g = self.generator
        if g is None or kv_cache is not g.kv_cache:
            raise ValueError("The exact vLLM-owned cache must be supplied")
        g.prepare()
        return g

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        self._prepare(kv_cache)

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        g = self._prepare(kv_cache)
        if enable_trace:
            g.capture()
        self.event("warmup", enable_trace=enable_trace, counters=dict(g.counters))

    def _sampling(self, params):
        if params is None:
            if not self.host_compat:
                raise RuntimeError(
                    "Host compatibility is disabled; explicitly set K2_VLLM_HOST_COMPAT=1 for unsupported request controls"
                )
            self.event("host_compatibility")
            return None
        values = {f.name: getattr(params, f.name) for f in fields(SamplingParams)}
        # The plugin uses -1 as a disabled-logprob sentinel; the common
        # generator uses zero with enable_log_probs=False.
        enabled = values["enable_log_probs"]
        counts = values["num_logprobs"]
        if isinstance(counts, list):
            values["num_logprobs"] = [n if enabled[i] else 0 for i, n in enumerate(counts)]
        elif not enabled:
            values["num_logprobs"] = 0
        return SamplingParams(**values)

    def _table(self, page_table):
        width = self.generator.pages_per_slot
        if page_table.shape[1] > width:
            # The plugin may include an output reservation beyond the logical
            # context. It cannot be addressed by a valid model position.
            return page_table[:, :width].contiguous()
        if page_table.shape[1] < width:
            return torch.nn.functional.pad(page_table, (0, width - page_table.shape[1]))
        return page_table

    def prefill_forward(self, tokens, *, sampling_params=None, page_table, kv_cache, **kwargs):
        return self.generator.prefill_forward(
            tokens,
            sampling_params=self._sampling(sampling_params),
            page_table=self._table(page_table),
            kv_cache=kv_cache,
            **kwargs,
        )

    def decode_forward(self, tokens, start_pos, *, sampling_params=None, page_table, kv_cache, **kwargs):
        # No per-token table copying or token/position reconstruction. The
        # generator executes the plugin's four explicit reload commands.
        if kwargs.get("reload_inputs", True) or kwargs.get("reload_page_table", False):
            page_table = self._table(page_table)
        return self.generator.decode_forward(
            tokens,
            start_pos,
            sampling_params=self._sampling(sampling_params),
            page_table=page_table,
            kv_cache=kv_cache,
            **kwargs,
        )

    def read_decode_output(self, output, async_read=False):
        return self.generator.read_decode_output(output, async_read=async_read)

    def process_decode_output_host(self, output, is_tokens=None):
        result = self.generator.process_decode_output_host(output, is_tokens=is_tokens)
        if not output.is_tokens:
            return result[:, None, :]
        return result

    def release_persistent_capture(self):
        if self.generator is not None:
            self.event("shutdown", counters=dict(self.generator.counters))
            self.generator.close()
