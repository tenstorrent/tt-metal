# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llama3Generator — thin vLLM adapter wrapping an executor.

Zero trace state, zero warmup state, zero execution logic.
Just signature adaptation for TTModelRunner.
"""

import ttnn
from models.common.models.llama3_8b.executor import LlamaExecutor, TracedLlamaExecutor
from models.common.models.llama3_8b.model import Llama3Transformer1D


class Llama3Generator:
    """vLLM-compatible adapter. Wraps any executor (typically traced).

    Usage:
        generator = Llama3Generator.initialize_vllm_model(hf_config, mesh_device, ...)
        kv_cache = generator.allocate_kv_cache(shape, dtype, num_layers)
        logits = generator.prefill_forward(tokens, page_table=..., kv_cache=kv_cache, ...)
        output = generator.decode_forward(tokens, start_pos, page_table=..., kv_cache=kv_cache, ...)
    """

    model_capabilities = {"supports_prefix_caching": True}

    def __init__(self, executor: LlamaExecutor | TracedLlamaExecutor, model_args=None):
        self.executor = executor
        self.model = executor.model
        self.model_args = model_args
        self.mesh_device = executor.mesh_device

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        optimizations="performance",
    ):
        """Build Llama3Transformer1D from HF config and wrap in traced executor.

        This is the entry point called by vLLM's TTModelRunner.
        """
        from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

        hf_model_name = hf_config._name_or_path
        instruct = "Instruct" in hf_model_name

        model_args = ModelArgs(
            mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            optimizations=(
                DecodersPrecision.from_string(optimizations)
                if optimizations is not None
                else DecodersPrecision.performance
            ),
            max_seq_len=max_seq_len,
        )

        if n_layers is not None:
            model_args.n_layers = n_layers

        state_dict = model_args.load_state_dict()
        dtype = ttnn.bfloat8_b

        model = Llama3Transformer1D.from_model_args(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )

        executor = TracedLlamaExecutor(model, mesh_device, model_args=model_args)
        return cls(executor, model_args=model_args)

    def prefill_forward(self, *args, **kwargs):
        return self.executor.prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return self.executor.decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return self.executor.allocate_kv_cache(*args, **kwargs)

    def warmup_model_prefill(self, *args, **kwargs):
        if hasattr(self.executor, "warmup_model_prefill"):
            return self.executor.warmup_model_prefill(*args, **kwargs)

    @property
    def cache_path(self):
        if self.model_args:
            return self.model_args.model_cache_path
        return None
