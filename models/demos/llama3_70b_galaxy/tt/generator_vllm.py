# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations, TtModelArgs
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.tt_transformers.tt.generator import create_submeshes


def initialize_vllm_text_transformer(
    hf_config,
    tt_data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
    optimizations=LlamaOptimizations.performance,
):
    # tt_data_parallel is the total number of DP kv caches, so need to divide by the DP factor of attention.
    dp_attention_factor = mesh_device.shape[1]
    assert (
        tt_data_parallel % dp_attention_factor == 0
    ), f"Total DP ({tt_data_parallel}) must be divisible by dp_attention_factor ({dp_attention_factor})"
    tt_data_parallel = tt_data_parallel // dp_attention_factor

    submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
    # Load model args, weights
    model_args = []
    for submesh in submesh_devices:
        model_args_i = TtModelArgs(
            submesh,
            instruct=(
                "Instruct" in hf_config._name_or_path or "DeepSeek-R1-Distill-Llama-70B" in hf_config._name_or_path
            ),
            max_batch_size=max_batch_size // tt_data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
        )

        if n_layers is not None:
            model_args_i.n_layers = n_layers

        model_args.append(model_args_i)

    state_dict = model_args[0].load_state_dict()

    tt_model = []
    for i, submesh in enumerate(submesh_devices):
        tt_model_i = TtTransformer(
            args=model_args[i],
            mesh_device=submesh,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args[i].weight_cache_path(dtype),
            create_kv_cache=False,
            mode="prefill",
            enable_prefetcher_performance_mode=True,
        )
        tt_model.append(tt_model_i)

    return tt_model, model_args


def initialize_vllm_text_transformer_qwen(
    hf_config,
    tt_data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
    optimizations=LlamaOptimizations.performance,
):
    # tt_data_parallel is the total number of DP kv caches, so need to divide by the DP factor of attention.
    dp_attention_factor = mesh_device.shape[1]
    assert (
        tt_data_parallel % dp_attention_factor == 0
    ), f"Total DP ({tt_data_parallel}) must be divisible by dp_attention_factor ({dp_attention_factor})"
    tt_data_parallel = tt_data_parallel // dp_attention_factor

    submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
    # Load model args, weights
    model_args = []
    for submesh in submesh_devices:
        model_args_i = TtQwenModelArgs(
            submesh,
            instruct=(
                "Instruct" in hf_config._name_or_path or "DeepSeek-R1-Distill-Llama-70B" in hf_config._name_or_path
            ),
            max_batch_size=max_batch_size // tt_data_parallel,
            # optimizations=optimizations,
            max_seq_len=max_seq_len,
        )

        if n_layers is not None:
            model_args_i.n_layers = n_layers

        model_args.append(model_args_i)

    state_dict = model_args[0].load_state_dict()

    tt_model = []
    for i, submesh in enumerate(submesh_devices):
        tt_model_i = TtTransformer(
            args=model_args[i],
            mesh_device=submesh,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args[i].weight_cache_path(dtype),
            create_kv_cache=False,
            mode="prefill",
            enable_prefetcher_performance_mode=True,
        )
        tt_model.append(tt_model_i)

    return tt_model, model_args


def input_processor_for_llama_text(ctx, inputs):
    return inputs


def input_processor_for_qwen_text(ctx, inputs):
    return inputs


# @INPUT_REGISTRY.register_input_processor(input_processor_for_llama_text)
class LlamaForCausalLM(Generator):
    # Class-level capabilities
    model_capabilities = {
        "supports_async_decode": True,
        "supports_prefix_caching": True,
        "supports_sample_on_device": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=131072,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        # max_seq_len = 128
        # n_layers = 1
        tt_model, model_args = initialize_vllm_text_transformer(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=LlamaOptimizations.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        # Model owns the cache: build one (shape, dtype, tensor_idx) per layer
        # (unique tensor_idx => one buffer per layer) and install into the model.
        per_layer_specs = [(kv_cache_shape, dtype, i) for i in range(num_layers)]
        return self.model.allocate_kv_cache(per_layer_specs)


# @INPUT_REGISTRY.register_input_processor(input_processor_for_qwen_text)
class QwenForCausalLM(Generator):
    # Class-level capabilities
    model_capabilities = {
        "supports_sample_on_device": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=131072,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        # max_seq_len = 128
        # n_layers = 1
        tt_model, model_args = initialize_vllm_text_transformer_qwen(
            hf_config,
            tt_data_parallel,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            optimizations=LlamaOptimizations.performance,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        per_layer_specs = [(kv_cache_shape, dtype, i) for i in range(num_layers)]
        return self.model.allocate_kv_cache(per_layer_specs)
