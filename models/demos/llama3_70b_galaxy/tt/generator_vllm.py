# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from tqdm import tqdm
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations, TtModelArgs
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.tt_transformers.tt.generator import create_submeshes


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: TtTransformer, tt_cache_path):
    submesh_devices = [model.mesh_device]
    kv_cache = []

    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for _ in tqdm(range(num_layers), desc=f"Allocating TT kv caches for each layer (submesh {mesh_idx+1})"):
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    # TODO: this could be ShardTensorToMesh, removing the need for vLLM to know about TP for num_kv_heads.
                    # Could affect other calculations which use TTCacheEngine.num_kv_heads, though.
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    # Separate cache files for K and V to avoid collision.
                    cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
                )
                for kv in ["k", "v"]
            ]

            kv_tt.append(kv_tt_i)
        kv_cache.append(kv_tt)
    return kv_cache


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
            use_paged_kv_cache=True,
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
            use_paged_kv_cache=True,
            mode="prefill",
            enable_prefetcher_performance_mode=True,
        )
        tt_model.append(tt_model_i)

    return tt_model, model_args


def initialize_vllm_text_transformer_olmo(
    hf_config,
    tt_data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len=33280,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
    optimizations=LlamaOptimizations.performance,
):
    # OLMo uses TP=32 (full Galaxy mesh), DP=1 - no Llama-style attention-DP splitting.
    # max_batch_size is always 32 regardless of vLLM concurrency: the TT decode trace is
    # fixed at batch=32 (on-device sampling requires target_len % 32 == 0), while vLLM
    # limits concurrent users to 1 via max_num_seqs=1 in the server config.
    TT_OLMO_BATCH_SIZE = 32
    submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
    model_args = []
    for submesh in submesh_devices:
        model_args_i = TtOlmoModelArgs(
            submesh,
            instruct=True,  # OLMo-3.1-32B-Think is always instruct/think mode
            max_batch_size=TT_OLMO_BATCH_SIZE,
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
            use_paged_kv_cache=True,
            mode="prefill",
            enable_prefetcher_performance_mode=False,  # OLMo uses custom prefetcher=False
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
        "supports_prefix_caching": False,
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

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)


# @INPUT_REGISTRY.register_input_processor(input_processor_for_qwen_text)
class QwenForCausalLM(Generator):
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

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)


class OLMo3ForCausalLM(Generator):
    """vLLM-compatible TT implementation of OLMo-3.1-32B-Think (Olmo3ForCausalLM arch)."""

    model_capabilities = {
        "supports_prefix_caching": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OLMo uses split-sampling (decode trace captures model ops only; sampling runs
        # eagerly outside the trace) but without the sampling module's own internal trace.
        # enable_internal_trace=True would cause the sampling module to call
        # begin_trace_capture inside the decode trace replay, causing a nested trace error.
        # With enable_internal_trace=False, sampling is called eagerly after each decode
        # trace execution, which is correct and matches the OLMo demo path.
        self.model.enable_internal_trace = False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=33280,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        tt_model, model_args = initialize_vllm_text_transformer_olmo(
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
        # Root-cause fix: never run on-device sampling during prefill for OLMo.
        #
        # The shared Generator.prefill_forward_text on-device-sampling branch
        # does: switch_mode("decode") -> sampling_module.reset_sampling_params
        # -> reset_prompt_tokens -> reset_output_state -> seed_manager.reset_seed.
        # reset_sampling_params calls reset_trace() whenever force_argmax_sampling
        # flips (models/common/sampling/generator.py L202-203), which wipes every
        # decode sampling trace captured during warmup. The previous workaround
        # (cycling modes and clearing decode/sampling traces after every request)
        # just triggered recapture on the next request, defeating warmup entirely.
        #
        # With sampling_params=None, prefill_forward_text sets return_logits=True
        # (generator.py L172), skips the on-device-sampling block, and leaves
        # decode + sampling module state untouched. Decode traces captured during
        # warmup stay valid across prefill->decode for every subsequent request.
        # Host-side argmax on the single last-token logits vector is cheap and
        # is what text_olmo_demo.py already does (verified up to 32K ISL).
        # Sampling params (temperature, top_k, top_p, penalties) are still
        # honored on-device from the second token onwards via decode sampling.
        kwargs.pop("sampling_params", None)
        logits = super().prefill_forward_text(*args, **kwargs)
        return logits[:, -1, :].argmax(dim=-1)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)
