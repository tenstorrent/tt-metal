# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from tqdm import tqdm
from models.demos.llama3_subdevices.tt.generator import Generator
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.model_config import LlamaOptimizations, TtModelArgs
from models.tt_transformers.tt.generator import create_submeshes
from vllm.inputs import INPUT_REGISTRY


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: TtTransformer, tt_cache_path):
    submesh_devices = [model.mesh_device]
    kv_cache = []

    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for _ in tqdm(range(num_layers), desc=f"Allocating TT kv caches for each layer (submesh {mesh_idx+1})"):
            kv_tt_i = [
                ttnn.as_tensor(
                    lp,
                    device=submesh,
                    # TODO: this could be ShardTensorToMesh, removing the need for vLLM to know about TP for num_kv_heads.
                    # Could affect other calculations which use TTCacheEngine.num_kv_heads, though.
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    cache_file_name=tt_cache_path / f"empty_cache_paged_attention{kv_cache_shape}",
                )
                for lp in (cache_kv, cache_kv)
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


def input_processor_for_llama_text(ctx, inputs):
    return inputs


@INPUT_REGISTRY.register_input_processor(input_processor_for_llama_text)
class LlamaForCausalLM(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=131072, n_layers=None, tt_data_parallel=1
    ):
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
        return super().decode_forward_text(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)
