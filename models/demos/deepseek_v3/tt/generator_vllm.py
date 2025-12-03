# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from tqdm import tqdm

import ttnn
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: DeepseekGenerator, tt_cache_path):
    """
    Allocate KV cache tensors for vLLM.

    Args:
        kv_cache_shape: Shape tuple (num_blocks, num_kv_heads, block_size, head_size)
        dtype: Data type for the cache
        num_layers: Number of layers to allocate cache for
        model: The DeepseekGenerator model instance
        tt_cache_path: Path for cache files

    Returns:
        List of [k_cache, v_cache] pairs, one per layer
    """
    kv_cache = []
    cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)

    for layer_num in tqdm(range(num_layers), desc="Allocating TT kv caches for each layer"):
        kv_tt_i = [
            ttnn.as_tensor(
                cache_kv,
                device=model.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                # Separate cache files for K and V to avoid collision.
                cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}_layer_{layer_num}",
            )
            for kv in ["k", "v"]
        ]
        kv_cache.append(kv_tt_i)

    return kv_cache


class DeepseekV3ForCausalLM(DeepseekGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations: str = None
    ):
        model_path = os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference")
        cache_dir = os.getenv("DEEPSEEK_V3_CACHE", "generated/deepseek_v3")
        tokenizer = load_tokenizer(model_path)

        model = cls(
            hf_config=hf_config,
            mesh_device=mesh_device,
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
            override_num_layers=5,
        )
        model._prepare_run_configs("prefill")
        model._prepare_run_configs("decode")
        return model

    @property
    def cache_path(self):
        return self.cache_dir

    def prefill_forward(self, *args, **kwargs):
        """Prefill forward pass, following vLLM interface."""
        return super().prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        """Decode forward pass, following vLLM interface."""
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        """Allocate KV cache for vLLM"""
        return allocate_vllm_kv_cache(*args, **kwargs, model=self, tt_cache_path=self.cache_path)
