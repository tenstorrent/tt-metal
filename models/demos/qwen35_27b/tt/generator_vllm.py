# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from tqdm import tqdm

import ttnn
from models.demos.qwen35_27b.tt.generator import Generator
from models.demos.qwen35_27b.tt.model import create_qwen35_model


class Qwen35ForCausalLM(Generator):
    """vLLM adapter for Qwen3.5-27B on Tenstorrent hardware.

    Wraps the TT Qwen3.5 model with the vLLM-expected interface:
    initialize_vllm_model, prefill_forward, decode_forward, allocate_kv_cache.
    """

    model_capabilities = {
        "supports_prefix_caching": False,  # GDN recurrence state not compatible with prefix caching
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        model = create_qwen35_model(
            mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            use_paged_kv_cache=True,
        )
        args = model.args
        return cls(model, args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        layer_types = self.model_args[0].layer_types
        mesh = self.model[0].mesh_device
        cache_path = self.cache_path

        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for layer_idx in tqdm(range(num_layers), desc="Allocating TT kv caches"):
            if layer_types and layer_types[layer_idx] == "linear_attention":
                # GDN layers use internal recurrence state, no KV cache needed
                kv_tt.append(None)
            else:
                kv_tt_i = [
                    ttnn.as_tensor(
                        cache_kv,
                        device=mesh,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b,
                        cache_file_name=cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
                    )
                    for kv in ["k", "v"]
                ]
                kv_tt.append(kv_tt_i)

        # Wrap in list for data-parallel indexing (DP=1)
        return [kv_tt]
