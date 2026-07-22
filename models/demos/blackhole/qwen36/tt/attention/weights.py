# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class AttentionWeights:
    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor  # +1 pre-offset (zero-centered RMSNorm)
    k_norm: ttnn.Tensor


def load_attention_weights(mesh_device, state_dict, tensor_cache_path=None) -> AttentionWeights:
    def load_2d(name):
        t = state_dict[f"{name}.weight"].T.contiguous()
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight") if tensor_cache_path else None,
        )

    def load_norm(name):
        t = state_dict[f"{name}.weight"] + 1.0
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight_offset") if tensor_cache_path else None,
        )

    return AttentionWeights(
        q_proj=load_2d("q_proj"),
        k_proj=load_2d("k_proj"),
        v_proj=load_2d("v_proj"),
        o_proj=load_2d("o_proj"),
        q_norm=load_norm("q_norm"),
        k_norm=load_norm("k_norm"),
    )
