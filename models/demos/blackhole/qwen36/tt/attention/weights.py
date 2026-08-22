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
    # On a multi-device mesh, weights replicate (the MTP drafter head runs this
    # single-device attention fully replicated on TP). No-op on one device.
    mesh_kwargs = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if mesh_device.get_num_devices() > 1 else {}

    def load_2d(name):
        t = state_dict[f"{name}.weight"].T.contiguous()
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight") if tensor_cache_path else None,
            **mesh_kwargs,
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
            **mesh_kwargs,
        )

    return AttentionWeights(
        q_proj=load_2d("q_proj"),
        k_proj=load_2d("k_proj"),
        v_proj=load_2d("v_proj"),
        o_proj=load_2d("o_proj"),
        q_norm=load_norm("q_norm"),
        k_norm=load_norm("k_norm"),
    )
