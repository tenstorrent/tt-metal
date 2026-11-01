# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, replace

import torch
import ttnn

def _dense_wdtype(default):
    """Weight dtype for a dense projection. QWEN36_1C_ALL_BFP4=1 forces bfloat4_b everywhere, which is
    the dense-bfp4 bar the exact-ternary path has to beat. Default returns `default` unchanged."""
    import os

    import ttnn

    return ttnn.bfloat4_b if os.environ.get("QWEN36_1C_ALL_BFP4") == "1" else default



@dataclass(frozen=True)
class AttentionWeights:
    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor  # +1 pre-offset (zero-centered RMSNorm)
    k_norm: ttnn.Tensor
    fused_qkv: ttnn.Tensor = None  # [dim, qg|k|v] one in-projection weight (QWEN36_1C_FUSED_QKV)


def load_attention_weights(mesh_device, state_dict, tensor_cache_path=None) -> AttentionWeights:
    def load_2d(name):
        return ttnn.as_tensor(
            state_dict[f"{name}.weight"],
            dtype=_dense_wdtype(ttnn.bfloat8_b),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight") if tensor_cache_path else None,
            preprocess=lambda t: t.T.contiguous(),  # [in, out] for ttnn.linear; cache-miss only
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

    w = AttentionWeights(
        q_proj=load_2d("q_proj"),
        k_proj=load_2d("k_proj"),
        v_proj=load_2d("v_proj"),
        o_proj=load_2d("o_proj"),
        q_norm=load_norm("q_norm"),
        k_norm=load_norm("k_norm"),
    )
    # 2a (QWEN36_1C_FUSED_QKV): one [dim, qg+k+v] weight so decode runs one in-projection instead of
    # three. k and v are only head_dim*num_kv_heads wide, so at decode they are mostly fixed cost.
    if os.environ.get("QWEN36_1C_FUSED_QKV") == "1":
        fused = torch.cat(
            [state_dict["q_proj.weight"], state_dict["k_proj.weight"], state_dict["v_proj.weight"]], dim=0
        )
        w = replace(
            w,
            fused_qkv=ttnn.as_tensor(
                fused,
                dtype=_dense_wdtype(ttnn.bfloat8_b),
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=(tensor_cache_path / "self_attn.qkv_fused.weight")
                if tensor_cache_path
                else None,
                preprocess=lambda t: t.T.contiguous(),  # [in, out] for ttnn.linear
            ),
        )
    return w
