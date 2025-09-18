# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Optional, Tuple

import torch
from pydantic import BaseModel, Field

import ttnn
from models.common.utility_functions import nearest_32
from models.demos.siglip.tests.common import flatten_state_dict
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_out_subblock_w
from models.tt_transformers.tt.multimodal.llama_image_attention import TtLlamaImageAttention


def find_largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def matmul_config(
    tile_size,
    m: int,
    k: int,
    n: int,
    grid_size: Tuple[int, int],
    in0_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
    per_core_M=None,
    per_core_N=None,
):
    if per_core_M is None:
        per_core_M = math.ceil(m / (tile_size * grid_size[1]))
    if per_core_N is None:
        per_core_N = math.ceil(n / (tile_size * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)

    if in0_block_w is None:
        assert k % (tile_size * grid_size[1]) == 0, f"Input width must be divisible by tile size times grid size"
        in0_block_w = find_largest_divisor(k // (tile_size * grid_size[1]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


class AttentionConfig(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
    }
    tile_size: Optional[int] = 32
    num_devices: int
    vision_dim: int
    vision_head_dim: int
    vision_attn_n_heads: int
    max_grid_size: ttnn.CoreGrid
    compute_kernel_config_hifi2: ttnn.WormholeComputeKernelConfig = Field(
        default_factory=lambda: ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    )
    compute_kernel_config_hifi4: ttnn.WormholeComputeKernelConfig = Field(
        default_factory=lambda: ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    )
    compute_kernel_config_sdpa: ttnn.WormholeComputeKernelConfig = Field(
        default_factory=lambda: ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    )
    sdpa_cfg: ttnn.SDPAProgramConfig = Field(
        default_factory=lambda: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
    )
    dummy_weights: bool = False
    VISION_MAX_MM_SEQ: int = 32

    def get_model_config(self):
        return {
            "IMAGE_ATTN_OUT_PROGCFG": lambda seq_len, max_seq: matmul_config(
                tile_size=self.tile_size,
                m=min(seq_len, max_seq),
                k=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3) // self.num_devices,
                n=self.vision_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            ),
            "IMAGE_ATTN_QKV_PROGCFG": lambda seq_len, max_seq: matmul_config(
                tile_size=self.tile_size,
                m=min(seq_len, max_seq),
                k=self.vision_dim,
                n=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3)
                // self.num_devices,  # Head dim was padded to nearest 32
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            ),
        }


def siglip_attention_ttnn(
    mesh_device,
    hidden_states: torch.Tensor,
    state_dict: Dict,
    state_dict_prefix: str,
    weight_cache_path: str,
    dtype=ttnn.bfloat16,
    vision_dim: int = 1152,
    num_heads: int = 16,
    dropout: float = 0.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    state_dict = flatten_state_dict(state_dict)
    grid = mesh_device.compute_with_storage_grid_size()
    attention_config = AttentionConfig(
        num_devices=mesh_device.get_num_devices() if mesh_device else 0,
        vision_dim=vision_dim,
        vision_head_dim=vision_dim // num_heads,
        vision_attn_n_heads=num_heads,
        max_grid_size=ttnn.CoreGrid(x=grid.x, y=grid.y),
    )

    ttnn_attention = TtLlamaImageAttention(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
        configuration=attention_config,
    )
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = ttnn.from_torch(
            hidden_states,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        output = ttnn_attention(hidden_states, mask=attention_mask)
        output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    else:
        output = ttnn_attention(hidden_states, mask=attention_mask)
    return output, None
