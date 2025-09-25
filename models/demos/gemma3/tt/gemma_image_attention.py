# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Optional, Tuple

import torch
from pydantic import BaseModel, Field

import ttnn
from models.tt_transformers.tt.common import get_out_subblock_w
from models.tt_transformers.tt.multimodal.llama_image_attention import TtLlamaImageAttention
from models.utility_functions import nearest_32


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


class GemmaAttentionConfig(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
    }
    tile_size: Optional[int] = 32
    num_devices: int
    vision_dim: int
    vision_head_dim: int
    vision_attn_n_heads: int
    max_grid_size: ttnn.CoreGrid
    base_model_name: str = ""
    VISION_MAX_MM_SEQ: int = 32
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

    def get_model_config(self):
        return {
            "IMAGE_ATTN_OUT_PROGCFG": lambda seq_len, max_seq: matmul_config(
                tile_size=self.tile_size,
                m=min(seq_len, max_seq),
                k=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads)
                // self.num_devices,  # Correctly divided by num_devices
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


def transform_gemma_weights_for_llama(state_dict: Dict, state_dict_prefix: str) -> Dict:
    """
    Transorm weight names to fit llama format
    """
    wq_str = f"{state_dict_prefix}wq.weight"
    wk_str = f"{state_dict_prefix}wk.weight"
    wv_str = f"{state_dict_prefix}wv.weight"
    wo_str = f"{state_dict_prefix}wo.weight"

    bq_str = f"{state_dict_prefix}wq.bias"
    bk_str = f"{state_dict_prefix}wk.bias"
    bv_str = f"{state_dict_prefix}wv.bias"
    bo_str = f"{state_dict_prefix}wo.bias"

    # Create new state dict with Llama-compatible keys
    llama_state_dict = {}

    if wq_str in state_dict and wk_str in state_dict and wv_str in state_dict:
        wq = state_dict[wq_str]
        wk = state_dict[wk_str]
        wv = state_dict[wv_str]

        # Store individual Q, K, V weights - LlamaImageAttention will fuse them internally
        llama_state_dict["wq.weight"] = wq
        llama_state_dict["wk.weight"] = wk
        llama_state_dict["wv.weight"] = wv
        llama_state_dict["wo.weight"] = state_dict[wo_str]

        # Handle biases if they exist
        if bq_str in state_dict and bk_str in state_dict and bv_str in state_dict:
            bq = state_dict[bq_str]
            bk = state_dict[bk_str]
            bv = state_dict[bv_str]

            llama_state_dict["wq.bias"] = bq
            llama_state_dict["wk.bias"] = bk
            llama_state_dict["wv.bias"] = bv

        if bo_str in state_dict:
            llama_state_dict["wo.bias"] = state_dict[bo_str]

    return llama_state_dict


class TtGemmaImageAttention:
    """
    Gemma Image Attention class using TtLlamaImageAttention internally.
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict: Dict,
        state_dict_prefix: str,
        weight_cache_path: str,
        dtype=ttnn.bfloat16,
        configuration=None,
    ):
        """
        Initialize the Gemma Image Attention.

        Args:
            mesh_device: Device mesh
            tt_ccl: CCL instance
            state_dict: Model weights dictionary
            state_dict_prefix: Prefix for weight keys
            weight_cache_path: Path for weight caching
            dtype: Data type for computation
            configuration: Model configuration (ModelArgs instance)
        """
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.state_dict = state_dict
        self.state_dict_prefix = state_dict_prefix
        self.weight_cache_path = weight_cache_path
        self.dtype = dtype
        self.configuration = configuration

        # Extract vision parameters from configuration
        vision_dim = configuration.vision_dim if configuration else 1152
        num_heads = configuration.vision_attn_n_heads if configuration else 16
        base_model_name = configuration.base_model_name if configuration else ""
        vision_max_mm_seq = configuration.VISION_MAX_MM_SEQ if hasattr(configuration, "VISION_MAX_MM_SEQ") else 32

        # Transform Gemma weights to Llama format
        llama_state_dict = transform_gemma_weights_for_llama(state_dict, state_dict_prefix)

        grid = mesh_device.compute_with_storage_grid_size()
        attention_config = GemmaAttentionConfig(
            num_devices=mesh_device.get_num_devices() if mesh_device else 0,
            vision_dim=vision_dim,
            vision_head_dim=vision_dim // num_heads,
            vision_attn_n_heads=num_heads,
            max_grid_size=ttnn.CoreGrid(x=grid.x, y=grid.y),
            base_model_name=base_model_name,
            VISION_MAX_MM_SEQ=vision_max_mm_seq,
        )

        # Initialize the underlying TtLlamaImageAttention
        self.ttnn_attention = TtLlamaImageAttention(
            mesh_device,
            tt_ccl=tt_ccl,
            state_dict=llama_state_dict,  # Use transformed weights
            state_dict_prefix="",  # Empty prefix since keys are already transformed
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=attention_config,
        )

    def __call__(self, hidden_states, attention_mask=None):
        """
        Forward pass of the attention layer.

        Args:
            hidden_states: Input tensor (torch.Tensor or ttnn.Tensor)
            attention_mask: Optional attention mask

        Returns:
            Output tensor
        """
        if isinstance(hidden_states, torch.Tensor):
            # Ensure the tensor has the correct 4D shape for Gemma3: (batch, 1, seq_len, dimensions)
            if len(hidden_states.shape) == 3:  # (batch, seq_len, hidden_dim)
                hidden_states = hidden_states.unsqueeze(1)  # (batch, 1, seq_len, hidden_dim)

            hidden_states = ttnn.from_torch(
                hidden_states,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            output = self.ttnn_attention(hidden_states, mask=attention_mask)
            output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0]
        else:
            if hidden_states.shape != (1, 1, hidden_states.shape[-2], hidden_states.shape[-1]):
                hidden_states = ttnn.reshape(hidden_states, [1, 1, hidden_states.shape[-2], hidden_states.shape[-1]])
            output = self.ttnn_attention(hidden_states, mask=attention_mask)

        return output
