# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.experimental.tt_dit.models.transformers.attention_sd35_medium import SD35JointAttention


class RMSNorm(torch.nn.Module):
    """Reference RMSNorm matching MM-DiT implementation"""

    def __init__(self, dim, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        return x


class SelfAttention(torch.nn.Module):
    """Reference PyTorch implementation matching MM-DiT SelfAttention"""

    def __init__(self, query_dim, head_dim, heads, out_dim, context_pre_only=False):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.context_pre_only = context_pre_only

        inner_dim = heads * head_dim

        # Fused QKV projections
        self.to_qkv = torch.nn.Linear(query_dim, 3 * inner_dim, bias=True, dtype=torch.bfloat16)
        self.add_qkv_proj = torch.nn.Linear(query_dim, 3 * inner_dim, bias=True, dtype=torch.bfloat16)

        # RMSNorm for Q and K (matching reference with elementwise_affine=True)
        self.norm_q = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.norm_added_q = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)

        # Output projections
        self.to_out = torch.nn.Linear(inner_dim, out_dim, bias=True, dtype=torch.bfloat16)
        if not context_pre_only:
            self.to_add_out = torch.nn.Linear(inner_dim, out_dim, bias=True, dtype=torch.bfloat16)

    def forward(self, spatial, prompt):
        B, N, _ = spatial.shape
        _, L, _ = prompt.shape

        # Fused QKV projection and split
        qkv = self.to_qkv(spatial).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        prompt_qkv = self.add_qkv_proj(prompt).reshape(B, L, 3, self.heads, self.head_dim)
        prompt_q, prompt_k, prompt_v = prompt_qkv.unbind(2)

        # Apply RMSNorm to Q and K (matching reference pre_attention)
        q = self.norm_q(q)
        k = self.norm_k(k)
        prompt_q = self.norm_added_q(prompt_q)
        prompt_k = self.norm_added_k(prompt_k)

        # Transpose to [B, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        prompt_q = prompt_q.transpose(1, 2)
        prompt_k = prompt_k.transpose(1, 2)
        prompt_v = prompt_v.transpose(1, 2)

        # Concatenate for joint attention
        joint_q = torch.cat([q, prompt_q], dim=2)
        joint_k = torch.cat([k, prompt_k], dim=2)
        joint_v = torch.cat([v, prompt_v], dim=2)

        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(joint_q, joint_k, joint_v, scale=self.scale)

        # Split back
        spatial_out = attn_output[:, :, :N, :]
        prompt_out = attn_output[:, :, N:, :]

        # Transpose and reshape
        spatial_out = spatial_out.transpose(1, 2).reshape(B, N, -1)
        prompt_out = prompt_out.transpose(1, 2).reshape(B, L, -1)

        # Project output (matching reference post_attention)
        spatial_out = self.to_out(spatial_out)
        if not self.context_pre_only:
            prompt_out = self.to_add_out(prompt_out)
        else:
            prompt_out = None

        return spatial_out, prompt_out


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d",
    [
        (1, 24, 1024, 77, 64),
        (1, 24, 4096, 333, 64),
    ],
    ids=["sd35_med_1k", "sd35_med_4k"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_joint_attention(device, b, nh, seq_len, joint_seq_len, d, dtype, reset_seeds):
    """Test SD3.5 Medium joint attention matching MM-DiT reference"""
    torch.manual_seed(1234)

    query_dim = nh * d

    # Create reference model
    reference_model = SelfAttention(
        query_dim=query_dim,
        head_dim=d,
        heads=nh,
        out_dim=query_dim,
        context_pre_only=False,
    )
    reference_model.eval()

    # Create parallel config for N150
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=None),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=None),
    )

    # Create TTNN model
    tt_model = SD35JointAttention(
        query_dim=query_dim,
        head_dim=d,
        heads=nh,
        out_dim=query_dim,
        bias=True,
        context_pre_only=False,
        eps=1e-6,
        mesh_device=device,
        ccl_manager=None,
        parallel_config=parallel_config,
        padding_config=None,
    )

    # Load weights from reference
    state_dict = reference_model.state_dict()
    tt_model.load_state_dict(state_dict)

    # Create inputs
    spatial_input = torch.randn(1, b, seq_len, query_dim, dtype=torch.bfloat16)
    prompt_input = torch.randn(1, b, joint_seq_len, query_dim, dtype=torch.bfloat16)

    # Reference forward
    with torch.no_grad():
        ref_spatial_out, ref_prompt_out = reference_model(spatial_input.squeeze(0), prompt_input.squeeze(0))

    # TTNN forward
    tt_spatial_input = ttnn.from_torch(spatial_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_prompt_input = ttnn.from_torch(prompt_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_spatial_out, tt_prompt_out = tt_model(tt_spatial_input, tt_prompt_input, seq_len)

    # Convert back and compare
    tt_spatial_out_torch = ttnn.to_torch(tt_spatial_out)[0, :b, :seq_len, :query_dim]
    tt_prompt_out_torch = ttnn.to_torch(tt_prompt_out)[0, :b, :joint_seq_len, :query_dim]

    spatial_passing, spatial_pcc = comp_pcc(ref_spatial_out, tt_spatial_out_torch, 0.99)
    prompt_passing, prompt_pcc = comp_pcc(ref_prompt_out, tt_prompt_out_torch, 0.99)

    logger.info(f"Spatial PCC: {spatial_pcc}, Prompt PCC: {prompt_pcc}")

    assert spatial_passing and prompt_passing, f"PCC check failed"
