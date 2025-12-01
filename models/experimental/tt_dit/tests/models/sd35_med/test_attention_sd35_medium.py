# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.models.transformers.sd35_med.attention_sd35_medium import SD35MediumSelfAttention


class RMSNorm(torch.nn.Module):
    """Reference RMSNorm matching MM-DiT"""

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
    """Reference matching MM-DiT SelfAttention"""

    def __init__(self, dim, num_heads, qkv_bias=False, pre_only=False, qk_norm="rms"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pre_only = pre_only

        # Fused QKV
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=torch.bfloat16)

        # Output projection
        if not pre_only:
            self.proj = torch.nn.Linear(dim, dim, dtype=torch.bfloat16)

        # QK norm
        self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, x):
        B, L, C = x.shape

        # Fused QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Apply RMSNorm
        q = self.ln_q(q)
        k = self.ln_k(k)

        # Transpose to [B, num_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Transpose back and reshape
        attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)

        # Output projection
        if not self.pre_only:
            attn_out = self.proj(attn_out)

        return attn_out


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "dim, num_heads, seq_len, batch_size",
    [
        (1536, 24, 1024, 1),
        (1536, 24, 512, 1),
    ],
    ids=["sd35_med_1k", "sd35_med_512"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_self_attention(device, dim, num_heads, seq_len, batch_size, dtype, reset_seeds):
    """Test SD3.5 Medium self-attention matching MM-DiT reference"""
    torch.manual_seed(1234)

    # Create reference model
    reference_model = SelfAttention(dim=dim, num_heads=num_heads, qkv_bias=True, pre_only=False)
    reference_model.eval()

    # Create TTNN model
    tt_model = SD35MediumSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        pre_only=False,
        qk_norm="rms",
        eps=1e-6,
        mesh_device=device,
    )

    # Load weights
    state_dict = reference_model.state_dict()
    tt_model.load_state_dict(state_dict)

    # Create input
    x_input = torch.randn(1, batch_size, seq_len, dim, dtype=torch.bfloat16)

    # Reference forward
    with torch.no_grad():
        ref_output = reference_model(x_input.squeeze(0))

    # TTNN forward
    tt_x_input = ttnn.from_torch(x_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_model(tt_x_input, seq_len)

    # Convert back and compare
    tt_output_torch = ttnn.to_torch(tt_output)[0, :batch_size, :seq_len, :dim]

    passing, pcc = comp_pcc(ref_output, tt_output_torch, 0.99)
    logger.info(f"Self-Attention PCC: {pcc}")

    assert passing, f"PCC check failed: {pcc}"
