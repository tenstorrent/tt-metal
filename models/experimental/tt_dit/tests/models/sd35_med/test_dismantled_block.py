# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from typing import Optional
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.experimental.tt_dit.models.transformers.sd35_med.dismantled_block_sd35_medium import (
    SD35MediumDismantledBlock,
)
from models.experimental.tt_dit.tests.models.sd35_med.test_attention_sd35_medium import SelfAttention, RMSNorm
from models.experimental.tt_dit.tests.models.sd35_med.test_mlp import Mlp
from models.experimental.tt_dit.tests.models.sd35_med.test_swiglu import SwiGLUFeedForward

# from models.experimental.tt_dit.tests.models.sd35_med.test_attention_sd35_medium import attention


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, seq_len, n_heads, dim_head = q.shape

    # Transpose to [batch, num_heads, seq_len, head_dim] for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    # Transpose back to [batch, seq_len, num_heads, head_dim]
    out = out.transpose(1, 2)

    # Reshape to [batch, seq_len, heads * dim_head]
    return out.reshape(b, seq_len, heads * dim_head)


class DismantledBlock(torch.nn.Module):
    """Reference PyTorch implementation matching MM-DiT DismantledBlock"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: Optional[str] = None,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        x_block_self_attn: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pre_only = pre_only
        self.scale_mod_only = scale_mod_only
        self.x_block_self_attn = x_block_self_attn

        # Norm1
        if rmsnorm:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=torch.bfloat16)

        # Primary attention
        self.attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pre_only=pre_only,
            qk_norm=qk_norm,
        )

        # Dual attention for x_block_self_attn mode
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.attn2 = SelfAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                pre_only=False,
                qk_norm=qk_norm,
            )

        # Norm2 and MLP (only if not pre_only)
        if not pre_only:
            if rmsnorm:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            else:
                self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=torch.bfloat16)

            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            if not swiglu:
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=torch.nn.GELU,
                    act_kwargs={"approximate": "tanh"},
                    dtype=torch.bfloat16,
                )
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)

        # AdaLN modulation
        if x_block_self_attn:
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1

        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=torch.bfloat16)
        )

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        """Pre-attention: normalization, modulation, and full attention computation"""
        assert x is not None, "pre_attention called with None input"

        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
                    6, dim=1
                )
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)

            # Return full attention output for standard blocks
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            return attn_out, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)

            # Return full attention output for pre_only blocks
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            return attn_out, None

    def pre_attention_qkv(self, x: torch.Tensor, c: torch.Tensor):
        """Pre-attention: return QKV tuples for joint attention"""
        assert x is not None, "pre_attention_qkv called with None input"

        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
                    6, dim=1
                )
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)

            # Get QKV tensors for joint attention
            modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
            B, L, C = modulated_x.shape
            qkv = self.attn.qkv(modulated_x).reshape(B, L, 3, self.attn.num_heads, self.attn.head_dim)
            q, k, v = qkv.unbind(2)

            # Apply RMSNorm
            q = self.attn.ln_q(q)
            k = self.attn.ln_k(k)

            return (q, k, v), (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)

            # Get QKV tensors for joint attention
            modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
            B, L, C = modulated_x.shape
            qkv = self.attn.qkv(modulated_x).reshape(B, L, 3, self.attn.num_heads, self.attn.head_dim)
            q, k, v = qkv.unbind(2)

            # Apply RMSNorm
            q = self.attn.ln_q(q)
            k = self.attn.ln_k(k)

            return (q, k, v), None

    def pre_attention_x(self, x: torch.Tensor, c: torch.Tensor):
        """Pre-attention for dual attention mode (x_block_self_attn)"""
        assert self.x_block_self_attn
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)

        x_norm = self.norm1(x)

        # Get QKV tuples for first attention (for joint attention)
        modulated_x1 = modulate(x_norm, shift_msa, scale_msa)
        B, L, C = modulated_x1.shape
        qkv1 = self.attn.qkv(modulated_x1).reshape(B, L, 3, self.attn.num_heads, self.attn.head_dim)
        q1, k1, v1 = qkv1.unbind(2)
        q1 = self.attn.ln_q(q1)
        k1 = self.attn.ln_k(k1)

        # Ensure QKV tensors have correct 4D shape (B, L, num_heads, head_dim)
        assert q1.shape == (
            B,
            L,
            self.attn.num_heads,
            self.attn.head_dim,
        ), f"q1 shape: {q1.shape}, expected: {(B, L, self.attn.num_heads, self.attn.head_dim)}"
        assert k1.shape == (B, L, self.attn.num_heads, self.attn.head_dim), f"k1 shape: {k1.shape}"
        assert v1.shape == (B, L, self.attn.num_heads, self.attn.head_dim), f"v1 shape: {v1.shape}"

        qkv1_tuple = (q1, k1, v1)

        # Get QKV tuples for second attention
        modulated_x2 = modulate(x_norm, shift_msa2, scale_msa2)
        qkv2 = self.attn2.qkv(modulated_x2).reshape(B, L, 3, self.attn2.num_heads, self.attn2.head_dim)
        q2, k2, v2 = qkv2.unbind(2)
        q2 = self.attn2.ln_q(q2)
        k2 = self.attn2.ln_k(k2)

        # Ensure QKV tensors have correct 4D shape
        assert q2.shape == (B, L, self.attn2.num_heads, self.attn2.head_dim), f"q2 shape: {q2.shape}"
        assert k2.shape == (B, L, self.attn2.num_heads, self.attn2.head_dim), f"k2 shape: {k2.shape}"
        assert v2.shape == (B, L, self.attn2.num_heads, self.attn2.head_dim), f"v2 shape: {v2.shape}"

        qkv2_tuple = (q2, k2, v2)

        return (
            qkv1_tuple,
            qkv2_tuple,
            (x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2),
        )

    def post_attention(self, attn_out, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        """Post-attention: apply gating, residual, and MLP"""
        assert not self.pre_only
        # attn_out is already the full attention output, no need for post_attention call
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def post_attention_x(
        self,
        attn_out,
        attn_out2,
        x,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        gate_msa2,
        attn1_dropout: float = 0.0,
    ):
        """Post-attention for dual attention mode"""
        assert not self.pre_only

        # If attn_out is a tuple (QKV tuple), compute attention output from it
        if isinstance(attn_out, tuple):
            q1, k1, v1 = attn_out
            # Compute attention output from QKV
            B, L, num_heads, head_dim = q1.shape
            # Transpose to [B, num_heads, L, head_dim] for SDPA
            q1 = q1.transpose(1, 2)
            k1 = k1.transpose(1, 2)
            v1 = v1.transpose(1, 2)
            # Scaled dot-product attention
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q1, k1, v1, scale=self.attn.scale, is_causal=False
            )
            # Transpose back and reshape to [B, L, hidden_size]
            attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)
            # Output projection
            attn_out = self.attn.proj(attn_out)

        # If attn_out2 is a tuple (QKV tuple), compute attention output from it
        if isinstance(attn_out2, tuple):
            q2, k2, v2 = attn_out2
            # Compute attention output from QKV
            B, L, num_heads, head_dim = q2.shape
            # Transpose to [B, num_heads, L, head_dim] for SDPA
            q2 = q2.transpose(1, 2)
            k2 = k2.transpose(1, 2)
            v2 = v2.transpose(1, 2)
            # Scaled dot-product attention
            attn_out2 = torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v2, scale=self.attn2.scale, is_causal=False
            )
            # Transpose back and reshape to [B, L, hidden_size]
            attn_out2 = attn_out2.transpose(1, 2).reshape(B, L, -1)
            # Output projection
            attn_out2 = self.attn2.proj(attn_out2)

        if attn1_dropout > 0.0:
            attn1_dropout_mask = torch.bernoulli(
                torch.full((attn_out.size(0), 1, 1), 1 - attn1_dropout, device=attn_out.device)
            )
            attn_ = gate_msa.unsqueeze(1) * attn_out * attn1_dropout_mask
        else:
            attn_ = gate_msa.unsqueeze(1) * attn_out

        x = x + attn_
        attn2_ = gate_msa2.unsqueeze(1) * attn_out2
        x = x + attn2_
        mlp_ = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + mlp_
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass orchestrating pre/post attention"""
        if self.pre_only:
            # Pre-only mode: just return attention output
            attn_out, _ = self.pre_attention(x, c)
            return attn_out
        else:
            # Standard mode: full block with attention + MLP
            assert not self.pre_only

            if self.x_block_self_attn:
                attn_out, attn_out2, intermediates = self.pre_attention_x(x, c)
                return self.post_attention_x(attn_out, attn_out2, *intermediates)
            else:
                attn_out, intermediates = self.pre_attention(x, c)
                return self.post_attention(attn_out, *intermediates)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "hidden_size, num_heads, seq_len, batch_size, mlp_ratio, pre_only, x_block_self_attn, swiglu",
    [
        (1536, 24, 1024, 1, 4.0, False, False, False),  # SD3.5 Medium standard block
        (1536, 24, 1024, 1, 4.0, False, True, False),  # SD3.5 Medium dual attention block
        (1536, 24, 512, 1, 4.0, False, False, True),  # SD3.5 Medium with SwiGLU
        (1536, 24, 77, 1, 4.0, True, False, False),  # SD3.5 Medium pre_only block (attention only, no MLP)
    ],
    ids=["standard", "dual_attn", "swiglu", "pre_only"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_dismantled_block(
    device,
    dtype,
    hidden_size,
    num_heads,
    seq_len,
    batch_size,
    mlp_ratio,
    pre_only,
    x_block_self_attn,
    swiglu,
    reset_seeds,
):
    """Test SD3.5 Medium DismantledBlock forward pass"""
    torch.manual_seed(1234)

    # Create reference model
    reference_model = DismantledBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        pre_only=pre_only,
        scale_mod_only=False,
        swiglu=swiglu,
        x_block_self_attn=x_block_self_attn,
    )
    reference_model.eval()

    # Create parallel config for N150
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=None),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=None),
    )

    # Create TTNN model
    tt_model = SD35MediumDismantledBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        pre_only=pre_only,
        scale_mod_only=False,
        swiglu=swiglu,
        qk_norm="rms",
        x_block_self_attn=x_block_self_attn,
        mesh_device=device,
        ccl_manager=None,
        parallel_config=parallel_config,
    )

    # Load weights
    state_dict = reference_model.state_dict()
    tt_model.load_state_dict(state_dict)

    # Create inputs
    x_input = torch.randn(1, batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    c_input = torch.randn(1, batch_size, hidden_size, dtype=torch.bfloat16)

    # Reference forward
    with torch.no_grad():
        x_ref = x_input.squeeze(0)
        c_ref = c_input.squeeze(0)
        ref_output = reference_model(x_ref, c_ref)

    # TTNN forward
    tt_x_input = ttnn.from_torch(x_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_c_input = ttnn.from_torch(c_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_model(tt_x_input, tt_c_input)
    tt_output_torch = ttnn.to_torch(tt_output)[0, :batch_size, :seq_len, :hidden_size]

    # Compare final outputs
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_required)
    print(f"Final output PCC: {pcc_message}")

    assert passing, f"Block output does not meet PCC requirement {pcc_required}."

    logger.info("SD3.5 Medium DismantledBlock test passed!")
