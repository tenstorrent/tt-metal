# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Optional
import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.experimental.tt_dit.tests.models.sd35_med.test_dismantled_block import (
    DismantledBlock,
    attention,
)
from models.experimental.tt_dit.models.transformers.sd35_med.joint_block_sd35_medium import SD35MediumJointBlock


def block_mixing(context, x, context_block, x_block, c):
    """Reference block mixing implementation"""
    assert context is not None, "block_mixing called with None context input"

    # Use pre_attention_qkv for joint attention
    context_qkv, context_intermediates = context_block.pre_attention_qkv(context, c)

    if x_block.x_block_self_attn:
        x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
    else:
        x_qkv, x_intermediates = x_block.pre_attention_qkv(x, c)

    # If x_qkv[0] is 2D, it might be attention output instead of QKV tuple
    # Reshape if needed: (L, hidden_size) -> (1, L, num_heads, head_dim)
    if len(x_qkv[0].shape) == 2:
        # This is likely an attention output, not a QKV tuple
        # This shouldn't happen if pre_attention_x is correct
        raise ValueError(
            f"x_qkv[0] has wrong shape {x_qkv[0].shape}. Expected 4D (B, L, num_heads, head_dim), got 2D. This suggests pre_attention_x is returning attention outputs instead of QKV tuples."
        )

    assert len(context_qkv[0].shape) == 4, f"context_qkv[0] shape: {context_qkv[0].shape}"
    assert len(x_qkv[0].shape) == 4, f"x_qkv[0] shape: {x_qkv[0].shape}"

    q, k, v = tuple(torch.cat(tuple(qkv[i] for qkv in [context_qkv, x_qkv]), dim=1) for i in range(3))
    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = (
        attn[:, : context_qkv[0].shape[1]],
        attn[:, context_qkv[0].shape[1] :],
    )

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None

    if x_block.x_block_self_attn:
        x_q2, x_k2, x_v2 = x_qkv2
        attn2 = attention(x_q2, x_k2, x_v2, x_block.attn2.num_heads)
        x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
    else:
        x = x_block.post_attention(x_attn, *x_intermediates)

    return context, x


class JointBlock(torch.nn.Module):
    """Reference PyTorch implementation matching MM-DiT JointBlock"""

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
        self.context_block = DismantledBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            pre_only=pre_only,
            rmsnorm=rmsnorm,
            scale_mod_only=scale_mod_only,
            swiglu=swiglu,
            x_block_self_attn=False,
        )
        self.x_block = DismantledBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            pre_only=False,
            rmsnorm=rmsnorm,
            scale_mod_only=scale_mod_only,
            swiglu=swiglu,
            x_block_self_attn=x_block_self_attn,
        )

    def forward(self, context: torch.Tensor, x: torch.Tensor, c: torch.Tensor):
        return block_mixing(context, x, context_block=self.context_block, x_block=self.x_block, c=c)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "hidden_size, num_heads, context_seq_len, x_seq_len, batch_size, mlp_ratio, pre_only, x_block_self_attn, swiglu",
    [
        (1536, 24, 77, 1024, 1, 4.0, False, True, False),  # SD3.5 Medium dual attention joint block (0-12)
        (
            1536,
            24,
            77,
            1024,
            1,
            4.0,
            False,
            False,
            False,
        ),  # SD3.5 Medium standard joint block (both blocks standard) (13-22)
        (
            1536,
            24,
            77,
            1024,
            1,
            4.0,
            True,
            False,
            False,
        ),  # SD3.5 Medium last joint block (context pre_only, x full) (23)
    ],
    # ids=["dual_attn"],
    ids=["dual_attn", "standard", "pre_only"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_joint_block(
    device,
    dtype,
    hidden_size,
    num_heads,
    context_seq_len,
    x_seq_len,
    batch_size,
    mlp_ratio,
    pre_only,
    x_block_self_attn,
    swiglu,
    reset_seeds,
):
    """Test SD3.5 Medium JointBlock forward pass"""
    torch.manual_seed(1234)

    # Create reference model
    reference_model = JointBlock(
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
    tt_model = SD35MediumJointBlock(
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
    context_input = torch.randn(1, batch_size, context_seq_len, hidden_size, dtype=torch.bfloat16)
    x_input = torch.randn(1, batch_size, x_seq_len, hidden_size, dtype=torch.bfloat16)
    c_input = torch.randn(1, batch_size, hidden_size, dtype=torch.bfloat16)

    # Reference forward
    with torch.no_grad():
        ref_context_output, ref_x_output = reference_model(
            context_input.squeeze(0), x_input.squeeze(0), c_input.squeeze(0)
        )

    # TTNN forward
    tt_context_input = ttnn.from_torch(context_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_x_input = ttnn.from_torch(x_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_c_input = ttnn.from_torch(c_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_context_output, tt_x_output = tt_model(tt_context_input, tt_x_input, tt_c_input)

    # Compare outputs
    pcc_required = 0.99

    # Handle context output (may be None if pre_only)
    if pre_only:
        assert tt_context_output is None, "Context output should be None when pre_only=True"
        assert ref_context_output is None, "Reference context output should be None when pre_only=True"
    else:
        # Convert back and compare
        tt_context_output_torch = ttnn.to_torch(tt_context_output)[0, :batch_size, :context_seq_len, :hidden_size]
        passing_context, _ = comp_pcc(ref_context_output, tt_context_output_torch, pcc_required)
        assert passing_context, f"Context output does not meet PCC requirement {pcc_required}."

    # Convert back and compare x output
    tt_x_output_torch = ttnn.to_torch(tt_x_output)[0, :batch_size, :x_seq_len, :hidden_size]
    passing_x, pcc_x = comp_pcc(ref_x_output, tt_x_output_torch, pcc_required)
    assert passing_x, f"X output does not meet PCC requirement {pcc_required}."

    # Print final joint block PCC
    print(f"JointBlock PCC: {pcc_x}")
