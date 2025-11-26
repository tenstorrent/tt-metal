# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium DismantledBlock Implementation

This module implements the DismantledBlock transformer block for SD3.5 Medium,
matching the MM-DiT reference implementation with adaptive layer normalization.
"""

import ttnn
from ...layers.normalization import LayerNorm, RMSNorm
from ...layers.linear import Linear
from ...utils.substate import substate
from models.experimental.tt_dit.models.transformers.attention_sd35_medium import SD35MediumSelfAttention
from models.experimental.tt_dit.models.transformers.mlp_sd35_medium import SD35MediumMlp
from models.experimental.tt_dit.models.transformers.swiglu_sd35_medium import SD35MediumSwiGLU


def modulate(x, shift, scale):
    """Apply modulation: x * (1 + scale) + shift"""
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class SD35MediumDismantledBlock:
    """
    DismantledBlock for SD3.5 Medium with adaptive layer normalization.
    Supports standard, dual attention, and pre-only modes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: str = None,
        x_block_self_attn: bool = False,
        eps: float = 1e-6,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.pre_only = pre_only
        self.scale_mod_only = scale_mod_only
        self.x_block_self_attn = x_block_self_attn
        self.mesh_device = mesh_device

        # Norm1
        if rmsnorm:
            self.norm1 = RMSNorm(
                embedding_dim=hidden_size,
                norm_eps=eps,
                norm_elementwise_affine=False,
                bias=False,
                mesh_device=mesh_device,
            )
        else:
            self.norm1 = LayerNorm(
                embedding_dim=hidden_size,
                norm_eps=eps,
                norm_elementwise_affine=False,
                bias=False,
                mesh_device=mesh_device,
            )

        # Primary attention
        self.attn = SD35MediumSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pre_only=pre_only,
            qk_norm=qk_norm,
            eps=eps,
            mesh_device=mesh_device,
        )

        # Dual attention (for blocks 0-12 in SD3.5 Medium)
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.attn2 = SD35MediumSelfAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                pre_only=False,
                qk_norm=qk_norm,
                eps=eps,
                mesh_device=mesh_device,
            )

        # MLP/SwiGLU
        if not pre_only:
            if rmsnorm:
                self.norm2 = RMSNorm(
                    embedding_dim=hidden_size,
                    norm_eps=eps,
                    norm_elementwise_affine=False,
                    bias=False,
                    mesh_device=mesh_device,
                )
            else:
                self.norm2 = LayerNorm(
                    embedding_dim=hidden_size,
                    norm_eps=eps,
                    norm_elementwise_affine=False,
                    bias=False,
                    mesh_device=mesh_device,
                )

            if swiglu:
                self.mlp = SD35MediumSwiGLU(
                    dim=hidden_size,
                    hidden_dim=int(hidden_size * mlp_ratio),
                    multiple_of=256,
                    mesh_device=mesh_device,
                )
            else:
                self.mlp = SD35MediumMlp(
                    in_features=hidden_size,
                    hidden_features=int(hidden_size * mlp_ratio),
                    out_features=hidden_size,
                    bias=True,
                    mesh_device=mesh_device,
                )

        # AdaLN modulation
        # Calculate number of parameters based on mode
        if pre_only:
            if scale_mod_only:
                adaln_out_features = hidden_size  # scale only
            else:
                adaln_out_features = 2 * hidden_size  # shift + scale
        elif x_block_self_attn:
            adaln_out_features = 9 * hidden_size  # 3 sets of (shift, scale, gate) for attn, attn2, mlp
        else:
            adaln_out_features = 6 * hidden_size  # 2 sets of (shift, scale, gate) for attn, mlp

        self.adaln_linear = Linear(
            in_features=hidden_size,
            out_features=adaln_out_features,
            mesh_device=mesh_device,
        )

        # Compute kernel config
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x, c):
        """
        Forward pass through the DismantledBlock.

        Args:
            x: [1, B, N, hidden_size] - input tensor
            c: [1, B, hidden_size] - conditioning tensor

        Returns:
            [1, B, N, hidden_size] - output tensor
        """
        # Apply SiLU activation to conditioning
        c = ttnn.silu(c, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # AdaLN modulation
        adaln_out = self.adaln_linear(c, compute_kernel_config=self.compute_kernel_config)

        if self.pre_only:
            return self._forward_pre_only(x, adaln_out)
        elif self.x_block_self_attn:
            return self._forward_dual_attention(x, adaln_out)
        else:
            return self._forward_standard(x, adaln_out)

    def _forward_standard(self, x, adaln_out):
        """Standard forward: attention + MLP"""
        B = x.shape[1]
        N = x.shape[2]

        # Split adaln_out into 6 chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        chunk_size = self.hidden_size
        shift_msa = adaln_out[:, :, :chunk_size]
        scale_msa = adaln_out[:, :, chunk_size : 2 * chunk_size]
        gate_msa = adaln_out[:, :, 2 * chunk_size : 3 * chunk_size]
        shift_mlp = adaln_out[:, :, 3 * chunk_size : 4 * chunk_size]
        scale_mlp = adaln_out[:, :, 4 * chunk_size : 5 * chunk_size]
        gate_mlp = adaln_out[:, :, 5 * chunk_size : 6 * chunk_size]

        # Attention path
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)

        # Attention forward
        attn_out = self.attn(x_mod, N)

        # Apply gating and residual
        gate_msa_expanded = ttnn.reshape(gate_msa, (1, B, 1, self.hidden_size))
        gate_msa_expanded = ttnn.repeat(gate_msa_expanded, ttnn.Shape([1, 1, N, 1]))
        x = ttnn.add(x, ttnn.multiply(gate_msa_expanded, attn_out))

        # MLP path
        x_norm_mlp = self.norm2(x)
        x_mod_mlp = modulate(x_norm_mlp, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod_mlp)

        # Apply gating and residual
        gate_mlp_expanded = ttnn.reshape(gate_mlp, (1, B, 1, self.hidden_size))
        gate_mlp_expanded = ttnn.repeat(gate_mlp_expanded, ttnn.Shape([1, 1, N, 1]))
        x = ttnn.add(x, ttnn.multiply(gate_mlp_expanded, mlp_out))

        return x

    def _forward_dual_attention(self, x, adaln_out):
        """Dual attention forward: attn + attn2 + MLP"""
        B = x.shape[1]
        N = x.shape[2]

        # Split adaln_out into 9 chunks
        chunk_size = self.hidden_size
        shift_msa = adaln_out[:, :, :chunk_size]
        scale_msa = adaln_out[:, :, chunk_size : 2 * chunk_size]
        gate_msa = adaln_out[:, :, 2 * chunk_size : 3 * chunk_size]
        shift_msa2 = adaln_out[:, :, 3 * chunk_size : 4 * chunk_size]
        scale_msa2 = adaln_out[:, :, 4 * chunk_size : 5 * chunk_size]
        gate_msa2 = adaln_out[:, :, 5 * chunk_size : 6 * chunk_size]
        shift_mlp = adaln_out[:, :, 6 * chunk_size : 7 * chunk_size]
        scale_mlp = adaln_out[:, :, 7 * chunk_size : 8 * chunk_size]
        gate_mlp = adaln_out[:, :, 8 * chunk_size : 9 * chunk_size]

        # First attention
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        attn_out = self.attn(x_mod, N)

        gate_msa_expanded = ttnn.reshape(gate_msa, (1, B, 1, self.hidden_size))
        gate_msa_expanded = ttnn.repeat(gate_msa_expanded, ttnn.Shape([1, 1, N, 1]))
        x = ttnn.add(x, ttnn.multiply(gate_msa_expanded, attn_out))

        # Second attention
        x_mod2 = modulate(x_norm, shift_msa2, scale_msa2)
        attn2_out = self.attn2(x_mod2, N)

        gate_msa2_expanded = ttnn.reshape(gate_msa2, (1, B, 1, self.hidden_size))
        gate_msa2_expanded = ttnn.repeat(gate_msa2_expanded, ttnn.Shape([1, 1, N, 1]))
        x = ttnn.add(x, ttnn.multiply(gate_msa2_expanded, attn2_out))

        # MLP path
        x_norm_mlp = self.norm2(x)
        x_mod_mlp = modulate(x_norm_mlp, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod_mlp)

        gate_mlp_expanded = ttnn.reshape(gate_mlp, (1, B, 1, self.hidden_size))
        gate_mlp_expanded = ttnn.repeat(gate_mlp_expanded, ttnn.Shape([1, 1, N, 1]))
        x = ttnn.add(x, ttnn.multiply(gate_mlp_expanded, mlp_out))

        return x

    def _forward_pre_only(self, x, adaln_out):
        """Pre-only mode: just normalization and modulation"""
        if not self.scale_mod_only:
            chunk_size = self.hidden_size
            shift_msa = adaln_out[:, :, :chunk_size]
            scale_msa = adaln_out[:, :, chunk_size : 2 * chunk_size]
        else:
            shift_msa = None
            scale_msa = adaln_out

        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)

        return x_mod

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict"""
        self.norm1.load_torch_state_dict(substate(state_dict, "norm1"))
        self.attn.load_state_dict(substate(state_dict, "attn"))

        if self.x_block_self_attn:
            self.attn2.load_state_dict(substate(state_dict, "attn2"))

        if not self.pre_only:
            self.norm2.load_torch_state_dict(substate(state_dict, "norm2"))
            self.mlp.load_torch_state_dict(substate(state_dict, "mlp"))

        # Load adaLN modulation linear layer (skip SiLU, load only the Linear at index 1)
        self.adaln_linear.load_torch_state_dict(substate(state_dict, "adaLN_modulation.1"))
