# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Dismantled Block Implementation
Modified to support both standard and joint block operations.
"""

import ttnn
from ...layers.linear import Linear
from ...layers.normalization import RMSNorm
from models.experimental.tt_dit.models.transformers.mlp_sd35_medium import SD35MediumMlp as TTNNMlp
from models.experimental.tt_dit.models.transformers.swiglu_sd35_medium import SD35MediumSwiGLU
from ...utils.substate import substate


class SD35MediumDismantledBlock:
    """Dismantled block for SD3.5 Medium with separate pre_attention methods."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pre_only: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: str = "rms",
        x_block_self_attn: bool = False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pre_only = pre_only
        self.scale_mod_only = scale_mod_only
        self.x_block_self_attn = x_block_self_attn
        self.mesh_device = mesh_device

        # Import attention here to avoid circular imports
        from .attention_sd35_medium import SD35MediumSelfAttention

        self.attn = SD35MediumSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pre_only=pre_only,
            qk_norm=qk_norm,
            mesh_device=mesh_device,
        )

        # Dual attention for x_block_self_attn mode
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.attn2 = SD35MediumSelfAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                pre_only=False,
                qk_norm=qk_norm,
                mesh_device=mesh_device,
            )

        # Norm layers
        self.norm1 = RMSNorm(
            embedding_dim=hidden_size,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
        )

        if not pre_only:
            self.norm2 = RMSNorm(
                embedding_dim=hidden_size,
                norm_eps=1e-6,
                norm_elementwise_affine=False,
                bias=False,
                mesh_device=mesh_device,
            )

            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            if swiglu:
                self.mlp = SD35MediumSwiGLU(
                    dim=hidden_size,
                    hidden_dim=mlp_hidden_dim,
                    multiple_of=256,
                    bias=False,
                    mesh_device=mesh_device,
                )
            else:
                self.mlp = TTNNMlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    mesh_device=mesh_device,
                )

        # AdaLN modulation
        if x_block_self_attn:
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1

        # AdaLN modulation: Sequential(SiLU(), Linear()) matching reference
        # First apply SiLU to context
        self.adaLN_modulation_silu = lambda c: ttnn.silu(c, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Then apply Linear
        self.adaLN_modulation = Linear(hidden_size, n_mods * hidden_size, bias=True, mesh_device=mesh_device)

        # Compute config
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def pre_attention(self, x, c):
        """Pre-attention: normalization, modulation, and full attention computation"""
        # Apply modulation
        if not self.scale_mod_only:
            # Apply SiLU first, then Linear (matching Sequential(SiLU(), Linear()))
            c_silu = self.adaLN_modulation_silu(c)
            modulation = self.adaLN_modulation(c_silu)
            if not self.pre_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ttnn.chunk(modulation, 6, dim=2)
            else:
                shift_msa, scale_msa = ttnn.chunk(modulation, 2, dim=2)
                shift_mlp = None
                gate_mlp = None

            # Unsqueeze for broadcasting: [1, B, hidden_size] -> [1, B, 1, hidden_size]
            scale_msa = ttnn.unsqueeze(scale_msa, 2)
            shift_msa = ttnn.unsqueeze(shift_msa, 2)
            if not self.pre_only:
                gate_msa = ttnn.unsqueeze(gate_msa, 2)
                shift_mlp = ttnn.unsqueeze(shift_mlp, 2)
                scale_mlp = ttnn.unsqueeze(scale_mlp, 2)
                gate_mlp = ttnn.unsqueeze(gate_mlp, 2)
            modulated_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        else:
            modulated_x = self.norm1(x)

        # Return full attention output for standard blocks
        seq_len = modulated_x.shape[2]
        attn_out = self.attn(modulated_x, seq_len)

        if not self.pre_only:
            return attn_out, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            return attn_out, None

    def pre_attention_qkv(self, x, c):
        """Pre-attention: return QKV tuples for joint attention"""
        # Apply modulation
        if not self.scale_mod_only:
            # Apply SiLU first, then Linear (matching Sequential(SiLU(), Linear()))
            c_silu = self.adaLN_modulation_silu(c)
            modulation = self.adaLN_modulation(c_silu)
            if not self.pre_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ttnn.chunk(modulation, 6, dim=2)
            else:
                shift_msa, scale_msa = ttnn.chunk(modulation, 2, dim=2)
                shift_mlp = None
                gate_mlp = None

            # Unsqueeze for broadcasting: [1, B, hidden_size] -> [1, B, 1, hidden_size]
            scale_msa = ttnn.unsqueeze(scale_msa, 2)
            shift_msa = ttnn.unsqueeze(shift_msa, 2)
            if not self.pre_only:
                gate_msa = ttnn.unsqueeze(gate_msa, 2)
                shift_mlp = ttnn.unsqueeze(shift_mlp, 2)
                scale_mlp = ttnn.unsqueeze(scale_mlp, 2)
                gate_mlp = ttnn.unsqueeze(gate_mlp, 2)
            modulated_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        else:
            modulated_x = self.norm1(x)

        # Get QKV tensors for joint attention
        _, B, seq_len, C = modulated_x.shape

        # Access the attention module's QKV projection directly
        qkv = self.attn.qkv(modulated_x)
        qkv = ttnn.reshape(qkv, (1, B, seq_len, 3, self.num_heads, self.attn.head_dim))
        q, k, v = ttnn.chunk(qkv, 3, dim=3)
        # Squeeze dimension 3 to get [1, B, seq_len, num_heads, head_dim]
        q = ttnn.squeeze(q, 3)
        k = ttnn.squeeze(k, 3)
        v = ttnn.squeeze(v, 3)

        # Apply RMSNorm if available
        if hasattr(self.attn, "ln_q") and self.attn.ln_q is not None:
            q = self.attn.ln_q(q)
        if hasattr(self.attn, "ln_k") and self.attn.ln_k is not None:
            k = self.attn.ln_k(k)

        # Reshape to [B, num_heads, seq_len, head_dim] format expected by joint SDPA
        # Current shape: [1, B, seq_len, num_heads, head_dim]
        # Target shape: [B, num_heads, seq_len, head_dim]
        # First squeeze dim 0, then permute dims 1 and 2
        q = ttnn.squeeze(q, 0)  # [B, seq_len, num_heads, head_dim]
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)

        # Permute to [B, num_heads, seq_len, head_dim]
        q = ttnn.permute(q, (0, 2, 1, 3))  # [B, num_heads, seq_len, head_dim]
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Remove padding on head_dim dimension (dim 3) for joint SDPA compatibility
        # Convert to torch, extract exact logical shape, then convert back
        q_torch = ttnn.to_torch(q)
        k_torch = ttnn.to_torch(k)
        v_torch = ttnn.to_torch(v)
        # Slice to exact logical shape (remove any padding on head_dim)
        q_torch = q_torch[:, :, :, : self.attn.head_dim]
        k_torch = k_torch[:, :, :, : self.attn.head_dim]
        v_torch = v_torch[:, :, :, : self.attn.head_dim]
        # Convert back and reshape with explicit logical and padded shapes matching
        logical_shape = ttnn.Shape([B, self.num_heads, seq_len, self.attn.head_dim])
        q = ttnn.from_torch(q_torch, dtype=q.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        k = ttnn.from_torch(k_torch, dtype=k.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        v = ttnn.from_torch(v_torch, dtype=v.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        # Reshape to explicitly set both logical and padded shapes to be the same (no padding)
        q = ttnn.reshape(q, logical_shape, logical_shape)
        k = ttnn.reshape(k, logical_shape, logical_shape)
        v = ttnn.reshape(v, logical_shape, logical_shape)

        if not self.pre_only:
            return (q, k, v), (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            return (q, k, v), None

    def pre_attention_x(self, x, c):
        """Pre-attention for dual attention mode (x_block_self_attn)"""
        assert self.x_block_self_attn

        # Apply SiLU first, then Linear (matching Sequential(SiLU(), Linear()))
        c_silu = self.adaLN_modulation_silu(c)
        modulation = self.adaLN_modulation(c_silu)
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
        ) = ttnn.chunk(modulation, 9, dim=2)

        x_norm = self.norm1(x)

        # Unsqueeze for broadcasting: [1, B, hidden_size] -> [1, B, 1, hidden_size]
        scale_msa = ttnn.unsqueeze(scale_msa, 2)
        shift_msa = ttnn.unsqueeze(shift_msa, 2)
        scale_msa2 = ttnn.unsqueeze(scale_msa2, 2)
        shift_msa2 = ttnn.unsqueeze(shift_msa2, 2)
        gate_msa = ttnn.unsqueeze(gate_msa, 2)
        shift_mlp = ttnn.unsqueeze(shift_mlp, 2)
        scale_mlp = ttnn.unsqueeze(scale_mlp, 2)
        gate_mlp = ttnn.unsqueeze(gate_mlp, 2)
        gate_msa2 = ttnn.unsqueeze(gate_msa2, 2)

        # First attention QKV
        modulated_x1 = x_norm * (1 + scale_msa) + shift_msa
        _, B, seq_len, C = modulated_x1.shape
        qkv1 = self.attn.qkv(modulated_x1)
        qkv1 = ttnn.reshape(qkv1, (1, B, seq_len, 3, self.num_heads, self.attn.head_dim))
        q1, k1, v1 = ttnn.chunk(qkv1, 3, dim=3)
        # Squeeze dimension 3 to get [1, B, seq_len, num_heads, head_dim]
        q1 = ttnn.squeeze(q1, 3)
        k1 = ttnn.squeeze(k1, 3)
        v1 = ttnn.squeeze(v1, 3)

        if hasattr(self.attn, "ln_q") and self.attn.ln_q is not None:
            q1 = self.attn.ln_q(q1)
        if hasattr(self.attn, "ln_k") and self.attn.ln_k is not None:
            k1 = self.attn.ln_k(k1)

        # Second attention QKV
        modulated_x2 = x_norm * (1 + scale_msa2) + shift_msa2
        qkv2 = self.attn2.qkv(modulated_x2)
        qkv2 = ttnn.reshape(qkv2, (1, B, seq_len, 3, self.num_heads, self.attn2.head_dim))
        q2, k2, v2 = ttnn.chunk(qkv2, 3, dim=3)
        # Squeeze dimension 3 to get [1, B, seq_len, num_heads, head_dim]
        q2 = ttnn.squeeze(q2, 3)
        k2 = ttnn.squeeze(k2, 3)
        v2 = ttnn.squeeze(v2, 3)

        if hasattr(self.attn2, "ln_q") and self.attn2.ln_q is not None:
            q2 = self.attn2.ln_q(q2)
        if hasattr(self.attn2, "ln_k") and self.attn2.ln_k is not None:
            k2 = self.attn2.ln_k(k2)

        # Compute attention outputs from QKV
        # Reshape to [B, seq_len, num_heads, head_dim]
        q1 = ttnn.reshape(q1, (B, seq_len, self.num_heads, self.attn.head_dim))
        k1 = ttnn.reshape(k1, (B, seq_len, self.num_heads, self.attn.head_dim))
        v1 = ttnn.reshape(v1, (B, seq_len, self.num_heads, self.attn.head_dim))

        # Transpose to [B, num_heads, seq_len, head_dim]
        q1 = ttnn.permute(q1, (0, 2, 1, 3))
        k1 = ttnn.permute(k1, (0, 2, 1, 3))
        v1 = ttnn.permute(v1, (0, 2, 1, 3))

        # Scaled dot-product attention
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.attn.core_grid,
            q_chunk_size=64,
            k_chunk_size=64,
        )

        attn_out1 = ttnn.transformer.scaled_dot_product_attention(
            q1,
            k1,
            v1,
            is_causal=False,
            scale=self.attn.scale,
            program_config=program_config,
            compute_kernel_config=self.attn.compute_kernel_config,
        )

        # Transpose back and reshape
        attn_out1 = ttnn.permute(attn_out1, (0, 2, 1, 3))
        attn_out1 = ttnn.reshape(attn_out1, (1, B, seq_len, self.attn.inner_dim))

        # Output projection
        attn_out1 = self.attn.proj(attn_out1)

        # Same for second attention
        q2 = ttnn.reshape(q2, (B, seq_len, self.num_heads, self.attn2.head_dim))
        k2 = ttnn.reshape(k2, (B, seq_len, self.num_heads, self.attn2.head_dim))
        v2 = ttnn.reshape(v2, (B, seq_len, self.num_heads, self.attn2.head_dim))

        q2 = ttnn.permute(q2, (0, 2, 1, 3))
        k2 = ttnn.permute(k2, (0, 2, 1, 3))
        v2 = ttnn.permute(v2, (0, 2, 1, 3))

        program_config2 = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.attn2.core_grid,
            q_chunk_size=64,
            k_chunk_size=64,
        )

        attn_out2 = ttnn.transformer.scaled_dot_product_attention(
            q2,
            k2,
            v2,
            is_causal=False,
            scale=self.attn2.scale,
            program_config=program_config2,
            compute_kernel_config=self.attn2.compute_kernel_config,
        )

        attn_out2 = ttnn.permute(attn_out2, (0, 2, 1, 3))
        attn_out2 = ttnn.reshape(attn_out2, (1, B, seq_len, self.attn2.inner_dim))
        attn_out2 = self.attn2.proj(attn_out2)

        return (
            attn_out1,
            attn_out2,
            (x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2),
        )

    def post_attention(self, attn_out, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        """Post-attention: apply gating, residual, and MLP"""
        assert not self.pre_only

        # Apply attention gating and residual
        x = x + gate_msa * attn_out

        # Apply MLP
        mlp_input = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp * mlp_out

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
    ):
        """Post-attention for dual attention mode"""
        assert not self.pre_only

        # Apply first attention
        x = x + gate_msa * attn_out

        # Apply second attention
        x = x + gate_msa2 * attn_out2

        # Apply MLP
        mlp_input = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(mlp_input)
        x = x + gate_mlp * mlp_out

        return x

    def __call__(self, x, c):
        """Forward pass for standard blocks"""
        if self.x_block_self_attn:
            attn_out, attn_out2, intermediates = self.pre_attention_x(x, c)
            return self.post_attention_x(attn_out, attn_out2, *intermediates)
        else:
            attn_out, intermediates = self.pre_attention(x, c)
            if self.pre_only:
                return attn_out
            else:
                return self.post_attention(attn_out, *intermediates)

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        # Handle attention weights
        if hasattr(self, "attn"):
            attn_state_dict = substate(state_dict, "attn")
            if not attn_state_dict:
                # Try alternative key patterns for SwiGLU
                attn_state_dict = {k.replace("self_attn.", ""): v for k, v in state_dict.items() if "self_attn." in k}
            self.attn.load_state_dict(attn_state_dict)

        # Handle second attention for x_block_self_attn mode
        if hasattr(self, "attn2"):
            attn2_state_dict = substate(state_dict, "attn2")
            if attn2_state_dict:
                self.attn2.load_state_dict(attn2_state_dict)

        # Load normalization weights
        self.norm1.load_torch_state_dict(substate(state_dict, "norm1"))

        if not self.pre_only:
            self.norm2.load_torch_state_dict(substate(state_dict, "norm2"))

            # Handle MLP weights
            if hasattr(self, "mlp"):
                mlp_state_dict = substate(state_dict, "mlp")
                # If no substate found, try extracting mlp.* keys directly
                if not mlp_state_dict:
                    mlp_state_dict = {}
                    for k, v in state_dict.items():
                        if "mlp." in k:
                            key = k.replace("mlp.", "")
                            mlp_state_dict[key] = v

                self.mlp.load_torch_state_dict(mlp_state_dict)

        # Handle AdaLN modulation weights
        # Reference model uses Sequential(SiLU(), Linear()), so keys are "1.weight" and "1.bias"
        adaLN_state_dict = substate(state_dict, "adaLN_modulation")
        if adaLN_state_dict:
            # Strip the "1." prefix if present (from Sequential)
            normalized_adaLN_state_dict = {}
            for k, v in adaLN_state_dict.items():
                if k.startswith("1."):
                    normalized_adaLN_state_dict[k[2:]] = v  # Remove "1." prefix
                else:
                    normalized_adaLN_state_dict[k] = v
            self.adaLN_modulation.load_torch_state_dict(normalized_adaLN_state_dict)
        else:
            self.adaLN_modulation.load_torch_state_dict({})
