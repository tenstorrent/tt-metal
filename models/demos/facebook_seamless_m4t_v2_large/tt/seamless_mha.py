# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the base SeamlessM4Tv2 multi-head attention block.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::seamless_mha_forward``,
which is the BART-style 4-projection MHA (q_proj/k_proj/v_proj/out_proj all
with bias=True) used throughout SeamlessM4T-v2 (NLLB text encoder/decoder,
T2U encoder/decoder).

The forward supports BOTH self-attention (encoder_hidden_states=None) and
cross-attention (encoder_hidden_states is the encoder output that K/V are
projected from).

This port follows the standard MHA pattern using ttnn.linear for the four
projections and ttnn.transformer.scaled_dot_product_attention for the
QK^T -> scale -> softmax -> @V kernel. The Whisper-decoder attention in
``models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py`` is
the closest reference.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class SeamlessMHA(LightweightModule):
    """SeamlessM4Tv2 base multi-head attention (BART-style, 4 projections, bias=True).

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for seamless-m4t-v2-large).
        num_heads: number of attention heads (16 for large).
        head_dim: per-head dim, must satisfy num_heads * head_dim == embed_dim.
        state_dict: nested mapping with q_proj/k_proj/v_proj/out_proj ->
            {"weight": torch.Tensor, "bias": torch.Tensor}. weight shape is
            (embed_dim, embed_dim), bias shape (embed_dim,). All four
            projections have bias for SeamlessM4T-v2.
        weight_dtype: storage dtype for the four projection weights/biases.
        weight_memory_config: where to place projection weights/biases.
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        state_dict,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        # Linear weight in HF is (out, in); ttnn.linear expects (in, out), so
        # transpose on load. Biases are 1D.
        def _load_proj(name):
            w = state_dict[name]["weight"]
            b = state_dict[name].get("bias")
            w_tt = ttnn.from_torch(
                w.transpose(0, 1).contiguous(),
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
            )
            b_tt = None
            if b is not None:
                b_tt = ttnn.from_torch(
                    b.reshape(1, -1),
                    device=device,
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=weight_memory_config,
                )
            return w_tt, b_tt

        self.q_weight, self.q_bias = _load_proj("q_proj")
        self.k_weight, self.k_bias = _load_proj("k_proj")
        self.v_weight, self.v_bias = _load_proj("v_proj")
        self.o_weight, self.o_bias = _load_proj("out_proj")

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _project_and_split(self, x: ttnn.Tensor, weight, bias, batch, seq_len):
        """Project x with linear(weight,bias), reshape to [B, num_heads, S, head_dim]."""
        proj = ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        # proj is [B, S, embed_dim]. Reshape -> [B, S, num_heads, head_dim] -> transpose to [B, num_heads, S, head_dim].
        proj = ttnn.reshape(proj, (batch, seq_len, self.num_heads, self.head_dim))
        proj = ttnn.transpose(proj, 1, 2)
        return proj

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor = None,
        attention_mask: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """Run MHA. hidden_states is [B, T, embed_dim]; encoder is [B, S, embed_dim] or None.

        attention_mask, if provided, is an additive log-mask broadcast-compatible
        with [B, 1, T, S].
        Returns: [B, T, embed_dim].
        """
        # hidden_states / encoder_hidden_states are TILE_LAYOUT, DRAM, [B, X, embed_dim].
        batch = hidden_states.shape[0]
        tgt_len = hidden_states.shape[1]

        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        src_len = current_states.shape[1]

        # Q from hidden_states (always); K, V from current_states (encoder for cross-attn).
        q = self._project_and_split(hidden_states, self.q_weight, self.q_bias, batch, tgt_len)
        k = self._project_and_split(current_states, self.k_weight, self.k_bias, batch, src_len)
        v = self._project_and_split(current_states, self.v_weight, self.v_bias, batch, src_len)

        # Use the fused QK^T -> scale -> softmax -> @V kernel.
        sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            attn_mask=attention_mask,
            is_causal=False,
            compute_kernel_config=sdpa_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # attn_output: [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, embed_dim]
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (batch, tgt_len, self.embed_dim))

        # Output projection.
        out = ttnn.linear(
            attn_output,
            self.o_weight,
            bias=self.o_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_output)
        return out
