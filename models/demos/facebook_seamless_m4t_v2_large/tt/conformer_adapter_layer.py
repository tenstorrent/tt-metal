# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Conformer adapter layer.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::conformer_adapter_layer_forward``,
which reproduces one full block of HuggingFace
``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2ConformerAdapterLayer``.

This adapter sits between the W2v-BERT-2.0 speech encoder stack and the rest
of the SeamlessM4T-v2 pipeline. Its purpose is to *down-sample* the temporal
axis by ``stride`` (=8) using a pair of strided Conv1d + GLU stacks (one for
the residual path and one for the self-attention path), and then apply a
standard MHA + FFN block on top of the down-sampled hidden states.

Op sequence (matches HF exactly):

    # 1. Residual path: LN -> transpose -> Conv1d(stride=s) -> GLU(dim=1) -> transpose
    residual = LN_residual(hidden_states)
    residual = Conv1d_residual(residual.transpose(1, 2)).transpose(1, 2 after GLU)
    residual = GLU(dim=1)(residual)              # halves channel dim back to C

    # 2. Main path: LN -> transpose -> Conv1d(stride=s) -> GLU(dim=1) -> transpose
    h = LN_self_attn(hidden_states)
    h = GLU(dim=1)(Conv1d_self_attn(h.transpose(1, 2)))

    # 3. Self-attention (no positional embeddings).
    h = self_attn(h, attention_mask=attention_mask) + residual

    # 4. Pre-norm FFN (ReLU activation, NOT swish).
    h = ffn(LN_ffn(h)) + h

Implementation notes (TTNN port):

* The two adapter Conv1d layers have ``hidden`` -> ``2 * hidden`` channels,
  kernel ``kernel_size`` (=8), stride ``stride`` (=8), and symmetric padding
  ``stride // 2`` (=4) on both sides. Bias is ``True``. Both are realised as
  ``ttnn.conv1d`` calls in NHWC ``[B, 1, T, C]`` layout (the same pattern used
  by :class:`ConformerConvolutionModule` and :class:`VariancePredictor`).
* The GLU after each conv runs along the channel dim. In TTNN we move the
  conv output (which is laid out as ``[B, 1, T_sub, 2C]`` NHWC) into a
  ``ttnn.glu(dim=-1)`` call, producing ``[B, 1, T_sub, C]`` -- mathematically
  identical to ``F.glu(dim=1)`` on the channel-first reference.
* The inner self-attention reuses :class:`ConformerSelfAttention` with
  ``position_embeddings_type=None`` (no relative-key bias). The reference
  passes ``distance_embedding_weight=None`` for the adapter, so we mirror
  that here.
* The post-attention FFN reuses :class:`ConformerFfn` with ``act_fn="relu"``
  (NOT the default swish used inside the encoder layers).
* Input ``hidden_states`` enters as ``[B, T, C]`` and the output is
  ``[B, T_sub, C]`` where ``T_sub = floor((T + 2 * (stride//2) - kernel) / stride) + 1``.
  For ``T=128, kernel=stride=8`` -> ``T_sub = 17`` (NOT 16).

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - speech_encoder_attention_heads = 16  (head_dim = 64)
    - adaptor_kernel_size = 8
    - adaptor_stride = 8
    - layer_norm_eps = 1e-5
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_ffn import ConformerFfn
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_self_attention import ConformerSelfAttention
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm


class ConformerAdapterLayer(LightweightModule):
    """SeamlessM4T-v2 Conformer adapter layer in TTNN.

    Args:
        device: ttnn device or mesh device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        seq_len: pre-downsample sequence length. The conv's strided output is
            computed at runtime; ``seq_len`` is only used to size the
            self-attention's positional bias table layout. Since the adapter
            uses ``position_embeddings_type=None``, no bias table is actually
            built, but ``ConformerSelfAttention`` still needs a non-zero value
            for its ``seq_len`` argument -- we pass the *post*-downsample
            length so that its internal reshape sizes are right at construct
            time (we recompute the actual T_sub during forward anyway).
        sub_seq_len: post-downsample sequence length (== output time dim of the
            strided conv). For ``T=128, kernel=stride=8`` this is 17.
        state_dict: nested mapping containing all sub-block weights as
            produced by ``_extract_adapter_layer_state_dict`` in the reference
            test -- keys ``residual_layer_norm``, ``residual_conv``,
            ``self_attn_layer_norm``, ``self_attn_conv``, ``self_attn``,
            ``ffn_layer_norm``, ``ffn``.
        kernel_size: Conv1d kernel size (default 8).
        stride: Conv1d stride (default 8). Output time dim is
            ``floor((seq_len + 2*(stride//2) - kernel_size) / stride) + 1``.
        eps: LayerNorm epsilon (default 1e-5).
        batch_size: forward batch size (default 1).
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM).
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        seq_len: int,
        sub_seq_len: int,
        state_dict,
        kernel_size: int = 8,
        stride: int = 8,
        eps: float = 1e-5,
        batch_size: int = 1,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        self.device = device
        self.embed_dim = int(embed_dim)
        self.eps = float(eps)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.pad = self.stride // 2  # symmetric padding on both sides
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.sub_seq_len = int(sub_seq_len)

        two_hidden = 2 * self.embed_dim
        # Sanity checks against the supplied state dict.
        assert tuple(state_dict["residual_conv"]["weight"].shape) == (two_hidden, embed_dim, self.kernel_size), (
            f"residual_conv weight shape {tuple(state_dict['residual_conv']['weight'].shape)} != "
            f"({two_hidden}, {embed_dim}, {self.kernel_size})"
        )
        assert tuple(state_dict["self_attn_conv"]["weight"].shape) == (two_hidden, embed_dim, self.kernel_size)

        # 1. Residual-branch LayerNorm.
        self.residual_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["residual_layer_norm"]["weight"],
            bias=state_dict["residual_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. Residual-branch strided Conv1d (hidden -> 2 * hidden) with bias.
        # ttnn.conv1d follows the PyTorch ``[out_C, in_C/groups, K]`` weight
        # convention, so no reshape is needed.
        self.residual_conv_weight = ttnn.from_torch(
            state_dict["residual_conv"]["weight"],
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.residual_conv_bias = ttnn.from_torch(
            state_dict["residual_conv"]["bias"].reshape(1, 1, 1, two_hidden),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # 3. Main-branch (self-attention input) LayerNorm.
        self.self_attn_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["self_attn_layer_norm"]["weight"],
            bias=state_dict["self_attn_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 4. Main-branch strided Conv1d (hidden -> 2 * hidden) with bias.
        self.self_attn_conv_weight = ttnn.from_torch(
            state_dict["self_attn_conv"]["weight"],
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.self_attn_conv_bias = ttnn.from_torch(
            state_dict["self_attn_conv"]["bias"].reshape(1, 1, 1, two_hidden),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # 5. Self-attention (NO positional embeddings -- distance_embedding=None).
        self.self_attn = ConformerSelfAttention(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=self.sub_seq_len,  # post-downsample T_sub for the attention
            state_dict=state_dict["self_attn"],
            distance_embedding_weight=None,
            position_embeddings_type=None,
            batch_size=self.batch_size,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 6. Pre-FFN LayerNorm.
        self.ffn_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["ffn_layer_norm"]["weight"],
            bias=state_dict["ffn_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 7. Post-attention FFN with ReLU (NOT swish).
        self.ffn = ConformerFfn(
            device=device,
            intermediate_weight=state_dict["ffn"]["intermediate_dense"]["weight"],
            intermediate_bias=state_dict["ffn"]["intermediate_dense"]["bias"],
            output_weight=state_dict["ffn"]["output_dense"]["weight"],
            output_bias=state_dict["ffn"]["output_dense"]["bias"],
            act_fn="relu",
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # Precision / compute config: HiFi4 + fp32 dest accum.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Conv1d configs (shared between both branches).
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weight_dtype,
            shard_layout=None,  # auto-pick
            deallocate_activation=False,
        )
        self.conv_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _strided_conv_glu(
        self,
        x: ttnn.Tensor,
        weight_attr: str,
        bias_attr: str,
        batch: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """LN'd ``[B, T, C]`` -> conv1d(stride) -> GLU -> ``[B, T_sub, C]``.

        Realises the channel-first reference op sequence

            x: (B, T, C)
            -> x.transpose(1, 2)              # (B, C, T)
            -> Conv1d(C, 2C, k=K, stride=S, padding=K//2)
            -> GLU(dim=1)                     # (B, C, T_sub)
            -> x.transpose(1, 2)              # (B, T_sub, C)

        without doing any channel-first transposes: instead we drive
        ``ttnn.conv1d`` in NHWC ``[B, 1, T, C]`` layout (which is the
        operator's required input format) and run ``ttnn.glu(dim=-1)``
        on its NHWC output ``[B, 1, T_sub, 2C]`` -- the resulting
        ``[B, 1, T_sub, C]`` tensor is then reshaped back to ``[B, T_sub, C]``.

        The output time dim ``T_sub`` is read from the conv's reported
        ``out_length`` (returned via ``return_output_dim=True``) so that
        floor-division semantics in the conv kernel match the host-side
        computation.
        """
        in_channels = self.embed_dim
        out_channels = 2 * self.embed_dim

        # Move to ROW_MAJOR ``[B, 1, T, C]`` NHWC for ttnn.conv1d.
        x_rm = x
        if x_rm.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_rm = ttnn.to_layout(x_rm, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (batch, 1, seq_len, in_channels))

        weight_tt = getattr(self, weight_attr)
        bias_tt = getattr(self, bias_attr)
        out, out_length, [new_w, new_b] = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=weight_tt,
            device=self.device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch,
            input_length=seq_len,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[self.pad, self.pad],
            dilation=1,
            groups=1,
            bias_tensor=bias_tt,
            conv_config=self.conv_config,
            compute_config=self.conv_compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        # Cache the prepared weight+bias for subsequent calls.
        setattr(self, weight_attr, new_w)
        setattr(self, bias_attr, new_b)
        ttnn.deallocate(x_rm)

        # ``out`` is NHWC ``[B, 1, T_sub, 2C]`` in TILE_LAYOUT, but T_sub
        # (=17 for the v2-Large adapter at T=128) is not a multiple of 32 so
        # the tile layout pads the time axis up to 32. Running ``ttnn.glu`` on
        # the tile-padded tensor would silently promote those padding rows to
        # real data and corrupt the time axis. Move to ROW_MAJOR first so the
        # GLU sees the true logical ``T_sub`` rows, then come back to TILE for
        # the downstream LayerNorm / matmul ops.
        t_sub = int(out_length)
        if out.layout != ttnn.ROW_MAJOR_LAYOUT:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)

        # GLU along the channel (last) dim. ``ttnn.glu(dim=-1)`` halves the
        # last dim (2C -> C), matching ``F.glu(dim=1)`` on the channel-first
        # reference.
        glu_out = ttnn.glu(out, dim=-1)
        ttnn.deallocate(out)

        # Drop the H=1 NHWC dim and return to TILE layout for the downstream
        # LayerNorm / matmul ops.
        glu_out_3d = ttnn.reshape(glu_out, (batch, t_sub, in_channels))
        ttnn.deallocate(glu_out)
        glu_out_3d = ttnn.to_layout(glu_out_3d, ttnn.TILE_LAYOUT)
        return glu_out_3d

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the Conformer adapter layer.

        Args:
            hidden_states: ttnn tensor of shape ``[B, T, embed_dim]`` in
                TILE_LAYOUT.
            attention_mask: optional ttnn tensor broadcastable to
                ``[B, 1, T_sub, T_sub]`` representing an additive log-mask
                computed by the caller on the *post-downsample* axis (the
                reference accepts the already-prepared 4D mask; see
                ``conformer_adapter_layer_forward`` docstring).

        Returns:
            ttnn tensor of shape ``[B, T_sub, embed_dim]`` in TILE_LAYOUT.
        """
        batch = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])

        # 1. Residual branch: LayerNorm -> strided Conv1d -> GLU.
        residual_pre = self.residual_layer_norm(hidden_states)
        residual = self._strided_conv_glu(
            residual_pre,
            "residual_conv_weight",
            "residual_conv_bias",
            batch,
            seq_len,
        )
        ttnn.deallocate(residual_pre)

        # 2. Main branch: LayerNorm -> strided Conv1d -> GLU.
        x_pre = self.self_attn_layer_norm(hidden_states)
        x = self._strided_conv_glu(
            x_pre,
            "self_attn_conv_weight",
            "self_attn_conv_bias",
            batch,
            seq_len,
        )
        ttnn.deallocate(x_pre)

        # 3. Self-attention (no positional bias) + residual.
        x_attn = self.self_attn(x, attention_mask=attention_mask)
        ttnn.deallocate(x)
        x_attn = ttnn.add(x_attn, residual)
        ttnn.deallocate(residual)

        # 4. Pre-norm FFN with ReLU + residual.
        ffn_pre = self.ffn_layer_norm(x_attn)
        ffn_out = self.ffn(ffn_pre)
        ttnn.deallocate(ffn_pre)
        out = ttnn.add(ffn_out, x_attn)
        ttnn.deallocate(ffn_out)
        ttnn.deallocate(x_attn)
        return out
