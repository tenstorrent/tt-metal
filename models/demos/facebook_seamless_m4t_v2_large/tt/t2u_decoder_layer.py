# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 Text-to-Unit decoder layer.

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::t2u_decoder_layer_forward``,
which reproduces one full ``SeamlessM4Tv2TextToUnitDecoderLayer`` -- one layer
of the **non-autoregressive** Text-to-Unit decoder used by SeamlessM4T-v2's
speech generation path.

Unlike the NLLB text decoder this layer:

  1. Uses **post-norm** residual placement: LayerNorm AFTER ``residual + sub_layer``
     (NOT pre-norm like ``text_decoder_layer``/``text_encoder_layer``).
  2. Replaces the FFN with a **two-Conv1d + ReLU** "conv" branch. Both Conv1d
     layers have ``kernel_size = 7`` and ``padding = "same"`` (symmetric pad=3
     since 7 is odd). Both have bias.
  3. Skips cross-attention entirely (the T2U decoder is NAR and conditions on
     encoder hidden states upstream via character expansion).

Op sequence (matches HF exactly)::

    residual = x
    x = self_attn(x, attention_mask=attention_mask)   # BART-style MHA (no causal mask inside)
    x = residual + x
    x = self_attn_layer_norm(x)                       # POST-NORM

    residual = x
    if padding_mask is not None:
        x = x * padding_mask                          # zero pad positions
    x = conv1(x.transpose(1, 2)).transpose(1, 2)      # Conv1d k=7, padding='same'
    if padding_mask is not None:
        x = x * padding_mask
    x = ReLU(x)
    x = conv2(x.transpose(1, 2)).transpose(1, 2)      # Conv1d k=7, padding='same'
    x = residual + x
    x = conv_layer_norm(x)                            # POST-NORM

    return x

This block is implemented as a thin composition over the already-verified
TTNN leaf modules :class:`LayerNorm` and :class:`SeamlessMHA`, plus inline
``ttnn.conv1d`` calls (k=7, symmetric pad=3, NOT causal -- this is the
bidirectional NAR T2U decoder, not the conformer convolution module).

For SeamlessM4T-v2-Large the relevant config values are:
    - hidden_size = 1024
    - decoder_attention_heads = 16  (head_dim = 64)
    - activation_function = "relu"
    - layer_norm_eps = 1e-5
    - conv_kernel_size = 7 (hardcoded in the HF class)
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm
from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_mha import SeamlessMHA


class T2UDecoderLayer(LightweightModule):
    """One full SeamlessM4T-v2 Text-to-Unit decoder layer (POST-norm, conv FFN) in TTNN.

    Args:
        device: ttnn device.
        embed_dim: hidden_size (1024 for v2-Large).
        num_heads: number of attention heads (16 for v2-Large).
        head_dim: per-head dim (64 for v2-Large). Requires
            ``num_heads * head_dim == embed_dim``.
        state_dict: nested mapping with keys
            ``{"self_attn", "self_attn_layer_norm", "conv1", "conv2",
                "conv_layer_norm"}`` matching the layout produced by
            ``_extract_t2u_decoder_layer_state_dict`` in the reference test.
        conv_kernel_size: Conv1d kernel size (default 7, hardcoded in HF).
            Must be odd for the symmetric ``padding='same'`` mapping.
        eps: LayerNorm epsilon (1e-5).
        weight_dtype: storage dtype for all sub-block weights/biases.
        weight_memory_config: where to place weights (default DRAM) for the
            sub-blocks; the conv1/conv2 weights are kept in row-major as
            required by ttnn.conv1d (it preprocesses them on first call).
    """

    def __init__(
        self,
        device,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        state_dict,
        conv_kernel_size: int = 7,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != embed_dim:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
        if conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd for padding='same', got {conv_kernel_size}")
        self.device = device
        self.embed_dim = embed_dim
        self.eps = float(eps)
        self.conv_kernel_size = int(conv_kernel_size)
        self.pad = self.conv_kernel_size // 2

        # 1. Self-attention (BART-style 4-proj MHA with bias). T2U self-attn is
        #    bidirectional by default; the caller supplies any additive log-mask
        #    via attention_mask (e.g. a triangular causal mask or padding mask).
        self.self_attn = SeamlessMHA(
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            state_dict=state_dict["self_attn"],
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. Post-self-attention LayerNorm.
        self.self_attn_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["self_attn_layer_norm"]["weight"],
            bias=state_dict["self_attn_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 3. Conv1 (k=7, padding='same', bias=True). HF weight shape is
        #    (embed_dim, embed_dim, K) -- already in the [out, in_per_group, K]
        #    layout ttnn.conv1d expects, so no reshape needed.
        conv1_w = state_dict["conv1"]["weight"]
        conv1_b = state_dict["conv1"]["bias"]
        assert int(conv1_w.shape[0]) == embed_dim, conv1_w.shape
        assert int(conv1_w.shape[1]) == embed_dim, conv1_w.shape
        assert int(conv1_w.shape[2]) == self.conv_kernel_size, conv1_w.shape
        assert tuple(conv1_b.shape) == (embed_dim,), conv1_b.shape
        self.conv1_weight = ttnn.from_torch(
            conv1_w,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # Bias must be [1, 1, 1, out_C] in row-major for ttnn.conv1d.
        self.conv1_bias = ttnn.from_torch(
            conv1_b.reshape(1, 1, 1, embed_dim),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # 4. Conv2 (k=7, padding='same', bias=True). Same shape as conv1.
        conv2_w = state_dict["conv2"]["weight"]
        conv2_b = state_dict["conv2"]["bias"]
        assert int(conv2_w.shape[0]) == embed_dim, conv2_w.shape
        assert int(conv2_w.shape[1]) == embed_dim, conv2_w.shape
        assert int(conv2_w.shape[2]) == self.conv_kernel_size, conv2_w.shape
        assert tuple(conv2_b.shape) == (embed_dim,), conv2_b.shape
        self.conv2_weight = ttnn.from_torch(
            conv2_w,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.conv2_bias = ttnn.from_torch(
            conv2_b.reshape(1, 1, 1, embed_dim),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # 5. Post-conv LayerNorm.
        self.conv_layer_norm = LayerNorm(
            device=device,
            dim=embed_dim,
            weight=state_dict["conv_layer_norm"]["weight"],
            bias=state_dict["conv_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # Conv1d compute / config: HiFi4 + fp32 dest accum, matching the other
        # conv-based blocks in this model (variance_predictor,
        # conformer_convolution_module). Explicit BLOCK_SHARDED requested for
        # the production NAR shape ([B=1, T>=32, C=1024]): C=1024 is tile-
        # aligned across an 8-column grid (128 channels/col), satisfying the
        # block-sharded conv kernel's "C >= 256 and per-col tile-multiple"
        # requirement. This avoids the auto-pick falling back to a slower
        # height-sharded path on the wider-than-AR sequences used by the T2U
        # decoder.
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weight_dtype,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=False,
        )
        self.conv_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _conv1d_block(
        self,
        x: ttnn.Tensor,
        weight_attr: str,
        bias_attr: str,
        batch: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """Run a single Conv1d (k=conv_kernel_size, symmetric pad) on a ``[B, T, C]`` input.

        Internally moves to ROW_MAJOR ``[B, 1, T, C]`` NHWC for ``ttnn.conv1d``,
        and returns the conv output reshaped back to ``[B, T, C]`` TILE_LAYOUT.
        """
        x_rm = x
        if x_rm.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_rm = ttnn.to_layout(x_rm, ttnn.ROW_MAJOR_LAYOUT)
        # [B, T, C] -> [B, 1, T, C] (NHWC with H=1).
        x_rm = ttnn.reshape(x_rm, (batch, 1, seq_len, self.embed_dim))

        weight_tt = getattr(self, weight_attr)
        bias_tt = getattr(self, bias_attr)
        out, _out_len, [new_w, new_b] = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=weight_tt,
            device=self.device,
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            batch_size=batch,
            input_length=seq_len,
            kernel_size=self.conv_kernel_size,
            stride=1,
            padding=[self.pad, self.pad],  # symmetric pad (NOT causal)
            dilation=1,
            groups=1,
            bias_tensor=bias_tt,
            conv_config=self.conv_config,
            compute_config=self.conv_compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        # Cache the prepared weight + bias for subsequent calls.
        setattr(self, weight_attr, new_w)
        setattr(self, bias_attr, new_b)
        ttnn.deallocate(x_rm)

        # ttnn.conv1d returns [1, 1, B*T, out_C] flattened NHWC. Move to TILE
        # layout and reshape back to [B, T, C] for downstream LayerNorm/ReLU.
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.reshape(out, (batch, seq_len, self.embed_dim))
        return out

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        padding_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run one T2U decoder layer.

        Args:
            hidden_states: ttnn tensor of shape ``[B, T, embed_dim]`` in
                TILE_LAYOUT.
            attention_mask: optional ttnn tensor broadcastable to ``[B, 1, T, T]``
                representing an additive log-mask for the self-attention. Pass
                a triangular ``-inf`` mask here for causal behaviour.
            padding_mask: optional ttnn tensor broadcastable to ``[B, T, 1]``
                where 0.0 marks padded positions to zero out around the conv
                branch (1.0 means "keep"). The caller is responsible for
                converting the boolean HF mask to this float form.

        Returns:
            ttnn tensor of shape ``[B, T, embed_dim]``.
        """
        mc_dram = ttnn.DRAM_MEMORY_CONFIG
        batch = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])

        # 1. Self-attention residual (POST-norm).
        residual = hidden_states
        x = self.self_attn(hidden_states, encoder_hidden_states=None, attention_mask=attention_mask)
        x = ttnn.add(x, residual, memory_config=mc_dram)
        ttnn.deallocate(residual)
        x = self.self_attn_layer_norm(x)

        # 2. Conv branch residual (POST-norm). Replaces the standard FFN.
        residual = x

        # Pre-conv1 padding-mask zeroing.
        y = x
        if padding_mask is not None:
            y = ttnn.multiply(y, padding_mask, memory_config=mc_dram)

        # Conv1 (k=7, padding=3 symmetric).
        y = self._conv1d_block(
            y,
            weight_attr="conv1_weight",
            bias_attr="conv1_bias",
            batch=batch,
            seq_len=seq_len,
        )

        # Pre-activation padding-mask zeroing (matches HF: AFTER conv1, BEFORE ReLU/conv2).
        if padding_mask is not None:
            y = ttnn.multiply(y, padding_mask, memory_config=mc_dram)

        # ReLU.
        y = ttnn.relu(y, memory_config=mc_dram)

        # Conv2 (k=7, padding=3 symmetric).
        y = self._conv1d_block(
            y,
            weight_attr="conv2_weight",
            bias_attr="conv2_bias",
            batch=batch,
            seq_len=seq_len,
        )

        # conv_dropout is a no-op at eval -> skipped.

        out = ttnn.add(y, residual, memory_config=mc_dram)
        ttnn.deallocate(residual)
        ttnn.deallocate(y)
        out = self.conv_layer_norm(out)
        return out
