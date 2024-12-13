# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional
import torch.nn.functional as F
import torch

from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_ada_layernorm_continuous import (
    ttnn_AdaLayerNormContinuous,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_ada_layernorm_zero import ttnn_AdaLayerNormZero
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_sd35_ada_layernorm_zerox import (
    ttnn_SD35AdaLayerNormZeroX,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_feed_forward import ttnn_FeedForward
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_attention import (
    ttnn_Attention,
    ttnn_JointAttnProcessor2_0,
)


class ttnn_JointTransformerBlock:
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        parameters=None,
    ):
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        if use_dual_attention:
            self.norm1 = ttnn_SD35AdaLayerNormZeroX(dim)
        else:
            self.norm1 = ttnn_AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = ttnn_AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = ttnn_AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        if hasattr(F, "scaled_dot_product_attention"):
            processor = ttnn_JointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = ttnn_Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
            parameters=parameters["attn"],
        )

        if use_dual_attention:
            self.attn2 = ttnn_Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=1e-6,
                parameters=parameters["attn2"],
            )
        else:
            self.attn2 = None

        self.norm2 = ttnn.layer_norm  # eps=1e-6
        self.ff = ttnn_FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = ttnn.layer_norm  # eps=1e-6
            self.ff_context = ttnn_FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def __call__(
        self, hidden_states: ttnn.Tensor, encoder_hidden_states: ttnn.Tensor, temb: ttnn.Tensor, parameters=None
    ):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb, parameters=parameters["norm1"]
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb, parameters=parameters["norm1"]
            )

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(
                encoder_hidden_states, temb, parameters=parameters["norm1_context"]
            )
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb, parameters=parameters["norm1_context"]
            )

        # Attention.

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            device=norm_hidden_states.device(),
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = ttnn.unsqueeze(gate_msa, 1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2, device=norm_hidden_states2.device())

            attn_output2 = ttnn.unsqueeze(gate_msa2, 1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states, epsilon=1e-6)
        norm_hidden_states = norm_hidden_states * (
            1 + ttnn.reshape(scale_mlp, (scale_mlp.shape[0], 1, scale_mlp.shape[1]))
        ) + ttnn.reshape(
            shift_mlp, (shift_mlp.shape[0], 1, shift_mlp.shape[1])
        )  # scale_mlp[:, None],shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states, parameters=parameters["ff"])
        ff_output = ttnn.unsqueeze(gate_mlp, 1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = ttnn.unsqueeze(c_gate_msa, 1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states, epsilon=1e-6)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (
                1 + ttnn.reshape(c_scale_mlp, (c_scale_mlp.shape[0], 1, c_scale_mlp.shape[1]))
            ) + ttnn.reshape(
                c_shift_mlp, (c_shift_mlp.shape[0], 1, c_shift_mlp.shape[1])
            )  # c_scale_mlp[:, None],c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states, parameters=parameters["ff_context"])
            encoder_hidden_states = encoder_hidden_states + ttnn.unsqueeze(c_gate_mlp, 1) * context_ff_output

        return encoder_hidden_states, hidden_states
