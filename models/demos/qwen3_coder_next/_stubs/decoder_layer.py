# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextDecoderLayer`, tensor-parallel over TP chips.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextDecoderLayer`.

    residual = x
    x = input_layernorm(x)
    x = linear_attn(x)          # or self_attn(x), depending on config.layer_types[layer_idx]
    x = residual + x
    residual = x
    x = post_attention_layernorm(x)
    x = mlp(x)                  # Qwen3NextSparseMoeBlock, or a dense Qwen3NextMLP
    x = residual + x

The layer is a composite, so the tensor-parallel scheme is its parts' schemes: the token mixer and
the MLP each end in a collective that restores the full model dim, and the pieces BETWEEN them --
the two RMSNorms and both residual adds -- are per-element over the model dim and therefore
REPLICATED. That is what makes the composition valid: every chip re-enters the next sub-block
holding the identical full-width activation, so no extra collective is needed at the seams.

The sub-modules are the sibling stubs, not copies:
  * `_stubs/gated_delta_net.py`     -- head-sharded delta net, all_reduce on out_proj
  * `_stubs/attention.py`           -- head-sharded GQA attention, all_reduce on o_proj
  * `_stubs/sparse_moe_block.py`    -- expert-parallel MoE, all_reduce on the down projection
  * `_stubs/r_m_s_norm.py`          -- BOTH layernorms; replicated, no collective
"""
from __future__ import annotations

import ttnn

from models.demos.qwen3_coder_next._stubs.attention import TtQwen3NextAttention
from models.demos.qwen3_coder_next._stubs.gated_delta_net import TtQwen3NextGatedDeltaNet
from models.demos.qwen3_coder_next._stubs.m_l_p import TtQwen3NextMLP
from models.demos.qwen3_coder_next._stubs.r_m_s_norm import TtQwen3NextRMSNorm
from models.demos.qwen3_coder_next._stubs.sparse_moe_block import TtQwen3NextSparseMoeBlock


class TtQwen3NextDecoderLayer:
    """Native ttnn Qwen3-Next decoder layer: token mixer + MLP, both tensor-parallel."""

    def __init__(self, device, *, mixer, mlp, input_layernorm, post_attention_layernorm, hidden_size) -> None:
        self.device = device
        self.mixer = mixer
        self.mlp = mlp
        # Both norms ARE the graduated `r_m_s_norm` stub (the port of `layers.*.input_layernorm`),
        # which folds HF's `(1.0 + weight)` in at build time exactly as this layer used to inline.
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.hidden_size = hidden_size
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("decoder_layer stub needs the torch reference module for its weights")

        if hasattr(torch_module, "linear_attn"):
            mixer = TtQwen3NextGatedDeltaNet.build(device, torch_module.linear_attn)
        elif hasattr(torch_module, "self_attn"):
            mixer = TtQwen3NextAttention.build(device, torch_module.self_attn)
        else:
            raise RuntimeError("decoder layer exposes neither `linear_attn` nor `self_attn`")

        torch_mlp = torch_module.mlp
        if hasattr(torch_mlp, "experts"):
            mlp = TtQwen3NextSparseMoeBlock.build(device, torch_mlp)
        else:
            mlp = TtQwen3NextMLP.build(device, torch_mlp)

        return cls(
            device,
            mixer=mixer,
            mlp=mlp,
            input_layernorm=TtQwen3NextRMSNorm.build(device, torch_module.input_layernorm),
            post_attention_layernorm=TtQwen3NextRMSNorm.build(device, torch_module.post_attention_layernorm),
            hidden_size=int(torch_module.hidden_size),
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        seq = int(hidden_states.shape[-2])
        residual = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))

        x = self.input_layernorm(residual)
        mixed = self.mixer(
            x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        residual = ttnn.add(residual, ttnn.reshape(mixed, (1, 1, seq, self.hidden_size)))

        x = self.post_attention_layernorm(residual)
        out = self.mlp(x)
        if isinstance(out, tuple):
            out = out[0]
        out = ttnn.add(residual, ttnn.reshape(out, (1, 1, seq, self.hidden_size)))
        return ttnn.reshape(out, (1, seq, self.hidden_size))


def build(device, torch_module=None):
    return TtQwen3NextDecoderLayer.build(device, torch_module)


def decoder_layer(device, torch_module=None):
    return TtQwen3NextDecoderLayer.build(device, torch_module)
