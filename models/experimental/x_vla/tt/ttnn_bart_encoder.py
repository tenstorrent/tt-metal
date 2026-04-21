# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of the 12-layer Florence-2 BART encoder used by X-VLA.

This is the language-and-vision encoder that consumes the merged
text+image embeddings from `XVLAModel.forward_vlm` and produces
`vlm_features` for the SoftPromptedTransformer.

Each Florence2EncoderLayer is **post-LN** BART-style:

    x = x + self_attn(x, mask)
    x = self_attn_layer_norm(x)
    x = x + ffn(x)
    x = final_layer_norm(x)

Differences from the SoftPromptedTransformer block stack we already ported:
  - Post-LN, not pre-LN — LayerNorm AFTER residual.
  - Q, K, V come as three separate Linear projections in HF's BART; we
    fuse them into one [D, 3D] matmul on the device side and then
    `split_query_key_value_and_split_heads`.
  - Attention mask is non-trivial (padded language tokens get -inf added
    to their score columns). We honor it via an `add(scores, mask)`
    before softmax.
  - inputs_embeds + learned positional embeddings + layernorm_embedding
    happens before the layers.

The encoder is called as
`encoder(attention_mask=..., inputs_embeds=...)[0]`, so we return a
`BaseModelOutput` for type compatibility (subscriptable as a tuple).
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


def _bf16_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat16, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


def _bfp8_tile(ttnn_mod, t: torch.Tensor, device):
    return ttnn_mod.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn_mod.bfloat8_b, layout=ttnn_mod.TILE_LAYOUT, device=device,
    )


class TTNNBartEncoder(nn.Module):
    """Drop-in replacement for `Florence2Encoder`.

    All 12 BART encoder layers + the embedding LayerNorm run on the
    Blackhole device. Embeddings (token / positional) stay on torch CPU
    because they are integer-indexed lookups of small tables.
    """

    def __init__(self, torch_encoder: nn.Module, device) -> None:
        super().__init__()
        import ttnn

        self._ttnn = ttnn
        self.device = device
        self.config = torch_encoder.config
        self.embed_dim = self.config.d_model
        self.num_heads = self.config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.head_dim_inv_sqrt = float(self.head_dim) ** -0.5

        # Keep token + positional embeddings on torch — they are tiny lookups.
        self.embed_tokens = torch_encoder.embed_tokens
        self.embed_positions = torch_encoder.embed_positions
        self.padding_idx = torch_encoder.padding_idx

        # Embedding LayerNorm on device.
        self.ln_emb_w = _bf16_tile(ttnn, torch_encoder.layernorm_embedding.weight.detach(), device)
        self.ln_emb_b = _bf16_tile(ttnn, torch_encoder.layernorm_embedding.bias.detach(), device)

        # Per-layer weight bundles (12 of them).
        self._layers: List[dict] = [self._bundle_layer(layer) for layer in torch_encoder.layers]

    # -- weight loading ------------------------------------------------------

    def _bundle_layer(self, layer: nn.Module) -> dict:
        ttnn = self._ttnn
        dev = self.device
        d = self.embed_dim
        attn = layer.self_attn

        # Fuse Q/K/V: three [d, d] torch weights -> one [d, 3d] device weight.
        q_w = attn.q_proj.weight.detach()
        k_w = attn.k_proj.weight.detach()
        v_w = attn.v_proj.weight.detach()
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0).t().contiguous()
        q_b = attn.q_proj.bias.detach() if attn.q_proj.bias is not None else torch.zeros(d)
        k_b = attn.k_proj.bias.detach() if attn.k_proj.bias is not None else torch.zeros(d)
        v_b = attn.v_proj.bias.detach() if attn.v_proj.bias is not None else torch.zeros(d)
        qkv_b = torch.cat([q_b, k_b, v_b])

        out_w = attn.out_proj.weight.detach().t().contiguous()
        out_b = attn.out_proj.bias.detach() if attn.out_proj.bias is not None else torch.zeros(d)

        return dict(
            qkv_w=_bf16_tile(ttnn, qkv_w, dev),
            qkv_b=_bf16_tile(ttnn, qkv_b, dev),
            out_w=_bf16_tile(ttnn, out_w, dev),
            out_b=_bf16_tile(ttnn, out_b, dev),
            sa_ln_w=_bf16_tile(ttnn, layer.self_attn_layer_norm.weight.detach(), dev),
            sa_ln_b=_bf16_tile(ttnn, layer.self_attn_layer_norm.bias.detach(), dev),
            # MLP weights are the biggest in the layer (1024 x 4096) -> bfp8
            fc1_w=_bfp8_tile(ttnn, layer.fc1.weight.detach().t().contiguous(), dev),
            fc1_b=_bf16_tile(ttnn, layer.fc1.bias.detach(), dev),
            fc2_w=_bfp8_tile(ttnn, layer.fc2.weight.detach().t().contiguous(), dev),
            fc2_b=_bf16_tile(ttnn, layer.fc2.bias.detach(), dev),
            final_ln_w=_bf16_tile(ttnn, layer.final_layer_norm.weight.detach(), dev),
            final_ln_b=_bf16_tile(ttnn, layer.final_layer_norm.bias.detach(), dev),
        )

    # -- forward -------------------------------------------------------------

    def _layer_forward(self, x_tt, wb: dict, mask_tt):
        ttnn = self._ttnn

        # --- Self-attention (post-LN) ---
        qkv = ttnn.linear(x_tt, wb["qkv_w"], bias=wb["qkv_b"])
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads
        )
        ttnn.deallocate(qkv)
        scores = ttnn.matmul(q, k)  # k pre-transposed by split op
        ttnn.deallocate(q); ttnn.deallocate(k)
        scores = ttnn.multiply(scores, self.head_dim_inv_sqrt)
        if mask_tt is not None:
            scores = ttnn.add(scores, mask_tt)  # broadcasts over heads
        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        attn_out = ttnn.matmul(probs, v)
        ttnn.deallocate(probs); ttnn.deallocate(v)
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = ttnn.linear(attn_out, wb["out_w"], bias=wb["out_b"])
        x_tt = ttnn.add(x_tt, attn_out)
        ttnn.deallocate(attn_out)
        x_tt = ttnn.layer_norm(x_tt, weight=wb["sa_ln_w"], bias=wb["sa_ln_b"])

        # --- FFN (post-LN) ---
        h = ttnn.linear(x_tt, wb["fc1_w"], bias=wb["fc1_b"], activation="gelu")
        h = ttnn.linear(h, wb["fc2_w"], bias=wb["fc2_b"])
        x_tt = ttnn.add(x_tt, h)
        ttnn.deallocate(h)
        x_tt = ttnn.layer_norm(x_tt, weight=wb["final_ln_w"], bias=wb["final_ln_b"])
        return x_tt

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
        from transformers.modeling_outputs import BaseModelOutput

        ttnn = self._ttnn

        # Mirror upstream: build inputs_embeds + positional + layernorm_embedding.
        if inputs_embeds is None:
            assert input_ids is not None, "input_ids or inputs_embeds required"
            inputs_embeds = self.embed_tokens(input_ids)
            input_for_pos = input_ids
        else:
            # Florence2's trick: use the last channel of inputs_embeds as
            # the "input" handed to embed_positions — only the SHAPE is
            # consulted by the learned positional embedding.
            input_for_pos = inputs_embeds[:, :, -1]

        embed_pos = self.embed_positions(input_for_pos).to(inputs_embeds.device).to(inputs_embeds.dtype)
        hidden = inputs_embeds + embed_pos

        # Upload + apply layernorm_embedding on device.
        x_tt = _bf16_tile(ttnn, hidden, self.device)
        x_tt = ttnn.layer_norm(x_tt, weight=self.ln_emb_w, bias=self.ln_emb_b)

        # Build attention mask on device if provided.
        mask_tt = None
        if attention_mask is not None:
            mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
            # Shape [batch, 1, tgt_len, src_len], additive (-inf at padded src cols).
            mask_tt = _bf16_tile(ttnn, mask_4d, self.device)

        # Run layers.
        for wb in self._layers:
            x_tt = self._layer_forward(x_tt, wb, mask_tt)

        if mask_tt is not None:
            ttnn.deallocate(mask_tt)

        out = ttnn.to_torch(x_tt).to(inputs_embeds.dtype)
        ttnn.deallocate(x_tt)

        if return_dict is False:
            return (out,)
        return BaseModelOutput(last_hidden_state=out, hidden_states=None, attentions=None)
