# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL text model (`Qwen2_5_VLTextModel`,
`model.language_model`), prefill without KV cache -- the text-encoder path of Qwen-Image-Edit.

    x = inputs_embeds (or embed_tokens(input_ids))
    cos, sin = mrope(position_ids)          default positions: arange(S) on all three (t, h, w) axes
    x = layer_i(x, causal & attention_mask, cos, sin) for 28 layers
    return norm(x)                          last_hidden_state

TP scheme per layer (see layer.py): attention split by KV group (TP=4 -> 1 kv head + 7 q heads per
chip), MLP intermediate split; all_reduce after each row-parallel projection. Embedding table, norms
and rotary tables replicated. The rotary tables depend only on integer positions and are precomputed
on host (as tt_transformers does); the residual stream stays in fp32 on device across all layers.
"""

from __future__ import annotations

import numpy as np

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs import language_model_layers_0_mlp, v_l_decoder_layer
from models.demos.qwen_image_edit_text_encoder._stubs.attention import mesh_shape, pad_to_tile
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtRMSNorm, _fp32
from models.demos.qwen_image_edit_text_encoder._stubs.token_embed import TtTokenEmbed


class TtTextModel:
    """Composed of the graduated ports: 28 x v_l_decoder_layer, each with its language_model_layers_0_mlp.

    Row-staged on a 2-row mesh (row_stages=True, the default there): every layer keeps its TP split over
    the 4 columns, but layers [0, n/2) live on row 0 and [n/2, n) on row 1, so each chip holds 1/8 of
    the stack instead of 1/4. One slot i of `self.layers` is the pair (layer i on row 0, layer n/2 + i
    on row 1). The forward runs the slots twice: pass 1 is valid on row 0, then row 0's activation is
    handed to row 1 (all_gather over the row axis, exact), and pass 2 is valid on row 1. On a 1-row mesh
    (or row_stages=False) it is the plain TP stack.
    """

    def __init__(self, device, torch_module, embed_tokens=True, row_stages=None):
        self.device = device
        self.embed_tokens = TtTokenEmbed(device, torch_module.embed_tokens) if embed_tokens else None
        rows, _ = mesh_shape(device)
        layers = list(torch_module.layers)
        if row_stages is None:
            row_stages = rows == 2 and len(layers) >= 2
        self.row_stages = bool(row_stages) and rows == 2
        self.num_layers = len(layers)
        self.layers = []
        if self.row_stages:
            if len(layers) % 2:
                raise ValueError(f"row-staged text stack needs an even depth, got {len(layers)}")
            half = len(layers) // 2
            for i in range(half):
                a, b = layers[i], layers[half + i]
                mlp = language_model_layers_0_mlp.build(device, a.mlp, pair=b.mlp)
                self.layers.append(v_l_decoder_layer.build(device, a, mlp=mlp, pair=b))
        else:
            for lyr in layers:
                mlp = language_model_layers_0_mlp.build(device, lyr.mlp)
                self.layers.append(v_l_decoder_layer.build(device, lyr, mlp=mlp))
        self.norm = TtRMSNorm(device, torch_module.norm)
        rot = torch_module.rotary_emb
        self.inv_freq = rot.inv_freq.detach().float().numpy()
        self.rope_scale = float(getattr(rot, "attention_scaling", 1.0))

    def _position_embeddings(self, position_ids, b, s):
        """HF Qwen2_5_VLTextModel positions -> Qwen2_5_VLRotaryEmbedding (cos, sin) [3, B, S, D]."""
        if position_ids is None:
            pos = np.broadcast_to(np.arange(s)[None, None, :], (3, b, s))
        else:
            pos = position_ids.numpy()
            if pos.ndim == 2:
                pos = np.broadcast_to(pos[None], (3,) + pos.shape)
            elif pos.shape[0] == 4:
                pos = pos[1:]
        freqs = pos.astype(np.float32)[..., None] * self.inv_freq[None, None, None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb) * self.rope_scale, np.sin(emb) * self.rope_scale

    @staticmethod
    def _take_row(x, row):
        """Every chip gets row `row`'s copy of x (all_gather over the row axis moves bits exactly)."""
        b = x.shape[0]
        g = ttnn.all_gather(x, dim=0, cluster_axis=0, num_links=1, topology=ttnn.Topology.Linear)
        start = [row * b] + [0] * (len(x.shape) - 1)
        end = [(row + 1) * b] + list(x.shape)[1:]
        return ttnn.slice(g, start, end)

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask):
        """Device path: x [B, 1, s_pad, C] fp32 -> norm(layers(x)) [B, 1, s_pad, C], replicated. Rotary
        tables ([B|1, 1, s_pad, D]) and the additive causal/padding mask are prepared outside."""
        for lyr in self.layers:
            x = lyr.forward_padded(x, tt_cos, tt_sin, tt_mask)
        if self.row_stages:
            x = self._take_row(x, 0)  # layers [0, n/2) done on row 0 -> hand to row 1
            for lyr in self.layers:
                x = lyr.forward_padded(x, tt_cos, tt_sin, tt_mask)
            x = self._take_row(x, 1)  # layers [n/2, n) done on row 1 -> everyone
        return self.norm(x)

    def __call__(self, inputs_embeds=None, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        b, s, c = x.shape
        s_pad = pad_to_tile(s)
        attn0 = self.layers[0].self_attn
        tt_cos, tt_sin = attn0.rope_tables(self._position_embeddings(position_ids, b, s), s_pad)
        tt_mask = attn0.mask(attention_mask, b, s, s_pad)

        x = ttnn.reshape(x, (b, 1, s, c))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        x = self.forward_padded(_fp32(x), tt_cos, tt_sin, tt_mask)
        if s_pad != s:
            x = ttnn.slice(x, [0, 0, 0, 0], [b, 1, s, c])
        return ttnn.reshape(x, (b, s, c))


def build(device, torch_module=None):
    return TtTextModel(device, torch_module)


def v_l_text_model(device, torch_module=None):
    return build(device, torch_module)
