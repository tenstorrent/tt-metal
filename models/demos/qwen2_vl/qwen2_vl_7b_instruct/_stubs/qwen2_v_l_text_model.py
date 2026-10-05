# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_v_l_text_model` of Qwen/Qwen2-VL-7B-Instruct.

Reference: `language_model`, a `Qwen2VLTextModel`:

    inputs_embeds = embed_tokens(input_ids)
    position_ids = arange(seq_len) broadcast to (3, B, S)   # temporal/height/width
    cos, sin = rotary_emb(inputs_embeds, position_ids)      # mrope table, host-side (cheap)
    for layer in layers:
        hidden_states = layer(hidden_states, position_embeddings=(cos, sin), ...)
    hidden_states = norm(hidden_states)

The per-layer math (attention, mrope rotation, MLP) is delegated to the
already-graduated native `qwen2_v_l_decoder_layer` port so this file only
needs to add the token embedding, the (host-side, cheap) rotary-table
construction, and the final RMSNorm as ttnn ops.
"""

from __future__ import annotations

import torch

import ttnn

from . import qwen2_v_l_decoder_layer as _decoder_stub

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    model = torch_module  # Qwen2VLTextModel
    embed = model.embed_tokens
    layers = model.layers
    norm = model.norm
    rotary_emb = model.rotary_emb
    eps = float(norm.variance_epsilon)

    embed_w = ttnn.from_torch(embed.weight.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    norm_w = ttnn.from_torch(norm.weight.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    layer_forwards = [_decoder_stub.build(device, layer) for layer in layers]

    def forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        cos_dev=None,
        sin_dev=None,
        mask_dev=None,
        **kwargs,
    ):
        # Trace/2CQ path: when `cos_dev`/`sin_dev`/`mask_dev` are supplied as
        # pre-uploaded persistent device buffers, the rotary table + causal mask
        # are NOT rebuilt on host here -- every layer runs pure ttnn (host-op-free).
        if inputs_embeds is not None:
            # VLM path: the caller has already embedded tokens and scattered the
            # vision features into the image-token positions -- use it directly
            # (device tensor, shape (batch, seq_len, hidden)).
            hidden_states = inputs_embeds
            batch, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[-2])
        else:
            tok_torch = ttnn.to_torch(input_ids).to(torch.long)
            batch, seq_len = tok_torch.shape[0], tok_torch.shape[-1]
            tok_torch = tok_torch.reshape(batch, seq_len)
            tok_dev = ttnn.from_torch(tok_torch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            hidden_states = ttnn.embedding(tok_dev, embed_w, layout=ttnn.TILE_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (batch, seq_len, embed_w.shape[-1]))

        if cos_dev is not None:
            position_embeddings = None
        else:
            if position_ids is None:
                pos = torch.arange(seq_len).view(1, 1, -1).expand(3, batch, -1)
            else:
                pos = position_ids
            # `rotary_emb` only needs `x` for its dtype/device, not its content.
            dummy_hidden = torch.zeros(batch, seq_len, 1, dtype=torch.float32)
            position_embeddings = rotary_emb(dummy_hidden, pos)

        for layer_forward in layer_forwards:
            hidden_states = layer_forward(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                cos_dev=cos_dev,
                sin_dev=sin_dev,
                mask_dev=mask_dev,
            )

        hidden_states = ttnn.rms_norm(hidden_states, weight=norm_w, epsilon=eps)
        return hidden_states

    # Expose the graduated per-layer forwards + final norm/embed so a caller can
    # drive a fixed-capacity KV-cache decode (seq=1 steps) without rebuilding
    # weights. The full-sequence `forward` above is unchanged (e2e gate path).
    forward.layers = layer_forwards
    forward.norm_w = norm_w
    forward.norm_eps = eps
    forward.embed_w = embed_w
    return forward


def qwen2_v_l_text_model(*args, **kwargs):
    raise RuntimeError(
        "qwen2_v_l_text_model requires build(device, torch_module) to bind trained weights; "
        "the bare callable has no parameters."
    )
