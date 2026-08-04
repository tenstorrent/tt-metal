# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `qwen2_vision_transformer_pretrained_model` of
Qwen/Qwen2-VL-7B-Instruct.

Reference: `visual`, a `Qwen2VisionTransformerPretrainedModel`:

    position_ids = get_vision_position_ids(grid_thw, spatial_merge_size)
    cu_seqlens = get_vision_cu_seqlens(grid_thw)
    hidden_states = patch_embed(hidden_states)                     # flattened-patch matmul
    rotary_pos_emb = rotary_pos_emb(position_ids)                  # (N, head_dim//2), host-side table
    emb = cat(rotary_pos_emb, rotary_pos_emb, dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    for blk in blocks:
        hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
    merged_hidden_states = merger(hidden_states)

This file only wires together the already-graduated native ports
(`patch_embed`, `qwen2_v_l_vision_block`, `patch_merger`) plus the cheap
host-side position-id / rotary-table construction (`get_vision_position_ids`
/ `get_vision_cu_seqlens` / `VisionRotaryEmbedding` are index/table math, not
model compute) -- the actual tensor math all runs as ttnn ops on device via
those per-component ports.
"""

from __future__ import annotations

from transformers.models.qwen2_vl.modeling_qwen2_vl import get_vision_cu_seqlens, get_vision_position_ids

from . import patch_embed as _patch_embed_stub
from . import patch_merger as _patch_merger_stub
from . import qwen2_v_l_vision_block as _vision_block_stub


def build(device, torch_module):
    visual = torch_module  # Qwen2VisionTransformerPretrainedModel
    spatial_merge_size = visual.spatial_merge_size
    rotary_pos_emb_mod = visual.rotary_pos_emb

    patch_embed_forward = _patch_embed_stub.build(device, visual.patch_embed)
    block_forwards = [_vision_block_stub.build(device, blk) for blk in visual.blocks]
    merger_forward = _patch_merger_stub.build(device, visual.merger)

    def forward(hidden_states, grid_thw=None, cos_dev=None, sin_dev=None, bounds=None, **kwargs):
        # Trace/on-device path: `cos_dev`/`sin_dev` (device buffers, (seq,1,hd))
        # and `bounds` (python list) pre-built by the caller -> pure ttnn, no
        # host position-id / rotary / cu_seqlens construction inside.
        if cos_dev is None:
            position_ids = get_vision_position_ids(grid_thw, spatial_merge_size)
            cu_seqlens = get_vision_cu_seqlens(grid_thw)
            rotary_pos_emb = rotary_pos_emb_mod(position_ids)
            emb = rotary_pos_emb.repeat(1, 2)
            position_embeddings = (emb.cos(), emb.sin())
        else:
            position_embeddings = None
            cu_seqlens = None

        hidden_states = patch_embed_forward(hidden_states)
        for block_forward in block_forwards:
            hidden_states = block_forward(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                cos_dev=cos_dev,
                sin_dev=sin_dev,
                bounds=bounds,
            )

        # The VLM consumes the MERGED (pooler) output as `image_embeds` -- that is
        # what `get_image_features(...).pooler_output` returns and what gets
        # scattered into the text sequence -- so return the merger output.
        return merger_forward(hidden_states)

    return forward


def qwen2_vision_transformer_pretrained_model(*args, **kwargs):
    raise RuntimeError(
        "qwen2_vision_transformer_pretrained_model requires build(device, torch_module) to bind "
        "trained weights; the bare callable has no parameters."
    )
