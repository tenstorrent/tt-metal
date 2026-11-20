# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ends of a decode step around the fused layers WITHOUT the host: the existing ttnn modules (tt/embedding.py,
tt/ihc.py TtIHCHead, tt/lm_head.py) writing and reading the root chip's streams on the device, so that embedding,
layers, head and greedy token are one trace and the token of step t can feed step t + 1 (PLAN.md section 13, form A).

    IN    token ids (uint32 on every chip) -> TtHy4Embedding rows -> the whole row on every chip (two all_reduce)
          -> the 4 equal streams in the site's page layout (ihc_site/layout.py streams_to_pages) -> an fp32 tensor
          width-sharded over the root's 16 site cores: the streams tensor the fused programs are launched with
    OUT   that tensor -> DRAM -> x mask (1 on the root, 0 elsewhere: only the root's copy went through the layers)
          -> all_reduce over both axes (= the root's streams on every chip) -> each chip's slice of the hidden axis
          -> the page layout undone -> [1, 1, 8, 4 x 6144 / 4] as the ttnn modules take streams
          -> TtIHCHead -> final norm, LM head -> greedy ids (uint32 on every chip)

The page layout of streams x [r, j, C] (8 rows, 4 streams): page[ja * 16 + h * 8 + r, t * 32 + jb * 16 + c] =
x[r, 2 ja + jb, t * 32 + h * 16 + c] (t: a 32-column tile, h: its two 16-column faces).
"""

from __future__ import annotations

import torch

import ttnn

ROWS, N, FACE, TILE = 8, 4, 16, 32


def root_mask(mesh_device, root_chip: int = 0):
    """[1, 1, 1, 1] fp32 per chip: 1 on the root, 0 on the others."""
    values = torch.zeros(mesh_device.get_num_devices(), 1, 1, 1)
    values[root_chip] = 1.0
    return ttnn.from_torch(
        values,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def embedding_whole(plan, embedding, token_ids):
    """Device token ids (uint32 [1, 1, 1, 8], the same on every chip) -> the embedding rows [1, 1, 8, C], whole on
    every chip (TtHy4Embedding.spread without its partition)."""
    x = embedding.lookup(embedding.table_rows(token_ids))
    return plan.all_reduce(plan.all_reduce(x, plan.head_axis), plan.hidden_axis)


def pages_from_embedding(whole, memory_config):
    """[1, 1, 8, C] (any float dtype) -> the 4 equal streams as fp32 pages [1, 1, 32, C] in ``memory_config`` (the
    site's width-sharded L1 config)."""
    C = whole.shape[-1]
    x = ttnn.typecast(whole, ttnn.float32) if whole.dtype != ttnn.float32 else whole
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, [ROWS, C // TILE, 2, FACE])  # r, t, h, c
    x = ttnn.permute(x, [2, 0, 1, 3])  # h, r, t, c
    x = ttnn.concat([x, x], dim=3)  # (jb): both faces of a page's tile hold the stream's face h
    x = ttnn.reshape(x, [1, 1, 2 * ROWS, C])
    x = ttnn.concat([x, x], dim=2)  # (ja)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    return ttnn.to_memory_config(x, memory_config)


def streams_from_pages(plan, pages, mask):
    """The root's streams pages (an L1 width-sharded fp32 tensor, valid on the root only) -> [1, 1, 8, 4 C / tp] per
    chip, as tt/decoder_layer.py streams_to_device lays the streams out for the ttnn modules."""
    C = pages.shape[-1]
    x = ttnn.to_memory_config(pages, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.multiply(x, mask)
    x = plan.all_reduce(plan.all_reduce(x, plan.head_axis), plan.hidden_axis)
    x = plan.partition(x, dim=3, axis=plan.hidden_axis)  # chip k of the hidden axis: its C / tp columns
    width = x.shape[-1]
    tiles = width // TILE
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, [4, ROWS, tiles, TILE])  # (ja, h), r, t, (jb, c)
    streams = []
    for ja in (0, 1):
        for jb in (0, 1):
            faces = [
                ttnn.slice(x, [2 * ja + h, 0, 0, FACE * jb], [2 * ja + h + 1, ROWS, tiles, FACE * jb + FACE])
                for h in (0, 1)
            ]
            s = ttnn.concat(faces, dim=3)  # r, t, (h, c)
            streams.append(ttnn.reshape(s, [1, 1, ROWS, width]))
    out = ttnn.concat(streams, dim=3)  # j-major, then the chip's hidden slice
    return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


def off_shards(device_logits: torch.Tensor, tail_logits: torch.Tensor, chips: int = 32, floor: float = 0.999):
    """The detector of PLAN.md section 42. device_logits [T, vocab] as the ttnn ends gave them, tail_logits the
    reference tail (host) on the SAME step's streams. Returns [(chip, min pcc over the rows)] of the chips whose
    shard of the vocabulary (vocab / chips consecutive ids) leaves the tail below ``floor``: empty on a step whose
    ends were right; one mesh row of four chips at 0.96 to 0.99 when a ring collective of the ends handed the LM
    head a wrong chunk of the hidden state."""
    shard = device_logits.shape[1] // chips
    out = []
    for c in range(chips):
        a, b = (
            device_logits[:, c * shard : (c + 1) * shard].double(),
            tail_logits[:, c * shard : (c + 1) * shard].double(),
        )
        a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
        pcc = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-30)
        if float(pcc.min()) < floor:
            out.append((c, round(float(pcc.min()), 4)))
    return out
