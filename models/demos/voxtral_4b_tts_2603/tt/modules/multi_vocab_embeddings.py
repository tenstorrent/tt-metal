# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `multi_vocab_embeddings` (`audio_tokenizer.audio_token_embedding`).

37 audio codebooks packed end to end into ONE table (8194 semantic rows + 36 x 23 acoustic rows,
padded up to 9088), so a lookup is `embeddings(input_ids + offsets[None, :, None])`.

Two details the op layer forces:

* **The offset add is done in float32, not bfloat16.** The offsets reach 9000, and bfloat16 has an
  8-bit mantissa -- it represents integers exactly only up to 256, so adding an offset in bfloat16
  would silently land on the wrong codebook row. float32 is exact to 2^24. (Verified: the widened
  add reproduces the integer result bit for bit.)
* **The ids are flattened to 2-D before the lookup.** `ttnn.embedding` reads a 3-D `[B, C, T]` input
  as `[B, T]` and returns `[B, T, dim]`, silently dropping the codebook axis; reshaping to
  `[B * C, T]` first and restoring the shape afterwards keeps all 37 codebooks.
"""

from __future__ import annotations

import torch

import ttnn


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def build(device, torch_module):
    emb = torch_module
    n_codebooks = len(emb.codebook_sizes)
    embed_dim = int(emb.embeddings.embedding_dim)

    table = _from_torch(emb.embeddings.weight.detach().contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    offsets = _from_torch(
        emb.offsets.detach().reshape(1, n_codebooks, 1).to(torch.float32),
        device,
        dtype=ttnn.float32,
    )

    def multi_vocab_embeddings(input_ids, codebooks_as_rows=False, **kwargs):
        batch, codebooks, frames = (int(v) for v in input_ids.shape)

        widened = ttnn.typecast(ttnn.to_layout(input_ids, ttnn.TILE_LAYOUT), ttnn.float32)
        shifted = ttnn.add(widened, offsets)
        ids = ttnn.to_layout(ttnn.typecast(shifted, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)

        if codebooks_as_rows and frames == 1:
            # `[B, 1, 37, D]`: one frame's 37 lookups share a tile column as ROWS. Gathered as
            # `[B * 37, 1]` instead, every lookup pads to its own 32-row tile.
            looked_up = ttnn.embedding(ttnn.reshape(ids, [batch, codebooks]), table, layout=ttnn.TILE_LAYOUT)
            return ttnn.reshape(looked_up, [batch, 1, codebooks, embed_dim])

        flat = ttnn.reshape(ids, [batch * codebooks, frames])
        looked_up = ttnn.embedding(flat, table, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(looked_up, [batch, codebooks, frames, embed_dim])

    return multi_vocab_embeddings
