# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mistral_audio_codebook` (`audio_tokenizer.quantizer`) -- its `decode`.

Two different quantizers side by side, concatenated on the channel axis into the codec's 292-wide
latent:

  * row 0 is a SEMANTIC index into an 8192-entry Euclidean codebook -> `ttnn.embedding`. The table
    is **not a parameter**: it is `embedding_sum / cluster_usage.clamp(min=1e-5)`, and both of those
    are BUFFERS, which is why a parameter-only census misses 2.1 M of this checkpoint. It is
    materialised on the host in `build`.
  * rows 1..36 are ACOUSTIC finite-scalar-quantization levels -> `codes * 2 / (n_levels - 1) - 1`,
    weight-free. Written centred as `(codes - (n_levels - 1) / 2) * (2 / (n_levels - 1))` so the
    subtraction is exact on the integer codes and only the single scale multiply rounds.

The codes arrive as integers, so the acoustic rows are tilized and widened to bfloat16 before the
arithmetic, while the semantic row stays ROW_MAJOR uint32 for the embedding lookup.
"""

from __future__ import annotations

import torch

import ttnn


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


# THE SEMANTIC CODEBOOK IS GATHERED IN TWO HALVES.
# `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`), and this table
# is the codec's INPUT -- everything downstream amplifies whatever it gets wrong. Measured on this
# checkpoint: a single bfloat16 table put the quantizer latent at 1.613e-3 relative, and the
# 292->1024 convolution that consumes it amplified that to 7.7e-3, which then dominated every
# later stage. So the table is split the way a compensated matmul splits a weight --
# `hi = bf16(t)`, `lo = bf16(t - hi)` -- and the two gathers are added back in float32. Two
# bfloat16 mantissas end to end is ~1e-5 on a table this size, for one extra gather and one add
# on 416 rows.
def _split_table(weight, device):
    hi = weight.to(torch.bfloat16)
    lo = (weight - hi.float()).to(torch.bfloat16)
    return (
        _from_torch(hi.contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
        _from_torch(lo.contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
    )


def _split_embedding(ids, tables, layout=None):
    hi, lo = tables
    layout = ttnn.TILE_LAYOUT if layout is None else layout
    return ttnn.add(
        ttnn.typecast(ttnn.embedding(ids, hi, layout=layout), ttnn.float32),
        ttnn.typecast(ttnn.embedding(ids, lo, layout=layout), ttnn.float32),
    )


def build(device, torch_module):
    quant = getattr(torch_module, "inner", torch_module)
    semantic = quant.semantic_codebook
    acoustic = quant.acoustic_codebook

    n_semantic = int(semantic.num_codebooks)
    n_acoustic = int(acoustic.num_codebooks)
    n_levels = int(acoustic.n_levels)
    shift = (n_levels - 1) / 2.0
    scale = 2.0 / (n_levels - 1)

    table = _split_table(semantic.embedding.detach(), device)

    def mistral_audio_codebook(codes, **kwargs):
        batch, rows, frames = (int(v) for v in codes.shape)

        sem_codes = ttnn.reshape(
            ttnn.slice(codes, [0, 0, 0], [batch, n_semantic, frames]), [batch, frames]
        )
        sem = ttnn.transpose(
            _split_embedding(sem_codes, table, layout=ttnn.TILE_LAYOUT), -2, -1
        )

        aco_codes = ttnn.typecast(
            ttnn.to_layout(
                ttnn.slice(codes, [0, n_semantic, 0], [batch, n_semantic + n_acoustic, frames]),
                ttnn.TILE_LAYOUT,
            ),
            ttnn.bfloat16,
        )
        aco = ttnn.multiply(ttnn.subtract(aco_codes, shift), scale)

        # float32 on BOTH sides: the semantic half is a two-gather split now and comes back
        # float32, and `ttnn.concat` requires a single dtype.
        return ttnn.concat([sem, ttnn.typecast(aco, ttnn.float32)], dim=1)

    return mistral_audio_codebook
