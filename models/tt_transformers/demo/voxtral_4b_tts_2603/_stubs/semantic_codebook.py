# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `semantic_codebook` (`audio_tokenizer.quantizer.semantic_codebook`)
-- its `decode`.

An 8192 x 256 Euclidean codebook lookup, transposed to channels-first: `embedding(codes).permute`.

The table is **not a parameter**. It is `embedding_sum / cluster_usage.clamp(min=1e-5)`, and both
of those are registered BUFFERS -- they carry 2.1 M of this checkpoint's 4002.35 M, which a
parameter-only census misses entirely. It is materialised on the host in `build` (the module
exposes it as the `embedding` property, which caches it as a non-persistent buffer on first
access).
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
    codebook = getattr(torch_module, "inner", torch_module)
    # `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`).
    table = _split_table(codebook.embedding.detach(), device)

    def semantic_codebook(codes, **kwargs):
        batch, rows, frames = (int(v) for v in codes.shape)
        flat = ttnn.reshape(codes, [batch * rows, frames])
        looked_up = _split_embedding(flat, table, layout=ttnn.TILE_LAYOUT)
        return ttnn.transpose(looked_up, -2, -1)

    return semantic_codebook
