# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `decoder_head` -- the text backbone's `lm_head`.

A single bias-free `Linear(3072, 131072)`, tied to `model.embed_tokens`. The output row is 131072
wide, so the projection is run in column chunks and concatenated: one 3072x131072 matmul asks the
default program config for an output block far past what L1 holds, while a chunk width that is a
multiple of the tile width splits the columns exactly and changes no arithmetic (each output column
depends only on its own weight column).

The LEADING axis is the batch and it is read off the tensor -- `batch = hidden_states.shape[0]`,
never a literal 1, which is the shape at which samples 1..B-1 would go missing. A rank-2
`[seq, in]` input is one sample. Note that a projection is positionwise, so a caller holding one
row per sample is free to present it as `[1, B, in]` (B "positions" of a single batch) instead of
`[B, 1, in]`; both are correct here and the first streams the 131072-wide weight ONCE rather than
once per sample.
"""

from __future__ import annotations

import torch

import ttnn


_CHUNK_COLS = 32768

# The 3072-term dot product behind every one of the 131072 logits accumulates in DEST. Leaving
# `ttnn.linear` on its defaults leaves `fp32_dest_acc_en` OFF, so that accumulation rounds to
# bfloat16 every step -- and the greedy consumer of this head resolves top-1 against top-2, a gap
# far narrower than the logits themselves. Widening DEST is the fidelity rung that makes the
# argmax reproducible; the weights stay bfloat16.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def build(device, torch_module):
    head = torch_module
    in_dim = int(head.in_features)
    out_dim = int(head.out_features)

    weight = head.weight.detach().transpose(0, 1).contiguous()
    chunks = [
        _from_torch(weight[:, i : i + _CHUNK_COLS].contiguous(), device)
        for i in range(0, out_dim, _CHUNK_COLS)
    ]
    bias = None
    if head.bias is not None:
        bias = _from_torch(head.bias.detach().reshape(1, 1, 1, out_dim), device)

    def decoder_head(hidden_states, **kwargs):
        shape = list(hidden_states.shape)
        seq = int(shape[-2])
        batch = int(shape[0]) if len(shape) > 2 else 1
        x = ttnn.reshape(hidden_states, [batch, 1, seq, in_dim])
        parts = [ttnn.linear(x, w, compute_kernel_config=_COMPUTE) for w in chunks]
        out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=-1)
        if bias is not None:
            out = ttnn.add(out, bias)
        return ttnn.reshape(out, [batch, seq, out_dim])

    return decoder_head
