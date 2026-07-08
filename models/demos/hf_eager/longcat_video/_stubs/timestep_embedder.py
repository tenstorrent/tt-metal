# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn port of `timestep_embedder` (meituan-longcat/LongCat-Video's
`dit.t_embedder`, class `TimestepEmbedder` in the vendored
`longcat_video/modules/blocks.py`):

    mlp = Sequential(Linear(frequency_embedding_size, t_embed_dim, bias=True),
                      SiLU(),
                      Linear(t_embed_dim, t_embed_dim, bias=True))
    forward(t, dtype):
        t_freq = timestep_embedding(t, frequency_embedding_size)   # sinusoidal, deterministic
        return mlp(t_freq.to(dtype))

The sinusoidal frequency embedding's frequency TABLE has no learned weights and depends only
on `frequency_embedding_size` (same category as the RoPE tables in
`long_cat_single_stream_block`) -- built once at construction time with numpy (never fires a
torch/aten host op) and uploaded as a device constant. The actual per-call embedding (which
depends on the real input timestep `t`, so it can't be precomputed) is computed entirely on
device: the outer product `t * freqs` is expressed as a `[M,1] @ [1,half]` matmul (same
avoid-an-explicit-broadcast technique `rotary_positional_embedding.py` uses for rotate_half),
then `ttnn.cos`/`ttnn.sin`/`ttnn.concat` build the embedding; the 2-layer MLP is native ttnn.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn


class TtTimestepEmbedder:
    def __init__(self, device: ttnn.Device, torch_module) -> None:
        self.device = device
        self.dtype = ttnn.bfloat16
        self.frequency_embedding_size = torch_module.frequency_embedding_size
        self.half = self.frequency_embedding_size // 2
        self.odd = self.frequency_embedding_size % 2

        state = torch_module.state_dict()
        self.w0 = ttnn.from_torch(
            state["mlp.0.weight"].transpose(0, 1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.b0 = ttnn.from_torch(
            state["mlp.0.bias"].reshape(1, -1), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w1 = ttnn.from_torch(
            state["mlp.2.weight"].transpose(0, 1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.b1 = ttnn.from_torch(
            state["mlp.2.bias"].reshape(1, -1), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

        max_period = 10000.0
        freqs = np.exp(-np.log(max_period) * np.arange(0, self.half, dtype=np.float32) / self.half).reshape(1, -1)
        self.freqs_row = ttnn.from_torch(
            torch.from_numpy(freqs),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def _mesh(self):
        return isinstance(self.device, ttnn.MeshDevice)

    def __call__(self, t: ttnn.Tensor, dtype=None) -> ttnn.Tensor:
        t_col = ttnn.to_layout(ttnn.reshape(t, (-1, 1)), ttnn.TILE_LAYOUT)
        args = ttnn.matmul(t_col, self.freqs_row)  # outer product: [M,1] @ [1,half] -> [M,half]
        embedding = ttnn.concat([ttnn.cos(args), ttnn.sin(args)], dim=-1)
        if self.odd:
            embedding = ttnn.concat([embedding, ttnn.zeros_like(embedding[:, :1])], dim=-1)
        h = ttnn.silu(ttnn.linear(embedding, self.w0, bias=self.b0))
        return ttnn.linear(h, self.w1, bias=self.b1)


def build(device: ttnn.Device, torch_module) -> TtTimestepEmbedder:
    return TtTimestepEmbedder(device, torch_module)
