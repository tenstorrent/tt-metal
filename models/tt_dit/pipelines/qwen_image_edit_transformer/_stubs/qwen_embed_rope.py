# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the QwenImage 3-axis RoPE table (`pos_embed`, diffusers QwenEmbedRope).

The reference returns complex tables e^{i*angle}; ttnn has no complex dtype, so this returns the
same tables as float32 [S, 2*F] = [cos | sin] (F = sum(axes_dim)/2 = 64) -- the layout the joint
attention port consumes. For one image of (frame, height, width), token t = (f, y, x) gets

    angle[t] = [ (idx+f) * inv0 | pos_h(y) * inv1 | pos_w(x) * inv2 ]
    inv_k    = 1 / theta^(arange(0, axes_dim[k], 2) / axes_dim[k])

with scale_rope centring the height/width positions: pos(y) = y - (height - height // 2). Text tokens
use pos = max_vid_index + arange(L) on every column. The per-token integer positions are metadata
(built as a plain list); the angle products and cos/sin run on device in float32.
"""

from __future__ import annotations

import torch

import ttnn


class TtQwenEmbedRope:
    def __init__(self, device, torch_module):
        self.device = device
        self.theta = torch_module.theta
        self.axes_dim = list(torch_module.axes_dim)
        self.scale_rope = bool(torch_module.scale_rope)
        self.halves = [d // 2 for d in self.axes_dim]
        self.F = sum(self.halves)
        # Same float32 expression as QwenEmbedRope.rope_params.
        inv = torch.cat(
            [1.0 / torch.pow(self.theta, torch.arange(0, d, 2).to(torch.float32).div(d)) for d in self.axes_dim]
        )
        kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
        self.inv = ttnn.from_torch(
            inv.reshape(1, self.F), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, **kw
        )

    def _axis_pos(self, n):
        if self.scale_rope:
            off = n - n // 2
            return [p - off for p in range(n)]
        return list(range(n))

    def _table(self, positions):
        """positions: list of S rows, each F ints -> float32 [S, 2F] = [cos | sin] on device."""
        S = len(positions)
        flat = [float(v) for row in positions for v in row]
        pos = ttnn.Tensor(flat, [S, self.F], ttnn.float32, ttnn.TILE_LAYOUT, self.device)
        ang = ttnn.multiply(pos, self.inv)
        return ttnn.concat([ttnn.cos(ang), ttnn.sin(ang)], dim=-1)

    def __call__(self, *args, video_fhw=None, txt_seq_lens=None, device=None, max_txt_seq_len=None, **_unused):
        if video_fhw is None:
            video_fhw = next(a for a in args if isinstance(a, (list, tuple)))
        if max_txt_seq_len is None and txt_seq_lens is not None:
            max_txt_seq_len = max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens
        if max_txt_seq_len is None:
            raise ValueError("Either `max_txt_seq_len` or `txt_seq_lens` must be provided.")

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        h0, h1, h2 = self.halves
        rows = []
        max_vid_index = 0
        for idx, (frame, height, width) in enumerate(video_fhw):
            ph, pw = self._axis_pos(height), self._axis_pos(width)
            for f in range(frame):
                for y in range(height):
                    for x in range(width):
                        rows.append([idx + f] * h0 + [ph[y]] * h1 + [pw[x]] * h2)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        L = int(max_txt_seq_len)
        txt_rows = [[max_vid_index + i] * self.F for i in range(L)]
        return self._table(rows), self._table(txt_rows)


def build(device, torch_module=None):
    return TtQwenEmbedRope(device, torch_module)


def qwen_embed_rope(device, torch_module=None):
    return TtQwenEmbedRope(device, torch_module)
