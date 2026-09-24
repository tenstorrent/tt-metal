# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Partial neox RoPE (first ``rotary_dim`` = 64 of 256 channels), HF half-split convention.

Qwen3.5's interleaved M-RoPE reduces to 1D RoPE for text-only positions (checked against upstream in
tests/torch). Keeping the HF half-split layout (not the Meta-interleaved one minimax permutes into)
means the cached K is directly comparable with the golden trace, no permutation.

cos/sin for one chunk are built on host for each row's contiguous token slice
``[start + r*s_local, start + (r+1)*s_local)`` and SP-sharded; ``rotate_half`` is a tiny constant
matmul (exact +-1 entries) so the whole rotation stays on device.
"""

import torch

import ttnn
from models.demos.qwen_3_8_27b.config import Qwen38Config
from models.demos.qwen_3_8_27b.reference.qwen3_8_ref import rope_cos_sin
from models.demos.qwen_3_8_27b.tt.common import hifi4_fp32


class TtRope:
    def __init__(self, mesh_config, cfg: Qwen38Config):
        self.mc = mesh_config
        self.cfg = cfg
        self.rd = cfg.rotary_dim
        half = self.rd // 2
        R = torch.zeros(self.rd, self.rd)
        R[half:, :half] = -torch.eye(half)  # out[:half] = -x[half:]
        R[:half, half:] = torch.eye(half)  # out[half:] = x[:half]
        self.rot_mat = ttnn.from_torch(
            R[None, None],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_config.mesh_device,
            mesh_mapper=mesh_config.replicate(),
        )
        self.ckc = hifi4_fp32()

    def tables(self, start: int, s_local: int):
        """cos/sin ``[1, 1, s_local, rd]`` per device for the chunk starting at global position ``start``."""
        sp = self.mc.sp
        pos = start + torch.arange(sp * s_local)
        cos, sin = rope_cos_sin(self.cfg, pos, dtype=torch.float32)
        mk = lambda t: ttnn.from_torch(  # noqa: E731
            t.reshape(sp, 1, s_local, self.rd),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mc.mesh_device,
            mesh_mapper=self.mc.shard(0, None),
        )
        return mk(cos), mk(sin)

    def apply(self, x, cos, sin):
        """x ``[1, H, S, D]`` -> RoPE on the first rd channels, rest passed through."""
        _, H, S, D = x.shape
        x_rot = ttnn.slice(x, [0, 0, 0, 0], [1, H, S, self.rd])
        x_pass = ttnn.slice(x, [0, 0, 0, self.rd], [1, H, S, D])
        rh = ttnn.matmul(x_rot, self.rot_mat, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        xr = ttnn.typecast(x_rot, ttnn.float32)
        a = ttnn.multiply(xr, cos)
        b = ttnn.multiply(rh, sin)
        r = ttnn.add(a, b, dtype=x.dtype)
        for t in (x_rot, rh, xr, a, b):
            ttnn.deallocate(t)
        out = ttnn.concat([r, x_pass], dim=3)
        ttnn.deallocate(r)
        ttnn.deallocate(x_pass)
        return out
