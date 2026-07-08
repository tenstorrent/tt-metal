# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hunyuan_video15_rotary_pos_embed` of tencent/HunyuanVideo-1.5.

Reference submodule: `rope`, a `HunyuanVideo15RotaryPosEmbed`. It is
PARAMETERLESS — the output depends only on the input's *spatial shape*:

    rope_sizes = [F // patch_t, H // patch, W // patch]
    grid = stack(meshgrid(arange(rope_sizes[i]) for i), 0)          # [3, ...]
    for i in 3:
        freqs = 1 / theta ** (arange(0, rope_dim[i], 2) / rope_dim[i])
        ang_i = outer(grid[i].reshape(-1), freqs)                    # [N, rope_dim[i]/2]
        cos_i = ang_i.cos().repeat_interleave(2, dim=1)              # [N, rope_dim[i]]
        sin_i = ang_i.sin().repeat_interleave(2, dim=1)
    freqs_cos = cat(cos_i, dim=1); freqs_sin = cat(sin_i, dim=1)     # [N, sum(rope_dim)]
    return freqs_cos, freqs_sin

Native ttnn strategy
--------------------
The angle table `outer(positions, freqs)` is a constant built from an integer
position meshgrid and constant frequencies — exactly how the reference builds it
on host (`torch.arange` / `meshgrid` / `torch.outer`). We build the per-axis
angle matrices (already `repeat_interleave`d, since cos(dup)=dup(cos)) and
concatenate them on host, then compute the transcendental `cos`/`sin` NATIVELY on
device with `ttnn.cos` / `ttnn.sin` in float32. The result lives on device.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def build(device, torch_module):
    """Read the rope config and return a native ttnn forward (cos/sin on device)."""
    import torch

    r = torch_module
    patch = int(getattr(r, "patch_size", 1))
    patch_t = int(getattr(r, "patch_size_t", 1))
    rope_dim = list(getattr(r, "rope_dim", [16, 56, 56]))
    theta = float(getattr(r, "theta", 256.0))

    def _angle_table(F, H, W):
        """Reproduce the reference's host-built angle table (repeat-interleaved)."""
        rope_sizes = [F // patch_t, H // patch, W // patch]
        grids = [torch.arange(0, s, dtype=torch.float32) for s in rope_sizes]
        mesh = torch.meshgrid(*grids, indexing="ij")  # 3 x [rF, rH, rW]
        grid = torch.stack(mesh, dim=0)  # [3, rF, rH, rW]
        cols = []
        for i in range(3):
            dim = int(rope_dim[i])
            pos = grid[i].reshape(-1)  # [N]
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # [dim/2]
            ang = torch.outer(pos, freqs)  # [N, dim/2]
            ang = ang.repeat_interleave(2, dim=1)  # [N, dim]  (cos(dup)=dup(cos))
            cols.append(ang)
        return torch.cat(cols, dim=1)  # [N, sum(rope_dim)]

    def forward(hidden_states, *args, **kwargs):
        if isinstance(hidden_states, ttnn.Tensor):
            shape = [int(d) for d in hidden_states.shape]
        else:
            shape = list(hidden_states.shape)
        _, _, F, H, W = shape

        ang = _angle_table(F, H, W)  # host constant [N, D]
        ang_t = ttnn.from_torch(ang, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        freqs_cos = ttnn.cos(ang_t)
        freqs_sin = ttnn.sin(ang_t)
        return freqs_cos, freqs_sin

    return forward


def hunyuan_video15_rotary_pos_embed(*args, **kwargs):
    raise RuntimeError(
        "hunyuan_video15_rotary_pos_embed requires build(device, torch_module) to read the "
        "rope config; the bare callable has no parameters."
    )
