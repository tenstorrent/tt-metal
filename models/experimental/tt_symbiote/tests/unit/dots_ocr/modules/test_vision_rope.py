# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: :class:`TTNNDotsVision2DRoPE`.

This is a non-TTNNModule helper that builds cos/sin tables and cu_seqlens.
We test by:

1. Building the cos/sin pair via the production class.
2. Independently computing the expected 2D RoPE cos/sin in PyTorch using
   the same grid layout (factored h/w with spatial-merge reshuffling).
3. Asserting PCC(host_cos, ttnn_cos) and the same for sin.

The ttnn cos/sin are read back via :func:`gather_replicated_first`. The
cu_seqlens output is a Python list of ints — assert exact equality with
the cumulative sum of token counts per grid row.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
    TTNNDotsVision2DRoPE,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.module_helpers import (
    gather_replicated_first,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import assert_op_pcc


def _torch_2d_rope(head_dim: int, grid_thw: torch.Tensor, spatial_merge_size: int, theta: float):
    """Reference 2D RoPE matching :meth:`TTNNDotsVision2DRoPE.build` exactly."""
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    sms = spatial_merge_size

    h_segs, w_segs = [], []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_ids = torch.arange(h, dtype=torch.float32)
        w_ids = torch.arange(w, dtype=torch.float32)
        h_grid = h_ids.unsqueeze(1).expand(h, w)
        w_grid = w_ids.unsqueeze(0).expand(h, w)
        h_grid = h_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)
        w_grid = w_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)
        if t > 1:
            h_grid = h_grid.repeat(t)
            w_grid = w_grid.repeat(t)
        h_segs.append(h_grid)
        w_segs.append(w_grid)
    h_all = torch.cat(h_segs)
    w_all = torch.cat(w_segs)
    freqs_h = h_all.unsqueeze(1) * inv_freq.unsqueeze(0)
    freqs_w = w_all.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos_h = torch.cos(freqs_h)
    sin_h = torch.sin(freqs_h)
    cos_w = torch.cos(freqs_w)
    sin_w = torch.sin(freqs_w)
    cos_half = torch.cat([cos_h, cos_w], dim=-1)
    sin_half = torch.cat([sin_h, sin_w], dim=-1)
    cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_full = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos_full.to(torch.bfloat16), sin_full.to(torch.bfloat16)


_SHAPES = [
    {"id": "vis_rope_t1_h16_w16", "thw": (1, 16, 16)},
    {"id": "vis_rope_t1_h32_w32", "thw": (1, 32, 32)},
]


@pytest.mark.parametrize("row", _SHAPES, ids=[r["id"] for r in _SHAPES])
def test_vision_2d_rope(row, mesh_device_t3k_dp):
    head_dim = 128
    sms = 2
    theta = 10000.0
    grid_thw = torch.tensor([list(row["thw"])], dtype=torch.int64)
    t, h, w = row["thw"]
    seq_len = t * h * w

    # Reference cos/sin computed in pure PyTorch
    ref_cos, ref_sin = _torch_2d_rope(head_dim, grid_thw, sms, theta)

    rope = TTNNDotsVision2DRoPE(
        device=mesh_device_t3k_dp,
        head_dim=head_dim,
        spatial_merge_size=sms,
        theta=theta,
    )
    (cos_tt, sin_tt), cu = rope.build(grid_thw, seq_len)

    assert cu == [0, seq_len], f"Unexpected cu_seqlens={cu} for single segment"

    cos_host = gather_replicated_first(cos_tt, mesh_device_t3k_dp).to(torch.float32)
    sin_host = gather_replicated_first(sin_tt, mesh_device_t3k_dp).to(torch.float32)

    # The TT tensors may pad/round to a tile multiple internally — slice to the
    # logical shape for comparison.
    cos_host = cos_host[..., :seq_len, :head_dim].reshape(ref_cos.shape)
    sin_host = sin_host[..., :seq_len, :head_dim].reshape(ref_sin.shape)

    assert_op_pcc(
        ref_cos.to(torch.float32),
        cos_host,
        threshold=0.999,
        op_name="TTNNDotsVision2DRoPE.cos",
        row_id=row["id"],
    )
    assert_op_pcc(
        ref_sin.to(torch.float32),
        sin_host,
        threshold=0.999,
        op_name="TTNNDotsVision2DRoPE.sin",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] cos/sin PCC ≥ 0.999")
