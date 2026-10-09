# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side, per-image tables that depend only on ``grid_thw`` (computed before the device forward).

Both are built with HF's own helpers (``transformers.vision_utils``) and op order, so they equal what
``Qwen3_5VisionModel.forward`` computes (checked against the golden by tests/vision/test_vision_inputs.py):
- learned position embedding: bilinear 4-corner interpolation of the 48 x 48 table to the image grid,
  in merge-block order, ``(table[idx] * w[:, :, None]).sum(0)`` in fp32 (BF16 table x fp32 weights);
- 2D rotary: ``(row, col) * inv_freq`` (18 fp32 inverse frequencies, theta 1e4) -> [n, 36],
  ``cat(x, x)`` -> [n, 72] neox layout, cos / sin in fp32.
The BF16 casts happen where HF casts (pos embed before the add) or at the device boundary (cos/sin).
"""

from __future__ import annotations

import torch

from models.demos.pplx_decider_v1_27b.tt.vision.config import PplxVisionArgs


def vision_inv_freq(args: PplxVisionArgs) -> torch.Tensor:
    """``Qwen3_5VisionRotaryEmbedding(head_dim // 2).inv_freq`` (fp32, 18 values)."""
    dim = args.rotary_dim // 2
    return 1.0 / (args.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))


def host_tables(grid_thw, args: PplxVisionArgs, pos_table: torch.Tensor) -> dict[str, torch.Tensor]:
    """``pos_embed`` [n, 1152] fp32 and ``rotary_cos`` / ``rotary_sin`` [n, 72] fp32 for one image."""
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids

    grid = torch.as_tensor(grid_thw, dtype=torch.int64).reshape(1, 3)
    idx, wts = get_vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=args.num_grid_per_side, spatial_merge_size=args.spatial_merge_size
    )
    pos = (torch.nn.functional.embedding(idx, pos_table) * wts[:, :, None]).sum(0)
    pos_ids = get_vision_position_ids(grid, args.spatial_merge_size)
    freqs = (pos_ids.unsqueeze(-1) * vision_inv_freq(args)).flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return {"pos_embed": pos, "rotary_cos": emb.cos(), "rotary_sin": emb.sin()}
