# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal 3D rope index: port of the Cosmos3 Edge `get_rope_index` (Qwen2-VL routine, transformers 5)."""

import itertools

import torch


def _vision_position_ids(start: torch.Tensor | int, grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """[3, t * h/merge * w/merge] positions of one visual span; every axis is offset by `start`."""
    t, h, w = grid_thw[0].item(), grid_thw[1].item() // merge, grid_thw[2].item() // merge
    grid_t, grid_h, grid_w = torch.meshgrid(
        torch.arange(t) + start, torch.arange(h) + start, torch.arange(w) + start, indexing="ij"
    )
    return torch.stack([grid_t, grid_h, grid_w], dim=0).reshape(3, -1)


def get_rope_index(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    merge: int = 2,
    video_grid_thw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (position_ids [3, B, S], rope_deltas [B, 1]).

    `mm_token_type_ids` marks text (0), image (1) and video (2) tokens; each run of equal types is one span, and
    grids are consumed in order across the whole batch. Text advances every axis by 1 per token; a visual span
    advances by max(h, w) / merge. Padding (attention_mask == 0) gets position 0 and is excluded from the delta.
    """
    if video_grid_thw is not None:
        # The processor emits one timestamped span per frame, so each video contributes t grids of one frame.
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        video_grid_thw[:, 0] = 1

    position_ids = torch.zeros(3, *input_ids.shape, dtype=input_ids.dtype)
    grids = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    deltas = []
    for b, ids in enumerate(input_ids):
        types = mm_token_type_ids[b]
        if attention_mask is not None:
            keep = attention_mask[b].bool()
            ids, types = ids[keep], types[keep]

        spans = []
        current = 0
        for modality, group in itertools.groupby(enumerate(types.tolist()), lambda x: x[1]):
            group = list(group)
            length = group[-1][0] + 1 - group[0][0]
            if modality == 0:
                spans.append(torch.arange(length).view(1, -1).expand(3, -1) + current)
                current += length
            else:
                grid = next(grids[modality])
                spans.append(_vision_position_ids(current, grid, merge))
                current += max(grid[1], grid[2]) // merge
        positions = torch.cat(spans, dim=1).reshape(3, -1)

        if attention_mask is not None:
            position_ids[:, b, attention_mask[b].bool()] = positions.to(position_ids.dtype)
        else:
            position_ids[:, b] = positions.to(position_ids.dtype)
        deltas.append(positions.max() + 1 - len(ids))
    return position_ids, torch.stack(deltas).view(-1, 1)
