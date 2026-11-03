# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Multimodal rope positions of the Qwen vision-language models.

Qwen2.5-VL and Qwen3-VL assign image tokens the same positions. Videos follow Qwen3-VL, which
separates frames by timestamps; Qwen2.5-VL instead scales the temporal axis by the frame interval.
"""

from __future__ import annotations

import itertools

import torch


def mrope_position_ids(
    mm_token_type_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    spatial_merge_size: int,
) -> torch.Tensor:
    """The (3, batch, sequence) multimodal rope positions of a prompt without padding.

    `mm_token_type_ids` marks every token as text (0), image (1) or video (2), and the grids give
    the temporal, height and width extent of every image and video in the order their runs appear.
    A text run counts up on all three axes. A vision run takes its grid coordinates, offset by the
    position after the preceding text, and the text after it resumes at that offset plus the
    run's largest spatial extent. Follows `Qwen3VLModel.get_rope_index` of transformers, where a
    video's frames are separated by text and each is its own run.
    """
    grids = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(_video_frame_grids(video_grid_thw)) if video_grid_thw is not None else None,
    }

    batch_size, sequence_length = mm_token_type_ids.shape
    position_ids = torch.zeros([3, batch_size, sequence_length], dtype=torch.long)

    for batch_idx, token_types in enumerate(mm_token_type_ids.tolist()):
        current_pos = 0
        parts = []
        for token_type, group in itertools.groupby(token_types):
            length = len(list(group))

            if token_type == 0:
                parts.append(torch.arange(length).expand(3, -1) + current_pos)
                current_pos += length
                continue

            grid_iter = grids.get(token_type)
            grid = next(grid_iter, None) if grid_iter is not None else None
            if grid is None:
                msg = f"no matching grid for the vision run of {length} tokens of type {token_type}"
                raise ValueError(msg)

            block = vision_position_ids(current_pos, grid, spatial_merge_size=spatial_merge_size)
            if block.shape[1] != length:
                msg = f"a vision run of {length} tokens does not match the {block.shape[1]} tokens of grid {grid.tolist()}"
                raise ValueError(msg)
            parts.append(block)
            current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size

        position_ids[:, batch_idx] = torch.cat(parts, dim=1)

    return position_ids


def vision_position_ids(start_position: int, grid_thw: torch.Tensor, *, spatial_merge_size: int) -> torch.Tensor:
    """The (3, tokens) positions of one image or video frame block, offset by `start_position`.

    Follows `Qwen3VLModel.get_vision_position_ids` of transformers.
    """
    t = int(grid_thw[0])
    h = int(grid_thw[1]) // spatial_merge_size
    w = int(grid_thw[2]) // spatial_merge_size

    temporal = torch.arange(t) + start_position
    height = torch.arange(h) + start_position
    width = torch.arange(w) + start_position

    grid = torch.meshgrid(temporal, height, width, indexing="ij")
    return torch.stack(grid, dim=0).reshape(3, -1)


def _video_frame_grids(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """One (1, h, w) grid per frame: timestamps separate a video's frames into separate runs."""
    frames = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    frames[:, 0] = 1
    return frames
