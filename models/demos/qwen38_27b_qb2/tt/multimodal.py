# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU request metadata for visual prefill; device/KV positions remain separate.

No TTNN or vLLM imports: these invariants can be checked against the pinned HF
reference without constructing a language model or opening a device.
"""

from dataclasses import dataclass
from itertools import groupby

import torch


def validate_grid(grid, *, merge=2):
    grid = torch.as_tensor(grid)
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.numel() == 0:
        raise ValueError("Visual grid must be nonempty [items, 3]")
    if grid.dtype == torch.bool or grid.is_floating_point() or (grid <= 0).any():
        raise ValueError("Visual grid dimensions must be positive integers")
    if merge < 1 or (grid[:, 1:] % merge).any():
        raise ValueError("Visual spatial dimensions must divide by the merge size")
    return grid.to(dtype=torch.int64, device="cpu")


def vision_boundaries(grid, padded_length, *, merge=2):
    """One attention window per temporal patch, and a disjoint padding window."""
    grid = validate_grid(grid, merge=merge)
    lengths = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0])
    boundaries = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    actual = int(boundaries[-1])
    if type(padded_length) is not int or padded_length < actual:
        raise ValueError("Padded length must cover all visual patches")
    if padded_length > actual:
        boundaries = torch.cat((boundaries, torch.tensor([padded_length])))
    if len(boundaries) > 1024:
        raise ValueError("Too many vision windows for the native SDPA boundary tensor")
    return boundaries.to(torch.int32)


def modality_positions(input_ids, image_grid, video_grid, *, image_token_id, video_token_id, merge=2):
    """Qwen3.5 interleaved M-RoPE indices, including per-frame timestamp text.

    Matches Transformers 5.12.1 Qwen3_5Model.get_rope_index for one unpadded
    request. Validate every placeholder run before assigning positions instead
    of allowing malformed pixels/grids to produce plausible text-only output."""
    ids = torch.as_tensor(input_ids, dtype=torch.int64).reshape(-1)
    if ids.numel() == 0:
        raise ValueError("A multimodal prompt must contain tokens")
    grids = {1: [], 2: []}
    if image_grid is not None:
        grids[1] = validate_grid(image_grid, merge=merge).tolist()
    if video_grid is not None:
        grids[2] = [[1, h, w] for t, h, w in validate_grid(video_grid, merge=merge).tolist() for _ in range(t)]
    types = torch.zeros_like(ids)
    types[ids == image_token_id] = 1
    types[ids == video_token_id] = 2
    positions = torch.empty(3, len(ids), dtype=torch.int64)
    used = {1: 0, 2: 0}
    cursor = 0
    for modality, entries in groupby(enumerate(types.tolist()), key=lambda entry: entry[1]):
        indices = [i for i, _ in entries]
        start, end = indices[0], indices[-1] + 1
        if modality == 0:
            positions[:, start:end] = torch.arange(end - start) + cursor
            cursor += end - start
            continue
        if used[modality] >= len(grids[modality]):
            raise ValueError("Visual placeholders have no corresponding image/video grid")
        t, h, w = grids[modality][used[modality]]
        used[modality] += 1
        h, w = h // merge, w // merge
        if end - start != t * h * w:
            raise ValueError("Visual placeholder run does not match its grid")
        positions[0, start:end] = torch.arange(t).repeat_interleave(h * w) + cursor
        positions[1, start:end] = torch.arange(h).repeat_interleave(w).repeat(t) + cursor
        positions[2, start:end] = torch.arange(w).repeat(t * h) + cursor
        cursor += max(h, w)
    if any(used[modality] != len(grids[modality]) for modality in (1, 2)):
        raise ValueError("Visual grids have no corresponding placeholder tokens")
    delta = int(positions.max()) + 1 - len(ids)
    return positions, delta


@dataclass(frozen=True)
class MultimodalChunk:
    vision_values: torch.Tensor
    vision_mask: torch.Tensor
    rope_positions: torch.Tensor


@dataclass(frozen=True)
class MultimodalPlan:
    request_id: str
    item_identity: tuple
    prompt_ids: torch.Tensor
    positions: torch.Tensor
    rope_delta: int
    feature_positions: torch.Tensor
    features: torch.Tensor

    def chunk(self, start, length):
        if not 0 <= start < start + length <= len(self.prompt_ids):
            raise ValueError("Multimodal chunk lies outside its complete prompt")
        selected = (self.feature_positions >= start) & (self.feature_positions < start + length)
        rows = self.feature_positions[selected] - start
        values = torch.zeros(1, length, self.features.shape[-1], dtype=self.features.dtype)
        mask = torch.zeros(1, length, 1, dtype=torch.bfloat16)
        values[0, rows] = self.features[selected]
        mask[0, rows] = 1
        return MultimodalChunk(values, mask, self.positions[:, None, start : start + length])

    def matches(self, request_id, item_identity, prompt_ids):
        return (
            self.request_id == request_id
            and self.item_identity == item_identity
            and torch.equal(self.prompt_ids, torch.as_tensor(prompt_ids, dtype=torch.int64).reshape(-1))
        )


def item_identity(items):
    result = []
    for item in items:
        if item.get("modality") not in ("image", "video"):
            raise ValueError("Unsupported multimodal item")
        if type(item.get("offset")) is not int or type(item.get("length")) is not int:
            raise ValueError("Multimodal spans require integer offset and length")
        if item["offset"] < 0 or item["length"] <= 0:
            raise ValueError("Invalid multimodal item span")
        result.append((item["modality"], item.get("identifier"), item["offset"], item["length"]))
    return tuple(result)


def gather_media(pixel_values, grids, *, merge=2):
    """Normalize one scheduled row, preserving modality occurrence order."""
    if pixel_values is None:
        if grids is not None:
            raise ValueError("Visual grids were supplied without pixels")
        return None
    pixels = pixel_values if isinstance(pixel_values, (list, tuple)) else [pixel_values]
    grid_items = grids if isinstance(grids, (list, tuple)) else [grids]
    if (
        not pixels
        or len(pixels) != len(grid_items)
        or any(p is None for p in pixels)
        or any(g is None for g in grid_items)
    ):
        raise ValueError("Missing multimodal pixels/grid; no matching cached request encoding")
    checked = [validate_grid(g, merge=merge) for g in grid_items]
    for patch, grid in zip(pixels, checked):
        if not isinstance(patch, torch.Tensor) or patch.ndim != 2 or patch.shape[0] != int(grid.prod(-1).sum()):
            raise ValueError("Visual pixel patches do not match the grid")
    return torch.cat(pixels, dim=0), torch.cat(checked, dim=0)


def build_plan(request_id, identity, input_ids, image, video, encoder, *, config):
    ids = torch.as_tensor(input_ids, dtype=torch.int64).reshape(-1).clone()
    positions, delta = modality_positions(
        ids,
        None if image is None else image[1],
        None if video is None else video[1],
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        merge=config.vision_config.spatial_merge_size,
    )
    feature_positions, features = [], []
    for modality, media, token_id in (("image", image, config.image_token_id), ("video", video, config.video_token_id)):
        spans = [(offset, offset + length) for kind, _, offset, length in identity if kind == modality]
        if media is None:
            if spans:
                raise ValueError("Media identity has no matching pixel payload")
            continue
        rows = (ids == token_id).nonzero().reshape(-1)
        if len(spans) != len(media[1]) or any(
            sum(start <= int(row) < end for start, end in spans) != 1 for row in rows
        ):
            raise ValueError("Media item spans do not cover the matching visual placeholders exactly once")
        encoded = encoder(*media)
        if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != (len(rows), config.text_config.hidden_size):
            raise ValueError("Vision encoder output does not match placeholder count and text hidden width")
        if not torch.isfinite(encoded).all():
            raise ValueError("Vision encoder produced nonfinite features")
        feature_positions.append(rows)
        features.append(encoded.to(device="cpu", dtype=torch.bfloat16))
    if not features:
        raise ValueError("Cannot create a visual plan without visual inputs")
    return MultimodalPlan(
        request_id, identity, ids, positions, delta, torch.cat(feature_positions), torch.cat(features)
    )
