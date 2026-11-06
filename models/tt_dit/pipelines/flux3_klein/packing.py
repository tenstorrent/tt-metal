# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Token ids and lengths of the FLUX 3 Klein DiT inputs (reference sampling.py / pipeline.py)."""

from __future__ import annotations

import torch

DOWNSAMPLING_RATIO = 16
IMAGE_ROPE_SCALE = 0.5
TEXT_PAD_MULTIPLE = 80
MAX_TEXT_LENGTH = 4096
# Text rows a prompt is padded to: the smallest bucket holding its attended rows (rows past them are masked keys).
# Per (size, reference token bucket; 0 = no reference): at 4096 with a reference in the 4096-token bucket only the 2048
# text bucket splits into whole k = 640 chunks.
TEXT_BUCKETS = (1024, 2048, MAX_TEXT_LENGTH)
TEXT_BUCKETS_BY_LAYOUT = {(4096, 4096): (2048, MAX_TEXT_LENGTH)}


def latent_shape(height: int, width: int) -> tuple[int, int]:
    return height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO


def image_ids(h: int, w: int) -> torch.Tensor:
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    zeros = torch.zeros_like(rows)
    ids = torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4).to(torch.float32)
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def reference_ids(h: int, w: int, index: int) -> torch.Tensor:
    ids = image_ids(h, w)
    ids[:, 3] = -(1 + index)
    return ids


def text_ids(length: int) -> torch.Tensor:
    positions = torch.arange(length)
    return torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)


def padded_text_length(real_length: int) -> int:
    return min(-(-real_length // TEXT_PAD_MULTIPLE) * TEXT_PAD_MULTIPLE, MAX_TEXT_LENGTH)


def text_bucket(attended: int, buckets: tuple[int, ...] = TEXT_BUCKETS) -> int:
    return next((b for b in buckets if b >= attended), buckets[-1])


def pad_rows(rows: torch.Tensor, length: int) -> torch.Tensor:
    return torch.cat([rows, rows.new_zeros((length - rows.shape[0], *rows.shape[1:]))])
