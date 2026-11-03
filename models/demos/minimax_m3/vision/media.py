# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module for the encoder service: image checks, resize, token expansion and stored values.

The vision encoder is not implemented yet: encode writes deterministic stub embeddings, and only when the encoder
config's model_options set stub_embeddings.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.vision import format

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 4 * FACTOR * FACTOR
MAX_PIXELS = 576 * FACTOR * FACTOR
MAX_RATIO = 200
IMAGE_START = 200029
IMAGE_END = 200030


@dataclass(frozen=True)
class M3Image:
    pixels: np.ndarray
    """Resized RGB image, uint8 [height, width, 3]; height and width are multiples of 28."""
    grid_h: int
    """Image-token rows: height / 28 (14x14 patches merged 2x2)."""
    grid_w: int
    """Image-token columns: width / 28."""

    @property
    def n_tokens(self) -> int:
        return self.grid_h * self.grid_w


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Hugging Face MiniMaxM3VLImageProcessor's resized (height, width)."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"aspect ratio {max(height, width) / min(height, width):.1f} exceeds {MAX_RATIO}:1")
    h = max(FACTOR, round(height / FACTOR) * FACTOR)
    w = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h * w > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h = math.floor(height / beta / FACTOR) * FACTOR
        w = math.floor(width / beta / FACTOR) * FACTOR
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


class MiniMaxM3Media:
    def __init__(self, hf_config: dict):
        self.hidden = hf_config["text_config"]["hidden_size"]
        self.placeholder_token_ids = {"image": hf_config["image_token_index"]}
        assert hf_config["vision_config"]["patch_size"] == PATCH_SIZE, hf_config["vision_config"]
        self._stub = None

    def prepare(self, kind: str, media: np.ndarray) -> M3Image:
        if kind != "image":
            raise ValueError(f"MiniMax-M3 does not accept {kind}")
        if media.dtype != np.uint8 or media.ndim != 3 or media.shape[2] != 4:
            raise ValueError(f"expected uint8 RGBA [height, width, 4], got {media.dtype} {media.shape}")
        h, w = smart_resize(*media.shape[:2])
        rgb = torch.from_numpy(np.ascontiguousarray(media[..., :3])).permute(2, 0, 1)
        resized = tvF.resize(rgb, [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        return M3Image(resized.permute(1, 2, 0).contiguous().numpy(), h // FACTOR, w // FACTOR)

    def expansion(self, item: M3Image) -> list[tuple[int, int]]:
        return [(IMAGE_START, 1), (self.placeholder_token_ids["image"], item.n_tokens), (IMAGE_END, 1)]

    def value_size(self, item: M3Image) -> int:
        return format.value_size(item.n_tokens, self.hidden)

    def load(self, weights_path: str, options: dict) -> None:
        if not options.get("stub_embeddings"):
            raise NotImplementedError("the MiniMax-M3 vision encoder is not implemented; set stub_embeddings")
        self._stub = True

    def encode(self, items: list[M3Image], out: list[memoryview]) -> None:
        assert self._stub, "encode before load"
        for item, buf in zip(items, out, strict=True):
            h = format.Header(item.grid_h, item.grid_w, item.n_tokens, self.hidden)
            format.write_header(buf, h)
            seed = int.from_bytes(hashlib.sha256(item.pixels.tobytes()).digest()[:8], "little")
            rows = torch.randn(item.n_tokens, self.hidden, generator=torch.Generator().manual_seed(seed)) * 0.02
            format.embeddings(buf, h)[:] = rows.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

    def close(self) -> None:
        self._stub = None
