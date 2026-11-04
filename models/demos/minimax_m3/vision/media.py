# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module for the encoder service: image checks, resize, token expansion and stored values.

load builds the TT vision encoder (encoder.py) on one chip, and encode writes each image's vision tokens
into its stored value. With the encoder config's model_options.stub_embeddings, encode writes deterministic stub
embeddings instead and no chip is used. Only load imports ttnn, so a process that only prepares media runs without it.
"""

import gc
import hashlib
import math
import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.vision import format as media_format

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 4 * FACTOR * FACTOR
MAX_PIXELS = 576 * FACTOR * FACTOR
MAX_RATIO = 200
IMAGE_START = 200029  # "]<]start of image[>[" in the MiniMax-M3 tokenizer
IMAGE_END = 200030  # "]<]end of image[>["
# Resized images, (height, width) in pixels, that load runs once so that the chip's programs for their patch counts
# are compiled before the first request: 576 tokens (672 x 672, and 16:9 photos resized to 896 x 504) and 540 tokens
# (4:3 photos resized to 756 x 560). The programs depend on the patch count only; an image of another count compiles
# them on its first encode, 15 to 42 s with an empty kernel cache. model_options.warmup replaces this list.
WARMUP_SIZES = ((672, 672), (560, 756))


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


def _warmup_sizes(options: dict) -> list[tuple[int, int]]:
    """options["warmup"] (default WARMUP_SIZES) as (height, width) pairs of resized images: sides multiples of 28,
    4 to 576 tokens. Raises ValueError otherwise, before load opens a chip."""
    sizes = []
    for size in options.get("warmup", WARMUP_SIZES):
        if not (
            isinstance(size, (list, tuple))
            and len(size) == 2
            and all(isinstance(side, int) and side > 0 and side % FACTOR == 0 for side in size)
        ):
            raise ValueError(f"warmup sizes must be [height, width] in multiples of {FACTOR} pixels, got {size!r}")
        height, width = size
        if not MIN_PIXELS <= height * width <= MAX_PIXELS:
            raise ValueError(f"warmup size {list(size)} makes {height * width // FACTOR**2} tokens, not 4 to 576")
        sizes.append((height, width))
    return sizes


def _stub_rows(item: M3Image, hidden: int) -> np.ndarray:
    """Deterministic stand-in embeddings of item, bf16 bit patterns [n_tokens, hidden], seeded by its pixels."""
    digest = hashlib.sha256(np.asarray(item.pixels.shape).tobytes() + item.pixels.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rows = torch.randn(item.n_tokens, hidden, generator=torch.Generator().manual_seed(seed)) * 0.02
    return rows.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


class MiniMaxM3Media:
    def __init__(self, hf_config: dict):
        self.hf_config = hf_config
        self.hidden = hf_config["text_config"]["hidden_size"]
        self.placeholder_token_ids = {"image": hf_config["image_token_index"]}
        assert hf_config["vision_config"]["patch_size"] == PATCH_SIZE, hf_config["vision_config"]
        self._stub = False
        self._encoder = None
        self._own_mesh_device = None
        # One encoder on one chip, which is not thread safe, and read_embeddings releases the GIL while it waits for the
        # chip: one image's forward and read at a time.
        self._lock = threading.Lock()

    def prepare(self, kind: str, media: np.ndarray) -> M3Image:
        if kind != "image":
            raise ValueError(f"MiniMax-M3 does not accept {kind}")
        if media.dtype != np.uint8 or media.ndim != 3 or media.shape[2] != 4 or 0 in media.shape:
            raise ValueError(f"expected a non-empty uint8 RGBA [height, width, 4], got {media.dtype} {media.shape}")
        h, w = smart_resize(*media.shape[:2])
        rgb = torch.from_numpy(media).permute(2, 0, 1)[:3]
        resized = tvF.resize(rgb, [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        return M3Image(resized.permute(1, 2, 0).contiguous().numpy(), h // FACTOR, w // FACTOR)

    def expansion(self, item: M3Image) -> list[tuple[int, int]]:
        return [(IMAGE_START, 1), (self.placeholder_token_ids["image"], item.n_tokens), (IMAGE_END, 1)]

    def value_size(self, item: M3Image) -> int:
        return media_format.value_size(item.n_tokens, self.hidden)

    def load(self, weights_path: str, options: dict, mesh_device=None) -> None:
        """Get ready to encode: stub embeddings with options["stub_embeddings"], else the TT vision encoder with the
        vision weights of the MiniMax-M3 checkpoint in weights_path. The encoder runs on mesh_device, a mesh of one
        chip, or without one on a 1x1 mesh of the first visible chip that load opens and close closes (the service
        picks the chip, for example with TT_VISIBLE_DEVICES). Then it encodes one blank image of each warm-up size
        (options["warmup"], default WARMUP_SIZES) to compile the chip's programs. Raises ValueError for a weights_path
        that is not a directory or a bad warm-up size before it opens a chip."""
        if self._stub or self._encoder is not None:
            raise RuntimeError("load after load: close first")
        if options.get("stub_embeddings"):
            self._stub = True
            return
        if not weights_path or not os.path.isdir(weights_path):
            raise ValueError(f"weights_path must be a MiniMax-M3 checkpoint directory, got {weights_path!r}")
        warmup = _warmup_sizes(options)
        import ttnn
        from models.demos.minimax_m3.vision.encoder import TtMiniMaxM3VisionEncoder

        own_mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1)) if mesh_device is None else None
        encoder = None
        try:
            encoder = TtMiniMaxM3VisionEncoder.from_weights_path(
                own_mesh_device if mesh_device is None else mesh_device, weights_path, self.hf_config
            )
            for height, width in warmup:
                ttnn.deallocate(encoder.forward(np.zeros((height, width, 3), dtype=np.uint8)))
        except BaseException:
            encoder = None
            if own_mesh_device is not None:
                gc.collect()  # the encoder's device tensors go before their chip closes
                ttnn.close_mesh_device(own_mesh_device)
            raise
        self._encoder, self._own_mesh_device = encoder, own_mesh_device

    def encode(self, items: list[M3Image], out: list[memoryview]) -> None:
        assert self._stub or self._encoder is not None, "encode before load"
        for item, buf in zip(items, out, strict=True):
            h = media_format.Header(item.grid_h, item.grid_w, item.n_tokens, self.hidden)
            # A value has no valid header until its embeddings are in: a failed encode leaves none, not a previous one.
            buf[: media_format.HEADER_BYTES] = bytes(media_format.HEADER_BYTES)
            rows = media_format.embeddings(buf, h)
            if self._stub:
                rows[:] = _stub_rows(item, self.hidden)
            else:
                self._encode_into(item, rows)
            media_format.write_header(buf, h)

    def _encode_into(self, item: M3Image, rows: np.ndarray) -> None:
        """The vision tokens of item, read from the chip straight into rows (format.embeddings of its value)."""
        import ttnn
        from models.demos.minimax_m3.vision.encoder import read_embeddings

        with self._lock:
            tokens = self._encoder.forward(item.pixels)
            try:
                read_embeddings(tokens, rows)
            finally:
                ttnn.deallocate(tokens)

    def close(self) -> None:
        """Free the encoder, and close the chip if load opened it (once, even if closing raises)."""
        with self._lock:
            self._stub = False
            self._encoder = None
            mesh_device, self._own_mesh_device = self._own_mesh_device, None
        if mesh_device is not None:
            import ttnn

            gc.collect()  # the encoder's device tensors go before their chip closes
            ttnn.close_mesh_device(mesh_device)
