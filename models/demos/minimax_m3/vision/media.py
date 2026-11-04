# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module for the encoder service: image checks, resize, token expansion and stored values.

load builds the TT vision encoder (encoder.py) on the chip that the service gives it, and encode writes each image's
vision tokens into its stored value. With the encoder config's model_options.stub_embeddings, encode writes
deterministic stub embeddings instead and no chip is used. Only load imports ttnn, so a process that only prepares
media runs without it.
"""

import hashlib
import math
import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE
from models.demos.minimax_m3.tt.vision.preprocess import FACTOR, MAX_PIXELS, MAX_RATIO, MIN_PIXELS, PATCH_SIZE
from models.demos.minimax_m3.vision import format as media_format

# The resize's geometry is the vision encoder's (tt/vision/preprocess.py and image_input.py, which need no ttnn
# either), so prepare gives only images that the encoder's check_pixels takes.
IMAGE_START = 200029  # "]<]start of image[>[" in the MiniMax-M3 tokenizer
IMAGE_END = 200030  # "]<]end of image[>["


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
        # The service opens the chip with these. ttnn's defaults suit the encoder: it uses no L1_SMALL memory (no
        # convolution or pooling op), captures no trace and runs on one chip, without fabric.
        self.device_params = {}
        vision = hf_config["vision_config"]
        assert vision["patch_size"] == PATCH_SIZE and vision.get("spatial_merge_size", MERGE_SIZE) == MERGE_SIZE, vision
        self._stub = False
        self._encoder = None
        # One encoder on one chip, which is not thread safe, and read_embeddings releases the GIL while it waits for the
        # chip: one image's forward and read at a time.
        self._lock = threading.Lock()

    def prepare(self, kind: str, media: np.ndarray) -> M3Image:
        if kind != "image":
            raise ValueError(f"MiniMax-M3 does not accept {kind}")
        if media.dtype != np.uint8 or media.ndim != 3 or media.shape[2] != 3 or 0 in media.shape:
            raise ValueError(f"expected a non-empty uint8 RGB [height, width, 3], got {media.dtype} {media.shape}")
        h, w = smart_resize(*media.shape[:2])
        chw = torch.from_numpy(media).permute(2, 0, 1)
        resized = tvF.resize(chw, [h, w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        return M3Image(resized.permute(1, 2, 0).contiguous().numpy(), h // FACTOR, w // FACTOR)

    def expansion(self, item: M3Image) -> list[tuple[int, int]]:
        return [(IMAGE_START, 1), (self.placeholder_token_ids["image"], item.n_tokens), (IMAGE_END, 1)]

    def stored_nbytes(self, item: M3Image) -> int:
        return media_format.stored_nbytes(item.n_tokens, self.hidden)

    def load(self, mesh_device, weights_path: str, kind: str, options: dict) -> None:
        """Get ready to encode: stub embeddings with options["stub_embeddings"], else the TT vision encoder with the
        vision weights of the MiniMax-M3 checkpoint in weights_path, on mesh_device, a mesh of one chip that the caller
        opens (with device_params) and closes. The encoder raises ValueError for a mesh of more chips before it reads
        a weight. Raises ValueError for a weights_path that is not a directory, and for a mesh_device of None (load
        opens no chip), before it imports ttnn."""
        if kind not in self.placeholder_token_ids:
            raise ValueError(f"MiniMax-M3 does not accept {kind}")
        if self._stub or self._encoder is not None:
            raise RuntimeError("load after load: close first")
        if options.get("stub_embeddings"):
            self._stub = True
            return
        if not weights_path or not os.path.isdir(weights_path):
            raise ValueError(f"weights_path must be a MiniMax-M3 checkpoint directory, got {weights_path!r}")
        if mesh_device is None:
            raise ValueError("mesh_device is None: load runs the encoder on the chip that the caller opened")
        from models.demos.minimax_m3.vision.encoder import TtMiniMaxM3VisionEncoder

        self._encoder = TtMiniMaxM3VisionEncoder.from_weights_path(mesh_device, weights_path, self.hf_config)

    def encode(self, items: list[M3Image], out: list[memoryview]) -> None:
        assert self._stub or self._encoder is not None, "encode before load"
        # TODO: replace the stub embeddings with the vision encoder.
        for item, buf in zip(items, out, strict=True):
            media_format.Header("image").write(buf)
            rows = media_format.embeddings(buf, self.hidden)
            if self._stub:
                rows[:] = _stub_rows(item, self.hidden)
            else:
                self._encode_into(item, rows)

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
        """Free the encoder and its device tensors. The chip is the caller's: close leaves it open."""
        with self._lock:
            self._stub = False
            self._encoder = None
