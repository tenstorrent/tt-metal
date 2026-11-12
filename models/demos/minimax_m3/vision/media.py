# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module for the encoder service: image checks, resize, token expansion and stored values.

load builds the TT vision encoder (encoder.py) on the chip that the service gives it, and encode writes each image's
stored value: the media kind header, then the image's vision tokens. With the encoder config's
model_options.stub_embeddings, encode writes deterministic stub embeddings instead and no chip is used. Only load
imports ttnn, so a process that only prepares media runs without it.
"""

import gc
import hashlib
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE
from models.demos.minimax_m3.tt.vision.preprocess import (
    FACTOR,
    MAX_PIXELS,
    MAX_RATIO,
    MIN_PIXELS,
    PATCH_SIZE,
    check_size,
)
from models.demos.minimax_m3.vision import format as media_format

# The resize's geometry is the vision encoder's (tt/vision/preprocess.py and image_input.py, which need no ttnn
# either), so prepare gives only images that the encoder's check_pixels takes.
IMAGE_START = 200029  # "]<]start of image[>[" in the MiniMax-M3 tokenizer
IMAGE_END = 200030  # "]<]end of image[>["
# Resized images, (height, width) in pixels, that load runs once so that the chip's programs for their patch counts
# are compiled before the first request: 576 tokens (672 x 672, and 16:9 photos resized to 896 x 504) and 540 tokens
# (4:3 photos resized to 756 x 560). The programs depend on the patch count only; an image of another count compiles
# them on its first encode, 15 to 42 s with an empty kernel cache. model_options.warmup_sizes replaces this list.
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
    """options["warmup_sizes"] (default WARMUP_SIZES): a list of [height, width] sizes of resized images, each one
    that the encoder takes (check_size). Raises ValueError otherwise, before load imports ttnn."""
    sizes = options.get("warmup_sizes", WARMUP_SIZES)
    if not isinstance(sizes, (list, tuple)):
        raise ValueError(f"warmup_sizes must be a list of [height, width] sizes, got {sizes!r}")
    checked = []
    for size in sizes:
        if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(type(s) is int and s > 0 for s in size)):
            raise ValueError(f"warmup_sizes entries must be [height, width] in pixels, got {size!r}")
        try:
            check_size(*size)
        except ValueError as error:
            raise ValueError(f"warmup_sizes entry {list(size)}: {error}") from None
        checked.append(tuple(size))
    return checked


def _clear_frames(error: BaseException) -> None:
    """Clear the locals of the frames that error's traceback keeps, and those of the exceptions it chains: the frames
    of a failed load or encode hold the encoder's device tensors. The tracebacks keep their lines, not their locals."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending += [error.__cause__, error.__context__]


def _stub_rows(item: M3Image, hidden: int) -> np.ndarray:
    """Deterministic stand-in embeddings of item, bf16 bit patterns [n_tokens, hidden], seeded by its pixels."""
    digest = hashlib.sha256(np.asarray(item.pixels.shape).tobytes() + item.pixels.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rows = torch.randn(item.n_tokens, hidden, generator=torch.Generator().manual_seed(seed)) * 0.02
    return rows.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


class MiniMaxM3Media:
    """The MiniMax-M3 media module that the encoder service calls.

    The encoder service must not call load, encode or close from several threads at once.
    """

    def __init__(self, hf_config: dict):
        self.hf_config = hf_config
        self.hidden = hf_config["text_config"]["hidden_size"]
        self.placeholder_token_ids = {"image": hf_config["image_token_index"]}
        # The service opens the chip with these. ttnn's defaults suit the encoder: it uses no L1_SMALL memory (no
        # convolution or pooling op), captures no trace and runs on one chip, without fabric.
        self.device_params = {}
        vision = hf_config["vision_config"]
        patch_size, merge_size = vision["patch_size"], vision.get("spatial_merge_size", MERGE_SIZE)
        if (patch_size, merge_size) != (PATCH_SIZE, MERGE_SIZE):
            raise ValueError(
                f"the media module makes patches of {PATCH_SIZE} x {PATCH_SIZE} pixels, merged {MERGE_SIZE} x"
                f" {MERGE_SIZE}; vision_config has patch_size {patch_size} and spatial_merge_size {merge_size}"
            )
        self._stub = False
        self._encoder = None

    def prepare(self, kind: str, media: np.ndarray) -> M3Image:
        if kind != "image":
            raise ValueError(f"MiniMax-M3 does not accept {kind}")
        if media.dtype != np.uint8 or media.ndim != 3 or media.shape[2] != 3 or 0 in media.shape:
            raise ValueError(f"expected a non-empty uint8 RGB [height, width, 3], got {media.dtype} {media.shape}")
        h, w = smart_resize(*media.shape[:2])
        # torch.from_numpy warns about a read-only array and rejects negative strides, so copy those first.
        chw = torch.from_numpy(np.require(media, requirements=("C_CONTIGUOUS", "WRITEABLE"))).permute(2, 0, 1)
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
        a weight. Then load encodes one blank image of each warm-up size (options["warmup_sizes"], default
        WARMUP_SIZES) to compile the chip's programs. Raises ValueError for a weights_path that is not a directory, a
        bad warm-up size or a mesh_device of None (load opens no chip) before it imports ttnn. If building the encoder
        or a warm-up fails, load frees what it put on the chip and raises the error."""
        if kind not in self.placeholder_token_ids:
            raise ValueError(f"MiniMax-M3 does not accept {kind}")
        if self._stub or self._encoder is not None:
            raise RuntimeError("load after load: close first")
        if options.get("stub_embeddings"):
            self._stub = True
            return
        if not weights_path or not os.path.isdir(weights_path):
            raise ValueError(f"weights_path must be a MiniMax-M3 checkpoint directory, got {weights_path!r}")
        warmup_sizes = _warmup_sizes(options)
        if mesh_device is None:
            raise ValueError("mesh_device is None: load runs the encoder on the chip that the caller opened")
        import ttnn
        from models.demos.minimax_m3.vision.encoder import TtMiniMaxM3VisionEncoder

        try:
            encoder = TtMiniMaxM3VisionEncoder.from_weights_path(mesh_device, weights_path, self.hf_config)
            for height, width in warmup_sizes:
                ttnn.deallocate(encoder.forward(np.zeros((height, width, 3), dtype=np.uint8)))
        except BaseException as error:
            encoder = None
            _clear_frames(error)  # the failed call's frames hold the encoder's device tensors
            gc.collect()  # and so may reference cycles: all are freed before load raises
            raise
        self._encoder = encoder

    def encode(self, items: list[M3Image], out: list[memoryview]) -> None:
        """Write each item's stored value into its slot in out: format.Header("image"), then the item's vision tokens,
        bf16 [n_tokens, hidden].

        Raises ValueError for a slot that is not stored_nbytes(item) bytes before it writes into any slot. A slot may
        be a memoryview of any format, uint16 for example: its size is counted in bytes. The slots of a call that
        raises hold no defined value: the caller must not use them. If the chip work fails, encode frees that call's
        device tensors before it raises. The error's frames then hold none of the encoder's tensors, so close frees
        them even while the caller still holds the error. Raises RuntimeError before load or after close.
        """
        for item, buf in zip(items, out, strict=True):
            size = self.stored_nbytes(item)
            if buf.nbytes != size:
                raise ValueError(
                    f"a slot of {buf.nbytes} B for an image of {item.n_tokens} tokens, which needs {size} B"
                )
        if not (self._stub or self._encoder is not None):
            raise RuntimeError("encode before load, or after close")
        for item, buf in zip(items, out):
            buf = buf.cast("B")  # the header and the rows are at byte offsets, whatever the slot's format
            media_format.Header("image").write(buf)
            rows = media_format.embeddings(buf, self.hidden)
            if self._stub:
                rows[:] = _stub_rows(item, self.hidden)
            else:
                self._encode_into(item, rows)

    def _encode_into(self, item: M3Image, rows: np.ndarray) -> None:
        """The vision tokens of item, read from the chip straight into rows (format.embeddings of its value). If
        forward or the read fails, it clears the failed call's frames before it raises, as load does."""
        import ttnn
        from models.demos.minimax_m3.vision.encoder import read_embeddings

        try:
            tokens = self._encoder.forward(item.pixels)
            try:
                read_embeddings(tokens, rows)
            finally:
                ttnn.deallocate(tokens)
        except BaseException as error:
            _clear_frames(error)  # so that a close while the caller holds the error frees the encoder's device tensors
            raise

    def close(self) -> None:
        """Free the encoder and its device tensors. The chip is the caller's: close leaves it open."""
        loaded = self._encoder is not None
        self._stub = False
        self._encoder = None
        if loaded:
            gc.collect()  # the encoder's device tensors that only reference cycles still hold
