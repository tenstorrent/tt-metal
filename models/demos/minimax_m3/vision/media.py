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
import traceback
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from torchvision.transforms.v2 import functional as tvF

from models.demos.minimax_m3.tt.vision.preprocess import (
    FACTOR,
    MAX_PIXELS,
    MAX_RATIO,
    MIN_PIXELS,
    PATCH_SIZE,
    check_size,
)
from models.demos.minimax_m3.vision import format as media_format

# The resize's geometry is the vision encoder's (tt/vision/preprocess.py, which needs no ttnn either), so prepare gives
# only images that the encoder's check_pixels takes.
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
    """options["warmup"] (default WARMUP_SIZES): a list of [height, width] sizes of resized images, each one that the
    encoder takes (check_size). Raises ValueError otherwise, before load opens a chip."""
    sizes = options.get("warmup", WARMUP_SIZES)
    if not isinstance(sizes, (list, tuple)):
        raise ValueError(f"warmup must be a list of [height, width] sizes, got {sizes!r}")
    checked = []
    for size in sizes:
        if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(type(s) is int and s > 0 for s in size)):
            raise ValueError(f"warmup sizes must be [height, width] in pixels, got {size!r}")
        try:
            check_size(*size)
        except ValueError as error:
            raise ValueError(f"warmup size {list(size)}: {error}") from None
        checked.append(tuple(size))
    return checked


def _clear_frames(error: BaseException) -> None:
    """Clear the locals of the frames that error's traceback keeps, and those of the exceptions it chains: the frames
    of a failed load hold the encoder's device tensors. The tracebacks keep their lines, not their locals."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending += [error.__cause__, error.__context__]


def _free_device(mesh_device) -> None:
    """Free the encoder's device tensors that only reference cycles still hold, then close mesh_device if it is given:
    the tensors go before their chip closes."""
    gc.collect()
    if mesh_device is not None:
        import ttnn

        ttnn.close_mesh_device(mesh_device)


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
        # chip: load, each encode call and close run one at a time.
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
        that is not a directory or a bad warm-up size before it opens a chip. If building the encoder or a warm-up
        fails, load frees what it put on the chip, closes a chip it opened, and raises the error. encode and close
        wait for a load in progress."""
        with self._lock:
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
            try:
                encoder = TtMiniMaxM3VisionEncoder.from_weights_path(
                    own_mesh_device if mesh_device is None else mesh_device, weights_path, self.hf_config
                )
                for height, width in warmup:
                    ttnn.deallocate(encoder.forward(np.zeros((height, width, 3), dtype=np.uint8)))
            except BaseException as error:
                encoder = None
                _clear_frames(error)  # so that the encoder's device tensors are freed before the chip closes
                try:
                    _free_device(own_mesh_device)
                except Exception as close_error:  # report the load's error, the one to fix
                    logger.warning(f"freeing the chip after a failed load failed too: {close_error!r}")
                raise
            self._encoder, self._own_mesh_device = encoder, own_mesh_device

    def encode(self, items: list[M3Image], out: list[memoryview]) -> None:
        """Write each item's stored value into its slot in out: the header and the item's vision tokens, bf16.

        Raises ValueError for a slot that is not value_size(item) bytes before it writes into any slot. It clears every
        slot's header first and writes each one only after its item's embeddings, so a call that fails leaves valid
        values only for the items it finished, never a previous value's. Raises RuntimeError before load or after
        close; waits for a load or another encode in progress.
        """
        headers = []
        for item, buf in zip(items, out, strict=True):
            h = media_format.Header(item.grid_h, item.grid_w, item.n_tokens, self.hidden)
            size = media_format.value_size(h.n_tokens, h.hidden)
            if len(buf) != size:
                raise ValueError(f"a slot of {len(buf)} B for an image of {h.n_tokens} tokens, which needs {size} B")
            headers.append(h)
        with self._lock:
            if not (self._stub or self._encoder is not None):
                raise RuntimeError("encode before load, or after close")
            for buf in out:
                buf[: media_format.HEADER_BYTES] = bytes(media_format.HEADER_BYTES)
            for item, buf, h in zip(items, out, headers):
                rows = media_format.embeddings(buf, h)
                if self._stub:
                    rows[:] = _stub_rows(item, self.hidden)
                else:
                    self._encode_into(item, rows)
                media_format.write_header(buf, h)

    def _encode_into(self, item: M3Image, rows: np.ndarray) -> None:
        """The vision tokens of item, read from the chip straight into rows (format.embeddings of its value). The
        caller holds the lock."""
        import ttnn
        from models.demos.minimax_m3.vision.encoder import read_embeddings

        tokens = self._encoder.forward(item.pixels)
        try:
            read_embeddings(tokens, rows)
        finally:
            ttnn.deallocate(tokens)

    def close(self) -> None:
        """Free the encoder, and close the chip if load opened it (once, even if closing raises). Waits for a load or
        an encode in progress, and a load waits for it."""
        with self._lock:
            loaded = self._encoder is not None
            mesh_device, self._own_mesh_device = self._own_mesh_device, None
            self._stub = False
            self._encoder = None
            if loaded:
                _free_device(mesh_device)
