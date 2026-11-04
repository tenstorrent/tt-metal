# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module and stored-value format, host only."""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.demos.minimax_m3.vision import format as media_format
from models.demos.minimax_m3.vision import media as media_module
from models.demos.minimax_m3.vision.media import IMAGE_END, IMAGE_START, M3Image, MiniMaxM3Media

CONFIG = {"image_token_index": 200025, "text_config": {"hidden_size": 64}, "vision_config": {"patch_size": 14}}


def rgba(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)


@pytest.mark.parametrize("hw", [(672, 896), (560, 756), (30, 30), (4000, 3000), (28, 5000), (1, 150)])
def test_prepare_matches_hf(hw):
    from PIL import Image
    from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor

    proc = MiniMaxM3VLImageProcessor()
    media = MiniMaxM3Media(CONFIG)
    image = rgba(*hw)
    item = media.prepare("image", image)

    want = proc(images=[Image.fromarray(image)], return_tensors="pt")
    # HF leaves an image of the resized size as is, so ours gives the same patches only if the pixels match.
    got = proc(images=[item.pixels], return_tensors="pt")
    assert want["image_grid_thw"].tolist() == [[1, item.grid_h * 2, item.grid_w * 2]]
    assert torch.equal(got["pixel_values"], want["pixel_values"])
    assert media.expansion(item) == [(IMAGE_START, 1), (200025, item.n_tokens), (IMAGE_END, 1)]
    assert item.n_tokens <= 576


@pytest.mark.parametrize(
    "kind,image,message",
    [
        ("video", rgba(28, 28), "does not accept video"),
        ("image", rgba(1, 201), "aspect ratio"),
        ("image", rgba(28, 28)[..., :3], "RGBA"),
        ("image", rgba(0, 28), "non-empty"),
    ],
)
def test_prepare_rejects(expect_error, kind, image, message):
    with expect_error(ValueError, message):
        MiniMaxM3Media(CONFIG).prepare(kind, image)


def test_stub_encode_writes_a_value_prefill_reads(expect_error):
    media = MiniMaxM3Media(CONFIG)
    with expect_error(ValueError, "checkpoint directory"):
        media.load("", {})  # before any chip or ttnn import
    media.load("", {"stub_embeddings": True})
    with expect_error(RuntimeError, "load after load"):
        media.load("", {"stub_embeddings": True})
    item = media.prepare("image", rgba(100, 60))
    bufs = [memoryview(bytearray(media.value_size(item))) for _ in range(2)]
    media.encode([item], bufs[:1])
    media.encode([item], bufs[1:])

    h = media_format.read_header(bufs[0])
    assert h == media_format.Header(item.grid_h, item.grid_w, item.n_tokens, 64)
    assert bytes(bufs[0]) == bytes(bufs[1])
    transposed = M3Image(item.pixels.reshape(item.pixels.shape[1], item.pixels.shape[0], 3), item.grid_w, item.grid_h)
    other = memoryview(bytearray(media.value_size(transposed)))
    media.encode([transposed], [other])
    assert bytes(other[media_format.HEADER_BYTES :]) != bytes(bufs[0][media_format.HEADER_BYTES :])
    assert media_format.embeddings(bufs[0], h).any()


def test_format_rejects_bad_values(expect_error):
    h = media_format.Header(2, 3, 6, 64)
    buf = memoryview(bytearray(media_format.value_size(6, 64)))
    media_format.write_header(buf, h)
    assert media_format.read_header(buf) == h
    with expect_error(ValueError, "needs"):
        media_format.read_header(buf[:-2])
    buf[8:12] = (5).to_bytes(4, "little")
    with expect_error(ValueError, "grid_h"):
        media_format.read_header(buf)
    with expect_error(ValueError, "no embeddings"):
        media_format.read_header(memoryview(bytearray(media_format.HEADER_BYTES)))
    buf[8:12] = (6).to_bytes(4, "little")
    buf[40] = 1
    with expect_error(ValueError, "reserved"):
        media_format.read_header(buf)


def test_image_pickles_out_of_band():
    item = MiniMaxM3Media(CONFIG).prepare("image", rgba(672, 896))
    buffers = []
    stream = pickle.dumps(item, protocol=5, buffer_callback=buffers.append)
    assert len(stream) < 4096
    back = pickle.loads(stream, buffers=buffers)
    assert (back.grid_h, back.grid_w) == (item.grid_h, item.grid_w)
    assert np.array_equal(back.pixels, item.pixels)


def test_media_imports_without_ttnn():
    """The media module imports without ttnn, so a process that only prepares media runs without it; only load
    imports ttnn and the encoder."""
    repo = Path(__file__).resolve().parents[5]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(repo), os.environ.get("PYTHONPATH")]))}
    code = (
        "import sys; import models.demos.minimax_m3.vision.media; print(sorted(m for m in sys.modules if m == 'ttnn'))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stdout + out.stderr


def test_failed_encode_leaves_no_valid_header(monkeypatch, expect_error):
    """encode writes a value's header only after its embeddings: if they fail, the slot holds no valid header, not
    the one a previous value left there, so prefill rejects it."""
    media = MiniMaxM3Media(CONFIG)
    media.load("", {"stub_embeddings": True})
    item = media.prepare("image", rgba(100, 60))
    value = memoryview(bytearray(media.value_size(item)))
    media.encode([item], [value])
    media_format.read_header(value)  # a valid value in the slot

    def fail(item, hidden):
        raise RuntimeError("the chip went away")

    monkeypatch.setattr(media_module, "_stub_rows", fail)
    with expect_error(RuntimeError, "the chip went away"):
        media.encode([item], [value])
    with expect_error(ValueError, "no embeddings"):
        media_format.read_header(value)


@pytest.mark.parametrize(
    "warmup, message",
    [
        ([[100, 100]], "multiples of 28"),
        ([5], "multiples of 28"),
        ([[28, 28]], "makes 1 tokens"),
        ([[700, 700]], "makes 625"),
    ],
)
def test_load_rejects_bad_warmup_sizes(tmp_path, expect_error, warmup, message):
    """A warm-up size that no resized image has raises ValueError before load imports ttnn or opens a chip."""
    with expect_error(ValueError, message):
        MiniMaxM3Media(CONFIG).load(str(tmp_path), {"warmup": warmup})
