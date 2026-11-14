# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module and stored-value format, host only."""

import pickle

import numpy as np
import pytest
import torch

from models.common.multimedia_store.reader import FetchRequest, Missing
from models.demos.common.prefill.media_stub import stub_media_key
from models.demos.minimax_m3.vision import format as media_format
from models.demos.minimax_m3.vision.media import IMAGE_END, IMAGE_START, M3Image, MiniMaxM3Media
from models.demos.minimax_m3.vision.stub_reader import StubEmbeddingReader

CONFIG = {
    "image_token_index": 200025,
    "text_config": {"hidden_size": 64},
    "vision_config": {"patch_size": 14, "spatial_merge_size": 2},
}


def rgb(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("hw", [(672, 896), (560, 756), (30, 30), (4000, 3000), (28, 5000), (1, 150)])
def test_prepare_matches_hf(hw):
    from PIL import Image
    from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor

    proc = MiniMaxM3VLImageProcessor()
    media = MiniMaxM3Media(CONFIG)
    image = rgb(*hw)
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
        ("video", rgb(28, 28), "does not accept video"),
        ("image", rgb(1, 201), "aspect ratio"),
        ("image", np.zeros((28, 28, 4), np.uint8), "RGB"),
        ("image", rgb(0, 28), "non-empty"),
    ],
)
def test_prepare_rejects(expect_error, kind, image, message):
    with expect_error(ValueError, message):
        MiniMaxM3Media(CONFIG).prepare(kind, image)


def test_stub_encode_writes_a_value_prefill_reads(expect_error):
    media = MiniMaxM3Media(CONFIG)
    with expect_error(ValueError, "does not accept video"):
        media.load(None, "", "video", {"stub_embeddings": True})
    with expect_error(NotImplementedError, "stub_embeddings"):
        media.load(None, "", "image", {})
    media.load(None, "", "image", {"stub_embeddings": True})
    item = media.prepare("image", rgb(100, 60))
    bufs = [memoryview(bytearray(media.stored_nbytes(item))) for _ in range(2)]
    media.encode([item], bufs[:1])
    media.encode([item], bufs[1:])

    assert media_format.Header.read(bufs[0]) == media_format.Header("image")
    rows = media_format.embeddings(bufs[0], 64)
    assert rows.shape == (item.n_tokens, 64) and rows.any()
    assert bytes(bufs[0]) == bytes(bufs[1])
    transposed = M3Image(item.pixels.reshape(item.pixels.shape[1], item.pixels.shape[0], 3), item.grid_w, item.grid_h)
    other = memoryview(bytearray(media.stored_nbytes(transposed)))
    media.encode([transposed], [other])
    assert bytes(other) != bytes(bufs[0])


def test_format_rejects_bad_values(expect_error):
    buf = memoryview(bytearray(media_format.stored_nbytes(6, 64)))
    assert media_format.embeddings(buf, 64).shape == (6, 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:-2], 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:16], 64)
    with expect_error(ValueError, "header"):
        media_format.Header.read(buf[:15])
    buf[0] = len(media_format.KINDS)
    with expect_error(ValueError, "unknown media kind"):
        media_format.Header.read(buf)


def test_image_pickles_out_of_band():
    item = MiniMaxM3Media(CONFIG).prepare("image", rgb(672, 896))
    buffers = []
    stream = pickle.dumps(item, protocol=5, buffer_callback=buffers.append)
    assert len(stream) < 4096
    back = pickle.loads(stream, buffers=buffers)
    assert (back.grid_h, back.grid_w) == (item.grid_h, item.grid_w)
    assert np.array_equal(back.pixels, item.pixels)


def test_stub_reader_serves_stub_keys(expect_error):
    reader = StubEmbeddingReader(64)
    key = stub_media_key(6, 0)
    value = reader.fetch(key)
    assert media_format.embeddings(value, 64).shape == (6, 64)
    assert bytes(value) == bytes(reader.fetch(key)) != bytes(reader.fetch(stub_media_key(6, 1)))
    head, rest = reader.fetch_batch([FetchRequest(key, 0, 16), FetchRequest(key, 16)])
    assert bytes(head) + bytes(rest) == bytes(value)
    for key in (bytes(32), b"\0"):
        with expect_error(Missing, key.hex()):
            reader.fetch(key)
