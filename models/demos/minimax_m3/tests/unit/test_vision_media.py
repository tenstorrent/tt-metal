# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module and stored-value format, host only."""

import pickle

import numpy as np
import pytest
import torch

from models.common.multimedia_store.reader import FetchRequest, Missing
from models.demos.common.prefill.media_stub import stub_media_key
from models.demos.minimax_m3.vision import format
from models.demos.minimax_m3.vision.media import IMAGE_END, IMAGE_START, MiniMaxM3Media
from models.demos.minimax_m3.vision.stub_reader import StubEmbeddingReader

CONFIG = {"image_token_index": 200025, "text_config": {"hidden_size": 64}, "vision_config": {"patch_size": 14}}


def rgba(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)


@pytest.mark.parametrize("hw", [(672, 896), (560, 756), (30, 30), (4000, 3000), (28, 5000), (1, 150)])
def test_prepare_matches_hf(hw):
    hf = pytest.importorskip("transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl")
    from PIL import Image

    proc = hf.MiniMaxM3VLImageProcessor()
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
    ],
)
def test_prepare_rejects(expect_error, kind, image, message):
    with expect_error(ValueError, message):
        MiniMaxM3Media(CONFIG).prepare(kind, image)


def test_stub_encode_writes_a_value_prefill_reads(expect_error):
    media = MiniMaxM3Media(CONFIG)
    with expect_error(NotImplementedError, "stub_embeddings"):
        media.load("", {})
    media.load("", {"stub_embeddings": True})
    item = media.prepare("image", rgba(100, 60))
    bufs = [memoryview(bytearray(media.value_size(item))) for _ in range(2)]
    media.encode([item], bufs[:1])
    media.encode([item], bufs[1:])

    h = format.read_header(bufs[0])
    assert h == format.Header(item.grid_h, item.grid_w, item.n_tokens, 64)
    assert bytes(bufs[0]) == bytes(bufs[1])
    assert format.embeddings(bufs[0], h).any()


def test_format_rejects_bad_values(expect_error):
    h = format.Header(2, 3, 6, 64)
    buf = memoryview(bytearray(format.value_size(6, 64)))
    format.write_header(buf, h)
    assert format.read_header(buf) == h
    with expect_error(ValueError, "needs"):
        format.read_header(buf[:-2])
    buf[8:12] = (5).to_bytes(4, "little")
    with expect_error(ValueError, "grid_h"):
        format.read_header(buf)


def test_image_pickles_out_of_band():
    item = MiniMaxM3Media(CONFIG).prepare("image", rgba(672, 896))
    buffers = []
    stream = pickle.dumps(item, protocol=5, buffer_callback=buffers.append)
    assert len(stream) < 4096
    back = pickle.loads(stream, buffers=buffers)
    assert (back.grid_h, back.grid_w) == (item.grid_h, item.grid_w)
    assert np.array_equal(back.pixels, item.pixels)


def test_stub_reader_serves_stub_keys(expect_error):
    reader = StubEmbeddingReader(64)
    key = stub_media_key(6, 0).hex()
    value = reader.fetch(key)
    assert format.read_header(value) == format.Header(1, 6, 6, 64)
    assert bytes(value) == bytes(reader.fetch(key)) != bytes(reader.fetch(stub_media_key(6, 1).hex()))
    head, rows = reader.fetch_batch([FetchRequest(key, 0, 16), FetchRequest(key, format.HEADER_BYTES)])
    assert bytes(head) == bytes(value[:16]) and bytes(rows) == bytes(value[format.HEADER_BYTES :])
    with expect_error(Missing, "00"):
        reader.fetch(bytes(32).hex())
