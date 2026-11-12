# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module and stored-value format, host only."""

import os
import pickle
import subprocess
import sys
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from models.demos.minimax_m3.vision import format as media_format
from models.demos.minimax_m3.vision.media import IMAGE_END, IMAGE_START, M3Image, MiniMaxM3Media

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


def test_prepare_takes_read_only_and_flipped_images(monkeypatch):
    """prepare takes a read-only image without a warning, and a flipped or channel-swapped view as it takes a copy."""
    from_numpy = torch.from_numpy

    def from_numpy_warns_every_time(array):  # torch itself warns about a read-only array once per process
        if not array.flags.writeable:
            warnings.warn("The given NumPy array is not writable")
        return from_numpy(array)

    monkeypatch.setattr(torch, "from_numpy", from_numpy_warns_every_time)
    media = MiniMaxM3Media(CONFIG)
    image = rgb(100, 60)
    read_only = image.copy()
    read_only.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(media.prepare("image", read_only).pixels, media.prepare("image", image).pixels)
    for view in (image[::-1], image[:, ::-1], image[..., ::-1]):
        assert np.array_equal(media.prepare("image", view).pixels, media.prepare("image", view.copy()).pixels)


def test_stub_encode_writes_a_value_prefill_reads(expect_error):
    media = MiniMaxM3Media(CONFIG)
    with expect_error(ValueError, "does not accept video"):
        media.load(None, "", "video", {"stub_embeddings": True})
    with expect_error(ValueError, "checkpoint directory"):
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


def test_load_after_load(expect_error):
    """load raises RuntimeError when called again before close, and loads again after close."""
    media = MiniMaxM3Media(CONFIG)
    media.load(None, "", "image", {"stub_embeddings": True})
    with expect_error(RuntimeError, "load after load"):
        media.load(None, "", "image", {"stub_embeddings": True})
    media.close()
    media.load(None, "", "image", {"stub_embeddings": True})


def test_load_needs_the_callers_chip(tmp_path, expect_error):
    """Without stub_embeddings, load raises ValueError for a mesh_device of None: it opens no chip of its own."""
    with expect_error(ValueError, "mesh_device is None"):
        MiniMaxM3Media(CONFIG).load(None, str(tmp_path), "image", {})


def test_encode_writes_every_item_of_a_call():
    """One encode call with two images of different sizes writes each image's header and its own rows into its slot,
    all of them, as a call with that image alone does."""
    from models.demos.minimax_m3.vision.media import _stub_rows

    media = MiniMaxM3Media(CONFIG)
    media.load(None, "", "image", {"stub_embeddings": True})
    header = memoryview(bytearray(media_format.stored_nbytes(0, 64)))
    media_format.Header("image").write(header)
    items = [media.prepare("image", rgb(100, 60)), media.prepare("image", rgb(60, 150, seed=1))]
    together = [memoryview(bytearray(b"\xff" * media.stored_nbytes(item))) for item in items]
    media.encode(items, together)
    for item, value in zip(items, together):
        alone = memoryview(bytearray(b"\xff" * media.stored_nbytes(item)))
        media.encode([item], [alone])
        assert bytes(value) == bytes(alone)
        assert media_format.Header.read(value) == media_format.Header("image")
        assert bytes(value[: len(header)]) == bytes(header)
        assert np.array_equal(media_format.embeddings(value, 64), _stub_rows(item, 64))


def test_encode_checks_its_slots_and_state_first(expect_error):
    """encode raises ValueError for a slot that is not its item's stored size before it writes into any slot, and
    RuntimeError before load and after close."""
    media = MiniMaxM3Media(CONFIG)
    item = media.prepare("image", rgb(100, 60))
    good = memoryview(bytearray(b"\x01" * media.stored_nbytes(item)))
    with expect_error(RuntimeError, "encode before load"):
        media.encode([item], [good])
    media.load(None, "", "image", {"stub_embeddings": True})
    row = media_format.stored_nbytes(2, 64) - media_format.stored_nbytes(1, 64)
    for size in (media.stored_nbytes(item) + row, media.stored_nbytes(item) - row):  # one row more, one row less
        with expect_error(ValueError, "needs"):
            media.encode([item, item], [good, memoryview(bytearray(size))])
        assert bytes(good) == b"\x01" * len(good), "encode wrote into a slot before it checked them all"
    media.close()
    with expect_error(RuntimeError, "after close"):
        media.encode([item], [good])


def test_encode_counts_slot_sizes_in_bytes(expect_error):
    """encode takes a slot of another format, a uint16 memoryview for example, by its size in bytes, and writes it as
    it writes a slot of bytes. A slot one byte short raises ValueError."""
    media = MiniMaxM3Media(CONFIG)
    media.load(None, "", "image", {"stub_embeddings": True})
    item = media.prepare("image", rgb(100, 60))
    size = media.stored_nbytes(item)
    want = memoryview(bytearray(size))
    media.encode([item], [want])
    for slot in (memoryview(np.full(size // 2, 0xFFFF, np.uint16)), memoryview(bytearray(b"\xff" * size)).cast("H")):
        media.encode([item], [slot])
        assert slot.format == "H" and slot.tobytes() == want.tobytes()
    with expect_error(ValueError, "needs"):
        media.encode([item], [memoryview(bytearray(size - 1))])


@pytest.mark.parametrize("chained", [False, True], ids=["raised", "chained"])
def test_failed_load_frees_what_it_put_on_the_chip(tmp_path, monkeypatch, expect_error, chained):
    """If building the encoder fails, load frees the device tensors that the failed call's frames hold (the exception's
    traceback keeps those frames, and so does the traceback of an exception it chains) before it raises: they are
    freed while the caller still holds the error, and the caller's chip is not closed."""
    ttnn = pytest.importorskip("ttnn")
    from models.demos.minimax_m3.vision import encoder as encoder_module

    class DeviceTensor:  # stands in for a tensor on the chip
        pass

    tensors, closed = [], []

    def build():
        tensor = DeviceTensor()  # held only by this frame
        tensors.append(weakref.ref(tensor))
        raise RuntimeError("out of device memory")

    def from_weights_path(mesh_device, weights_path, config):
        if not chained:
            build()
        try:
            build()
        except RuntimeError as error:  # only the chained exception's traceback keeps build's frame
            raise RuntimeError("out of device memory, while building the encoder") from error

    monkeypatch.setattr(ttnn, "close_mesh_device", closed.append)
    monkeypatch.setattr(encoder_module.TtMiniMaxM3VisionEncoder, "from_weights_path", staticmethod(from_weights_path))
    with expect_error(RuntimeError, "out of device memory") as failed:
        MiniMaxM3Media(CONFIG).load("the caller's chip", str(tmp_path), "image", {"warmup_sizes": []})
    assert failed.value is not None
    assert tensors[0]() is None, "load raised with the failed call's tensors alive"
    assert closed == [], f"load closed the caller's chip: {closed}"


def test_failed_warmup_frees_the_encoder(tmp_path, monkeypatch, expect_error):
    """If a warm-up fails after the encoder is built, load frees the encoder and its device tensors before it raises:
    they are freed while the caller still holds the error, and the caller's chip is not closed."""
    ttnn = pytest.importorskip("ttnn")
    from models.demos.minimax_m3.vision import encoder as encoder_module

    class DeviceTensor:  # stands in for a tensor on the chip
        pass

    held, closed = [], []  # held: weak references to the encoder and its weights

    class Encoder:
        def __init__(self):
            self.weights = DeviceTensor()
            held.append(weakref.ref(self.weights))

        def forward(self, pixels):  # this frame holds the encoder, its self
            raise RuntimeError("out of device memory, in the warm-up")

    def from_weights_path(mesh_device, weights_path, config):
        encoder = Encoder()
        held.append(weakref.ref(encoder))
        return encoder

    monkeypatch.setattr(ttnn, "close_mesh_device", closed.append)
    monkeypatch.setattr(encoder_module.TtMiniMaxM3VisionEncoder, "from_weights_path", staticmethod(from_weights_path))
    with expect_error(RuntimeError, "in the warm-up") as failed:
        MiniMaxM3Media(CONFIG).load("the caller's chip", str(tmp_path), "image", {"warmup_sizes": [[224, 224]]})
    assert failed.value is not None
    assert all(ref() is None for ref in held), "load raised with the encoder of the failed warm-up alive"
    assert closed == [], f"load closed the caller's chip: {closed}"


@pytest.mark.parametrize("failing", ["forward", "read"])
def test_failed_encode_keeps_no_device_tensors_past_close(tmp_path, monkeypatch, expect_error, failing):
    """If the encoder's forward or the read of its tokens fails, encode clears the locals of the failed call's frames
    before it raises, as load does: a close while the caller still holds the error, in a finally for example, frees the
    encoder and the failed call's device tensors and leaves the caller's chip open. The tokens of a failed read are
    still freed, once."""
    ttnn = pytest.importorskip("ttnn")
    from models.demos.minimax_m3.vision import encoder as encoder_module

    class DeviceTensor:  # stands in for a tensor on the chip
        pass

    held, closed, freed = [], [], []  # held: weak references to the encoder and to tensors only the frames hold

    def all_freed():
        return all(ref() is None for ref in held)

    class Encoder:
        def forward(self, pixels):  # this frame holds the encoder, its self
            tensor = DeviceTensor()  # held only by this frame
            held.append(weakref.ref(tensor))
            if failing == "forward":
                raise RuntimeError("the chip went away")
            return "tokens"

    def read_embeddings(tokens, out):
        chip_tokens = DeviceTensor()  # held only by this frame, as read_embeddings' view of the tokens is
        held.append(weakref.ref(chip_tokens))
        raise RuntimeError("the chip went away")

    def from_weights_path(mesh_device, weights_path, config):
        encoder = Encoder()
        held.append(weakref.ref(encoder))
        return encoder

    monkeypatch.setattr(ttnn, "close_mesh_device", closed.append)
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    monkeypatch.setattr(encoder_module.TtMiniMaxM3VisionEncoder, "from_weights_path", staticmethod(from_weights_path))
    monkeypatch.setattr(encoder_module, "read_embeddings", read_embeddings)
    media = MiniMaxM3Media(CONFIG)
    media.load("the caller's chip", str(tmp_path), "image", {"warmup_sizes": []})
    item = media.prepare("image", rgb(56, 56))
    with expect_error(RuntimeError, "the chip went away") as failed:
        media.encode([item], [memoryview(bytearray(media.stored_nbytes(item)))])
    media.close()  # the caller still holds the error
    assert failed.value is not None
    assert all_freed(), "close left the encoder or the failed call's tensors alive while the caller holds the error"
    assert closed == [], f"close closed the caller's chip: {closed}"
    assert freed == (["tokens"] if failing == "read" else []), f"freed {freed}: the tokens of a failed read, once"


@pytest.mark.parametrize(
    "warmup_sizes, message",
    [
        ([[100, 100]], "multiples of 28"),
        ([5], r"\[height, width\] in pixels"),
        ([[28, 28]], r"got 1 \(1 down"),
        ([[700, 700]], "got 625"),
        ([[28, 16128]], "aspect ratio must be at most 339"),  # 576 tokens, but 576:1
        (None, "warmup_sizes must be a list"),
    ],
)
def test_load_rejects_bad_warmup_sizes(tmp_path, expect_error, warmup_sizes, message):
    """A warmup_sizes that is not a list of sizes, or a warm-up size that no resized image has, raises ValueError before
    load imports ttnn or builds the encoder."""
    with expect_error(ValueError, message):
        MiniMaxM3Media(CONFIG).load(None, str(tmp_path), "image", {"warmup_sizes": warmup_sizes})
