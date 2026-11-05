# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 media module and stored-value format, host only."""

import os
import pickle
import subprocess
import sys
import weakref
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


def test_failed_encode_leaves_no_stale_value(monkeypatch, expect_error):
    """encode clears every slot's header first and writes each value's header only after its embeddings: when an item
    fails, the slots of the items before it hold their new values and the others no valid header, never the value a
    previous call left there, so prefill rejects them."""
    media = MiniMaxM3Media(CONFIG)
    media.load("", {"stub_embeddings": True})
    items = [media.prepare("image", rgba(100, 60)), media.prepare("image", rgba(60, 100, seed=1))]
    values = [memoryview(bytearray(media.value_size(item))) for item in items]
    media.encode(items, values)  # a valid value in each slot
    encoded = [bytes(value) for value in values]
    stub_rows = media_module._stub_rows
    for failing in (1, 0):
        calls = []

        def fail_at(item, hidden):
            calls.append(item)
            if len(calls) - 1 == failing:
                raise RuntimeError("the chip went away")
            return stub_rows(item, hidden)

        with monkeypatch.context() as patch:
            patch.setattr(media_module, "_stub_rows", fail_at)
            with expect_error(RuntimeError, "the chip went away"):
                media.encode(items, values)
        assert all(bytes(value) == want for value, want in zip(values[:failing], encoded)), "a finished item's value"
        for value in values[failing:]:
            with expect_error(ValueError, "no embeddings"):
                media_format.read_header(value)


def test_encode_checks_its_slots_and_state_first(expect_error):
    """encode raises ValueError for a slot that is not its item's value size before it writes into any slot, and
    RuntimeError before load and after close."""
    media = MiniMaxM3Media(CONFIG)
    item = media.prepare("image", rgba(100, 60))
    good = memoryview(bytearray(b"\x01" * media.value_size(item)))
    with expect_error(RuntimeError, "encode before load"):
        media.encode([item], [good])
    media.load("", {"stub_embeddings": True})
    long = memoryview(bytearray(media.value_size(item) + 2))
    with expect_error(ValueError, "needs"):
        media.encode([item, item], [good, long])
    assert bytes(good) == b"\x01" * len(good), "encode wrote into a slot before it checked them all"
    media.close()
    with expect_error(RuntimeError, "after close"):
        media.encode([item], [good])


@pytest.mark.parametrize("chained", [False, True], ids=["raised", "chained"])
def test_failed_load_frees_the_encoder_before_closing_its_chip(tmp_path, monkeypatch, expect_error, chained):
    """If building the encoder fails on the chip that load opened, load frees the device tensors that the failed
    call's frames hold (the exception's traceback keeps those frames, and so does the traceback of an exception it
    chains) before it closes the chip, then raises."""
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

    monkeypatch.setattr(ttnn, "open_mesh_device", lambda shape: "chip")
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda mesh: closed.append((mesh, tensors[0]() is None)))
    monkeypatch.setattr(encoder_module.TtMiniMaxM3VisionEncoder, "from_weights_path", staticmethod(from_weights_path))
    with expect_error(RuntimeError, "out of device memory"):
        MiniMaxM3Media(CONFIG).load(str(tmp_path), {"warmup": []})
    assert closed == [("chip", True)], f"load closed its chip with the failed call's tensors alive: {closed}"


@pytest.mark.parametrize("failing", ["forward", "read"])
@pytest.mark.parametrize("own_chip", [True, False], ids=["owned", "borrowed"])
def test_failed_encode_keeps_no_device_tensors_past_close(tmp_path, monkeypatch, expect_error, failing, own_chip):
    """If the encoder's forward or the read of its tokens fails, encode clears the locals of the failed call's frames
    before it raises, as load does: a close while the caller still holds the error, in a finally for example, frees the
    encoder and the failed call's device tensors, before it closes a chip that load opened. The tokens of a failed read
    are still freed, once."""
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

    monkeypatch.setattr(ttnn, "open_mesh_device", lambda shape: "chip")
    monkeypatch.setattr(ttnn, "close_mesh_device", lambda mesh: closed.append((mesh, all_freed())))
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    monkeypatch.setattr(encoder_module.TtMiniMaxM3VisionEncoder, "from_weights_path", staticmethod(from_weights_path))
    monkeypatch.setattr(encoder_module, "read_embeddings", read_embeddings)
    media = MiniMaxM3Media(CONFIG)
    media.load(str(tmp_path), {"warmup": []}, mesh_device=None if own_chip else "the caller's chip")
    item = media.prepare("image", rgba(56, 56))
    with expect_error(RuntimeError, "the chip went away") as failed:
        media.encode([item], [memoryview(bytearray(media.value_size(item)))])
    media.close()  # the caller still holds the error
    assert failed.value is not None
    assert closed == ([("chip", True)] if own_chip else []), f"close closed its chip with them alive: {closed}"
    assert all_freed(), "close left the encoder or the failed call's tensors alive while the caller holds the error"
    assert freed == (["tokens"] if failing == "read" else []), f"freed {freed}: the tokens of a failed read, once"


@pytest.mark.parametrize(
    "warmup, message",
    [
        ([[100, 100]], "multiples of 28"),
        ([5], r"\[height, width\] in pixels"),
        ([[28, 28]], r"got 1 \(1 down"),
        ([[700, 700]], "got 625"),
        ([[28, 16128]], "aspect ratio must be at most 339"),  # 576 tokens, but 576:1
        (None, "must be a list"),
    ],
)
def test_load_rejects_bad_warmup_sizes(tmp_path, expect_error, warmup, message):
    """A warm-up size that no resized image has raises ValueError before load imports ttnn or opens a chip."""
    with expect_error(ValueError, message):
        MiniMaxM3Media(CONFIG).load(str(tmp_path), {"warmup": warmup})
