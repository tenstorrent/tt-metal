# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device tests of TtMiniMaxM3VisionEncoder (vision/encoder.py) against the 2-layer reference of conftest.py.

test_vision_encoder_vs_ref checks forward's output on noise images of the CI grids. M3_VISION_ALL_GRIDS=1 adds 84x56
pixels (h4w6), a size small images resize to. test_vision_encoder_load_and_encode uses the encoder as load() and
encode() do: one warm-up image, then two images read into their slots of one shared buffer.
test_vision_media_load_and_encode runs the media module's load(), with one warm-up size, then prepare() and encode().
Its chip is opened with the module's device_params, as the service opens it. The reference has seeded random weights;
only the media test writes them to a checkpoint. The encoder's host tests are in test_vision_encoder_api.py.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import ttnn
from models.demos.minimax_m3.tt.vision.preprocess import FACTOR, PATCH_SIZE, patch_rows
from models.demos.minimax_m3.vision import format as media_format
from models.demos.minimax_m3.vision.encoder import TtMiniMaxM3VisionEncoder, read_embeddings, vision_config
from models.demos.minimax_m3.vision.media import MiniMaxM3Media

from ..test_factory import parametrize_mesh_with_fabric
from .vision_checkpoint import write_checkpoint
from .vision_helpers import CI_GRIDS, MAX_REL_L2, VISION_GRIDS, assert_close, grid_name

# Resized image sizes in pixels as (height, width), by their patch grid: the CI grids of the vision tests, then h4w6.
SIZES = {
    grid_id: (h * PATCH_SIZE, w * PATCH_SIZE) for grid_id, (_, h, w) in VISION_GRIDS.items() if grid_id in CI_GRIDS
}
SIZES["h4w6"] = (56, 84)
HIDDEN = 6144  # values per vision token: MiniMax-M3's text hidden size, which the encoder's output must have
NUM_LAYERS = 2
# Guard bytes around the slots of the shared buffer, which read_embeddings must leave as they are.
GUARD_BYTES = 64
GUARD = 0xA5
# The media module's device_params, with which the service opens the chip: the media test opens its chip with them.
MEDIA_DEVICE_PARAMS = {}


def _parametrize_sizes():
    run_all = os.getenv("M3_VISION_ALL_GRIDS") == "1"
    params = [pytest.param(size, id=size_id) for size_id, size in SIZES.items() if run_all or size_id in CI_GRIDS]
    return pytest.mark.parametrize("size", params)


def _noise_pixels(height: int, width: int) -> np.ndarray:
    """A resized image of seeded noise, seeded from its patch grid as conftest.py seeds its pixel values."""
    h, w = height // PATCH_SIZE, width // PATCH_SIZE
    generator = torch.Generator().manual_seed(1000 * h + w)
    return torch.randint(0, 256, (height, width, 3), generator=generator, dtype=torch.uint8).numpy()


def _reference_tokens(reference, pixels: np.ndarray) -> torch.Tensor:
    """The reference's vision tokens for the pixels, from the bf16 copy of their patch rows that the chip gets."""
    pixel_values, grid_thw = patch_rows(pixels)
    tower = reference.tower(pixel_values=pixel_values.bfloat16().float(), image_grid_thw=grid_thw)
    return reference.projector(tower.last_hidden_state.squeeze(0))


def _read_plain(tokens: ttnn.Tensor) -> torch.Tensor:
    """forward's output on the host, read with ttnn.to_torch."""
    (chip_tokens,) = ttnn.get_device_tensors(tokens)
    return ttnn.to_torch(chip_tokens)


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@_parametrize_sizes()
def test_vision_encoder_vs_ref(mesh_device, device_params, size, vision_reference):
    """forward gives bf16 [n_tokens, 6144] row-major tokens in DRAM that pass assert_close against the reference."""
    height, width = size
    reference = vision_reference(NUM_LAYERS)
    pixels = _noise_pixels(height, width)
    ref = _reference_tokens(reference, pixels)
    n_tokens = (height // FACTOR) * (width // FACTOR)
    assert ref.shape == (n_tokens, HIDDEN)

    encoder = TtMiniMaxM3VisionEncoder(mesh_device, reference.weights, reference.config)
    tokens = encoder.forward(pixels)
    assert tuple(tokens.shape) == (n_tokens, HIDDEN), tuple(tokens.shape)
    assert tokens.dtype == ttnn.bfloat16 and tokens.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tokens.memory_config().buffer_type == ttnn.BufferType.DRAM
    out = _read_plain(tokens)
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    grid = f"{grid_name((1, height // PATCH_SIZE, width // PATCH_SIZE))} ({width}x{height} pixels)"
    assert_close(f"encoder {grid} layers={NUM_LAYERS}", ref, out.float(), 0.9998, MAX_REL_L2["final"])


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_vision_encoder_load_and_encode(mesh_device, device_params, vision_reference):
    """Used as load() and encode() use it, the encoder fills each slot of a shared buffer after its header.

    Each slot gets the chip's output bit for bit, which matches the reference at PCC >= 0.9998. Nothing else in the
    buffer changes. After a 224x224 warm-up, images of 448x112 and 112x448, with the same 256 patches, add no program
    to the cache.
    """
    from transformers import MiniMaxM3VLConfig

    reference = vision_reference(NUM_LAYERS)
    hf_config = MiniMaxM3VLConfig(vision_config={"num_hidden_layers": NUM_LAYERS})
    assert vision_config(hf_config) == reference.config
    encoder = TtMiniMaxM3VisionEncoder(mesh_device, reference.weights, hf_config)

    warm_up = encoder.forward(_noise_pixels(224, 224))  # load(): one image of the patch count to serve
    ttnn.deallocate(warm_up)
    cache_entries = mesh_device.num_program_cache_entries()
    assert cache_entries > 0, "the program cache holds no programs after the warm-up"

    images = [_noise_pixels(112, 448), _noise_pixels(448, 112)]
    sizes = [media_format.stored_nbytes((p.shape[0] // FACTOR) * (p.shape[1] // FACTOR), HIDDEN) for p in images]
    buffer = np.full(GUARD_BYTES + sum(size + GUARD_BYTES for size in sizes), GUARD, dtype=np.uint8)
    starts = np.cumsum([GUARD_BYTES] + [size + GUARD_BYTES for size in sizes])[:-1]
    slots = [memoryview(buffer)[start : start + size] for start, size in zip(starts, sizes)]
    for slot in slots:
        media_format.Header("image").write(slot)  # encode() writes each value's header, then its rows
    before = buffer.copy()
    row_bytes = np.zeros(buffer.shape, dtype=bool)  # the bytes of the rows that format.embeddings gives
    for i, (pixels, slot) in enumerate(zip(images, slots)):  # encode(): one item at a time
        embeddings = media_format.embeddings(slot, HIDDEN)
        offset = embeddings.ctypes.data - buffer.ctypes.data
        row_bytes[offset : offset + embeddings.nbytes] = True
        tokens = encoder.forward(pixels)
        read_embeddings(tokens, embeddings)
        plain = _read_plain(tokens)
        ttnn.deallocate(tokens)

        stored = torch.from_numpy(embeddings.copy()).view(torch.bfloat16)
        assert torch.equal(stored.view(torch.int16), plain.view(torch.int16)), f"image {i}: not the chip's output"
        name = f"encode image {i + 1} of 2 ({pixels.shape[1]}x{pixels.shape[0]} pixels) layers={NUM_LAYERS}"
        assert_close(name, _reference_tokens(reference, pixels), stored.float(), 0.9998, MAX_REL_L2["final"])
        assert mesh_device.num_program_cache_entries() == cache_entries, f"{name} added a program to the cache"

    outside = ~row_bytes  # the headers and the guard bytes
    assert np.array_equal(buffer[outside], before[outside]), "read_embeddings wrote outside the rows of its slot"
    for i, slot in enumerate(slots):
        assert media_format.Header.read(slot) == media_format.Header("image"), f"image {i}: the header changed"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device, device_params", [pytest.param((1, 1), MEDIA_DEVICE_PARAMS, id="1x1")], indirect=True
)
def test_vision_media_load_and_encode(mesh_device, device_params, vision_reference, tmp_path, monkeypatch):
    """The media module's load(), prepare() and encode() store the encoder's own tokens after the image's header.

    The chip is opened with the module's device_params, as the service opens it. load() reads the reference's weights
    from a checkpoint and warms up 224x224, so encoding a 224x224 image compiles nothing. The stored tokens equal the
    encoder's forward bit for bit, and match the reference at PCC >= 0.9998.
    """
    from transformers import MiniMaxM3VLConfig

    reference = vision_reference(NUM_LAYERS)
    hf_config = MiniMaxM3VLConfig(vision_config={"num_hidden_layers": NUM_LAYERS}).to_dict()
    assert vision_config(hf_config) == reference.config
    write_checkpoint(tmp_path, reference.weights, hf_config)
    built = []  # the encoder that load() builds, which the bit-exact check reuses
    from_weights_path = TtMiniMaxM3VisionEncoder.from_weights_path

    def keep_built(*args, **kwargs):
        built.append(from_weights_path(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(TtMiniMaxM3VisionEncoder, "from_weights_path", staticmethod(keep_built))
    media = MiniMaxM3Media(hf_config)
    assert media.device_params == MEDIA_DEVICE_PARAMS, "the test's chip is not opened as the service opens it"
    media.load(mesh_device, str(tmp_path), "image", {"warmup_sizes": [[224, 224]]})
    try:
        cache_entries = mesh_device.num_program_cache_entries()
        assert cache_entries > 0, "load's warm-up compiled no program"
        item = media.prepare("image", _noise_pixels(230, 230))
        assert item.pixels.shape == (224, 224, 3), item.pixels.shape
        value = memoryview(bytearray(b"\xff" * media.stored_nbytes(item)))
        media.encode([item], [value])
        assert mesh_device.num_program_cache_entries() == cache_entries, "encode compiled a program the warm-up had not"
        assert len(built) == 1, f"load() built {len(built)} encoders"
        tokens = built[0].forward(item.pixels)
        plain = _read_plain(tokens)
        ttnn.deallocate(tokens)
    finally:
        built.clear()  # so that close() frees the encoder
        media.close()

    assert media_format.Header.read(value) == media_format.Header("image"), "encode wrote no image header"
    rows = media_format.embeddings(value, HIDDEN)
    assert rows.shape == (item.n_tokens, HIDDEN), rows.shape
    stored = torch.from_numpy(rows.copy()).view(torch.bfloat16)
    assert torch.equal(stored.view(torch.int16), plain.view(torch.int16)), "not the encoder's own output"
    name = f"media encode (230x230 RGB -> 224x224 pixels) layers={NUM_LAYERS}"
    assert_close(name, _reference_tokens(reference, item.pixels), stored.float(), 0.9998, MAX_REL_L2["final"])
