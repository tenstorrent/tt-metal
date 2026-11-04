# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device tests of TtMiniMaxM3VisionEncoder (vision/encoder.py), the vision encoder as the encoder service calls it,
against the Torch reference, on one chip with the 2-layer reference of conftest.py (seeded random weights, no
checkpoint):
  - test_vision_encoder_vs_ref: forward(pixels) of a seeded noise image returns its vision tokens on the chip as bf16
    [n_tokens, 6144], row-major, in DRAM, at PCC >= 0.9998 and within the tower test's relL2 bound against the
    reference run on the same patch rows (patch_rows of the pixels, in the bf16 the chip gets).
  - test_vision_encoder_load_and_encode: the encoder used the way the service's load() and encode() will use it: built
    from a state_dict and a transformers config, one warm-up image, then two images of other grids with the warm-up's
    patch count, each read with read_embeddings into its slot of one preallocated buffer laid out like a stored value
    ([n_tokens, 6144] bf16, with no header), with guard bytes around the slots. Each slot must hold the chip's output
    bit for bit and match the reference, nothing else in the buffer may change, and the images after the warm-up must
    add no program to the cache.
The host tests of the encoder, which need no chip, are in test_vision_encoder_api.py.

Image sizes (width x height in pixels), named by their grid of 14-pixel patches as the other vision tests name
theirs: 224x224 (h16w16) and 756x560 (h40w54, not square) are the CI grids of the tower test, whose programs the
1-card leg builds anyway; M3_VISION_ALL_GRIDS=1 adds 84x56 (h4w6: 24 patches, under one tile of rows, 6 tokens), a
size small images resize to.

Run (the second line adds h4w6):
    pytest models/demos/minimax_m3/tests/unit/test_vision_encoder_vs_ref.py
    M3_VISION_ALL_GRIDS=1 pytest models/demos/minimax_m3/tests/unit/test_vision_encoder_vs_ref.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import ttnn
from models.demos.minimax_m3.tt.vision.preprocess import BLOCK, PATCH_SIZE, patch_rows
from models.demos.minimax_m3.vision import format as media_format
from models.demos.minimax_m3.vision.encoder import TtMiniMaxM3VisionEncoder, read_embeddings, vision_config

from ..test_factory import parametrize_mesh_with_fabric
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
    return reference(pixel_values.bfloat16().float(), grid_thw)


def _read_plain(tokens: ttnn.Tensor) -> torch.Tensor:
    """forward's output on the host, read with ttnn.to_torch."""
    (chip_tokens,) = ttnn.get_device_tensors(tokens)
    return ttnn.to_torch(chip_tokens)


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@_parametrize_sizes()
def test_vision_encoder_vs_ref(mesh_device, device_params, size, vision_reference):
    """forward(pixels) gives bf16 [n_tokens, 6144] row-major tokens in DRAM, at PCC >= 0.9998 and within the final
    relL2 bound of the tower test against the reference, with no NaN or Inf."""
    height, width = size
    reference = vision_reference(NUM_LAYERS)
    pixels = _noise_pixels(height, width)
    ref = _reference_tokens(reference, pixels)
    n_tokens = (height // BLOCK) * (width // BLOCK)
    assert ref.shape == (n_tokens, HIDDEN)

    encoder = TtMiniMaxM3VisionEncoder(mesh_device, reference.hf_state_dict(), reference.config)
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
    """Built as load() will build it and called as encode() will call it, the encoder writes each image's tokens into
    its slot of a shared buffer: bit for bit the chip's output, at PCC >= 0.9998 against the reference, with the guard
    bytes untouched; after one warm-up image of 224x224 (h16w16), images of 448x112 (h8w32) and 112x448 (h32w8),
    256 patches each, add no program to the cache."""
    from transformers import MiniMaxM3VLConfig

    reference = vision_reference(NUM_LAYERS)
    hf_config = MiniMaxM3VLConfig(vision_config={"num_hidden_layers": NUM_LAYERS})
    assert vision_config(hf_config) == reference.config
    encoder = TtMiniMaxM3VisionEncoder(mesh_device, reference.hf_state_dict(), hf_config)

    warm_up = encoder.forward(_noise_pixels(224, 224))  # load(): one image of the patch count to serve
    ttnn.deallocate(warm_up)
    cache_entries = mesh_device.num_program_cache_entries()
    assert cache_entries > 0, "the program cache holds no programs after the warm-up"

    images = [_noise_pixels(112, 448), _noise_pixels(448, 112)]
    sizes = [media_format.stored_nbytes((p.shape[0] // BLOCK) * (p.shape[1] // BLOCK), HIDDEN) for p in images]
    buffer = np.full(GUARD_BYTES + sum(size + GUARD_BYTES for size in sizes), GUARD, dtype=np.uint8)
    guard = buffer.copy()
    starts = np.cumsum([GUARD_BYTES] + [size + GUARD_BYTES for size in sizes])[:-1]
    for i, (pixels, start) in enumerate(zip(images, starts)):  # encode(): one item at a time
        embeddings = media_format.embeddings(memoryview(buffer)[start : start + sizes[i]], HIDDEN)
        tokens = encoder.forward(pixels)
        read_embeddings(tokens, embeddings)
        plain = _read_plain(tokens)
        ttnn.deallocate(tokens)

        stored = torch.from_numpy(embeddings.copy()).view(torch.bfloat16)
        assert torch.equal(stored.view(torch.int16), plain.view(torch.int16)), f"image {i}: not the chip's output"
        name = f"encode image {i + 1} of 2 ({pixels.shape[1]}x{pixels.shape[0]} pixels) layers={NUM_LAYERS}"
        assert_close(name, _reference_tokens(reference, pixels), stored.float(), 0.9998, MAX_REL_L2["final"])
        assert mesh_device.num_program_cache_entries() == cache_entries, f"{name} added a program to the cache"

    outside = np.ones(buffer.shape, dtype=bool)
    for start, size in zip(starts, sizes):
        outside[start : start + size] = False
    assert np.array_equal(buffer[outside], guard[outside]), "read_embeddings wrote outside the embeddings of its slot"
