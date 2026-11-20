# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: the Kimi-K3 media module's load, encode and close on the ring, as the encoder service calls them, against
the fp32 reference. Images of different sizes go through the one captured trace, and encode's read (untilized on the
chips, each chip's rows copied into the slot) matches the untraced forward gathered on the host bit for bit."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
from safetensors.torch import save_file

import ttnn

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.service import format as media_format
from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    MEDIA_CONFIG,
    REAL,
    REAL_MARK,
    RING_TRACE,
    SIMULATOR,
    SYNTHETIC,
    assert_quality,
    assert_same_as_untraced,
    random_array,
)
from models.experimental.kimi_k3_moonvit_v2.tests.goldens import to_fp32
from models.experimental.kimi_k3_moonvit_v2.tt.image_prep import pixels_to_patches

SHARD = "model-00001-of-00001.safetensors"
# (height, width) in pixels: 4 x 6 and 10 x 4 patches after the padding, so the second image has a grid the trace
# was not captured with; then the first again, which must not keep any of the second's rows.
IMAGES = [(56, 84), (130, 50), (56, 84)]


def write_checkpoint(reference, directory) -> str:
    """`reference`'s weights as a one-shard checkpoint folder, as `load_vision_state_dict` reads it."""
    state = {key: value.contiguous().clone() for key, value in reference.state_dict().items()}
    save_file(state, directory / SHARD)
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: SHARD for k in state}}))
    return str(directory)


def encode(media: KimiK3Media, array: np.ndarray):
    item = media.prepare("image", array)
    buf = memoryview(bytearray(media.stored_nbytes(item)))
    media.encode([item], [buf])
    assert media_format.Header.read(buf).kind == "image"
    rows = torch.from_numpy(media_format.embeddings(buf, media.hidden).copy()).view(torch.bfloat16)
    return item, bytes(buf), rows


def untraced_gather(media: KimiK3Media, item) -> torch.Tensor:
    """The untraced forward, gathered on the host with `gather_output`: what encode's traced replay and
    `read_output` must match."""
    model = media._model
    out = model(**model.prepare_inputs(*pixels_to_patches(item.pixels, model.config)))
    try:
        return model.gather_output(out, logical_n=item.n_tokens).to(torch.bfloat16)
    finally:
        ttnn.deallocate(out)


@RING_TRACE
@SYNTHETIC
@REAL
def test_load_encode_close(mesh_device, sp_axis, num_links, weights, reference, tmp_path, expect_error):
    weights_path = str(checkpoint_path()) if weights == REAL_MARK else write_checkpoint(reference, tmp_path)
    golden_model = to_fp32(copy.deepcopy(reference))
    media = KimiK3Media(MEDIA_CONFIG)

    media.load(mesh_device, weights_path, "image", {})
    try:
        values = []
        for hw in IMAGES:
            array = random_array(*hw)
            old = untraced_gather(media, media.prepare("image", array))
            item, value, rows = encode(media, array)
            values.append(value)
            assert rows.shape == (item.n_tokens, media.hidden)
            assert_same_as_untraced(old, rows, f"encode {hw}")
            with torch.no_grad():
                (golden,) = golden_model(*pixels_to_patches(item.pixels, media.config))
            assert_quality(golden, rows, relative_rmse=0.07)
        assert values[0] == values[2], "the same image after another gave a different value"

        # Two items in one call, and a slot that is not bytes.
        items = [media.prepare("image", random_array(*hw)) for hw in IMAGES[:2]]
        bufs = [memoryview(bytearray(media.stored_nbytes(items[0]))).cast("H")]
        bufs.append(memoryview(bytearray(media.stored_nbytes(items[1]))))
        media.encode(items, bufs)
        assert [bufs[0].tobytes(), bytes(bufs[1])] == values[:2]
    finally:
        media.close()
    media.close()  # a second close is a no-op

    item = media.prepare("image", random_array(*IMAGES[0]))
    with expect_error(RuntimeError, "encode before load"):
        media.encode([item], [memoryview(bytearray(media.stored_nbytes(item)))])


@SIMULATOR
def test_load_rejects_one_chip(mesh_device, sp_axis, num_links, tmp_path, expect_error):
    media = KimiK3Media(MEDIA_CONFIG)
    with expect_error(ValueError, r"sp factor 4 != mesh axis 1 size \(1, 1\)"):
        media.load(mesh_device, str(tmp_path), "image", {})
    media.load(None, "", "image", {"stub_embeddings": True})  # the failed load left nothing loaded
    media.close()
