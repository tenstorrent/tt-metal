# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Host-glue tests for the Instruct (I2I) input path (tt/vision/preprocess.py,
# tt/vision/i2i.py):
#   * the SigLIP2 image processor wrapper turns a PIL image into the
#     pixel_values / spatial_shapes / mask the device tower expects, and uploads
#     into a Siglip2VisionInputs bundle;
#   * <img> span lookup finds the contiguous placeholder run(s);
#   * the inject dispatcher (device-multi vs host fallback) matches the reference
#     masked scatter.
#
# The full SigLIP2 tower + aligner is already PCC-tested in test_siglip2_ttnn.py;
# here we validate the orchestration around it, so no real weights are needed.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/vision/test_cond_image_preprocess.py -v -s

import pytest
import torch
import ttnn
from loguru import logger
from PIL import Image

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vision.inject import scatter_vit_image_tokens
from models.experimental.hunyuan_image_3_0.tt.vision.i2i import inject_cond_vision
from models.experimental.hunyuan_image_3_0.tt.vision.preprocess import (
    build_cond_image_processor,
    find_image_token_spans,
    process_cond_image,
    to_vision_inputs,
)

H = 4096
IMG = 128006  # config.json image_token_id
PCC_THR = 0.999


def test_find_image_token_spans():
    # [text, <img>x4, text, <img>x2, text]
    ids = torch.tensor([5, 6, IMG, IMG, IMG, IMG, 7, IMG, IMG, 8])
    spans = find_image_token_spans(ids, IMG)
    assert spans == [slice(2, 6), slice(7, 9)], spans
    # span running to end of sequence
    ids2 = [1, IMG, IMG]
    assert find_image_token_spans(ids2, IMG) == [slice(1, 3)]
    # no image tokens
    assert find_image_token_spans([1, 2, 3], IMG) == []


def test_process_cond_image_shapes():
    proc = build_cond_image_processor()
    img = Image.new("RGB", (320, 192), (120, 30, 200))
    pixel_values, spatial_shapes_hw, mask = process_cond_image(proc, img)
    mnp = proc.max_num_patches
    patch_dim = 3 * proc.patch_size**2
    assert pixel_values.shape == (1, mnp, patch_dim), pixel_values.shape
    assert mask.shape == (1, mnp), mask.shape
    assert len(spatial_shapes_hw) == 1 and len(spatial_shapes_hw[0]) == 2
    th, tw = spatial_shapes_hw[0]
    # valid patch count == token grid area, and never exceeds the padded length
    assert 0 < th * tw <= mnp
    assert int(mask.sum()) == th * tw, (int(mask.sum()), th * tw)
    logger.info(
        f"processor: pixel_values={tuple(pixel_values.shape)} grid={spatial_shapes_hw[0]} valid={int(mask.sum())}"
    )


def test_to_vision_inputs_bundle(device):
    proc = build_cond_image_processor()
    img = Image.new("RGB", (256, 256), (10, 200, 90))
    pixel_values, spatial_shapes_hw, mask = process_cond_image(proc, img)
    vi = to_vision_inputs(device, pixel_values, spatial_shapes_hw, mask)
    assert vi.spatial_shapes_hw == spatial_shapes_hw
    assert tuple(vi.pixel_values.shape) == tuple(pixel_values.shape)
    assert tuple(vi.pixel_attention_mask.shape) == tuple(mask.shape)


@pytest.mark.parametrize(
    "S,spans",
    [
        (1088, [slice(32, 1056)]),  # single 1024-token cond-image span (the real layout)
        (256, [slice(32, 96), slice(160, 224)]),  # two aligned spans (device path)
        (130, [slice(7, 23), slice(50, 90)]),  # ragged, non-aligned (host fallback)
    ],
)
def test_inject_dispatch_matches_ref(device, S, spans):
    """inject_cond_vision (auto device-vs-host) == reference masked scatter."""
    torch.manual_seed(0)
    B = 1
    n_img = sum(s.stop - s.start for s in spans)
    hidden = torch.randn(B, S, H)
    image_embeds = torch.randn(B, n_img, H)

    image_mask = torch.zeros(B, S, dtype=torch.bool)
    for s in spans:
        image_mask[:, s.start : s.stop] = True
    ref = scatter_vit_image_tokens(hidden, image_embeds, image_mask)

    def up(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    out_tt = inject_cond_vision(up(hidden), up(image_embeds), img_slices=spans)
    out = ttnn.to_torch(out_tt).float()
    ttnn.deallocate(out_tt)

    assert tuple(out.shape) == tuple(ref.shape)
    passing, pcc = comp_pcc(ref, out, PCC_THR)
    logger.info(f"inject_cond_vision S={S} spans={spans} PCC={pcc} (passing={passing})")
    assert passing, f"PCC {pcc} < {PCC_THR}"
