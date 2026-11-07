# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The TTNN ViT (tt/vision) on one 1x8 row vs the fp32 reference (reference/vision.py), real weights.

Inputs come from vLLM's processor (reference/vision_inputs.py) on the repo's sample images: a 448x448 image, a
640x360 image, two images in one call, and a 32-frame 640x360 video (also split into several passes). Checks the PCC
of every block's output (row order) and of the final embeddings.

Run on one Galaxy row, as a row engine does:
    TT_VISIBLE_DEVICES=<the row's 8 chips> pytest models/experimental/mimo_v2_6_flash/tests/test_vision_tp8.py
The converted weights go to ``MIMO_VISION_CACHE`` (default: a ``vision`` cache next to the LLM's, per checkpoint).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import vision as ref
from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vin
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, model_path, pcc
from models.experimental.mimo_v2_6_flash.tt.vision.vit import TtMiMoVisionTransformer

BLOCK_PCC = 0.99
FINAL_PCC = 0.99


def vision_cache_dir():
    root = os.environ.get("MIMO_VISION_CACHE", "/home/zni/mimo/mm/vision/tt-cache")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path())))


def make_case(path, case):
    """(pixel_values, grid_thw, num_tokens) from the vLLM processor."""
    if case == "image_448":
        out = vin.process_images(path, [vin.sample_image(size=(448, 448))])
    elif case == "image_1024":
        out = vin.process_images(path, [vin.sample_image("demo.jpeg", size=(1024, 1024))])
    elif case == "image_640x360":
        out = vin.process_images(path, [vin.sample_image("house_in_field_1080p.jpg", size=(640, 360))])
    elif case == "two_images":
        images = [vin.sample_image(size=(448, 448)), vin.sample_image("demo.jpeg", size=(512, 384))]
        out = vin.process_images(path, images)
    elif case.startswith("video_32f"):
        out = vin.process_video(path, vin.sample_video(num_frames=32, size=(640, 360)))
        return out["pixel_values_videos"], out["video_grid_thw"], out["num_tokens"]
    else:
        raise ValueError(case)
    return out["pixel_values"], out["image_grid_thw"], out["num_tokens"]


@pytest.fixture(scope="module")
def reference():
    torch.set_num_threads(min(16, torch.get_num_threads()))
    path = model_path()
    return path, ref.MiMoVisionReference.from_model_path(path)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "case, max_rows",
    [
        ("image_448", None),
        ("image_640x360", None),
        ("image_1024", None),
        ("two_images", None),
        ("video_32f", None),
        ("video_32f_passes", 4096),
    ],
)
def test_vision_tp8(mesh_device, reference, case, max_rows):
    path, model_ref = reference
    pixel_values, grid, num_tokens = make_case(path, case)
    cfg = model_ref.cfg
    n_tokens = ref.num_merged_tokens(grid, cfg.spatial_merge_size)
    assert n_tokens == sum(num_tokens), (n_tokens, num_tokens)

    t0 = time.time()
    with torch.no_grad():
        expected, expected_hidden = model_ref(pixel_values, grid, return_hidden=True)
    logger.info(
        f"{case}: grid {ref.as_grid_list(grid)}, {pixel_values.shape[0]} patches, reference {time.time() - t0:.1f} s"
    )

    t0 = time.time()
    kwargs = {} if max_rows is None else {"max_rows": max_rows}
    model = TtMiMoVisionTransformer.from_checkpoint(mesh_device, path, cache_dir=vision_cache_dir(), **kwargs)
    logger.info(f"weights loaded in {time.time() - t0:.1f} s")

    hidden = []
    out = model(pixel_values, grid, hidden=hidden if max_rows is None else None)
    assert out.shape == (n_tokens, cfg.out_hidden_size) and out.dtype == torch.bfloat16, out.shape
    final = pcc(out.float(), expected)
    if hidden:
        assert len(hidden) == cfg.depth
        scores = [pcc(h, e) for h, e in zip(hidden, expected_hidden)]
        table = " ".join(f"{i}:{s:.4f}" for i, s in enumerate(scores))
        logger.info(f"{case}: per-block PCC {table}")
        worst = min(range(cfg.depth), key=lambda i: scores[i])
        logger.info(f"{case}: min block PCC {scores[worst]:.5f} (block {worst}), final PCC {final:.5f}")
        assert scores[worst] >= BLOCK_PCC, f"block {worst} PCC {scores[worst]:.5f}"
    else:
        logger.info(f"{case}: final PCC {final:.5f} ({max_rows} rows per pass)")
    assert final >= FINAL_PCC, f"final PCC {final:.5f}"
    ttnn.synchronize_device(mesh_device)
