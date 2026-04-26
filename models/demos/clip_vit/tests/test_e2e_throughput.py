# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import requests
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.clip_vit.tests.conftest import IMAGE_URLS, TEXT_QUERIES
from models.demos.clip_vit.tt.tt_clip_model import TtCLIPModel
from models.demos.clip_vit.tt.tt_clip_model_optimized import TtCLIPModelOptimized

BATCH_BF8 = 63
BATCH_BF16 = 35

BATCH_COUNTS_BF8 = [BATCH_BF8 * n for n in [8, 16, 32]]
BATCH_COUNTS_BF16 = [BATCH_BF16 * n for n in [8, 16, 32]]


@pytest.fixture(scope="session")
def image_pool():
    return [Image.open(requests.get(u, stream=True).raw) for u in IMAGE_URLS]


def _make_inputs(processor, image_pool, total_images, device):
    selected_imgs = [image_pool[i % len(image_pool)] for i in range(total_images)]
    selected_texts = [TEXT_QUERIES[i % len(TEXT_QUERIES)] for i in range(total_images)]
    proc = processor(
        selected_texts,
        selected_imgs,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )
    ids = ttnn.from_torch(
        proc["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    seq_len = proc["input_ids"].shape[1]
    pos = ttnn.from_torch(
        torch.arange(seq_len).unsqueeze(0).expand(total_images, -1),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    pv = ttnn.from_torch(
        proc["pixel_values"],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return ids, pos, pv


def _run_throughput(module, processor, image_pool, device, total_images, label, is_base=False):
    """One warmup run to populate program cache, then a timed run."""
    ids, pos, pv = _make_inputs(processor, image_pool, total_images, device)

    # Warmup
    if is_base:
        module(ids, pv, position_ids=pos)
    else:
        module(ids, pv)
    ttnn.synchronize_device(device)

    # Timed
    ids, pos, pv = _make_inputs(processor, image_pool, total_images, device)
    t0 = time.perf_counter()
    if is_base:
        module(ids, pv, position_ids=pos)
    else:
        module(ids, pv)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0

    logger.info(
        f"{label}: {total_images / elapsed:.1f} images/sec " f"| total={total_images} elapsed={elapsed*1000:.1f} ms"
    )


# ==========================================================================
#  bfloat16: baseline vs optimized
# ==========================================================================


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("total_images", BATCH_COUNTS_BF16)
def test_throughput_bf16_base(torch_model, processor, image_pool, device, total_images):
    module = TtCLIPModel(torch_model.config, torch_model, device)
    _run_throughput(
        module, processor, image_pool, device, total_images, label=f"Base bf16 (N={total_images})", is_base=True
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("total_images", BATCH_COUNTS_BF16)
def test_throughput_bf16_optimized(torch_model, processor, image_pool, device, total_images):
    module = TtCLIPModelOptimized(
        torch_model.config,
        torch_model,
        device,
        vision_batch=BATCH_BF16,
        text_batch=BATCH_BF16,
        dtype=ttnn.bfloat16,
    )
    _run_throughput(module, processor, image_pool, device, total_images, label=f"Optimized bf16 (N={total_images})")


# ==========================================================================
#  bfloat8_b: baseline vs optimized
# ==========================================================================


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("total_images", BATCH_COUNTS_BF8)
def test_throughput_bf8_base(torch_model, processor, image_pool, device, total_images):
    module = TtCLIPModel(torch_model.config, torch_model, device)
    _run_throughput(
        module, processor, image_pool, device, total_images, label=f"Base bf8 (N={total_images})", is_base=True
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("total_images", BATCH_COUNTS_BF8)
def test_throughput_bf8_optimized(torch_model, processor, image_pool, device, total_images):
    module = TtCLIPModelOptimized(
        torch_model.config,
        torch_model,
        device,
        vision_batch=BATCH_BF8,
        text_batch=BATCH_BF8,
        dtype=ttnn.bfloat8_b,
    )
    _run_throughput(module, processor, image_pool, device, total_images, label=f"Optimized bf8 (N={total_images})")
