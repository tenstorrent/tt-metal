# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import time

import pytest
import requests
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.clip_vit.tests.pcc.conftest import IMAGE_URLS, TEXT_BATCH_SIZE, TEXT_QUERIES, VISION_BATCH_SIZE
from models.demos.clip_vit.tt.tt_clip_model import TtCLIPModel
from models.demos.clip_vit.tt.tt_clip_model_optimized import TtCLIPModelOptimized

NUM_WARMUP = 3
NUM_RUNS = 50

# N = LCM(VISION_BATCH_SIZE, TEXT_BATCH_SIZE) keeps both encoders fed with full
# batches (no partial-batch edge cases in the optimized wrapper).
LATENCY_BATCH_SIZE = math.lcm(VISION_BATCH_SIZE, TEXT_BATCH_SIZE)


def _bench(fn, device, batch_size, name):
    for _ in range(NUM_WARMUP):
        fn()
    ttnn.synchronize_device(device)

    times_ms = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms.sort()
    median = times_ms[len(times_ms) // 2]
    mean = sum(times_ms) / len(times_ms)
    p95 = times_ms[int(len(times_ms) * 0.95)]
    imgs_per_sec = batch_size / (median / 1000)

    logger.info(
        f"{name}: median {median:.3f} ms | mean {mean:.3f} ms | p95 {p95:.3f} ms "
        f"| throughput {imgs_per_sec:.1f} items/sec | batch {batch_size} | runs {NUM_RUNS}"
    )


def _repeated_to_batch(items, batch_size):
    return (items * ((batch_size + len(items) - 1) // len(items)))[:batch_size]


def _build_inputs(processor, batch_size):
    image_pool = [Image.open(requests.get(u, stream=True).raw) for u in IMAGE_URLS]
    images = _repeated_to_batch(image_pool, batch_size)
    texts = _repeated_to_batch(TEXT_QUERIES, batch_size)
    return processor(texts, images, padding="max_length", max_length=77, return_tensors="pt")


def _prepare_device_inputs(inputs, batch_size, device):
    ids = ttnn.from_torch(
        inputs["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    seq_len = inputs["input_ids"].shape[1]
    pos = ttnn.from_torch(
        torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    pv = ttnn.from_torch(
        inputs["pixel_values"],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return ids, pos, pv


def _run_full_optimized_latency(torch_model, processor, device, dtype, label):
    batch_size = LATENCY_BATCH_SIZE
    inputs = _build_inputs(processor, batch_size)

    module = TtCLIPModelOptimized(
        torch_model.config,
        torch_model,
        device,
        vision_batch=VISION_BATCH_SIZE,
        text_batch=TEXT_BATCH_SIZE,
        dtype=dtype,
    )
    ids, _, pv = _prepare_device_inputs(inputs, batch_size, device)

    _bench(lambda: module(ids, pv), device, batch_size, f"CLIP optimized full model ({label})")


@pytest.mark.models_performance_bare_metal
def test_latency_full_model_base(torch_model, processor, device):
    batch_size = LATENCY_BATCH_SIZE
    inputs = _build_inputs(processor, batch_size)

    module = TtCLIPModel(torch_model.config, torch_model, device)
    ids, pos, pv = _prepare_device_inputs(inputs, batch_size, device)

    _bench(lambda: module(ids, pv, position_ids=pos), device, batch_size, "CLIP base full model")


@pytest.mark.models_performance_bare_metal
def test_latency_full_model_optimized_bf16(torch_model, processor, device):
    _run_full_optimized_latency(
        torch_model,
        processor,
        device,
        dtype=ttnn.bfloat16,
        label="bf16",
    )


@pytest.mark.models_performance_bare_metal
def test_latency_full_model_optimized_bf8(torch_model, processor, device):
    _run_full_optimized_latency(
        torch_model,
        processor,
        device,
        dtype=ttnn.bfloat8_b,
        label="bf8",
    )
