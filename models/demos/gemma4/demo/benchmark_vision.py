# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma-4 vision-only micro-benchmark.

Builds ONLY the on-device vision tower + multimodal embedder (no LLM on device)
and measures their per-iteration latency on a real image. This isolates the
vision-encoder + projection cost from the text model so it can be profiled on
its own.

What is measured (device-synchronized, per iteration):
    - ``VisionTower.forward``      (patch embed -> encoder -> pool -> standardize)
    - ``Gemma4MultimodalEmbedder`` (scale-free RMSNorm + Linear -> text space)
    - the sum of the two

Usage:
    HF_MODEL=google/gemma-4-31B-it pytest \\
        models/demos/gemma4/demo/benchmark_vision.py -k "1x8" -sv

    # Override image / prompt / iteration count / vision dtype:
    HF_MODEL=google/gemma-4-31B-it \\
        GEMMA4_VISION_IMAGE=models/tt_transformers/demo/sample_prompts/llama_models/dog.jpg \\
        GEMMA4_VISION_PROMPT="Describe this image." \\
        GEMMA4_VISION_BENCH_ITERS=20 \\
        GEMMA4_VISION_DTYPE=bfloat8_b \\
        pytest models/demos/gemma4/demo/benchmark_vision.py -k "1x8" -sv

    # Profile a single vision encoder layer under tracy (per-op device time):
    HF_MODEL=google/gemma-4-31B-it \\
        GEMMA4_VISION_NUM_LAYERS=1 GEMMA4_VISION_BENCH_ITERS=3 \\
        pytest models/demos/gemma4/demo/benchmark_vision.py -k "1x1" -sv
"""

import os
import time

import pytest
import torch
from loguru import logger
from PIL import Image as PIL_Image

import ttnn
from models.demos.gemma4.demo.vision_demo import IMG_PATH, _device_params, _model_path, encode_multimodal
from models.demos.gemma4.tt.generator import _build_vision_state_dict, _vision_encoder_seq_len
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.vision.multimodal_embedder import Gemma4MultimodalEmbedder
from models.demos.gemma4.tt.vision.vision_model_config import VisionModelArgs
from models.demos.gemma4.tt.vision.vision_tower import VisionTower
from models.tt_transformers.tt.ccl import TT_CCL


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 4))
    ],
    indirect=True,
)
def test_benchmark_vision(mesh_device, batch_size, reset_seeds):
    """Benchmark the Gemma-4 vision tower + multimodal embedder (no LLM)."""
    from transformers import AutoProcessor

    model_path = _model_path()
    os.environ.setdefault("HF_MODEL", model_path)

    vision_dtype_name = os.environ.get("GEMMA4_VISION_DTYPE", "bfloat8_b")
    vision_dtype = getattr(ttnn, vision_dtype_name, ttnn.bfloat8_b)
    num_iters = int(os.environ.get("GEMMA4_VISION_BENCH_ITERS", 2))
    # Optional encoder-depth override for profiling: build only N vision transformer blocks
    # (e.g. GEMMA4_VISION_NUM_LAYERS=1) so the run is small enough to capture a clean
    # tracy/per-op-device-time trace. Weights for the dropped layers are simply unused.
    _nl_env = os.environ.get("GEMMA4_VISION_NUM_LAYERS", 1)
    vision_num_layers = int(_nl_env) if _nl_env else None

    image_file = os.environ.get("GEMMA4_VISION_IMAGE", str(IMG_PATH / "dog.jpg"))
    prompt = os.environ.get("GEMMA4_VISION_PROMPT", "Write a short summary about this image.")
    logger.info(
        f"Vision benchmark: image={image_file}, prompt={prompt!r}, "
        f"dtype={vision_dtype_name}, iters={num_iters}, batch_size={batch_size}, "
        f"vision_layers={vision_num_layers if vision_num_layers else 'all'}"
    )
    image = PIL_Image.open(image_file).convert("RGB")

    # ── Load checkpoint + build vision-only components (no LLM on device) ───
    logger.info(f"Loading checkpoint + vision weights from {model_path}...")
    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
    # Text hidden size comes from the text sub-config; derive it the same way the
    # LLM build does (the top-level Gemma4Config has no `hidden_size` itself).
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    text_hidden_size = model_args.hidden_size

    vision_args = VisionModelArgs(mesh_device, dummy_weights=True, max_batch_size=1, max_seq_len=8192)
    if vision_num_layers is not None:
        full_vision_layers = vision_args.hf_config.vision_config.num_hidden_layers
        vision_args.hf_config.vision_config.num_hidden_layers = vision_num_layers
        logger.info(f"Vision encoder depth override: {full_vision_layers} -> {vision_num_layers} layers (profiling)")
    vision_state_dict = _build_vision_state_dict(state_dict, vision_args)
    vision_tower = VisionTower(
        args=vision_args,
        dtype=vision_dtype,
        state_dict=vision_state_dict,
        tt_ccl=TT_CCL(mesh_device),
        weight_cache_path=vision_args.weight_cache_path(vision_dtype),
    )
    embed_vision = Gemma4MultimodalEmbedder.from_state_dict(
        mesh_device=mesh_device,
        state_dict=state_dict,
        vision_args=vision_args,
        text_hidden_size=text_hidden_size,
        dtype=ttnn.bfloat16,
        weight_cache_path=vision_args.weight_cache_path(ttnn.bfloat16),
    )

    # ── Encode the image (host preprocessing via HF Gemma4Processor) ────────
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, do_convert_rgb=True)
    _input_ids, pixel_values, image_position_ids = encode_multimodal(prompt, image, processor)
    pixel_values = pixel_values.to(torch.bfloat16)
    image_position_ids = image_position_ids.to(torch.long)
    num_patches = image_position_ids.shape[1]
    seq_len = _vision_encoder_seq_len(num_patches)

    # The tower is tensor-parallel across every device and runs one image at a time,
    # so a batch is just a loop over users.
    is_mesh = hasattr(mesh_device, "shape")
    num_devices = mesh_device.get_num_devices() if is_mesh else 1
    # Replicate the single encoded image to the full batch.
    pixel_values_batch = pixel_values.repeat(batch_size, 1, 1) if batch_size > 1 else pixel_values
    image_position_ids_batch = image_position_ids.repeat(batch_size, 1, 1) if batch_size > 1 else image_position_ids
    logger.info(
        f"Image: num_patches={num_patches}, encoder seq_len={seq_len}, text_hidden={text_hidden_size}, "
        f"batch={batch_size}, num_devices={num_devices}, tp={vision_args.tp}"
    )

    def _sync():
        ttnn.synchronize_device(mesh_device)

    def _run_image(index):
        """Run tower + embedder on one image (TP across the whole mesh).

        Returns (tower_ms, embed_ms, pooled_shape, proj_shape) and deallocates intermediates.
        """
        pv = pixel_values_batch[index : index + 1]
        pi = image_position_ids_batch[index : index + 1]
        t0 = time.perf_counter()
        pooled_tt, _mask = vision_tower(pv, pi, seq_len)
        _sync()
        t1 = time.perf_counter()
        projected_tt = embed_vision(pooled_tt)
        _sync()
        t2 = time.perf_counter()
        p_shape = tuple(pooled_tt.shape)
        pr_shape = tuple(projected_tt.shape)
        if hasattr(pooled_tt, "deallocate"):
            pooled_tt.deallocate(True)
        if hasattr(projected_tt, "deallocate"):
            projected_tt.deallocate(True)
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, p_shape, pr_shape

    # ── Warmup (compile both tower + embedder kernels) ────────────────────
    logger.info("Warming up (compile)...")
    _run_image(0)
    logger.info("Warmup complete")

    # ── Timed loop ────────────────────────────────────────────────────────
    # Each iteration processes the whole batch (one image per user). Per-iteration
    # totals are the sum over users, so per-image latency = total_min / batch_size and
    # aggregate throughput = batch_size / (total_min / 1000).
    tower_times, embed_times, total_times = [], [], []
    last_pooled_shape = last_proj_shape = None
    logger.info(f"Running {num_iters} timed iterations (batch={batch_size}, one image at a time)...")
    for _i in range(num_iters):
        iter_tower = 0.0
        iter_embed = 0.0
        for u in range(batch_size):
            t_ms, e_ms, p_shape, pr_shape = _run_image(u)
            iter_tower += t_ms
            iter_embed += e_ms
            last_pooled_shape, last_proj_shape = p_shape, pr_shape
        tower_times.append(iter_tower)
        embed_times.append(iter_embed)
        total_times.append(iter_tower + iter_embed)

    def _stats(name, samples):
        s = sorted(samples)
        mean = sum(s) / len(s)
        mn = s[0]
        logger.info(f"{name}: mean={mean:.2f} ms, min={mn:.2f} ms (n={len(s)})")
        return mean, mn

    logger.info("")
    logger.info(f"=== Vision benchmark results (batch_size={batch_size}, tp={vision_args.tp}) ===")
    _stats("vision_tower", tower_times)
    _stats("embed_vision", embed_times)
    total_mean, total_min = _stats("vision_tower+embed_vision", total_times)
    logger.info(f"Output shapes: pooled={last_pooled_shape}, projected={last_proj_shape}")
    per_image_ms = total_min / max(batch_size, 1)
    aggregate_img_s = batch_size * 1000.0 / max(total_min, 1e-9)
    logger.info(
        f"vision_tower+embed_vision min: {total_min:.2f} ms/batch -> "
        f"{per_image_ms:.2f} ms/image, {aggregate_img_s:.2f} img/s aggregate (batch={batch_size})"
    )
