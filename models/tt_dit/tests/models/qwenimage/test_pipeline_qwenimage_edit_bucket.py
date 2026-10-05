# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prompt-length bucketing on Qwen-Image-Edit: accuracy vs unpadded, and trace reuse.

Builds the CFG-parallel pipeline once. For each edit it runs unpadded and padded (toggling
``pipeline.prompt_bucket``) and compares the images, then runs several instructions of
different lengths with bucketing on and reports which calls re-captured the denoise trace.

Run:
  TT_DIT_CACHE_DIR=<cache> python_env/bin/python -m pytest -s --timeout=0 \
    models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage_edit_bucket.py
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.qwenimage_edit.pipeline_qwenimage_edit import QwenImageEditPipeline
from ....utils.test import line_params_req_exact_devices

CAT_IMAGE = "models/sample_data/huggingface_cat_image.jpg"
LANDSCAPE_IMAGE = "models/sample_data/house_in_field_1080p.jpg"
OUT_DIR = "models/tt_dit/pipelines/qwenimage_edit"

EDITS = [
    ("a", CAT_IMAGE, "Give the cat a blue wizard hat."),
    ("b", CAT_IMAGE, "Replace the two TV remotes with two slices of pepperoni pizza."),
    ("c", CAT_IMAGE, "Turn this photo into a Van Gogh style oil painting."),
    ("d", LANDSCAPE_IMAGE, "Make it a night scene with a full moon in the sky."),
]
BUCKET = 64
STRESS_BUCKET = 256


def _pcc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x.astype(np.float64).ravel(), y.astype(np.float64).ravel())[0, 1])


def _psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(((x.astype(np.float64) - y.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "trace_region_size": 130000000}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "num_inference_steps"),
    [pytest.param((4, 8), 20, id="4x8_cfgpar_devvae_full_20steps")],
    indirect=["mesh_device"],
)
def test_qwenimage_edit_prompt_bucket(*, mesh_device: ttnn.MeshDevice, num_inference_steps: int) -> None:
    pipeline = QwenImageEditPipeline.create_pipeline(mesh_device=mesh_device, cfg_parallel=True, prompt_bucket=None)

    def run(tag: str, image_path: str, prompt: str, bucket: int | None) -> np.ndarray:
        pipeline.prompt_bucket = bucket
        tracers_before = [b._tracer for b in pipeline._branches]  # noqa: SLF001
        t = time.time()
        images = pipeline(
            image=Image.open(image_path).convert("RGB"),
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            true_cfg_scale=4.0,
            seed=0,
        )
        wall = time.time() - t
        branches = pipeline._branches  # noqa: SLF001
        recaptured = any(b._tracer is not before for b, before in zip(branches, tracers_before))  # noqa: SLF001
        steps = pipeline.last_step_times
        images[0].save(f"{OUT_DIR}/edit_bucket_{tag}.png")
        logger.info(
            f"[bucket] {tag}: bucket={bucket} prompt_len={branches[0]._sig[1]} "  # noqa: SLF001
            f"trace {'RE-CAPTURED' if recaptured else 'reused'} | wall {wall:.1f} s | "
            f"step0 {steps[0] * 1000:.0f} ms | warm {np.mean(steps[1:]) * 1000:.1f} ms/step"
        )
        return np.asarray(images[0].convert("RGB")), recaptured

    # 1. accuracy: unpadded vs padded, same pipeline
    worst = 1.0
    for name, path, prompt in EDITS:
        unpadded, _ = run(f"{name}_none", path, prompt, None)
        padded, _ = run(f"{name}_b{BUCKET}", path, prompt, BUCKET)
        pcc = _pcc(unpadded, padded)
        worst = min(worst, pcc)
        logger.info(
            f"[bucket] {name}: unpadded vs bucket {BUCKET}: pcc={pcc:.6f} psnr={_psnr(unpadded, padded):.2f} dB"
        )
    unpadded_a, _ = run("a_none_again", *EDITS[0][1:], None)
    stress, _ = run(f"a_b{STRESS_BUCKET}", *EDITS[0][1:], STRESS_BUCKET)
    logger.info(
        f"[bucket] a: unpadded vs bucket {STRESS_BUCKET} (stress): pcc={_pcc(unpadded_a, stress):.6f} "
        f"psnr={_psnr(unpadded_a, stress):.2f} dB"
    )

    # 2. trace reuse across instructions of different lengths (bucket on throughout)
    recaptures = {}
    for tag, (name, path, prompt) in zip(("r1", "r2", "r3", "r4"), (EDITS[0], EDITS[1], EDITS[2], EDITS[0])):
        _, recaptures[tag] = run(f"{tag}_{name}", path, prompt, BUCKET)
    logger.info(f"[bucket] re-captures with bucket {BUCKET} (a, b, c, a): {recaptures}")
    logger.info(f"[bucket] worst unpadded-vs-bucket{BUCKET} pcc: {worst:.6f}")

    assert not any(list(recaptures.values())[1:]), "instructions within one bucket must reuse the trace"
