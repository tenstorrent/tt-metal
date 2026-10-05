# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Many edits on one Qwen-Image-Edit pipeline, the way the media server uses it.

Builds the CFG-parallel pipeline once, then runs several edits back to back (different prompts,
prompt lengths and input images) and checks that a repeated request reproduces its first output, so
prompt and image inputs do not leak between calls through the persistent trace buffers. Reports the
wall time of every call and which calls re-captured the denoise trace.

Run:
  TT_DIT_CACHE_DIR=<cache> python_env/bin/python -m pytest -s --timeout=0 \
    models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage_edit_multi.py
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

REQUESTS = [
    ("a", CAT_IMAGE, "Give the cat a blue wizard hat."),
    ("b", CAT_IMAGE, "Replace the two TV remotes with two slices of pepperoni pizza."),
    ("c", CAT_IMAGE, "Turn this photo into a Van Gogh style oil painting."),
    ("d", LANDSCAPE_IMAGE, "Make it a night scene with a full moon in the sky."),
    ("e", CAT_IMAGE, "Give the cat a blue wizard hat."),
]


def _pcc(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    return float(np.corrcoef(x, y)[0, 1])


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
def test_qwenimage_edit_pipeline_multi_request(*, mesh_device: ttnn.MeshDevice, num_inference_steps: int) -> None:
    pipeline = QwenImageEditPipeline.create_pipeline(mesh_device=mesh_device, cfg_parallel=True)

    outputs = {}
    for name, image_path, prompt in REQUESTS:
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
        recaptured = any(
            b._tracer is not before for b, before in zip(pipeline._branches, tracers_before)
        )  # noqa: SLF001
        steps = pipeline.last_step_times
        prompt_len = pipeline._branches[0]._sig[1]  # noqa: SLF001
        out_path = f"{OUT_DIR}/edit_multi_{name}.png"
        images[0].save(out_path)
        outputs[name] = np.asarray(images[0].convert("RGB"))
        logger.info(
            f"[multi] {name}: wall {wall:.1f} s | step0 {steps[0] * 1000:.0f} ms | "
            f"warm per-step {np.mean(steps[1:]) * 1000:.1f} ms | prompt_len {prompt_len} | "
            f"trace {'RE-CAPTURED' if recaptured else 'reused'} | '{prompt}' on {image_path} -> {out_path}"
        )

    identical = np.array_equal(outputs["a"], outputs["e"])
    pcc_ae = _pcc(outputs["a"], outputs["e"])
    max_abs = int(np.abs(outputs["a"].astype(np.int16) - outputs["e"].astype(np.int16)).max())
    logger.info(f"[multi] a vs e: identical={identical} pcc={pcc_ae:.6f} max|diff|={max_abs}")
    for other in ("b", "c", "d"):
        logger.info(f"[multi] a vs {other}: pcc={_pcc(outputs['a'], outputs[other]):.4f}")

    assert identical or pcc_ae > 0.999, f"repeated request differs from the first: pcc={pcc_ae}"
