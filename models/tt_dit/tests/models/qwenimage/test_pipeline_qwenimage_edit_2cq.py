# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit with two command queues: identical output to one CQ, and its timing.

Opens the mesh with two command queues and builds the CFG-parallel pipeline once. Each edit
runs with the per-step latent write / output read on CQ1 (``use_2cq``) and with everything on
CQ0; the images must be identical. Then several warm calls per mode compare the timings.

Run:
  TT_DIT_CACHE_DIR=<cache> python_env/bin/python -m pytest -s --timeout=0 \
    models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage_edit_2cq.py
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
OUT_DIR = "models/tt_dit/pipelines/qwenimage_edit"
EDITS = [
    ("a", "Give the cat a blue wizard hat."),
    ("b", "Replace the two TV remotes with two slices of pepperoni pizza."),
]
WARM_CALLS = 3


@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "trace_region_size": 130000000, "num_command_queues": 2}],
    ids=["line_2cq"],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "num_inference_steps"),
    [pytest.param((4, 8), 20, id="4x8_cfgpar_devvae_full_20steps")],
    indirect=["mesh_device"],
)
def test_qwenimage_edit_2cq(*, mesh_device: ttnn.MeshDevice, num_inference_steps: int) -> None:
    pipeline = QwenImageEditPipeline.create_pipeline(mesh_device=mesh_device, cfg_parallel=True, use_2cq=True)
    image = Image.open(CAT_IMAGE).convert("RGB")

    def run(tag: str, prompt: str, use_2cq: bool):
        for branch in pipeline._branches:  # noqa: SLF001
            branch._use_2cq = use_2cq  # noqa: SLF001
        t = time.time()
        out = pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            true_cfg_scale=4.0,
            seed=0,
        )
        wall = time.time() - t
        steps = pipeline.last_step_times
        out[0].save(f"{OUT_DIR}/edit_2cq_{tag}.png")
        warm = float(np.mean(steps[1:])) * 1000
        logger.info(
            f"[2cq] {tag}: use_2cq={use_2cq} wall {wall:.2f} s | denoise {sum(steps):.2f} s | "
            f"step0 {steps[0] * 1000:.0f} ms | warm {warm:.1f} ms/step"
        )
        return np.asarray(out[0].convert("RGB")), wall, sum(steps), warm

    # correctness: 1CQ vs 2CQ on the same pipeline and trace
    for name, prompt in EDITS:
        one, *_ = run(f"{name}_1cq", prompt, False)
        two, *_ = run(f"{name}_2cq", prompt, True)
        same = np.array_equal(one, two)
        logger.info(f"[2cq] {name}: 1CQ vs 2CQ identical={same} max|d|={int(np.abs(one.astype(int) - two).max())}")
        assert same, f"2CQ output differs from 1CQ for {name}"

    # timing: warm calls of one edit, alternating modes (trace already captured for this prompt)
    stats = {False: [], True: []}
    for i in range(WARM_CALLS):
        for mode in (False, True):
            _, wall, denoise, warm = run(f"t{i}_{'2cq' if mode else '1cq'}", EDITS[0][1], mode)
            stats[mode].append((wall, denoise, warm))
    for mode, rows in stats.items():
        w, d, s = (np.median([r[k] for r in rows]) for k in range(3))
        logger.info(
            f"[2cq] median over {WARM_CALLS}: use_2cq={mode} wall {w:.2f} s | denoise {d:.2f} s | {s:.1f} ms/step"
        )
