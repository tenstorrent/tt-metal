# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile

from loguru import logger

DEFAULT_THRESHOLDS = {
    "subject_consistency": 0.85,
    "background_consistency": 0.90,
    "motion_smoothness": 0.95,
    "dynamic_degree": 0.50,
}


def assert_vbench_quality(
    video_path: str,
    *,
    prompt: str | None = None,
    thresholds: dict[str, float] | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    try:
        from vbench import VBench
    except ImportError:
        logger.warning("VBench is not installed, skipping quality evaluation")
        return {}

    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    dimension_list = list(thresholds.keys())

    with tempfile.TemporaryDirectory() as tmp_dir:
        name = "eval"
        prompt_list = [prompt] if prompt is not None else []

        bench = VBench(device=device, full_info_dir="", output_path=tmp_dir)
        bench.evaluate(
            videos_path=video_path,
            name=name,
            dimension_list=dimension_list,
            prompt_list=prompt_list,
            mode="custom_input",
        )

        results_path = os.path.join(tmp_dir, f"{name}_eval_results.json")
        with open(results_path) as f:
            raw_results = json.load(f)

    scores: dict[str, float] = {}
    for metric, value in raw_results.items():
        scores[metric] = value[0]

    for metric, score in scores.items():
        logger.info(f"VBench {metric} = {score:.4f}")

    for metric, minimum in thresholds.items():
        if metric not in scores:
            continue
        if scores[metric] < minimum:
            msg = f"VBench {metric} = {scores[metric]:.4f} < {minimum:.4f}"
            raise Exception(msg)  # noqa: TRY002

    return scores
