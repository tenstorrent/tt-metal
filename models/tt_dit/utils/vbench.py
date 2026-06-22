# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile

from loguru import logger


def assert_vbench_quality(
    video_path: str,
    *,
    prompt: str | None = None,
    thresholds: dict[str, float],
    device: str = "cpu",
) -> dict[str, float]:
    try:
        from vbench import VBench
    except ImportError as e:
        logger.warning(f"VBench import failed ({e}), skipping quality evaluation")
        return {}

    # VBench 0.1.5 checkpoints contain typing.OrderedDict which is rejected by
    # torch.load's weights_only=True default (PyTorch 2.6+).
    import typing

    import torch

    torch.serialization.add_safe_globals([typing.OrderedDict])

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
            raise AssertionError(msg)

    return scores
