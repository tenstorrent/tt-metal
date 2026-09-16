# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile

from loguru import logger

# Suffixes vbench 0.1.5 accepts when it lists a directory in custom_input mode
# (vbench/__init__.py, build_full_info_json).
_VBENCH_VIDEO_SUFFIXES = (".mp4", ".gif", ".jpg", ".png")


def vbench_prompt_list(video_path: str, prompt: str | None) -> list[str] | dict[str, str]:
    """Shape the prompt for vbench's custom_input mode.

    vbench 0.1.5 takes two shapes depending on ``videos_path``: a single file wants a one-element
    list; a directory wants a dict keyed by the bare filename (it does
    ``{os.path.join(videos_path, k): prompt_list[k] for k in prompt_list}`` and then matches each
    listed video by absolute path), so handing a directory a list raises
    ``TypeError: list indices must be integers or slices, not str``. Every video in the directory is
    scored against the same prompt.
    """
    if prompt is None:
        return []
    if os.path.isdir(video_path):
        return {
            name: prompt
            for name in sorted(os.listdir(video_path))
            if os.path.splitext(name)[1].lower() in _VBENCH_VIDEO_SUFFIXES
        }
    return [prompt]


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
        # Never silently pass: a requested quality gate with no vbench must surface, not no-op.
        # Callers that treat missing vbench as skippable should guard with pytest.importorskip.
        raise RuntimeError("VBench quality gate requested but `vbench` is not installed") from e

    # VBench 0.1.5 checkpoints contain typing.OrderedDict which is rejected by
    # torch.load's weights_only=True default (PyTorch 2.6+).
    import typing

    import torch

    torch.serialization.add_safe_globals([typing.OrderedDict])

    dimension_list = list(thresholds.keys())

    with tempfile.TemporaryDirectory() as tmp_dir:
        name = "eval"
        prompt_list = vbench_prompt_list(video_path, prompt)

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

    failures = []
    for metric, minimum in thresholds.items():
        if metric not in scores:
            # A requested threshold with no returned score is an ungated dimension, not a pass.
            failures.append(f"{metric}: no score returned (ungated dimension)")
        elif scores[metric] < minimum:
            failures.append(f"{metric} = {scores[metric]:.4f} < {minimum:.4f}")

    if failures:
        raise AssertionError("VBench quality gate failed:\n  " + "\n  ".join(failures))

    return scores
