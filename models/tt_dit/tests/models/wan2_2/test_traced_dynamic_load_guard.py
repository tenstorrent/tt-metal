# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`WanPipeline.__call__` rejects traced calls under dynamic_load.

A captured trace replays the weight addresses it saw at capture, and dynamic_load evicts and
reloads weights within every request, so the combination is rejected up front. Runs without a
device: the guard is the first statement of ``__call__``, so a bare instance exercises it.
"""

import pytest

from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline


def _bare_pipeline(dynamic_load: bool) -> WanPipeline:
    """A `WanPipeline` with only the attribute the guard reads; `__init__` needs a mesh device."""
    pipeline = object.__new__(WanPipeline)
    pipeline.dynamic_load = dynamic_load
    return pipeline


def test_traced_with_dynamic_load_raises():
    pipeline = _bare_pipeline(dynamic_load=True)
    with pytest.raises(ValueError, match="traced=True is unsupported with dynamic_load=True"):
        pipeline(prompts=["prompt"], num_inference_steps=1, traced=True)


def test_traced_without_dynamic_load_passes_guard():
    pipeline = _bare_pipeline(dynamic_load=False)
    # Past the guard, the uninitialized instance fails on its first missing attribute.
    with pytest.raises(AttributeError):
        pipeline(prompts=["prompt"], num_inference_steps=1, traced=True)


def test_untraced_with_dynamic_load_passes_guard():
    pipeline = _bare_pipeline(dynamic_load=True)
    with pytest.raises(AttributeError):
        pipeline(prompts=["prompt"], num_inference_steps=1, traced=False)
