# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the step count `run_warm_generation` gives its compile pass."""

from contextlib import nullcontext

import pytest

from ....pipelines.minimax_h3.hyperflow_minimax_h3 import MiniMaxH3HyperFlow
from .common_av import run_warm_generation

NINE_POINT_GRID = (1.0, 0.95, 0.9, 0.8, 0.7, 0.55, 0.4, 0.2, 0.0)


class _FakePipeline:
    """Records each call's step count and, like the turbo pipeline, refuses one a contract forbids."""

    def __init__(self, hyperflow):
        self.hyperflow = hyperflow
        self.steps = []
        self.last_program_cache_misses = {}

    def quiet(self):
        return nullcontext()

    def __call__(self, prompt, *, seed, on_event=None, num_inference_steps=None, **kwargs):
        if getattr(self, "hyperflow", None) is not None:
            self.hyperflow.assert_forwards(num_inference_steps)
        self.steps.append(num_inference_steps)
        return "video"


def _hyperflow():
    return MiniMaxH3HyperFlow(version="1", gate=1.0, sigmas=NINE_POINT_GRID, video_shift=5.0, audio_shift=5.0)


@pytest.mark.parametrize("requested", [9, None])
def test_fixed_grid_warm_pass_keeps_the_callers_steps(requested):
    pipeline = _FakePipeline(_hyperflow())
    assert run_warm_generation(pipeline, "p", seed=0, num_inference_steps=requested) == "video"
    assert pipeline.steps == [requested, requested]


def test_free_schedule_warm_pass_stays_short():
    pipeline = _FakePipeline(None)
    run_warm_generation(pipeline, "p", seed=0, num_inference_steps=5)
    assert pipeline.steps == [3, 5]


def test_pipeline_without_a_contract_attribute_stays_short():
    pipeline = _FakePipeline(None)
    del pipeline.hyperflow
    run_warm_generation(pipeline, "p", seed=0, num_inference_steps=50)
    assert pipeline.steps == [3, 50]
