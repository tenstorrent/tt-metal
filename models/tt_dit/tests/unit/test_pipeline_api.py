# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free coverage for PipelineAPIMixin.run_single_prompt: which keyword arguments reach the pipeline."""

from __future__ import annotations

from ...pipelines.pipeline_api import PipelineAPIMixin


class _RecordingPipeline(PipelineAPIMixin):
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return "image"


def test_defaults_pass_only_the_original_arguments():
    pipeline = _RecordingPipeline()
    assert pipeline.run_single_prompt(prompt="a cat", num_inference_steps=4) == "image"
    assert pipeline.kwargs == {
        "prompts": ["a cat"],
        "negative_prompts": None,
        "num_inference_steps": 4,
        "seed": 0,
        "traced": True,
        "on_event": None,
    }


def test_guidance_and_second_prompts_are_forwarded_as_lists():
    pipeline = _RecordingPipeline()
    pipeline.run_single_prompt(
        prompt="a cat",
        prompt_2="a tabby cat",
        negative_prompt="blurry",
        negative_prompt_2="noisy",
        num_inference_steps=4,
        guidance_scale=4.5,
        seed=7,
    )
    assert pipeline.kwargs["prompts"] == ["a cat"]
    assert pipeline.kwargs["prompts_2"] == ["a tabby cat"]
    assert pipeline.kwargs["negative_prompts"] == ["blurry"]
    assert pipeline.kwargs["negative_prompts_2"] == ["noisy"]
    assert pipeline.kwargs["guidance_scale"] == 4.5
    assert pipeline.kwargs["seed"] == 7
