# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 with FastVideo's FastH3 4-step DMD student merged in: LoRA, dense diffs and, under
`vsa_config`, the VSA gates (see `MiniMaxH3TurboPipeline._read_safetensors`).

The student was distilled on `dmd_denoising_steps = [999, 749, 500, 250]` of 1000. Its grid is that
ladder plus the terminal 0, pushed through each modality's shift; the base `linspace` grid differs at
the first two points.
"""

from __future__ import annotations

import torch

from .pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline
from .scheduler import MiniMaxH3Scheduler, shift_sigmas

FASTH3_BASE_LADDER = (0.999, 0.749, 0.5, 0.25, 0.0)
FASTH3_NUM_GRID_POINTS = len(FASTH3_BASE_LADDER)


class MiniMaxH3FastPipeline(MiniMaxH3TurboPipeline):
    """`MiniMaxH3TurboPipeline` on the FastH3 sigma ladder; `num_inference_steps` is fixed at 5."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.hyperflow is not None:
            raise ValueError(
                f"{self.lora_path} is a HyperFlow adapter, which fixes its own grid; use the Turbo pipeline"
            )

    def __call__(self, prompt: str, *, num_inference_steps: int | None = None, **kwargs):
        if num_inference_steps is None:
            num_inference_steps = FASTH3_NUM_GRID_POINTS
        # Warmup's forced generations run 2 grid points on the base grid; they only compile.
        if num_inference_steps != FASTH3_NUM_GRID_POINTS and not self._warming:
            raise ValueError(
                f"FastH3 was distilled for {FASTH3_NUM_GRID_POINTS} grid points, got num_inference_steps={num_inference_steps}"
            )
        return super().__call__(prompt, num_inference_steps=num_inference_steps, **kwargs)

    def _build_schedulers(self, num_inference_steps: int) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
        if self._warming:
            return super()._build_schedulers(num_inference_steps)
        ladder = torch.tensor(FASTH3_BASE_LADDER, dtype=torch.float32)
        scheduler = MiniMaxH3Scheduler(shift=self.video_shift)
        audio_scheduler = MiniMaxH3Scheduler(shift=self.audio_shift)
        scheduler.set_timesteps(sigmas=shift_sigmas(ladder, self.video_shift))
        audio_scheduler.set_timesteps(sigmas=shift_sigmas(ladder, self.audio_shift))
        return scheduler, audio_scheduler


__all__ = ["FASTH3_BASE_LADDER", "FASTH3_NUM_GRID_POINTS", "MiniMaxH3FastPipeline"]
