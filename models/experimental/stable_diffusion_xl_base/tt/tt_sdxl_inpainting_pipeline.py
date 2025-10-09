# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from models.experimental.stable_diffusion_xl_base.tests.test_common import get_timesteps
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig


@dataclass
class TtSDXLInpaintingPipelineConfig(TtSDXLPipelineConfig):
    strength: float = 0.99


class TtSDXLInpaintingPipeline(TtSDXLPipeline):
    def __init__(self, ttnn_device, torch_pipeline, pipeline_config: TtSDXLInpaintingPipelineConfig):
        super().__init__(ttnn_device, torch_pipeline, pipeline_config)

    def _prepare_timesteps(self):
        super()._prepare_timesteps()

        print("Calling get_timesteps from derived class")
        self.ttnn_timesteps, self.pipeline_config.num_inference_steps = get_timesteps(
            self.torch_pipeline.scheduler, self.pipeline_config.num_inference_steps, self.pipeline_config.strength, None
        )
