# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig


class TtSDXLInpaintingPipeline(TtSDXLPipeline):
    def __init__(self, ttnn_device, torch_pipeline, pipeline_config: TtSDXLPipelineConfig):
        super().__init__(ttnn_device, torch_pipeline, pipeline_config)
