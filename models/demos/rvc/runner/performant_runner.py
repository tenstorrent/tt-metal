# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.demos.rvc.utils.f0 import F0Method


@dataclass(frozen=True)
class RVCModelConfig:
    if_f0: bool = True
    version: str = "v1"
    num: str = "48k"


@dataclass(frozen=True)
class RVCInferenceConfig:
    num_secs: float
    speaker_id: int = 0
    f0_up_key: int = 0
    f0_method: F0Method = F0Method.RAPT
    index_rate: float = 0.75
    rms_mix_rate: float = 0.25
    protect: float = 0.33


def load_ttnn_pipeline(
    device,
    model_config: RVCModelConfig,
    inference_config: RVCInferenceConfig,
    batch_size: int = 1,
    validation=False,
    performance_runner=False,
):
    from models.demos.rvc.tt_impl.vc.pipeline import Pipeline as TTPipeline

    ttnn_pipeline = TTPipeline(
        device=device,
        batch_size=batch_size,
        if_f0=model_config.if_f0,
        version=model_config.version,
        num=model_config.num,
        speaker_id=inference_config.speaker_id,
        f0_up_key=inference_config.f0_up_key,
        f0_method=inference_config.f0_method,
        index_rate=inference_config.index_rate,
        rms_mix_rate=inference_config.rms_mix_rate,
        protect=inference_config.protect,
        validation=validation,
        performance_runner=performance_runner,
    )
    return ttnn_pipeline


class RVCRunner:
    def __init__(self, validation=False):
        self.test_infra = None
        self.device = None
        self.initialized = False

    def initialize_inference(
        self,
        device,
        config,
        batch_size: int = 1,
        model_config: RVCModelConfig | None = None,
        input_tensor_torch: torch.Tensor | None = None,
        validation=False,
        performance_runner=False,
    ):
        (
            inference_config,
            normalized_model_config,
        ) = self._normalize_runner_config(config, model_config)
        self.device = device
        self.num_devices = self.device.get_num_devices()
        self.batch_size = batch_size
        # self.batch_size = self.num_devices
        self.inference_config = inference_config
        self.model_config = normalized_model_config or RVCModelConfig()
        self.ttnn_pipeline = load_ttnn_pipeline(
            self.device,
            self.model_config,
            self.inference_config,
            batch_size=self.batch_size,
            validation=validation,
            performance_runner=performance_runner,
        )

        self.initialized = True

    def run(self, torch_input_tensor=None) -> np.ndarray:
        output = self.ttnn_pipeline.run(torch_input_tensor)
        return output

    @staticmethod
    def _normalize_runner_config(config, model_config):
        if isinstance(config, RVCInferenceConfig):
            return config, model_config

        if not isinstance(config, dict):
            raise TypeError(f"Expected config to be RVCInferenceConfig or dict, got {type(config)!r}")

        inference_section = config.get("inference", config)
        inferred_inference_config = (
            inference_section
            if isinstance(inference_section, RVCInferenceConfig)
            else RVCInferenceConfig(**inference_section)
        )

        if model_config is None and "model" in config:
            model_section = config["model"]
            model_config = (
                model_section if isinstance(model_section, RVCModelConfig) else RVCModelConfig(**model_section)
            )

        return inferred_inference_config, model_config
