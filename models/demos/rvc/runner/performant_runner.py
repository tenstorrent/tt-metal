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


def load_ttnn_pipeline(device, model_config: RVCModelConfig, inference_config: RVCInferenceConfig) -> object:
    from models.demos.rvc.tt_impl.vc.pipeline import Pipeline as TTPipeline

    ttnn_pipeline = TTPipeline(
        tt_device=device,
        if_f0=model_config.if_f0,
        version=model_config.version,
        num=model_config.num,
        speaker_id=inference_config.speaker_id,
        f0_up_key=inference_config.f0_up_key,
        f0_method=inference_config.f0_method,
        index_rate=inference_config.index_rate,
        rms_mix_rate=inference_config.rms_mix_rate,
        protect=inference_config.protect,
    )
    return ttnn_pipeline


class RVCTestInfra:
    def __init__(
        self,
        device,
        inference_config: RVCInferenceConfig,
        model_config: RVCModelConfig | None = None,
    ):
        self.device = device
        self.num_devices = self.device.get_num_devices()
        self.batch_size = self.num_devices
        self.inference_config = inference_config
        self.model_config = model_config or RVCModelConfig()

        self.ttnn_pipeline = load_ttnn_pipeline(self.device, self.model_config, self.inference_config)
        self.tt_output = None

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if isinstance(torch_input_tensor, torch.Tensor):
            return torch_input_tensor, None
        raise TypeError(f"Expected preprocessed audio tensor, got {type(torch_input_tensor)!r}")

    def setup_dram_sharded_input(self, device, torch_input_tensor=None):
        audio_tensor, input_mem_config = self.setup_l1_sharded_input(device, torch_input_tensor)
        return audio_tensor, None, input_mem_config

    def run(self, input_tensor=None) -> torch.Tensor:
        audio_tensor, _ = self.setup_l1_sharded_input(self.device, input_tensor)
        return self.ttnn_pipeline._run_pipeline(audio_tensor)


class RVCRunner:
    def __init__(self):
        self.test_infra = None
        self.device = None
        self.initialized = False

    def initialize_inference(
        self,
        device,
        config,
        model_config: RVCModelConfig | None = None,
    ):
        (
            inference_config,
            normalized_model_config,
        ) = self._normalize_runner_config(config, model_config)
        self.test_infra = RVCTestInfra(device, inference_config, normalized_model_config)
        self.device = device

        self.initialized = True

    def run(self, torch_input_tensor=None) -> np.ndarray:
        tt_inputs_host, _ = self.test_infra.setup_l1_sharded_input(self.device, torch_input_tensor)
        if not self.initialized:
            raise RuntimeError("Runner is not initialized. Call initialize_inference(...) first.")
        output = self.test_infra.run(tt_inputs_host)
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
