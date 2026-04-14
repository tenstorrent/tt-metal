# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger


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
    f0_method: str = "pm"
    index_rate: float = 0.75
    rms_mix_rate: float = 0.25
    protect: float = 0.33


@dataclass(frozen=True)
class RVCValidationConfig:
    require_non_silent_output: bool = True


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
        validation_config: RVCValidationConfig | None = None,
    ):
        self.device = device
        self.num_devices = self.device.get_num_devices()
        self.batch_size = self.num_devices
        self.inference_config = inference_config
        self.model_config = model_config or RVCModelConfig()
        self.validation_config = validation_config or RVCValidationConfig()

        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.non_silent_output = False

        self.ttnn_pipeline = self._load_pipeline()
        self.tt_output = None

    def _load_pipeline(self):
        return load_ttnn_pipeline(self.device, self.model_config, self.inference_config)

    def _prepare_audio_input(self, num_secs: float | None = None) -> torch.Tensor:
        num_secs = self.inference_config.num_secs if num_secs is None else num_secs
        num_samples = max(int(num_secs * 16000), 1)
        generator = torch.Generator().manual_seed(0)
        audio = torch.randn(num_samples, generator=generator, dtype=torch.float32)
        return audio.unsqueeze(0).repeat(self.batch_size, 1)

    @staticmethod
    def _to_numpy(audio) -> np.ndarray:
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if torch_input_tensor is None:
            return self._prepare_audio_input(), None
        if isinstance(torch_input_tensor, torch.Tensor):
            return torch_input_tensor, None
        raise TypeError(f"Expected preprocessed audio tensor, got {type(torch_input_tensor)!r}")

    def setup_dram_sharded_input(self, device, torch_input_tensor=None):
        audio_tensor, input_mem_config = self.setup_l1_sharded_input(device, torch_input_tensor)
        return audio_tensor, None, input_mem_config

    def run(self, input_tensor=None) -> np.ndarray:
        audio_tensor, _ = self.setup_l1_sharded_input(self.device, input_tensor)
        self.tt_output = self._to_numpy(self.ttnn_pipeline._run_pipeline(audio_tensor))
        return self.tt_output

    def validate(self, output_tensor=None) -> tuple[bool, str]:
        tt_output = self.tt_output if output_tensor is None else self._to_numpy(output_tensor)
        if tt_output is None:
            tt_output = self.run()

        if tt_output.size == 0:
            self.pcc_passed = False
            self.pcc_message = "Empty output tensor."
            return self.pcc_passed, self.pcc_message

        self.non_silent_output = float(np.max(np.abs(tt_output))) > 0.0
        self.pcc_passed = self.non_silent_output
        self.pcc_message = f"output_shape={tt_output.shape}, non_silent_output={self.non_silent_output}"
        logger.info(f"RVC, {self.pcc_message}")

        passed = self.pcc_passed
        if self.validation_config.require_non_silent_output:
            passed = passed and self.non_silent_output
        return passed, self.pcc_message


def create_test_infra(
    device,
    inference_config: RVCInferenceConfig,
    model_config: RVCModelConfig | None = None,
    validation_config: RVCValidationConfig | None = None,
) -> RVCTestInfra:
    return RVCTestInfra(device, inference_config, model_config, validation_config)
