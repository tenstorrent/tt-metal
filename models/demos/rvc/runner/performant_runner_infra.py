# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class RVCModelConfig:
    if_f0: bool = True
    version: str = "v1"
    num: str = "48k"


@dataclass(frozen=True)
class RVCInferenceConfig:
    audio_path: str
    speaker_id: int = 0
    f0_up_key: int = 0
    f0_method: str = "pm"
    index_rate: float = 0.75
    resample_sr: int = 0
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
        resample_sr=inference_config.resample_sr,
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
        self.inference_config = inference_config
        self.model_config = model_config or RVCModelConfig()
        self.validation_config = validation_config or RVCValidationConfig()
        self.audio_path = Path(self.inference_config.audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"RVC input audio path does not exist: {self.audio_path}")

        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.non_silent_output = False

        self.ttnn_pipeline = self._load_pipeline()
        self.tt_output = None

    def _load_pipeline(self):
        return load_ttnn_pipeline(self.device, self.model_config, self.inference_config)

    @staticmethod
    def _to_numpy(audio) -> np.ndarray:
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def run(self) -> np.ndarray:
        self.tt_output = self._to_numpy(self.ttnn_pipeline.infer(str(self.audio_path)))
        return self.tt_output

    def validate(self, output_tensor=None) -> tuple[bool, str]:
        tt_output = self.tt_output if output_tensor is None else self._to_numpy(output_tensor)
        if tt_output is None:
            tt_output = self.run()

        if tt_output.shape[0] == 0:
            self.pcc_passed = False
            self.pcc_message = "Empty output tensor."
            return self.pcc_passed, self.pcc_message

        self.non_silent_output = float(np.max(np.abs(tt_output))) > 0.0
        self.pcc_passed = self.non_silent_output
        self.pcc_message = f"num_samples={tt_output.shape[0]}, non_silent_output={self.non_silent_output}"
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
