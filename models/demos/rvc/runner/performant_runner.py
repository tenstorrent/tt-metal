# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

import ttnn
from models.demos.rvc.runner.performant_runner_infra import (
    RVCInferenceConfig,
    RVCModelConfig,
    RVCValidationConfig,
    create_test_infra,
)


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
        validation_config: RVCValidationConfig | None = None,
        warmup_runs: int = 1,
    ):
        (
            inference_config,
            normalized_model_config,
            normalized_validation_config,
            normalized_warmup_runs,
        ) = self._normalize_runner_config(config, model_config, validation_config, warmup_runs)
        self.test_infra = create_test_infra(
            device, inference_config, normalized_model_config, normalized_validation_config
        )
        self.device = device

        # Warm up the TT path to populate caches before repeated execution.
        for _ in range(max(normalized_warmup_runs, 0)):
            self.test_infra.run()

        self.initialized = True

    def execute_inference(self, tt_inputs_host=None) -> ttnn.Tensor | None:
        if not self.initialized:
            raise RuntimeError("Runner is not initialized. Call initialize_inference(...) first.")
        return self.test_infra.run(tt_inputs_host)

    def release_inference(self):
        self.initialized = False

    def validate(self):
        if not self.initialized:
            raise RuntimeError("Runner is not initialized. Call initialize_inference(...) first.")
        return self.test_infra.validate()

    def run(self, torch_input_tensor=None) -> np.ndarray:
        tt_inputs_host, _ = self.test_infra.setup_l1_sharded_input(self.device, torch_input_tensor)
        output = self.execute_inference(tt_inputs_host)
        self.validate()
        return output

    @staticmethod
    def _normalize_runner_config(config, model_config, validation_config, warmup_runs):
        if isinstance(config, RVCInferenceConfig):
            return config, model_config, validation_config, warmup_runs

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

        if validation_config is None and "validation" in config:
            validation_section = config["validation"]
            validation_config = (
                validation_section
                if isinstance(validation_section, RVCValidationConfig)
                else RVCValidationConfig(**validation_section)
            )

        normalized_warmup_runs = int(config.get("warmup_runs", warmup_runs))
        return inferred_inference_config, model_config, validation_config, normalized_warmup_runs
