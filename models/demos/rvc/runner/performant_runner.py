# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        inference_config: RVCInferenceConfig,
        model_config: RVCModelConfig | None = None,
        validation_config: RVCValidationConfig | None = None,
        warmup_runs: int = 1,
    ):
        self.test_infra = create_test_infra(device, inference_config, model_config, validation_config)
        self.device = device

        # Warm up the TT path to populate caches before repeated execution.
        for _ in range(max(warmup_runs, 0)):
            self.test_infra.run()

        self.initialized = True

    def execute_inference(self) -> ttnn.Tensor | None:
        if not self.initialized:
            raise RuntimeError("Runner is not initialized. Call initialize_inference(...) first.")
        return self.test_infra.run()

    def release_inference(self):
        self.initialized = False

    def validate(self):
        if not self.initialized:
            raise RuntimeError("Runner is not initialized. Call initialize_inference(...) first.")
        return self.test_infra.validate()

    def run(self):
        output = self.execute_inference()
        self.validate()
        return output
