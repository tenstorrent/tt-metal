# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from models.experimental.stable_diffusion_xl_base.tt.lora_weights_logger import lora_logger


class TtTimestepEmbedding(LightweightModule):
    def __init__(self, device, state_dict, module_path, linear_weights_dtype=ttnn.bfloat16):
        super().__init__()

        self.device = device

        # Log module initialization start
        lora_logger.log_module_start(module_path, "TtTimestepEmbedding")
        # LORA WEIGHT: Timestep embedding linear weights - occasionally targeted for temporal adaptation
        weights_1 = state_dict[f"{module_path}.linear_1.weight"]
        bias_1 = state_dict[f"{module_path}.linear_1.bias"]

        # LORA WEIGHT: Timestep embedding linear weights - occasionally targeted for temporal adaptation
        weights_2 = state_dict[f"{module_path}.linear_2.weight"]
        bias_2 = state_dict[f"{module_path}.linear_2.bias"]

        self.tt_weights_1, self.tt_bias_1, host_creation_ms, host_to_device_ms = prepare_linear_params(
            device, weights_1, bias_1, linear_weights_dtype, is_lora_impacted=True
        )
        # LORA WEIGHT LOG: Timestep embedding first linear weights
        lora_logger.log_weight_creation(
            module_path,
            "tt_weights_1",
            self.tt_weights_1.shape,
            linear_weights_dtype,
            device,
            "Timestep embedding first linear layer weights",
            tensor_obj=self.tt_weights_1,
            host_creation_time_ms=host_creation_ms,
            host_to_device_time_ms=host_to_device_ms,
        )

        self.tt_weights_2, self.tt_bias_2, host_creation_ms, host_to_device_ms = prepare_linear_params(
            device, weights_2, bias_2, linear_weights_dtype, is_lora_impacted=True
        )
        # LORA WEIGHT LOG: Timestep embedding second linear weights
        lora_logger.log_weight_creation(
            module_path,
            "tt_weights_2",
            self.tt_weights_2.shape,
            linear_weights_dtype,
            device,
            "Timestep embedding second linear layer weights",
            tensor_obj=self.tt_weights_2,
            host_creation_time_ms=host_creation_ms,
            host_to_device_time_ms=host_to_device_ms,
        )

        # Log module initialization end
        lora_logger.log_module_end(module_path, "TtTimestepEmbedding")

    def forward(self, sample):
        sample = ttnn.linear(sample, self.tt_weights_1, bias=self.tt_bias_1, activation="silu")
        sample = ttnn.linear(
            sample,
            self.tt_weights_2,
            bias=self.tt_bias_2,
        )
        return sample
