# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from models.experimental.stable_diffusion_xl_base.tt.lora_weights_logger import lora_logger


class TtFeedForward(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
    ):
        super().__init__()

        self.device = device

        # Log module initialization start
        lora_logger.log_module_start(module_path, "TtFeedForward")
        # LORA WEIGHT: GEGLU contains linear projections commonly targeted by LoRA
        self.tt_geglu = TtGEGLU(device, state_dict, f"{module_path}.net.0", model_config)

        # LORA WEIGHT: Feedforward output projection weights - commonly targeted by LoRA
        weights = state_dict[f"{module_path}.net.2.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.net.2.bias"]

        ff_weights_dtype = model_config.ff_weights_dtype
        self.tt_weights, self.tt_bias, host_creation_ms, host_to_device_ms = prepare_linear_params(
            device, weights, bias, ff_weights_dtype, is_lora_impacted=True
        )
        # LORA WEIGHT LOG: Feedforward output projection weights
        lora_logger.log_weight_creation(
            module_path,
            "tt_weights",
            self.tt_weights.shape,
            ff_weights_dtype,
            device,
            "Feedforward output projection weights (net.2)",
            tensor_obj=self.tt_weights,
            host_creation_time_ms=host_creation_ms,
            host_to_device_time_ms=host_to_device_ms,
        )

        self.ff2_model_config = model_config.get_matmul_config(f"{module_path}.net.2")
        self.default_compute_kernel_config = model_config.get_mm_compute_config(module_path)
        self.ff2_memory_config = model_config.get_mm_output_memory_config(f"{module_path}.net.2")

        # Log module initialization end
        lora_logger.log_module_end(module_path, "TtFeedForward")

    def forward(self, hidden_states):
        hidden_states = self.tt_geglu(hidden_states)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights,
            bias=self.tt_bias,
            program_config=self.ff2_model_config,
            memory_config=self.ff2_memory_config,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        return hidden_states
