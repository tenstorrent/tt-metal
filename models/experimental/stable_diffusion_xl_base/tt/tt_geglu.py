# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from models.experimental.stable_diffusion_xl_base.refiner.tt.model_configs import RefinerModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.lora_weights_logger import lora_logger


class TtGEGLU(LightweightModule):
    def __init__(self, device, state_dict, module_path, model_config):
        super().__init__()

        self.device = device

        self.module_path = module_path

        self.is_refiner = isinstance(model_config, RefinerModelOptimisations)

        # Log module initialization start
        lora_logger.log_module_start(module_path, "TtGEGLU")

        # LORA WEIGHT: GEGLU projection weights - commonly targeted by LoRA for feedforward adaptation
        weights = state_dict[f"{module_path}.proj.weight"]
        bias = state_dict[f"{module_path}.proj.bias"]
        w1, w2 = weights.chunk(2, dim=0)  # Each: [out_dim // 2, in_dim]
        b1, b2 = bias.chunk(2, dim=0)  # Each: [out_dim // 2]

        w1 = w1.unsqueeze(0).unsqueeze(0)  # [1, 1, out_dim // 2, in_dim]
        w2 = w2.unsqueeze(0).unsqueeze(0)  # same

        ff_weights_dtype = model_config.ff_weights_dtype
        self.tt_weights_1, self.tt_bias_1, host_creation_ms, host_to_device_ms = prepare_linear_params(
            device, w1, b1, ff_weights_dtype, is_lora_impacted=True
        )
        # LORA WEIGHT LOG: GEGLU first projection weights
        lora_logger.log_weight_creation(
            module_path,
            "tt_weights_1",
            self.tt_weights_1.shape,
            ff_weights_dtype,
            device,
            "GEGLU first projection weights (w1)",
            tensor_obj=self.tt_weights_1,
            host_creation_time_ms=host_creation_ms,
            host_to_device_time_ms=host_to_device_ms,
        )

        self.tt_weights_2, self.tt_bias_2, host_creation_ms, host_to_device_ms = prepare_linear_params(
            device, w2, b2, ff_weights_dtype, is_lora_impacted=True
        )
        # LORA WEIGHT LOG: GEGLU second projection weights
        lora_logger.log_weight_creation(
            module_path,
            "tt_weights_2",
            self.tt_weights_2.shape,
            ff_weights_dtype,
            device,
            "GEGLU second projection weights (w2, with GELU)",
            tensor_obj=self.tt_weights_2,
            host_creation_time_ms=host_creation_ms,
            host_to_device_time_ms=host_to_device_ms,
        )

        self.program_config = model_config.get_matmul_config(matmul_path=f"{module_path}.proj.split")
        self.program_config_gelu = model_config.get_matmul_config(matmul_path=f"{module_path}.proj.split.gelu")
        assert self.program_config_gelu is not None, "Program config for GELU linear is None"
        assert (
            self.program_config_gelu.fused_activation.op_type == ttnn.UnaryOpType.GELU
        ), "GELU isn't fused in program config for GELU linear"

        self.compute_config = model_config.get_mm_compute_config(f"{module_path}.proj")
        self.output_memory_config = model_config.get_mm_output_memory_config(f"{module_path}.proj.split")
        self.output_memory_config_gelu = model_config.get_mm_output_memory_config(f"{module_path}.proj.split.gelu")

        # Log module initialization end
        lora_logger.log_module_end(module_path, "TtGEGLU")

    def forward(self, input_tensor):
        hidden_states = ttnn.linear(
            input_tensor,
            self.tt_weights_1,
            bias=self.tt_bias_1,
            memory_config=self.output_memory_config,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
        )

        gate = ttnn.linear(
            input_tensor,
            self.tt_weights_2,
            bias=self.tt_bias_2,
            memory_config=self.output_memory_config_gelu,
            program_config=self.program_config_gelu,
            compute_kernel_config=self.compute_config,
        )

        ttnn.deallocate(input_tensor)
        if self.is_refiner and ("down_blocks.1" in self.module_path or "up_blocks.2" in self.module_path):
            # gate is L1 BS, hidden_states is L1 interleaved; mul_ will output its result in L1 BS
            hidden_states = ttnn.mul_(gate, hidden_states, use_legacy=False, fast_and_approximate_mode=True)
        else:
            hidden_states = ttnn.mul_(hidden_states, gate, use_legacy=False, fast_and_approximate_mode=True)
        return hidden_states
