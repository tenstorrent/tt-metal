# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.stable_diffusion_xl_base.tt.model_configs import get_image_resolution_from_model_config
from models.demos.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from models.demos.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU


class TtFeedForward(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        lora_weights_manager=None,
    ):
        super().__init__()

        self.device = device
        self.image_resolution = get_image_resolution_from_model_config(model_config)
        self.tt_geglu = TtGEGLU(
            device, state_dict, f"{module_path}.net.0", model_config, lora_weights_manager=lora_weights_manager
        )

        weights = state_dict[f"{module_path}.net.2.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.net.2.bias"]

        ff_weights_dtype = model_config.ff_weights_dtype
        if lora_weights_manager:
            self.tt_weights, self.tt_bias = lora_weights_manager.prepare_lora_linear_params(
                device, weights, bias, ff_weights_dtype, f"{module_path}.net.2"
            )
        else:
            self.tt_weights, self.tt_bias = prepare_linear_params(device, weights, bias, ff_weights_dtype)

        self.ff2_model_config = model_config.get_matmul_config(f"{module_path}.net.2")
        self.default_compute_kernel_config = model_config.get_mm_compute_config(module_path)
        self.ff2_memory_config = model_config.get_mm_output_memory_config(f"{module_path}.net.2")

    def forward(self, hidden_states):
        # Reshard hidden_states to the appropriate grid_size used in feedforward layer
        if self.image_resolution == (512, 512) and hidden_states.shape[-1] == 1280:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=ttnn.CoreGrid(x=8, y=4),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)

        # GEGLU
        hidden_states = self.tt_geglu(hidden_states)

        # ff.net.2 linear
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights,
            bias=self.tt_bias,
            program_config=self.ff2_model_config,
            memory_config=self.ff2_memory_config,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        # In order not to break the following layers, we need to reshard back to the original grid size
        if self.image_resolution == (512, 512) and hidden_states.shape[-1] == 1280:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=ttnn.CoreGrid(x=5, y=8),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)

        return hidden_states
