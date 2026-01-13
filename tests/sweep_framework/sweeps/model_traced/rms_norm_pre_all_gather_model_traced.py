# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader import MasterConfigLoader

TIMEOUT = 30

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("rms_norm_pre_all_gather", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_memory_config=None,
    program_config=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    shape = input_shape if isinstance(input_shape, (tuple, list)) else (1, 1, 32, 32)
    eps = 1e-5

    # Tensor creation
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )
    torch_weight = torch.randn(int(shape[-1]), dtype=torch.float32)
    torch_output = torch_input * torch_weight / torch.sqrt(torch.mean(torch_input**2, dim=-1, keepdim=True) + eps)
    torch_weight_padded = torch.cat(
        [torch_weight, torch.zeros(((torch_weight.numel() + 31) // 32) * 32 - torch_weight.numel())]
    ).reshape(1, 1, -1, 32)

    # Create input tensor - bfloat8_b and bfloat4_b require TILE layout
    input_layout = ttnn.TILE_LAYOUT if input_a_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b] else input_a_layout
    input_tensor = ttnn.from_torch(
        torch_input, dtype=input_a_dtype, layout=input_layout, device=device, memory_config=input_a_memory_config
    )
    # Weight tensor should always be ROW_MAJOR layout and bfloat16 dtype
    # This is required by rms_norm_post_all_gather operation
    weight_tensor = ttnn.from_torch(
        torch_weight_padded,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_b_memory_config or input_a_memory_config,
    )

    # Op call
    start_time = start_measuring_time()

    # Parse program_config if provided (from traced JSON)
    if program_config and isinstance(program_config, dict):
        # Create LayerNormShardedMultiCoreProgramConfig from dict
        compute_grid = program_config.get("compute_with_storage_grid_size", {})
        ttnn_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid.get("x", 1), compute_grid.get("y", 1)),
            subblock_w=program_config.get("subblock_w", 1),
            block_h=program_config.get("block_h", 1),
            block_w=program_config.get("block_w", 1),
            inplace=bool(program_config.get("inplace", 0)),
        )
        stats = ttnn.rms_norm_pre_all_gather(input_tensor, program_config=ttnn_program_config)
    else:
        stats = ttnn.rms_norm_pre_all_gather(input_tensor)

    output_tensor = ttnn.rms_norm_post_all_gather(input_tensor, stats, epsilon=eps, weight=weight_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
