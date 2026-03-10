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

TIMEOUT = 60

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("fill_cache", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """Custom device fixture for fill_cache with DispatchCoreConfig to free up more compute cores"""
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())
    device_name = ttnn.get_arch_name()

    yield (device, device_name)

    ttnn.close_device(device)
    del device


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    *,
    device,
    **kwargs,  # Accept any extra parameters the loader might pass
) -> list:
    torch.manual_seed(0)

    # Handle both sample suite (tuple/list) and model_traced suite (dict with 'self'/'other')
    if isinstance(input_shape, dict):
        cache_shape = tuple(input_shape.get("self", (1, 1, 32, 64)))
        input_tensor_shape = tuple(input_shape.get("other", (1, 1, 32, 64)))
    else:
        # Convert list to tuple if needed
        cache_shape = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        input_tensor_shape = cache_shape

    batch_idx = 0

    # Tensor creation
    torch_cache = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        cache_shape
    )
    torch_input = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype or input_a_dtype
    )(input_tensor_shape)
    torch_output = torch_cache.clone()
    if len(torch_cache.shape) >= 4 and len(torch_input.shape) >= 4 and torch_cache.shape[0] > batch_idx:
        seq_len = min(torch_input.shape[2], torch_cache.shape[2])
        torch_output[batch_idx, :, :seq_len, :] = torch_input[0, :, :seq_len, :]

    cache_tensor = ttnn.from_torch(
        torch_cache, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
    )
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=input_b_dtype or input_a_dtype,
        layout=input_b_layout or input_a_layout,
        device=device,
        memory_config=input_b_memory_config or input_a_memory_config,
    )

    # Op call
    start_time = start_measuring_time()
    output_tensor = ttnn.fill_cache(cache_tensor, input_tensor, batch_idx)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.95), e2e_perf]
