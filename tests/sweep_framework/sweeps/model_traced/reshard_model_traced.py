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
model_traced_params = loader.get_suite_parameters("reshard", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Adjust layout for non-tile-aligned shard shapes.
    If shard shape is not tile-aligned, we need ROW_MAJOR layout.
    """
    # No configs need to be skipped - we'll adjust them in run()
    return False, None


def mesh_device_fixture():
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
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    shape = input_shape if isinstance(input_shape, (tuple, list)) else (1, 1, 32, 32)

    # Check if output shard shape is tile-aligned
    # If not, we need to use ROW_MAJOR layout
    needs_row_major = False
    if hasattr(output_memory_config, "shard_spec") and output_memory_config.shard_spec:
        shard_spec = output_memory_config.shard_spec
        if hasattr(shard_spec, "shape"):
            shard_shape = shard_spec.shape
            if len(shard_shape) >= 2:
                height, width = shard_shape[-2], shard_shape[-1]
                if height % 32 != 0 or width % 32 != 0:
                    needs_row_major = True

    # Override layout if needed
    actual_layout = ttnn.ROW_MAJOR_LAYOUT if needs_row_major else input_a_layout

    # Tensor creation
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )
    torch_output = torch_input.clone()

    input_tensor = ttnn.from_torch(
        torch_input, dtype=input_a_dtype, layout=actual_layout, device=device, memory_config=input_a_memory_config
    )

    # Op call
    start_time = start_measuring_time()
    output_tensor = ttnn.reshard(input_tensor, output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
