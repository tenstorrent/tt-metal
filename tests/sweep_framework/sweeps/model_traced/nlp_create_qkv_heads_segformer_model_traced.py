# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_create_qkv_heads_segformer", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with Segformer-specific shape
    "model_traced_sample": {
        "input_shape": [(1, 1, 256, 256)],  # Segformer uses 256 hidden dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture for nlp_create_qkv_heads_segformer operation.
    Using explicit DispatchCoreConfig to handle sharded memory configs.
    """
    import ttnn

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

    # Handle tuple input_shape
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    batch, _, seq_len, hidden_dim = shape
    head_dim = 32
    heads_num = hidden_dim // head_dim

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # PyTorch reference: Segformer reshapes and transposes
    ref_q = torch_input_tensor_a
    ref_q = torch.reshape(ref_q, [batch, seq_len, heads_num, head_dim]).transpose(-3, -2)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    # nlp_create_qkv_heads_segformer doesn't support sharded output
    # Force to DRAM interleaved if output is sharded
    actual_output_mem_config = output_memory_config
    if isinstance(output_memory_config, dict):
        mem_layout = output_memory_config.get("memory_layout", "")
        if not mem_layout and "data" in output_memory_config:
            mem_layout = output_memory_config.get("data", {}).get("memory_layout", "")
        if "SHARDED" in str(mem_layout):
            actual_output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    elif hasattr(output_memory_config, "is_sharded") and callable(output_memory_config.is_sharded):
        if output_memory_config.is_sharded():
            actual_output_mem_config = ttnn.DRAM_MEMORY_CONFIG

    start_time = start_measuring_time()
    q = ttnn.experimental.nlp_create_qkv_heads_segformer(input_tensor_a, memory_config=actual_output_mem_config)[0]
    q_torch = ttnn.to_torch(q)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(ref_q, q_torch, 0.999)

    return [pcc, e2e_perf]
