# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30


def tilize_with_val_padding(x, output_tensor_shape, pad_value):
    """Golden function for tilize_with_val_padding.

    At the logical (PyTorch) level, tilize/tilize_with_val_padding preserve the
    original tensor values and shape; the padding and tilized layout are
    internal to the device representation and are not visible once we convert
    back with ttnn.to_torch.  This matches the behavior exercised in
    tests/ttnn/unit_tests/base_functionality/test_tilize_untilize_2D.py, where
    tilize_with_val_padding round‑trips the input.

    The sweep therefore uses the input tensor itself as the golden: we are
    validating that the TTNN op preserves values for the traced shapes and
    memory configs, not that we can re‑implement its internal tilizing logic.
    """
    # Logical golden is the original tensor; padded_shape and pad_value only
    # affect internal device layout.
    _ = output_tensor_shape
    _ = pad_value
    return x


# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("tilize_with_val_padding", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],  # tilize_with_val_padding requires ROW_MAJOR_LAYOUT
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "padded_shape": [(1, 1, 64, 64)],  # shape to pad to (must be multiples of 32 for tilize)
        "pad_value": [0.0],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    padded_shape,
    pad_value,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    torch_output_tensor = tilize_with_val_padding(torch_input_tensor_a, padded_shape, pad_value)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Check if we need to override layout for bfloat8_b/bfloat4_b
    # These dtypes REQUIRE TILE layout, so we use TILE even though
    # tilize_with_val_padding normally takes ROW_MAJOR input
    dtype_str = str(input_a_dtype).upper() if input_a_dtype else ""
    needs_tile_layout = "BFLOAT8_B" in dtype_str or "BFLOAT4_B" in dtype_str

    # Use TILE layout for bfloat8_b/bfloat4_b, otherwise use specified layout
    actual_layout = ttnn.TILE_LAYOUT if needs_tile_layout else input_a_layout

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": actual_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    start_time = start_measuring_time()
    output_tensor = ttnn.tilize_with_val_padding(
        input_tensor_a, padded_shape, pad_value, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
