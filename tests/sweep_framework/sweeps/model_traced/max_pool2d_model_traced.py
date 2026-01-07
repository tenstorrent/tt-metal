# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("max_pool2d", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Note: max_pool2d requires L1 memory config with BLOCK_SHARDED scheme
    # Sample test skipped - use model_traced suite only
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
    batch_size=None,
    input_h=None,
    input_w=None,
    channels=None,
    kernel_size=None,
    stride=None,
    padding=None,
    dilation=None,
    applied_shard_scheme=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    """
    Run max_pool2d test with parameters extracted from traced JSON.
    All parameters are now extracted from JSON including applied_shard_scheme.
    """
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # All parameters must be extracted from JSON - no fallbacks
    if batch_size is None or input_h is None or input_w is None or channels is None:
        raise ValueError(
            f"Missing required parameters from JSON: batch_size={batch_size}, input_h={input_h}, input_w={input_w}, channels={channels}"
        )

    N = batch_size
    H = input_h
    W = input_w
    C = channels

    # All parameters must be extracted from JSON - no fallbacks
    if kernel_size is None or not isinstance(kernel_size, list) or len(kernel_size) != 2:
        raise ValueError(f"Missing or invalid kernel_size from JSON: {kernel_size}")
    kH, kW = kernel_size[0], kernel_size[1]

    if stride is None or not isinstance(stride, list) or len(stride) != 2:
        raise ValueError(f"Missing or invalid stride from JSON: {stride}")
    stride_h, stride_w = stride[0], stride[1]

    if padding is None or not isinstance(padding, list) or len(padding) != 2:
        raise ValueError(f"Missing or invalid padding from JSON: {padding}")
    pad_h, pad_w = padding[0], padding[1]

    if dilation is None or not isinstance(dilation, list) or len(dilation) != 2:
        raise ValueError(f"Missing or invalid dilation from JSON: {dilation}")
    dil_h, dil_w = dilation[0], dilation[1]

    # Create input in [N, C, H, W] format for PyTorch
    torch_input_shape = [N, C, H, W]

    # Adjust height if needed for ROW_MAJOR layout
    if input_a_layout == ttnn.ROW_MAJOR_LAYOUT and H % 2 == 1:
        H += 1
        torch_input_shape[2] = H

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(torch_input_shape)

    torch_output_tensor = torch.nn.functional.max_pool2d(
        torch_input_tensor_a, (kH, kW), stride=(stride_h, stride_w), padding=pad_h, dilation=dil_h
    )

    # Convert to ttnn format: [NHW, C] -> [1, 1, N*H*W, C]
    torch_input_tensor_a = torch.permute(torch_input_tensor_a, (0, 2, 3, 1))
    torch_input_tensor_a = torch.reshape(torch_input_tensor_a, [1, 1, N * H * W, C])

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

    start_time = start_measuring_time()
    # applied_shard_scheme must be extracted from JSON - no fallbacks
    if applied_shard_scheme is None:
        raise ValueError(f"Missing applied_shard_scheme from JSON")

    if applied_shard_scheme == "BLOCK_SHARDED":
        applied_shard_scheme_ttnn = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    elif applied_shard_scheme == "HEIGHT_SHARDED":
        applied_shard_scheme_ttnn = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    elif applied_shard_scheme == "WIDTH_SHARDED":
        applied_shard_scheme_ttnn = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    elif applied_shard_scheme == "INTERLEAVED":
        applied_shard_scheme_ttnn = ttnn.TensorMemoryLayout.INTERLEAVED
    else:
        raise ValueError(f"Invalid applied_shard_scheme from JSON: {applied_shard_scheme}")

    result = ttnn.max_pool2d(
        input_tensor=input_tensor_a,
        batch_size=N,
        input_h=H,
        input_w=W,
        channels=C,
        kernel_size=[kH, kW],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dil_h, dil_w],
        memory_config=output_memory_config,
        applied_shard_scheme=applied_shard_scheme_ttnn,
    )

    result = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # Convert back to [N, C, H, W] format
    output_tensor = torch.permute(result, (0, 3, 1, 2))

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
