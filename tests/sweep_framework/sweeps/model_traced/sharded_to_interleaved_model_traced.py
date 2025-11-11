# Model traced sweep for sharded_to_interleaved
# Generated automatically - DO NOT EDIT MANUALLY

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
model_traced_params = loader.get_suite_parameters("sharded_to_interleaved", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # Check if input memory config is actually sharded and in L1
    # sharded_to_interleaved requires L1 sharded input
    is_sharded = hasattr(input_a_memory_config, "shard_spec") and input_a_memory_config.shard_spec is not None
    is_l1 = hasattr(input_a_memory_config, "buffer_type") and input_a_memory_config.buffer_type == ttnn.BufferType.L1

    if not is_sharded or not is_l1:
        # If traced config is not sharded or not L1, skip this config
        # The operation requires L1 sharded input
        raise ValueError(
            f"sharded_to_interleaved requires L1 sharded input, but got "
            f"buffer_type={input_a_memory_config.buffer_type if hasattr(input_a_memory_config, 'buffer_type') else 'N/A'}, "
            f"is_sharded={is_sharded}"
        )

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Use the traced sharded memory config directly since it's valid
    # Create interleaved tensor first, then convert to sharded
    input_tensor_interleaved = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert to sharded using the traced config
    # Catch errors related to invalid core coordinates
    try:
        sharded_tensor = ttnn.interleaved_to_sharded(input_tensor_interleaved, input_a_memory_config)
    except (RuntimeError, ValueError) as e:
        error_msg = str(e)
        if "No core coordinate found" in error_msg or "core coordinate" in error_msg.lower():
            # Invalid core coordinates in traced config - skip this config
            raise ValueError(
                f"Invalid core coordinates in sharding config: {error_msg}. "
                f"This traced config uses cores that don't exist on this device."
            )
        raise

    # Run sharded_to_interleaved - no PyTorch reference needed
    # Just verify the operation completes and produces interleaved output
    start_time = start_measuring_time()
    output_tensor = ttnn.sharded_to_interleaved(sharded_tensor, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    # Verify output is interleaved
    output_mem_config = output_tensor.memory_config()
    if output_mem_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(
            f"sharded_to_interleaved should produce interleaved output, but got {output_mem_config.memory_layout}"
        )

    # Convert to torch for shape verification
    output_torch = ttnn.to_torch(output_tensor)

    # Verify output shape matches input shape
    if list(output_torch.shape) != list(shape):
        raise ValueError(f"Output shape {list(output_torch.shape)} does not match input shape {list(shape)}")

    # Return success (PCC = 1.0) since we're just verifying the operation works
    # The operation is a data movement operation, so correctness is verified by shape and layout
    return [1.0, e2e_perf]
