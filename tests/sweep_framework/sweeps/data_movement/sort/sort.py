# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 60

TORCH_SEED = 0

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys,
# and lists of possible inputs as values.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 32, 64], [4, 4, 128, 128], [1, 1, 32, 64], 8)
        + gen_shapes([1, 32, 64], [4, 128, 128], [1, 32, 64], 8)
        + gen_shapes([32, 64], [128, 128], [32, 64], 8),
        # dim intentionally over-generates for lower-rank shapes (e.g. dim=2 for 2D inputs);
        # invalidate_vector prunes those combinations before they run.
        "dim": [-1, 0, 1, 2],
        "descending": [False, True],
        "input_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.uint16],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    shape = test_vector["input_shape"]
    rank = len(shape)
    dim = test_vector["dim"]

    # dim must be in [-rank, rank)
    if dim >= rank or dim < -rank:
        return True, f"dim={dim} is out of bounds for rank={rank}"

    return False, None


def run(
    input_shape,
    dim,
    descending,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(TORCH_SEED)

    # Generate input: use integer range for uint16, float for others.
    if input_dtype == ttnn.uint16:
        torch_input = torch.randint(0, 1000, input_shape, dtype=torch.int32)
    else:
        torch_input = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
        )(input_shape)

    torch_values, _ = torch.sort(torch_input, dim=dim, descending=descending)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    ttnn_values, ttnn_indices = ttnn.sort(
        ttnn_input, dim=dim, descending=descending, memory_config=output_memory_config
    )
    e2e_perf = stop_measuring_time(start_time)

    # Gather values via indices and compare with torch reference.
    torch_output = ttnn.to_torch(ttnn_values)
    if input_dtype == ttnn.uint16:
        torch_output = torch_output.to(torch.int32)

    torch_indices = ttnn.to_torch(ttnn_indices).to(torch.int64)
    torch_gathered = torch.gather(torch_input, dim, torch_indices)

    # Both sorted values and gathered values must match torch.sort reference.
    values_match = torch.equal(torch_output, torch_values)
    gathered_match = torch.equal(torch_gathered, torch_values)

    passing = values_match and gathered_match
    output_str = "passed" if passing else f"values_match={values_match}, gathered_match={gathered_match}"

    return [(passing, output_str), e2e_perf]
