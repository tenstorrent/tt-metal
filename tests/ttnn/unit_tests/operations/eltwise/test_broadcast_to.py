# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc
from functools import reduce
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

pytestmark = pytest.mark.use_module_device

input_bcast_shape_pairs = [
    ((1), (2, 10)),
    ((1, 1, 1, 1), (1, 1, 1, 67000)),
    ((1, 1, 1, 1), (1, 1, 67000, 4)),
    ((1, 1, 1, 1), (1, 1, 270, 270)),
    ((1, 1, 270, 270), (1, 1, 270, 270)),
    ((1, 1, 64, 64), (1, 310, 64, 64)),
    ((1, 1, 64, 64), (310, 1, 64, 64)),
    ((1, 1, 64, 64), (10, 31, 64, 64)),
    ((31, 33, 64, 64), (31, 33, 64, 64)),
    ((1, 1, 1, 1), (7, 17, 32, 64)),  # scalar to 4D tensor
    ((1, 3, 1, 1), (8, 3, 32, 64)),  # broadcast N, H, W (preserve C)
    ((2, 1, 4, 1), (2, 17, 4, 64)),  # broadcast C and W
    ((1, 1, 32, 32), (7, 17, 32, 32)),  # broadcast N and C (preserve H, W)
    ((1, 3, 1, 4), (7, 3, 32, 4)),  # broadcast N, H, W, C
]


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ),
)
@pytest.mark.parametrize(
    "memory_config_input",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
@pytest.mark.parametrize("shape_and_broadcast_spec", input_bcast_shape_pairs)
def test_broadcast_to(device, dtype_pt, dtype_tt, shape_and_broadcast_spec, memory_config_input):
    shape, broadcast_shape = shape_and_broadcast_spec

    # For float32, use values that expose TF32 precision loss (require >10 mantissa bits)
    # TF32 has only 10 mantissa bits vs FP32's 23 bits
    # if dtype_pt == torch.float32:
    # Use values with many significant digits that will lose precision in TF32
    # Example: 1.23456789 requires more than 10 mantissa bits for exact representation
    # torch_input_tensor = torch.randn(shape, dtype=dtype_pt) * 1000000.0 + 1.23456789
    # else:
    torch_input_tensor = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(shape)

    torch_result = torch_input_tensor.broadcast_to(broadcast_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_input
    )
    output = ttnn.experimental.broadcast_to(
        input_tensor, ttnn.Shape(broadcast_shape), memory_config=memory_config_input
    )
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)


input_bcast_shape_pairs = [
    ((1), (2, 10)),
    ((1, 1, 1, 1), (1, 1, 1, 67000)),
    ((1, 1, 1, 1), (1, 1, 67000, 4)),
    ((1, 1, 1, 1), (1, 1, 270, 270)),
    ((1, 1, 270, 270), (1, 1, 270, 270)),
    ((1, 1, 64, 64), (1, 310, 64, 64)),
    ((1, 1, 64, 64), (310, 1, 64, 64)),
    ((1, 1, 64, 64), (1, 310, 64, 64)),
    ((31, 33, 64, 64), (31, 33, 64, 64)),
]


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    (
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ),
)
@pytest.mark.parametrize(
    "memory_config_input",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
@pytest.mark.parametrize("shape_and_broadcast_spec", input_bcast_shape_pairs)
def test_broadcast_to_out(device, dtype_pt, dtype_tt, shape_and_broadcast_spec, memory_config_input):
    shape, broadcast_shape = shape_and_broadcast_spec
    torch_input_tensor = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(shape)
    torch_output_tensor = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(
        broadcast_shape
    )
    torch_result = torch_input_tensor.broadcast_to(broadcast_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_input
    )
    output_tensor = ttnn.from_torch(
        torch_output_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_input
    )
    ttnn.experimental.broadcast_to(
        input_tensor, ttnn.Shape(broadcast_shape), memory_config=memory_config_input, output=output_tensor
    )
    output = ttnn.to_torch(output_tensor)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)


profile_input_bcast_shape_pairs = [
    ((1, 1, 1, 1), (1, 65, 32, 64)),
    ((1, 1, 64, 64), (2, 16, 64, 64)),
    ((2, 16, 1, 1), (2, 16, 64, 64)),
    ((2, 1, 64, 1), (2, 16, 64, 64)),  # col
    ((2, 1, 1, 64), (2, 16, 64, 64)),  # row
    # following tests takes a bit longer to run
    # enable on demand
    # ((1, 1, 256, 256), (4, 128, 256, 256)),
    # ((4, 128, 1, 1), (4, 128, 256, 256)),
    # ((4, 1, 256, 1), (4, 128, 256, 256)), # col
    # ((4, 1, 1, 256), (4, 128, 256, 256)), # row
]


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ((torch.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "memory_config_input",
    [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
)
@pytest.mark.parametrize("shape_and_broadcast_spec", profile_input_bcast_shape_pairs)
def test_broadcast_to_profile(device, dtype_pt, dtype_tt, shape_and_broadcast_spec, memory_config_input):
    torch.manual_seed(0)
    shape, broadcast_shape = shape_and_broadcast_spec
    if dtype_pt == torch.bfloat16 and shape[-1] < 2 and broadcast_shape[-1] < 2:
        pytest.skip("bfloat16 needs 4 byte inner dim on the output.")

    torch_input_tensor = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(shape)

    torch_result = torch_input_tensor.broadcast_to(broadcast_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config_input
    )
    for _ in range(2):
        output = ttnn.experimental.broadcast_to(
            input_tensor, ttnn.Shape(broadcast_shape), memory_config=memory_config_input
        )
        output = ttnn.to_torch(output)

        assert (
            output.shape == torch_result.shape
        ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

        assert_with_pcc(torch_result, output, 0.9999)
        ttnn.synchronize_device(device)


input_bcast_shape_pairs = [
    ((1, 1, 1, 1), (1, 1, 1)),
    ((1, 1, 1, 1), (1, 1, 1, 1, 1)),
    ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
    ((2, 1), (3, 1)),
]


@pytest.mark.parametrize("dtype_pt, dtype_tt", ((torch.bfloat16, ttnn.bfloat16),))
@pytest.mark.parametrize("input, bcast", input_bcast_shape_pairs)
def test_broadcast_to_invalid(input, bcast, dtype_pt, dtype_tt, device):
    torch.manual_seed(0)

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=dtype_pt), dtype_tt)(input)
    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    with pytest.raises(RuntimeError):
        _ = ttnn.experimental.broadcast_to(a_tt, bcast)


input_bcast_shape_pairs = [
    ((1, 1), (7, 10, 32, 32)),
]


@pytest.mark.parametrize("input, bcast", input_bcast_shape_pairs)
def test_invalid_broadcast_to_sharding(input, bcast, device):
    torch.manual_seed(0)
    dtype_pt, dtype_tt = [torch.bfloat16, ttnn.bfloat16]
    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.float32), dtype_tt)(input)
    sharded_config = ttnn.create_sharded_memory_config(
        [10 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a_tt = ttnn.from_torch(
        a_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    with pytest.raises(RuntimeError):
        _ = ttnn.experimental.broadcast_to(a_tt, bcast, memory_config=sharded_config)


input_bcast_shape_pairs = [
    ((1), (2, 10)),
    ((1, 1, 1, 1), (1, 13, 1, 670)),
    ((1, 1, 1, 1), (1, 13, 670, 1)),
    ((1, 1, 1, 1), (1, 13, 270, 270)),
    ((1, 13, 270, 270), (1, 13, 270, 270)),
]


@pytest.mark.parametrize("shape_and_broadcast_spec", input_bcast_shape_pairs)
def test_broadcast_to_bf8_b(device, shape_and_broadcast_spec):
    pytest.skip("Skip for now, as it is not stable. Need to investigate.")
    shape, broadcast_shape = shape_and_broadcast_spec
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(shape)
    torch_result = torch_input_tensor.broadcast_to(broadcast_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    out_tt = ttnn.experimental.broadcast_to(input_tensor, ttnn.Shape(broadcast_shape))
    output = ttnn.to_torch(out_tt)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)
