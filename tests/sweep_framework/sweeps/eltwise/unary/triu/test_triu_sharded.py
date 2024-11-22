import torch
import random
import ttnn
import itertools
import pytest
import traceback
from loguru import logger
from functools import partial

from tests.sweep_framework.sweep_utils.utils import gen_shapes, get_device_grid_size, get_sharded_config
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import (
    gen_func_with_cast_tt,
    _gen_reshape_args_from_volume,
)
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


Y, X = get_device_grid_size()
DEVICE_GRID_SIZE = ttnn.CoreGrid(y=Y, x=X)


def gen_test_sweep_args(gen_unsafe, num_shapes, shard_orientation, sharding_strategy=None):
    if sharding_strategy:
        assert sharding_strategy in ["block", "height", "width"]

    assert shard_orientation in ["col_major", "row_major"]

    input_shape_list = []

    for i in range(num_shapes):
        for rank in [3, 4]:
            if sharding_strategy == "block":
                min_shard_size_y = 32 * Y
                min_shard_size_x = 32 * X

                mul_x = random.randint(1, 10)
                mul_y = random.randint(1, 64 // mul_x)

                shape = random.choice(
                    _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                )
                shape = list(shape["reshape_dims"])
                if gen_unsafe:
                    while shape[-1] % 32 == 0:
                        shape = random.choice(
                            _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                        )
                        shape = list(shape["reshape_dims"])
                else:
                    while shape[-1] % 32 != 0:
                        shape = random.choice(
                            _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                        )
                        shape = list(shape["reshape_dims"])
                shape.append(mul_x * min_shard_size_x)

                input_shape_list.append(shape)

            else:
                if sharding_strategy == "height":
                    min_shard_size_y = 32 * X * Y
                    min_shard_size_x = 32
                    mul_x = random.randint(1, 10)
                    mul_y = random.randint(1, 2)
                else:
                    min_shard_size_y = 32
                    min_shard_size_x = 32 * X * Y
                    mul_x = random.randint(1, 2)
                    mul_y = random.randint(1, 10)

                shape = random.choice(
                    _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                )
                shape = list(shape["reshape_dims"])
                if gen_unsafe:
                    while shape[-1] % 32 == 0:
                        shape = random.choice(
                            _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                        )
                        shape = list(shape["reshape_dims"])
                else:
                    while shape[-1] % 32 != 0:
                        shape = random.choice(
                            _gen_reshape_args_from_volume(mul_y * min_shard_size_y, step=1, out_dims=rank - 1)
                        )
                        shape = list(shape["reshape_dims"])

                shape.append(mul_x * min_shard_size_x)

                input_shape_list.append(shape)

    for input_shape, dtype in itertools.product(input_shape_list, [ttnn.bfloat16, ttnn.bfloat8_b]):
        data_seed = random.randint(0, 20000000)
        mem_cfg = get_sharded_config(input_shape, sharding_strategy, DEVICE_GRID_SIZE, shard_orientation)
        yield (input_shape, dtype, ttnn.TILE_LAYOUT, mem_cfg, data_seed)


def run_tril_sharded_tests(
    input_shape,
    dtype,
    dlayout,
    mem_cfg,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), dtype)(input_shape)

    low = -(input_shape[-2] - 2)
    high = input_shape[-1]
    diagonal = torch.randint(low, high, (1,)).item()

    try:
        ref_value = torch.tril(x, diagonal)

        tt_x = ttnn.from_torch(
            x,
            dtype=dtype,
            layout=dlayout,
            device=device,
            memory_config=mem_cfg,
        )
        tt_result = ttnn.tril(tt_x, diagonal=diagonal, memory_config=mem_cfg)
        tt_result = ttnn.to_torch(tt_result)
    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    passed, output_str = check_with_pcc(ref_value, tt_result, 0.999)
    assert passed, f"{output_str}, {input_shape}, {dtype}, {mem_cfg}"


test_sweep_args = (
    list(gen_test_sweep_args(False, 2, "row_major", "block"))
    + list(gen_test_sweep_args(False, 2, "col_major", "block"))
    + list(gen_test_sweep_args(False, 2, "row_major", "height"))
    + list(gen_test_sweep_args(False, 2, "col_major", "height"))
    + list(gen_test_sweep_args(False, 2, "row_major", "width"))
    + list(gen_test_sweep_args(False, 2, "col_major", "width"))
)


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, mem_cfg, data_seed",
    (test_sweep_args),
)
def test_tril_sharded(input_shape, dtype, dlayout, mem_cfg, data_seed, device):
    run_tril_sharded_tests(input_shape, dtype, dlayout, mem_cfg, data_seed, device)
