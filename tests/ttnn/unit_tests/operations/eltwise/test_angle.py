from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from functools import partial

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random


def run_angle_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_cfg,
    out_mem_cfg,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)

    x_real = gen_func_with_cast_tt(partial(torch_random, low=-10, high=10, dtype=torch.float32), dtype)(input_shape).to(
        torch.float32
    )
    x_imag = gen_func_with_cast_tt(partial(torch_random, low=-10, high=10, dtype=torch.float32), dtype)(input_shape).to(
        torch.float32
    )
    x = torch.complex(x_real, x_imag)

    try:
        golden_function = ttnn.get_golden_function(ttnn.angle)
        ref_value = golden_function(x)

        tt_x_real = ttnn.from_torch(
            x_real,
            dtype=dtype,
            layout=dlayout,
            device=device,
            memory_config=in_mem_cfg,
        )

        tt_x_imag = ttnn.from_torch(
            x_imag,
            dtype=dtype,
            layout=dlayout,
            device=device,
            memory_config=in_mem_cfg,
        )

        tt_x = ttnn.complex_tensor(tt_x_real, tt_x_imag)

        tt_result = ttnn.angle(tt_x, memory_config=out_mem_cfg)
        tt_result = ttnn.to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    torch.set_printoptions(sci_mode=False)
    print(tt_x_real)
    print(tt_x_imag)
    print(ref_value)
    print(tt_result)
    assert_with_pcc(ref_value, tt_result, 0.999)


test_sweep_args = [
    (
        [1, 1, 32, 32],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        0,
    ),
    # (
    #     [4, 7, 21, 133],
    #     ttnn.bfloat16,
    #     ttnn.TILE_LAYOUT,
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     1,
    # ),
    # (
    #     [4, 7, 21, 133],
    #     ttnn.bfloat8_b,
    #     ttnn.TILE_LAYOUT,
    #     ttnn.L1_MEMORY_CONFIG,
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     21312,
    # ),
    # (
    #     [4, 7, 21, 133],
    #     ttnn.bfloat16,
    #     ttnn.TILE_LAYOUT,
    #     ttnn.L1_MEMORY_CONFIG,
    #     ttnn.DRAM_MEMORY_CONFIG,
    #     124214,
    # ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_angle(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_angle_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)
