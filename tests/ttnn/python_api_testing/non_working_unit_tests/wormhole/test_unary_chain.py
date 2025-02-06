# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from functools import partial
import pytest
import torch
import ttnn
import traceback
from tests.sweep_framework.sweep_utils.utils import unary_ops_dict, gen_unary_op_args, tensor_to_dtype
from tests.ttnn.utils_for_testing import comp_equal, check_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random


def run_unary_chain_tests(
    input_shape,
    unary_op_list,
    low,
    high,
    dtype,
    dlayout,
    in_mem_cfg,
    out_mem_cfg,
    device,
):
    # torch.manual_seed(0)

    input_tensor = gen_func_with_cast_tt(partial(torch_random, low=low, high=high, dtype=torch.float32), dtype[0])(
        input_shape
    )

    try:
        # get ref result
        ref_value = input_tensor.detach().clone()
        unary_op_type_list = []
        for i in range(len(unary_op_list)):
            unary_op = unary_op_list[i]
            param_gen_func = unary_ops_dict[unary_op][-1]
            if param_gen_func is not None:
                param = param_gen_func()
                if isinstance(param, torch.Tensor):
                    param = param.item()
            else:
                param = None
            golden_function, unary_op_type = gen_unary_op_args(unary_op, ref_value, param)
            unary_op_type_list.append(unary_op_type)
            ref_value = tensor_to_dtype(golden_function(), dtype[0])

        tt_input_tensor = ttnn.from_torch(
            input_tensor,
            dtype=dtype[0],
            layout=dlayout,
            device=device,
            memory_config=in_mem_cfg,
        )

        tt_result = ttnn.unary_chain(tt_input_tensor, unary_op_type_list, memory_config=out_mem_cfg)
        tt_result = ttnn.to_torch(tt_result)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    passed, output_str = check_with_pcc(ref_value, tt_result, 0.999)
    assert passed, f"{output_str}, unary_ops: {unary_op_list}, params: {param}"


test_sweep_args = [
    (
        [2, 4, 139, 169],
        ["leaky_relu"],
        [ttnn.bfloat16],
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
test_sweep_args = []
for op in unary_ops_dict.keys():
    if op in ["atan", "asin", "acos"]:
        low, high = -1, 1
    else:
        low, high = -100, 100
    test_sweep_args.append(
        (
            [2, 4, 139, 169],
            [op],
            low,
            high,
            [ttnn.bfloat16],
            ttnn.TILE_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        )
    )


@pytest.mark.parametrize(
    "input_shape, unary_op_list, low, high, dtype, dlayout, in_mem_cfg, out_mem_cfg",
    (test_sweep_args),
)
def test_unary_chain(input_shape, unary_op_list, low, high, dtype, dlayout, in_mem_cfg, out_mem_cfg, device):
    run_unary_chain_tests(input_shape, unary_op_list, low, high, dtype, dlayout, in_mem_cfg, out_mem_cfg, device)
