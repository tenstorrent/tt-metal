# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from functools import partial
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128])),
        (torch.Size([5, 32, 32]), torch.Size([5, 32, 32])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.int32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.bfloat4_b]),
)
def test_binary_scalar_ops(input_shapes, dtype, device):
    torch.manual_seed(0)
    a_shape, b_shape = input_shapes

    a_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(b_shape)

    a_tt = ttnn.from_torch(
        a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_tt = ttnn.from_torch(
        b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    out_tt = ttnn.add(a_tt, b_tt, queue_id=cq_id, use_legacy=False)
    out_pt = a_pt + b_pt

    comp_pass = compare_pcc([out_tt], [out_pt])
    assert comp_pass
