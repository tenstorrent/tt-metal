# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    (
        (ttnn.gt),
        (ttnn.lt),
        (ttnn.ne),
        (ttnn.ge),
        (ttnn.le),
        (ttnn.eq),
        (ttnn.bias_gelu),
    ),
)
def test_binary_scalar_ops(input_shapes, device, ttnn_fn):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    cq_id = 0
    scalar = random.randint(-80, 80)
    ttnn_fn(input_tensor, scalar, output_tensor=output_tensor, queue_id=cq_id)
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_fn(in_data, scalar)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
