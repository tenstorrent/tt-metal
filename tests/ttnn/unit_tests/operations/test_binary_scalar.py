# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "fn_relational",
    (
        [ttnn.gt, torch.gt],
        [ttnn.lt, torch.lt],
        [ttnn.ne, torch.ne],
        [ttnn.ge, torch.ge],
        [ttnn.le, torch.le],
        [ttnn.eq, torch.eq],
    ),
)
def test_unary_relops_ttnn(input_shapes, device, fn_relational):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    ttnn_fn = fn_relational[0]
    torch_fn = fn_relational[1]
    cq_id = 0
    ttnn_fn(input_tensor, 4.5, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch_fn(in_data, 4.5)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
