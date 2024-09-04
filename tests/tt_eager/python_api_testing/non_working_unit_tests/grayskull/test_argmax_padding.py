# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 10])),
        (torch.Size([1, 1, 10, 20])),
        (torch.Size([1, 1, 30, 3])),
        (torch.Size([1, 4, 3, 5])),
        (torch.Size([5, 4, 3, 20])),
    ),
)
@pytest.mark.parametrize("dim", (3, 2, 1, 0, -1, -2, -3, -4))
@pytest.mark.parametrize("all", (True, False))
class TestArgmax:
    def test_argmax(self, input_shapes, dim, all, device):
        torch.manual_seed(10)
        input_data = torch.randn(input_shapes).bfloat16()
        input_tensor = ttnn.Tensor(input_data, ttnn.bfloat16).pad_to_tile(100).to(ttnn.TILE_LAYOUT).to(device)
        tt_output_tensor_on_device = ttnn.experimental.argmax(input_tensor, dim=dim, all=all)
        tt_out_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if all:
            golden_tensor = torch.argmax(input_data)
            tt_out_tensor = tt_out_tensor[0, 0, 0, 0]
        else:
            golden_tensor = torch.argmax(input_data, dim=dim)
            if dim == 1 or dim == -3 or dim == 0 or dim == -4:
                tt_out_tensor = tt_out_tensor[0, :, 0 : input_shapes[2], 0 : input_shapes[3]]
            else:
                if input_shapes[1] != 1 or input_shapes[0] != 1:
                    if dim == 2 or dim == -2:
                        tt_out_tensor = tt_out_tensor[0, :, :, 0 : input_shapes[3]]
                    else:
                        tt_out_tensor = tt_out_tensor[0, :, :, 0 : input_shapes[2]]
                else:
                    if dim == 2 or dim == -2:
                        tt_out_tensor = tt_out_tensor[0, 0, 0, 0 : input_shapes[3]]
                    else:
                        tt_out_tensor = tt_out_tensor[0, 0, 0, 0 : input_shapes[2]]
        pt_out_tensor = golden_tensor
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=0.99)
        comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=4, rtol=1e-1)
        logger.info(comp_pass)
        logger.info(comp_all)
        logger.info(comp_out)
        status = comp_pass | comp_all
        assert status
