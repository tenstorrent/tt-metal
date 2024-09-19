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
        (torch.Size([1, 1, 30, 4])),
        (torch.Size([1, 1, 3, 6])),
        (torch.Size([1, 1, 3, 20])),
        (torch.Size([1, 1, 3, 2])),
        (torch.Size([1, 1, 3, 8])),
        (torch.Size([1, 1, 1, 24])),
        (torch.Size([1, 1, 4, 8])),
        (torch.Size([1, 1, 2, 8])),
        (torch.Size([1, 1, 2, 4])),
    ),
)
@pytest.mark.parametrize("dim", (-1,))
@pytest.mark.parametrize("memconfig", (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG))
class TestArgmax:
    def test_argmax(self, input_shapes, dim, memconfig, device):
        torch.manual_seed(10)
        input_data = torch.randn(input_shapes).bfloat16()

        # DEBUG
        # input_data = torch.randn(input_shapes).bfloat16()
        # lin = torch.arange(24)
        # input_data = torch.reshape(lin, input_shapes).bfloat16()

        input_tensor = ttnn.Tensor(input_data, ttnn.bfloat16).to(device, memconfig)

        tt_output_tensor_on_device = ttnn.argmax(input_tensor, dim=dim)
        tt_out_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
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
        tt_out_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=0.99)
        comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=0, rtol=0)

        # DEBUG
        # print(pt_out_tensor)
        # print(tt_out_tensor)
        # flat = torch.flatten(input_data)
        # print(flat)
        # print(torch.topk(flat, 8))

        logger.info(comp_pass)
        logger.info(comp_all)
        logger.info(comp_out)
        status = comp_pass | comp_all
        assert status
