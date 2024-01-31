# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),  # Upsupported dimension
    ),
)
@pytest.mark.parametrize(
    "dim",
    (3, 2, 1, -1, -2, -3),
)
class TestArgmax:
    def test_argmax(self, input_shapes, dim, device):
        torch.manual_seed(0)
        # if input_shapes[0] * input_shapes[1] != 1:
        #     pytest.skip(f"Dim 0, and 1 in {input_shapes} not supported for argmax.")
        input_data = torch.randn(input_shapes).bfloat16()
        input_tensor = (
            tt_lib.tensor.Tensor(input_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )
        tt_output_tensor_on_device = tt_lib.tensor.argmax(input_tensor, dim=dim)

        tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
        if dim == 0:
            golden_tensor = torch.argmax(input_data)
            tt_out_tensor = tt_out_tensor[0, 0, 0, 0]

        else:
            golden_tensor = torch.argmax(input_data, dim=dim)
            if dim == 1 or dim == -3:
                tt_out_tensor = tt_out_tensor[0]
            else:
                if input_shapes[1] != 1:
                    tt_out_tensor = tt_out_tensor[0]
                else:
                    tt_out_tensor = tt_out_tensor[0, 0, 0]
        pt_out_tensor = golden_tensor
        print(tt_out_tensor.shape)
        print(pt_out_tensor.shape)
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=0.99)
        comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=4, rtol=1e-1)
        logger.info(comp_pass)
        logger.info(comp_all)
        logger.info(comp_out)
        status = comp_pass | comp_all
        assert status
