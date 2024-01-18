# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs


@pytest.mark.parametrize("end", (256, 300, 390, 1024))
def test_arange(end, device):
    torch.manual_seed(0)
    torch_data = torch.arange(250, end, 1)

    tt_output_tensor_on_device = tt_lib.tensor.arange(250, end, 1)

    tt_output = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    tt_output = tt_output.flatten()
    print("==============================")
    torch.set_printoptions(sci_mode=False, threshold=10000)

    print("********** torch  ****")
    print(torch_data)

    print("********** TT  ****")
    print(tt_output)

    print(torch.eq(torch_data, tt_output))
