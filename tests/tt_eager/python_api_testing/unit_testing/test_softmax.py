# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import tt_lib as ttl
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize("inplace", [True, False])
def test_softmax(device, inplace):
    torch.manual_seed(0)
    sm_op = ttl.operations.primary.softmax_in_place if inplace else ttl.tensor.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = (
            ttl.tensor.Tensor(input_tensor, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
        )
        tt_output_tensor_on_device = sm_op(tt_input_tensor)
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"


@skip_for_wormhole_b0()
@pytest.mark.parametrize("inplace", [True, False])
def test_softmax_with_program_cache(device, use_program_cache, inplace):
    torch.manual_seed(0)
    sm_op = ttl.operations.primary.softmax_in_place if inplace else ttl.tensor.softmax

    input_shapes = [(3, 64, 128, 96), (1, 64, 32, 32)]

    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).bfloat16()

        tt_input_tensor = (
            ttl.tensor.Tensor(input_tensor, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
        )
        tt_output_tensor_on_device = sm_op(tt_input_tensor)
        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        golden_output_tensor = torch.softmax(input_tensor, dim=-1)
        print_diff_argmax(tt_output_tensor, golden_output_tensor)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"
