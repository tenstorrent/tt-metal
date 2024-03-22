# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_pytorch_files():
    ttnn_output_torch = torch.load("gn2_output.pt")
    torch_output = torch.load("fallback_gn2_output.pt")

    # ttnn_output_torch = torch.load('block_input.pt')
    # torch_output = torch.load('block_input_fallback.pt')

    # ttnn_output_torch = torch.load('gn2_input.pt')
    # torch_output = torch.load('gn2_input_fallback.pt')

    # ttnn_output_torch = torch.load('gn1_output.pt')
    # torch_output = torch.load('fallback_gn1_output.pt')

    # ttnn_output_torch = torch.load('gn1_input.pt')
    # torch_output = torch.load('fallback_gn1_input.pt')

    ttnn_output_torch = torch.load("gn1_output.pt")
    torch_output = torch.load("gn1_ut_output.pt")
    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)
