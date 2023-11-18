# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skip(reason="Unable to slice large tensors")
def test_large_slicing(device):
    torch_a = torch.rand((1, 1, 42, 250880), dtype=torch.bfloat16)
    torch_output = torch_a[:, :, -1, :]
    a = ttnn.from_torch(torch_a)
    shape = a.shape
    a = ttnn.to_device(a, device)
    tt_output = a[:, :, -1, :]
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)
