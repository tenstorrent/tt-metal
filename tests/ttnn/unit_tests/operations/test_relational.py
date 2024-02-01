# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_relational_test(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor, 0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gtz(device, h, w):
    run_relational_test(device, h, w, ttnn.gtz, torch.gt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ltz(device, h, w):
    run_relational_test(device, h, w, ttnn.ltz, torch.lt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gez(device, h, w):
    run_relational_test(device, h, w, ttnn.gez, torch.ge)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lez(device, h, w):
    run_relational_test(device, h, w, ttnn.lez, torch.le)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nez(device, h, w):
    run_relational_test(device, h, w, ttnn.nez, torch.ne)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eqz(device, h, w):
    run_relational_test(device, h, w, ttnn.eqz, torch.eq)
