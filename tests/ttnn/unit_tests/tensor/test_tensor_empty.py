# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


def test_tensor_create_empty_tensor_with_shape_0(device):
    tt_tensor = ttnn.empty([0, 0], ttnn.float32, device=device)
    assert tt_tensor.empty() == True


def test_tensor_create_empty_tensor_with_shape(device):
    tt_tensor = ttnn.empty(shape=[2, 3], device=device)
    assert tt_tensor.empty() == False


def test_tensor_create_empty_tensor_with_shape_0_and_fill_value(device):
    tt_tensor = ttnn.full(shape=[0, 0], fill_value=0, device=device)
    assert tt_tensor.empty() == True


def test_tensor_create_empty_tensor_with_shape_2_3_and_fill_value(device):
    # A Tensor with zeroes is not empty!
    tt_tensor = ttnn.full(shape=[2, 3], fill_value=0, device=device)
    assert tt_tensor.empty() == False


def test_tensor_create_empty_tensor_with_shape_2_3_and_slice(device):
    tt_tensor = ttnn.empty(shape=[2, 3], device=device)
    tt_tensor = ttnn.slice(tt_tensor, [0, 0], [0, 0])
    assert tt_tensor.empty() == True


def test_tensor_create_empty_tensor_with_shape_2_3_and_deallocate(device):
    tt_tensor = ttnn.empty(shape=[2, 3], device=device)
    ttnn.deallocate(tt_tensor)
    assert tt_tensor.empty() == True
