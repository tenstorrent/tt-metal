# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn.deprecated
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from loguru import logger


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_embedding_bw(input_shapes, device):
    torch.manual_seed(1234)

    batch_size = input_shapes[0]
    no_of_embeddings = input_shapes[1] * input_shapes[2]
    embedding_dim = input_shapes[3]

    input_shape = [batch_size, 1, 1, no_of_embeddings]
    input_index = torch.reshape(torch.arange(0, batch_size * no_of_embeddings), shape=input_shape)
    weights_shape = [batch_size, 1, no_of_embeddings, embedding_dim]
    weights = torch.randn(weights_shape, requires_grad=True)
    grad_shape = [1, 1, batch_size * no_of_embeddings, embedding_dim]
    grad_data = torch.randn(grad_shape, requires_grad=True)

    grad_tensor = (
        ttnn.experimental.tensor.Tensor(grad_data, ttnn.experimental.tensor.DataType.BFLOAT16)
        .to(ttnn.experimental.tensor.Layout.ROW_MAJOR)
        .to(device)
    )

    input_tensor = ttnn.experimental.tensor.Tensor(input_index, ttnn.experimental.tensor.DataType.UINT32).to(device)

    weights_tensor = (
        ttnn.experimental.tensor.Tensor(weights, ttnn.experimental.tensor.DataType.BFLOAT16)
        .to(ttnn.experimental.tensor.Layout.ROW_MAJOR)
        .to(device)
    )

    tt_output_tensor_on_device = ttnn.embedding_bw(grad_tensor, input_tensor, weights_tensor)
    tt_output_tensor_a = tt_output_tensor_on_device[0].cpu().to(ttnn.experimental.tensor.Layout.ROW_MAJOR).to_torch()

    golden_function = ttnn.get_golden_function(ttnn.embedding_bw)
    golden_tensor = golden_function(grad_data, input_index, weights, input_shapes)

    comp_pass_a, comp_out_a = comparison_funcs.comp_pcc(golden_tensor, tt_output_tensor_a)

    logger.debug(comp_out_a)
    assert comp_pass_a
