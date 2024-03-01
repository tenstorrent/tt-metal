# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
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
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.ROW_MAJOR).to(device)
    )

    input_tensor = tt_lib.tensor.Tensor(input_index, tt_lib.tensor.DataType.UINT32).to(device)

    weights_tensor = (
        tt_lib.tensor.Tensor(weights, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.ROW_MAJOR).to(device)
    )

    tt_output_tensor_on_device = tt_lib.tensor.embedding_bw(grad_tensor, input_tensor, weights_tensor)
    tt_output_tensor_a = tt_output_tensor_on_device[0].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    weights.retain_grad()

    pyt_y = torch.nn.functional.embedding(
        input_index.reshape((batch_size, no_of_embeddings)),
        weights.reshape((batch_size * no_of_embeddings, embedding_dim)),
    ).reshape((1, 1, batch_size * no_of_embeddings, embedding_dim))

    pyt_y.backward(gradient=grad_data)

    golden_output_tensor_a = weights.grad

    comp_pass_a, comp_out_a = comparison_funcs.comp_pcc(golden_output_tensor_a, tt_output_tensor_a)

    logger.debug(comp_out_a)
    assert comp_pass_a
