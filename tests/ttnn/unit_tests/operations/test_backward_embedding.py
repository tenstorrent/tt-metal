# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from models.utility_functions import skip_for_grayskull
from loguru import logger


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim, num_embeddings",
    [
        (2, 64, 160, 96),
        (3, 32, 384, 320),
        (2, 1024, 4096, 3200),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
    ],
)
def test_embedding_bw(input_dtype, output_dtype, batch_size, seq_len, embedding_dim, num_embeddings, device):
    torch.manual_seed(1234)

    if input_dtype == ttnn.bfloat16 and num_embeddings > 256:
        pytest.skip("Skipping tests with large vocab sizes for bfloat16 indices!")

    input_shape = (batch_size, seq_len)
    input_index = torch.randint(0, num_embeddings, input_shape)
    input_tensor = ttnn.from_torch(input_index, dtype=input_dtype, device=device)

    weights_shape = (num_embeddings, embedding_dim)
    weights = torch.randn(weights_shape, requires_grad=True)
    weights_ttnn = ttnn.from_torch(weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    grad_shape = (1, 1, batch_size * seq_len, embedding_dim)
    grad_data = torch.randn(grad_shape, requires_grad=True)
    grad_tensor = ttnn.from_torch(grad_data, dtype=output_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output_tensor_on_device = ttnn.embedding_bw(input_tensor, weights_ttnn, grad_tensor, dtype=output_dtype)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device)

    # PyTorch reference
    weights.retain_grad()
    pyt_y = torch.nn.functional.embedding(input_index, weights).reshape(grad_shape)
    pyt_y.backward(gradient=grad_data)
    golden_output_tensor = weights.grad

    comp_pass, comp_out = comparison_funcs.comp_pcc(golden_output_tensor, tt_output_tensor)

    logger.debug(comp_out)
    assert comp_pass


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim, num_embeddings",
    [
        (2, 64, 160, 96),
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
    ],
)
def test_embedding_bw_with_program_cache(
    input_dtype, output_dtype, batch_size, seq_len, embedding_dim, num_embeddings, device, use_program_cache
):
    torch.manual_seed(1234)

    input_shape = (batch_size, seq_len)
    weights_shape = (num_embeddings, embedding_dim)
    grad_shape = (1, 1, batch_size * seq_len, embedding_dim)

    for _ in range(2):
        input_index = torch.randint(0, num_embeddings, input_shape)
        input_tensor = ttnn.from_torch(input_index, dtype=input_dtype, device=device)

        weights = torch.randn(weights_shape, requires_grad=True)
        weights_ttnn = ttnn.from_torch(weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        grad_data = torch.randn(grad_shape, requires_grad=True)
        grad_tensor = ttnn.from_torch(grad_data, dtype=output_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output_tensor_on_device = ttnn.embedding_bw(input_tensor, weights_ttnn, grad_tensor, dtype=output_dtype)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device)

        # PyTorch reference
        weights.retain_grad()
        pyt_y = torch.nn.functional.embedding(input_index, weights).reshape(grad_shape)
        pyt_y.backward(gradient=grad_data)
        golden_output_tensor = weights.grad

        comp_pass, comp_out = comparison_funcs.comp_pcc(golden_output_tensor, tt_output_tensor)

        logger.debug(comp_out)
        assert comp_pass

    assert device.num_program_cache_entries() == 1
