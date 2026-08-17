# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from models.common.utility_functions import skip_for_blackhole
from loguru import logger


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
    input_dtype, output_dtype, batch_size, seq_len, embedding_dim, num_embeddings, device
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

        with device.cache_entries_counter.measure():
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

    # embedding_bw dispatches two programs: the device primitive, which
    # accumulates into a private FLOAT32 result, and the typecast that converts
    # it once to the public output dtype. Both are cached on the first call and
    # neither is recreated on the second, so two entries is the cache-hit result.
    assert device.cache_entries_counter.total == 2


@pytest.mark.parametrize("collision_count", [32, 64, 128, 256, 512, 1024, 2048])
def test_embedding_bw_repeated_indices_accumulate_exactly(collision_count, device):
    batch_size = 32
    seq_len = 64
    embedding_dim = 128
    num_embeddings = 64
    target_index = 7
    total_positions = batch_size * seq_len

    non_target_indices = torch.tensor(
        [index for index in range(num_embeddings) if index != target_index], dtype=torch.int64
    )
    input_index = non_target_indices[torch.arange(total_positions).remainder(num_embeddings - 1)]
    target_positions = torch.div(
        torch.arange(collision_count) * total_positions, collision_count, rounding_mode="floor"
    )
    input_index[target_positions] = target_index
    input_index = input_index.reshape(batch_size, seq_len)

    hidden_values = 1 + torch.arange(embedding_dim, dtype=torch.float32).remainder(8)
    grad_data = (
        hidden_values.view(1, 1, 1, embedding_dim)
        .expand(1, 1, total_positions, embedding_dim)
        .div(2048)
        .to(torch.bfloat16)
        .contiguous()
    )
    golden = torch.zeros(num_embeddings, embedding_dim, dtype=torch.float32)
    golden.index_add_(0, input_index.reshape(-1), grad_data.float().reshape(-1, embedding_dim))

    input_tensor = ttnn.from_torch(input_index, dtype=ttnn.uint32, device=device)
    weights_tensor = ttnn.from_torch(
        torch.zeros(num_embeddings, embedding_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.embedding_bw(input_tensor, weights_tensor, grad_tensor, dtype=ttnn.bfloat16)
    output_torch = ttnn.to_torch(output).float()

    # Shape-agnostic on purpose: this test is about accumulation, not the
    # returned rank.
    assert torch.equal(golden, output_torch.reshape(num_embeddings, embedding_dim))
