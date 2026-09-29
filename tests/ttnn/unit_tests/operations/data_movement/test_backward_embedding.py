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
        # num_embeddings not a multiple of TILE_HEIGHT: the output is tile-padded, so
        # the trailing partial tile row still has to be zeroed before accumulation.
        (2, 64, 160, 9),
        (1, 32, 64, 33),
        (2, 32, 128, 100),
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


SENTINEL = 999.0


@pytest.mark.parametrize("num_embeddings", [9, 32, 33, 64, 96, 100, 320])
def test_embedding_bw_unindexed_rows_are_zero(num_embeddings, device):
    """Rows no index points at must come back exactly zero.

    The op zeroes its whole output before accumulating, so PCC over the full tensor
    hides a partially zeroed one; this asserts the untouched rows directly. When
    num_embeddings is not a multiple of TILE_HEIGHT its trailing partial tile row was
    left out of the zero-init pass and came back holding whatever the DRAM pages
    already contained, so the test first dirties those pages with a sentinel.
    """
    torch.manual_seed(1234)
    embedding_dim, seq_len = 64, 32

    indexed_rows = sorted(range(0, num_embeddings, 3))
    input_index = torch.tensor(indexed_rows, dtype=torch.int32)
    input_index = input_index.repeat((seq_len + len(indexed_rows) - 1) // len(indexed_rows))[:seq_len]
    input_tensor = ttnn.from_torch(input_index.reshape(1, seq_len), dtype=ttnn.uint32, device=device)

    grad_data = torch.randn(1, 1, seq_len, embedding_dim)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    weights = torch.zeros(num_embeddings, embedding_dim)
    weights_ttnn = ttnn.from_torch(weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Allocate an identically shaped tensor full of a sentinel and free it, so the
    # allocator hands the same pages to the output and a missed zero-init is visible.
    dirty = ttnn.from_torch(
        torch.full((1, 1, num_embeddings, embedding_dim), SENTINEL),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn.deallocate(dirty)

    out = ttnn.to_torch(ttnn.embedding_bw(input_tensor, weights_ttnn, grad_tensor, dtype=ttnn.bfloat16)).reshape(
        num_embeddings, embedding_dim
    )

    touched = set(input_index.tolist())
    untouched = [row for row in range(num_embeddings) if row not in touched]
    assert untouched, "test needs at least one row no index points at"
    nonzero = [row for row in untouched if out[row].abs().max() != 0]
    assert not nonzero, f"rows {nonzero} are not zero (num_embeddings={num_embeddings})"


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

    assert device.cache_entries_counter.total == 1
