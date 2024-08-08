# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, skip_for_wormhole_b0


def run_moreh_embedding(device, input_shape, embedding_dim, num_embeddings, dtype):
    torch.manual_seed(1234)

    torch_input = torch.randint(0, num_embeddings - 1, input_shape)
    torch_weight = torch_random((num_embeddings, embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)

    torch_output = torch.nn.functional.embedding(torch_input, torch_weight)

    tt_input = ttnn.to_device(
        ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT),
        device,
    )
    tt_weights = ttnn.to_device(ttnn.from_torch(torch_weight, dtype=dtype, layout=ttnn.TILE_LAYOUT), device)

    tt_output = ttnn.to_device(
        ttnn.from_torch(torch.empty_like(torch_output), dtype=dtype, layout=ttnn.TILE_LAYOUT), device
    )

    ttnn.moreh_embedding(tt_input, tt_weights, output=tt_output)
    tt_result = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_result)


@pytest.mark.parametrize("input_shape", [[2, 2], [2, 100], [2, 2, 2], [2, 50, 100]])
@pytest.mark.parametrize("embedding_dim", [5, 32, 100])
@pytest.mark.parametrize("num_embeddings", [90])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_moreh_embedding(
    device,
    input_shape,
    embedding_dim,
    num_embeddings,
    dtype,
):
    run_moreh_embedding(device, input_shape, embedding_dim, num_embeddings, dtype)


@pytest.mark.parametrize("input_shape", [[2, 50, 100]])
@pytest.mark.parametrize("embedding_dim", [100])
@pytest.mark.parametrize("num_embeddings", [90])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_moreh_embedding_callback(
    device,
    input_shape,
    embedding_dim,
    num_embeddings,
    dtype,
):
    for _ in range(2):
        run_moreh_embedding(device, input_shape, embedding_dim, num_embeddings, dtype)

        torch_dummy = torch.empty((32, 32), dtype=torch.bfloat16)
        weights = ttnn.to_device(ttnn.from_torch(torch_dummy, dtype=dtype, layout=ttnn.TILE_LAYOUT), device)
