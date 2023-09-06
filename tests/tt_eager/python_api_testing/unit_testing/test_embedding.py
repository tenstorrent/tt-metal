"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import math
from pathlib import Path
import sys
import time
import os

import torch

import tt_lib as ttl

from tt_lib.utils import (
    is_close,
)


def baseline_embeddings_list(num_rows, num_embeddings, embedding_dim, input, weights):
    input_list = input.reshape((num_rows,)).tolist()
    weight_list = weights.reshape((num_embeddings, embedding_dim)).tolist()

    selected_weights = []
    for inp_row in input_list:
        selected_weights.append(weight_list[inp_row])

    return selected_weights


def run_embeddings_tests(
    num_embeddings,
    embedding_dim,
    num_rows,
    dtype,
    in0_mem_config,
    out_mem_config,
    device,
):
    torch.manual_seed(1234)

    # Initialize the device
    tensor = ttl.tensor
    dev = device

    input_rows_shape = [1, 1, num_rows, 1]

    input_rows_torch = torch.randint(0, num_embeddings - 1, tuple(input_rows_shape))
    weights_shape = [1, 1, num_embeddings, embedding_dim]
    weights_torch = torch.randn(weights_shape)
    input_tensor = tensor.Tensor(input_rows_torch, ttl.tensor.DataType.UINT32).to(
        dev, in0_mem_config
    )
    weights_tensor = tensor.Tensor(weights_torch, dtype).to(dev, in0_mem_config)

    ttz = tensor.embeddings(input_tensor, weights_tensor, out_mem_config)
    tt_data = ttz.cpu().to_torch()
    tt_got_back = torch.Tensor(tt_data).reshape((1, 1, num_rows, embedding_dim))
    reference_list = baseline_embeddings_list(
        num_rows, num_embeddings, embedding_dim, input_rows_torch, weights_torch
    )
    reference_torch = torch.Tensor(reference_list).reshape(
        (1, 1, num_rows, embedding_dim)
    )

    assert is_close(tt_got_back, reference_torch)


import pytest


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)

# TODO add Llama Word embeddings of size 320000
# Enqueue write buffer too large
@pytest.mark.parametrize(
    "num_embeddings",
    (512, 30522, 2048),
    ids=[
        "Bert_Position_Embeddings_512",
        "Bert_Word_Embeddings_30522",
        "Llama_Position_Embeddings",
    ],
)
@pytest.mark.parametrize(
    "embedding_dim",
    (768, 4096),
    ids=["Bert_Num_Cols_768", "Llama_Num_Cols"],
)
@pytest.mark.parametrize(
    "num_rows",
    (4,),
    ids=["Num_Output_Rows_4"],
)
def test_embeddings(
    num_embeddings,
    embedding_dim,
    num_rows,
    dtype,
    in0_mem_config,
    out_mem_config,
    device,
):
    run_embeddings_tests(
        num_embeddings,
        embedding_dim,
        num_rows,
        dtype,
        in0_mem_config,
        out_mem_config,
        device,
    )
