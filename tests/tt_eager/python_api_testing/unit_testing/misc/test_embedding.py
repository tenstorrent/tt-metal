# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


from tt_py_test_utils_common.embedding_tests import run_embeddings_tests

import pytest
import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
)


@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "num_rows",
    (32, 256, 512),
    ids=["Num_Rows_32", "Num_Rows_256", "Num_Rows_512"],
)
@pytest.mark.parametrize(
    "num_embeddings",
    (512, 30522, 2048),
    ids=[
        "Bert_Position_Embeddings_512",
        "Bert_Word_Embeddings_30528",
        "Llama_Position_Embeddings",
    ],
)
@pytest.mark.parametrize(
    "embedding_dim",
    (768, 4096),
    ids=["Bert_Num_Cols_768", "Llama_Num_Cols"],
)
@pytest.mark.parametrize(
    "batch_size",
    (1, 8, 9),
    ids=["Batch_Size_1", "Batch_Size_8", "Batch_Size_9"],
)
@pytest.mark.parametrize(
    "tilized",
    (True,),
    ids=[
        "Tilized",
    ],
)
def test_embeddings(
    batch_size, num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config, device, tilized
):
    run_embeddings_tests(
        batch_size, num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config, device, tilized
    )
