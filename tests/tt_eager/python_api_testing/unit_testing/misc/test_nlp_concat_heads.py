# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.common.utility_functions import comp_pcc, tt2torch_tensor
import torch
from models.common.utility_functions import is_wormhole_b0
import ttnn
from ttnn import ttnn_dtype_to_torch_dtype


def run_nlp_concat_heads_test(batch, seq_len, num_heads, head_dim, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    in0_shape = [batch, num_heads, seq_len, head_dim]

    A = torch.randn(in0_shape)

    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    out = ttnn.experimental.nlp_concat_heads(in0_t, memory_config=out_mem_config)

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"out: {out.memory_config().buffer_type} and {out.get_dtype()}")

    assert list(out.padded_shape) == [batch, 1, seq_len, num_heads * head_dim]

    pyt_got_back_rm_out = tt2torch_tensor(out)

    ref_out = torch.transpose(A, -3, -2).reshape([batch, 1, seq_len, num_heads * head_dim])

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc, output_pcc = comp_pcc(pyt_got_back_rm_out, ref_out, pcc)
    logger.debug(f"passing={passing_pcc}")
    logger.debug(f"output pcc={output_pcc}")
    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32),
    ids=["BFLOAT8_B", "BFLOAT16", "FLOAT32"],
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 32), (1, 64), (1, 128)),
    ids=[
        "batch1_seq32",
        "batch1_seq64",
        "batch1_seq128",
    ],
)
@pytest.mark.parametrize("num_heads", (2, 8, 16), ids=["num_heads2", "num_heads8", "num_heads16"])
@pytest.mark.parametrize("head_dim", (64, 128), ids=["head_dim64", "head_dim128"])
def test_nlp_concat_heads_test(
    batch, seq_len, num_heads, head_dim, dtype, in0_mem_config, out_mem_config, request, device
):
    run_nlp_concat_heads_test(batch, seq_len, num_heads, head_dim, dtype, in0_mem_config, out_mem_config, device)


def run_nlp_concat_heads_subtile_head_dim_test(
    batch, seq_len, num_heads, head_dim, dtype, in0_mem_config, out_mem_config, device
):
    # Head dims that are not a multiple of 32 used to be silently mis-read by the TM
    # kernel, which walks each head as a whole number of 32-wide tiles.
    torch.manual_seed(1234)

    in0_shape = [batch, num_heads, seq_len, head_dim]
    # Round the input to the tensor dtype up front so the reference is what actually
    # reaches the device; comparing against a wider-typed reference never matches
    # bit-exactly.
    A = torch.randn(in0_shape).to(ttnn_dtype_to_torch_dtype(dtype))

    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)
    out = ttnn.experimental.nlp_concat_heads(in0_t, memory_config=out_mem_config)

    # Logical shape only: the last dim pads to a multiple of TILE_SIZE, so
    # num_heads * head_dim need not be a multiple of 32.
    assert list(out.shape) == [batch, 1, seq_len, num_heads * head_dim]

    # to_torch returns the logical region, so no padded tail is compared.
    pyt_got_back_rm_out = ttnn.to_torch(out)
    ref_out = torch.transpose(A, -3, -2).reshape([batch, 1, seq_len, num_heads * head_dim])

    # Exact compare: the fused TM path is bit-exact for these shapes, and so is the
    # row-major fallback the op routes sub-tile head dims through.
    assert torch.equal(
        pyt_got_back_rm_out, ref_out
    ), f"max abs diff {(pyt_got_back_rm_out.float() - ref_out.float()).abs().max().item()}"


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
@pytest.mark.parametrize("dtype", (ttnn.bfloat16,), ids=["BFLOAT16"])
@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 32), (2, 64)),
    ids=["batch1_seq32", "batch2_seq64"],
)
@pytest.mark.parametrize("head_dim", (80, 96, 100), ids=["head_dim80", "head_dim96", "head_dim100"])
@pytest.mark.parametrize("num_heads", (2, 5, 8), ids=["num_heads2", "num_heads5", "num_heads8"])
def test_nlp_concat_heads_subtile_head_dim(
    batch, seq_len, num_heads, head_dim, dtype, in0_mem_config, out_mem_config, device
):
    if num_heads * head_dim % ttnn.TILE_SIZE == 0 and head_dim % ttnn.TILE_SIZE == 0:
        pytest.skip("head_dim already TILE aligned, covered by test_nlp_concat_heads_test")
    run_nlp_concat_heads_subtile_head_dim_test(
        batch, seq_len, num_heads, head_dim, dtype, in0_mem_config, out_mem_config, device
    )


def test_nlp_concat_heads_with_program_cache(device):
    num_heads = 71
    head_dim = 64
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    for _ in range(2):
        run_nlp_concat_heads_test(1, 32, num_heads, head_dim, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    for _ in range(2):
        run_nlp_concat_heads_test(1, 32, num_heads, head_dim, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
