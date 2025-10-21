# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.common.utility_functions import tt2torch_tensor, comp_pcc
from models.common.utility_functions import is_grayskull
import torch
import ttnn
from models.experimental.stable_diffusion_35_large.tt.utils import assert_quality

"""
Test for nlp_create_qkv_heads_boltz operation
"""


def run_nlp_create_qkv_heads_boltz_test(batch, seq_len, head_dim, n_heads, dtype, in0_mem_config, device):
    torch.manual_seed(1234)

    ## input tensor
    heads_num = n_heads

    qkvg_shape = [batch, seq_len, seq_len, heads_num * head_dim * 3]
    qkvg = torch.randn(qkvg_shape)

    # Extract Q, K, V as contiguous blocks
    ref_q = qkvg[:, :, :, : heads_num * head_dim]  # [batch, 1, seq_len, heads_num*head_dim]
    ref_k = qkvg[:, :, :, heads_num * head_dim : 2 * heads_num * head_dim]  # [batch, 1, seq_len, heads_num*head_dim]
    ref_v = qkvg[:, :, :, 2 * heads_num * head_dim :]  # [batch, 1, seq_len, heads_num*head_dim]

    ref_q = torch.reshape(ref_q, [seq_len, seq_len, heads_num, head_dim]).permute(2, 0, 1, 3)
    ref_k = torch.reshape(ref_k, [seq_len, seq_len, heads_num, head_dim]).permute(2, 0, 1, 3)
    ref_v = torch.reshape(ref_v, [seq_len, seq_len, heads_num, head_dim]).permute(2, 0, 1, 3)

    ## ttnn
    qkvg_ttnn = ttnn.Tensor(qkvg, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    ## experimental op method
    q_ttnn, k_ttnn, v_ttnn = ttnn.experimental.nlp_create_qkv_heads_boltz(
        qkvg_ttnn,
        num_heads=heads_num,
        num_kv_heads=heads_num,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Check memory of inputs and outputs
    assert qkvg_ttnn.memory_config().buffer_type == in0_mem_config.buffer_type
    logger.debug(f"qkvg: {qkvg_ttnn.memory_config().buffer_type} and {qkvg_ttnn.get_dtype()}")
    logger.debug(f"q: {q_ttnn.memory_config().buffer_type} and {q_ttnn.get_dtype()}")
    logger.debug(f"k: {k_ttnn.memory_config().buffer_type} and {k_ttnn.get_dtype()}")
    logger.debug(f"v: {v_ttnn.memory_config().buffer_type} and {v_ttnn.get_dtype()}")

    # Convert TTNN tensors to PyTorch tensors for comparison
    q_from_ttnn = tt2torch_tensor(q_ttnn)
    k_from_ttnn = tt2torch_tensor(k_ttnn)
    v_from_ttnn = tt2torch_tensor(v_ttnn)

    # Quality checks - check all heads
    out_pass_q, output_pcc_q = comp_pcc(q_from_ttnn, ref_q, pcc=0.99)
    out_pass_k, output_pcc_k = comp_pcc(k_from_ttnn, ref_k, pcc=0.99)
    out_pass_v, output_pcc_v = comp_pcc(v_from_ttnn, ref_v, pcc=0.99)

    assert out_pass_q, f"Q tensor quality check failed with PCC: {output_pcc_q}"
    assert out_pass_k, f"K tensor quality check failed with PCC: {output_pcc_k}"
    assert out_pass_v, f"V tensor quality check failed with PCC: {output_pcc_v}"


@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["FLOAT32", "BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "batch, seq_len, n_heads, head_dim",
    [(1, 768, 4, 32), (1, 704, 4, 32), (1, 1408, 4, 32)],
    ids=[
        "batch1_seq768_heads4_headdim32",
        "batch1_seq704_heads4_headdim32",
        "batch1_seq1408_heads4_headdim32",
    ],
)

# Test with different configurations: batch size, sequence length, number of heads, head dimension
@pytest.mark.timeout(120)
def test_nlp_create_qkv_heads_boltz(batch, seq_len, n_heads, head_dim, dtype, in0_mem_config, request, device):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    run_nlp_create_qkv_heads_boltz_test(batch, seq_len, head_dim, n_heads, dtype, in0_mem_config, device)
