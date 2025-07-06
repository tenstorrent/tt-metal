# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc
from models.utility_functions import is_grayskull
import torch
import ttnn
from models.experimental.stable_diffusion_35_large.tt.utils import assert_quality

"""
Segformer shapes + functionality
"""


def run_nlp_create_qkv_heads_segformer_test(
    batch, seq_len, head_dim, n_heads, dtype, in0_mem_config, out_mem_config, device
):
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    torch.manual_seed(1234)

    ## input tensor
    seq_len = 768
    n_heads = 8
    heads_num = n_heads  # 8
    head_dim = 32

    qkvg_shape = [batch, 1, seq_len, heads_num * head_dim * 3]

    qkvg = torch.randn(qkvg_shape)

    qkvg_ttnn = ttnn.Tensor(qkvg, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)
    print("in0_shape", qkvg_ttnn.shape)

    # print(qkvg_ttnn)
    # print(qkvg)

    ## original method
    #  qkv [:, :, : head*n_heads]
    # first one
    # replicate for q k v each indicvi9udally

    # qkv = torch.unsqueeze(qkvg, 0)

    # Extract Q, K, V as contiguous blocks
    # ref_q = qkvg[:, : heads_num*head_dim]  # [S, heads_num*head_dim]
    # ref_k = qkvg[:, heads_num*head_dim : 2*heads_num*head_dim]  # [S, heads_num*head_dim]
    # ref_v = qkvg[:, 2*heads_num*head_dim : ]  # [S, heads_num*head_dim]

    # ref_q = torch.reshape(ref_q, [seq_len, seq_len, heads_num, head_dim]).transpose(-4, -2)
    # ref_k = torch.reshape(ref_k, [seq_len, seq_len, heads_num, head_dim]).transpose(-4, -2)
    # ref_v = torch.reshape(ref_v, [seq_len, seq_len, heads_num, head_dim]).transpose(-4, -2)

    # print("ref_q", ref_q.shape)
    # print("ref_k", ref_k.shape)
    # print("ref_v", ref_v.shape)

    ## experimental op method
    # qkv = qkvg[:, :, : 3 * head_dim * n_heads]
    # qkv_ttnn = ttnn.unsqueeze(qkvg_ttnn, 0)
    q_ttnn, k_ttnn, v_ttnn = ttnn.experimental.nlp_create_qkv_heads_boltz(
        qkvg_ttnn,
        num_heads=n_heads,
        num_kv_heads=n_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print("q_ttnn", q_ttnn.shape)
    print("k_ttnn", k_ttnn.shape)
    print("v_ttnn", v_ttnn.shape)

    q_from_ttnn = tt2torch_tensor(q_ttnn)
    k_from_ttnn = tt2torch_tensor(k_ttnn)
    v_from_ttnn = tt2torch_tensor(v_ttnn)

    # print(ref_q)
    # print(q_from_ttnn)

    # out_pass_q, output_pcc_q = comp_pcc(q_from_ttnn, ref_q, pcc=0.99)
    # print(out_pass_q, output_pcc_q)
    # out_pass_k, output_pcc_k = comp_pcc(k_from_ttnn, ref_k, pcc=0.99)
    # print(out_pass_k, output_pcc_k)
    # out_pass_v, output_pcc_v = comp_pcc(v_from_ttnn, ref_v, pcc=0.99)
    # print(out_pass_v, output_pcc_v)

    assert_quality(q_ttnn, ref_q, pcc=0.945)

    """
    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.get_dtype()}")

    assert list(q.padded_shape) == [batch, heads_num, seq_len, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)


    ref_q = A
    # Additional shuffling for Q,K,V heads
    ref_q = torch.reshape(ref_q, [batch, seq_len, heads_num, head_dim]).transpose(-3, -2)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    """


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        # ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM"],  # , "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        # ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in0_DRAM"],  # , "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),  # (ttnn.bfloat8_b,),#, ttnn.bfloat16),
    ids=["BFLOAT8_B"],  # , "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len, n_heads, head_dim",
    [(1, 768, 2, 32)],  # Changed to a list containing one tuple
    ids=[
        "batch1_seq768_head384",
    ],
)

# ( 1, 768, 4, 32)
def test_nlp_create_qkv_heads_segformer_test(
    batch, seq_len, n_heads, head_dim, dtype, in0_mem_config, out_mem_config, request, device
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    run_nlp_create_qkv_heads_segformer_test(
        batch, seq_len, head_dim, n_heads, dtype, in0_mem_config, out_mem_config, device
    )


"""
def test_nlp_create_qkv_heads_segformer_with_program_cache(device):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_segformer_test(1, 32, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_segformer_test(1, 32, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
"""
