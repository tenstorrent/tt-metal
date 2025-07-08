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
Test for nlp_create_qkv_heads_boltz operation
"""


def run_nlp_create_qkv_heads_boltz_test(
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
    # n_heads = 8
    heads_num = n_heads  # 8
    head_dim = 32

    qkvg_shape = [batch, 1, seq_len, heads_num * head_dim * 3]
    qkvg = torch.randn(qkvg_shape)

    # Extract Q, K, V as contiguous blocks
    ref_q = qkvg[:, :, :, : heads_num * head_dim]  # [batch, 1, seq_len, heads_num*head_dim]
    ref_k = qkvg[:, :, :, heads_num * head_dim : 2 * heads_num * head_dim]  # [batch, 1, seq_len, heads_num*head_dim]
    ref_v = qkvg[:, :, :, 2 * heads_num * head_dim :]  # [batch, 1, seq_len, heads_num*head_dim]

    # Remove batch and singleton dimensions: [seq_len, heads_num*head_dim]
    ref_q = ref_q.squeeze(0).squeeze(0)  # [seq_len, heads_num*head_dim]
    ref_k = ref_k.squeeze(0).squeeze(0)  # [seq_len, heads_num*head_dim]
    ref_v = ref_v.squeeze(0).squeeze(0)  # [seq_len, heads_num*head_dim]

    # Reshape to separate heads: [seq_len, heads_num, head_dim]
    ref_q = ref_q.reshape(seq_len, heads_num, head_dim)
    ref_k = ref_k.reshape(seq_len, heads_num, head_dim)
    ref_v = ref_v.reshape(seq_len, heads_num, head_dim)

    # Expand to create the seq_len x seq_len structure: [seq_len, seq_len, heads_num, head_dim]
    ref_q = ref_q.unsqueeze(1).expand(seq_len, seq_len, heads_num, head_dim)
    ref_k = ref_k.unsqueeze(1).expand(seq_len, seq_len, heads_num, head_dim)
    ref_v = ref_v.unsqueeze(1).expand(seq_len, seq_len, heads_num, head_dim)

    # Apply transpose to get [heads_num, seq_len, seq_len, head_dim]
    ref_q = ref_q.transpose(-4, -2)
    ref_k = ref_k.transpose(-4, -2)
    ref_v = ref_v.transpose(-4, -2)

    # Now convert the input to interleaved format for the TTNN operation
    # The TTNN operation expects data in interleaved format per head: Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, etc.
    # But our test data is in concatenated format: Q_all_heads, K_all_heads, V_all_heads
    # Convert from concatenated to interleaved format to match what the TTNN op expects

    # Extract the QKV data: [batch, seq_len, heads_num * head_dim * 3]
    qkv_data = qkvg.squeeze(1)  # Remove the dimension of size 1: [batch, seq_len, heads_num * head_dim * 3]

    # Extract Q, K, V from concatenated format
    q_data = qkv_data[:, :, : heads_num * head_dim]  # [batch, seq_len, heads_num * head_dim]
    k_data = qkv_data[:, :, heads_num * head_dim : 2 * heads_num * head_dim]  # [batch, seq_len, heads_num * head_dim]
    v_data = qkv_data[:, :, 2 * heads_num * head_dim :]  # [batch, seq_len, heads_num * head_dim]

    # Reshape to separate heads
    q_data = q_data.reshape(batch, seq_len, heads_num, head_dim)  # [batch, seq_len, heads_num, head_dim]
    k_data = k_data.reshape(batch, seq_len, heads_num, head_dim)  # [batch, seq_len, heads_num, head_dim]
    v_data = v_data.reshape(batch, seq_len, heads_num, head_dim)  # [batch, seq_len, heads_num, head_dim]

    # Convert to interleaved format: Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, etc.
    qkv_interleaved = torch.zeros(batch, seq_len, heads_num * head_dim * 3)
    for head in range(heads_num):
        # For each head, place Q, K, V data in interleaved positions
        start_idx = head * head_dim * 3
        qkv_interleaved[:, :, start_idx : start_idx + head_dim] = q_data[:, :, head, :]  # Q
        qkv_interleaved[:, :, start_idx + head_dim : start_idx + 2 * head_dim] = k_data[:, :, head, :]  # K
        qkv_interleaved[:, :, start_idx + 2 * head_dim : start_idx + 3 * head_dim] = v_data[:, :, head, :]  # V

    # Update the input tensor to use the interleaved format for TTNN
    qkvg_ttnn_input = qkv_interleaved.unsqueeze(
        1
    )  # Add back the dimension: [batch, 1, seq_len, heads_num * head_dim * 3]
    qkvg_ttnn = ttnn.Tensor(qkvg_ttnn_input, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    ## experimental op method
    q_ttnn, k_ttnn, v_ttnn = ttnn.experimental.nlp_create_qkv_heads_boltz(
        qkvg_ttnn,
        num_heads=heads_num,
        num_kv_heads=heads_num,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Convert TTNN tensors to PyTorch tensors for comparison
    q_from_ttnn = tt2torch_tensor(q_ttnn)  # [heads_num, seq_len, seq_len, head_dim] - Query tensor output
    k_from_ttnn = tt2torch_tensor(k_ttnn)  # [heads_num, seq_len, seq_len, head_dim] - Key tensor output
    v_from_ttnn = tt2torch_tensor(v_ttnn)  # [heads_num, seq_len, seq_len, head_dim] - Value tensor output

    # Check all heads
    print("Checking all heads")

    # Quality checks - check all heads
    out_pass_q, output_pcc_q = comp_pcc(q_from_ttnn, ref_q, pcc=0.99)
    out_pass_k, output_pcc_k = comp_pcc(k_from_ttnn, ref_k, pcc=0.99)
    out_pass_v, output_pcc_v = comp_pcc(v_from_ttnn, ref_v, pcc=0.99)

    print(f"Q PCC: {output_pcc_q}, Pass: {out_pass_q}")
    print(f"K PCC: {output_pcc_k}, Pass: {out_pass_k}")
    print(f"V PCC: {output_pcc_v}, Pass: {out_pass_v}")

    assert out_pass_q, f"Q tensor quality check failed with PCC: {output_pcc_q}"
    assert out_pass_k, f"K tensor quality check failed with PCC: {output_pcc_k}"
    assert out_pass_v, f"V tensor quality check failed with PCC: {output_pcc_v}"


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
def test_nlp_create_qkv_heads_boltz(
    batch, seq_len, n_heads, head_dim, dtype, in0_mem_config, out_mem_config, request, device
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    run_nlp_create_qkv_heads_boltz_test(
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
