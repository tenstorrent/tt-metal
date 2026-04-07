# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test flash_mla_prefill at specific sequence lengths.

Tests sequence lengths that are both small and large, covering
aligned multiples of the k_chunk_size. Uses DeepSeek V3-compatible
parameters (nh=8, kv_lora_rank=128) to stay within L1 limits on any device.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    nearest_n,
    nearest_pow_2,
    scaled_dot_product_attention_reference_prefill,
)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize(
    "seq_len",
    [
        128,
        256,
        384,
        512,
        768,
        1024,
    ],
)
@pytest.mark.parametrize("k_chunk_size", [128])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_prefill_positions(
    device,
    batch,
    seq_len,
    k_chunk_size,
    q_dtype,
    kv_dtype,
    reset_seeds,
):
    """
    Test flash_mla_prefill at various causal sequence lengths.

    Uses WH-compatible parameters (nh=8, kv_lora_rank=128) to stay within
    L1 limits. seq_len must be a multiple of both q_chunk_size and k_chunk_size.
    Works on any device — no hardcoded grid.
    """
    nh = 8
    nkv = 1
    kv_lora_rank = 128
    d_rope = 64
    kvpe_dim = kv_lora_rank + d_rope  # 192
    scale = kvpe_dim**-0.5

    # q_chunk_size is derived from nh, mirroring mla_test_utils.run_flash_mla_prefill_impl
    q_chunk_size = nearest_pow_2(nearest_n(nh, n=32))  # 32 for nh=8

    logger.info(
        f"seq_len={seq_len}, nh={nh}, kv_lora_rank={kv_lora_rank}, "
        f"q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}"
    )

    # Torch tensors: Q [batch, nh, seq_len, kvpe_dim], K [batch, nkv, seq_len, kvpe_dim]
    q = torch.randn(batch, nh, seq_len, kvpe_dim).float()
    k = torch.randn(batch, nkv, seq_len, kvpe_dim).float()

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # flash_mla_prefill expects Q as [batch, nh, seq_len, kvpe_dim]
    tt_q = ttnn.from_torch(
        q,
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        k,
        device=device,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = ttnn.transformer.flash_mla_prefill(
        tt_q,
        tt_k,
        head_dim_v=kv_lora_rank,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        attn_mask=None,
        is_causal=True,
    )

    v = k[..., :kv_lora_rank]
    ref_out = scaled_dot_product_attention_reference_prefill(q, k, v, scale, is_causal=True)

    # tt_out shape: [batch, nh_padded, seq_len, kv_lora_rank] -> slice to [batch, nh, seq_len, kv_lora_rank]
    tt_out_torch = ttnn.to_torch(tt_out)[:, :nh, :seq_len, :]

    pcc_threshold = 0.98 if kv_dtype == ttnn.bfloat8_b else 0.999
    out_pass, out_pcc = comp_pcc(tt_out_torch, ref_out, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")
    assert out_pass, f"PCC {out_pcc} < {pcc_threshold} at seq_len={seq_len}"
