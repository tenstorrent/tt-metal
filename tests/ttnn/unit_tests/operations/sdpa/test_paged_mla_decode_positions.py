# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test paged_flash_multi_latent_attention_decode at specific decode positions.

Tests both aligned and unaligned chunk boundaries using DeepSeek V3 parameters.
Like test_mla_decode_positions.py but uses paged KV cache via
ttnn.transformer.paged_flash_multi_latent_attention_decode.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import nearest_y
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    page_table_setup,
    scaled_dot_product_attention_reference,
    to_paged_cache,
)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize(
    "decode_position",
    [
        # Aligned chunk boundaries
        127,
        255,
        383,
        511,
        639,
        767,
        895,
        1023,
        # Unaligned
        0,
        1,
        2,
        7,
        8,
        15,
        16,
        128,
        564,
        1203,
    ],
)
@pytest.mark.parametrize("k_chunk_size", [128])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_paged_flash_mla_decode_positions(
    device,
    batch,
    decode_position,
    k_chunk_size,
    q_dtype,
    kv_dtype,
    reset_seeds,
):
    """
    Test paged_flash_multi_latent_attention_decode at specific aligned/unaligned decode positions.

    Uses a paged KV cache (block_size == k_chunk_size) and verifies output matches
    the non-paged reference at the given decode position.  Uses WH-compatible
    parameters (nh=8, kv_lora_rank=128) to stay within L1 limits.
    """
    nh = 8
    nkv = 1
    kv_lora_rank = 128
    d_rope = 64
    kvpe_dim = kv_lora_rank + d_rope  # 192
    scale = kvpe_dim**-0.5

    # block_size equals k_chunk_size so that seq_len is always a multiple of block_size
    block_size = k_chunk_size

    # seq_len must cover decode_position and be a multiple of block_size
    seq_len = max(2048, nearest_y(decode_position + 1, block_size) * 2)

    start_indices = [decode_position] * batch
    padded_layer_len = nearest_y(decode_position + 1, k_chunk_size)

    logger.info(
        f"decode_position={decode_position}, seq_len={seq_len}, "
        f"padded_layer_len={padded_layer_len}, block_size={block_size}, "
        f"nh={nh}, kv_lora_rank={kv_lora_rank}"
    )

    # Paged attention configuration
    max_num_blocks = seq_len // block_size * batch
    paged_attention_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    # Torch tensors: Q [batch, nh, 1, kvpe_dim], K/V [batch, nkv, seq_len, kvpe_dim/kv_lora_rank]
    q = torch.randn(batch, nh, 1, kvpe_dim).float()
    k = torch.randn(batch, nkv, seq_len, kvpe_dim).float()
    v = k[..., :kv_lora_rank]

    # Build page table and convert K/V to paged layout
    page_table = page_table_setup(batch, paged_attention_cfg)
    k_paged = to_paged_cache(k, page_table, paged_attention_cfg)
    v_paged = to_paged_cache(v, page_table, paged_attention_cfg)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # flash_multi_latent_attention_decode expects Q as [1, batch, nh, kvpe_dim]
    tt_q = ttnn.from_torch(
        q.permute(2, 0, 1, 3),
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Paged K/V: [max_num_blocks, nkv, block_size, dim]
    tt_k = ttnn.from_torch(
        k_paged,
        device=device,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v = ttnn.from_torch(
        v_paged,
        device=device,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_page_table = ttnn.from_torch(
        page_table,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_start_indices = ttnn.from_torch(
        torch.tensor(start_indices),
        device=device,
        dtype=ttnn.int32,
    )

    tt_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        head_dim_v=kv_lora_rank,
        page_table_tensor=tt_page_table,
        cur_pos_tensor=tt_start_indices,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ref_out = scaled_dot_product_attention_reference(q, k, v, start_indices, padded_layer_len, scale)

    # tt_out shape: [1, batch, nh_padded, kv_lora_rank] -> slice nh and permute to [batch, nh, 1, kv_lora_rank]
    tt_out_torch = ttnn.to_torch(tt_out)[..., :nh, :].permute(1, 2, 0, 3)

    pcc_threshold = 0.98 if kv_dtype == ttnn.bfloat8_b else 0.999
    out_pass, out_pcc = comp_pcc(tt_out_torch, ref_out, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")
    assert out_pass, f"PCC {out_pcc} < {pcc_threshold} at decode_position={decode_position}"
