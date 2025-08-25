# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_flash_multi_latent_attention_decode import (
    page_table_setup,
    to_paged_cache,
    from_paged_cache,
)
import ttnn
from loguru import logger
import pytest

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
)


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def scaled_dot_product_attention_reference(Q, K, V, scale, is_causal=True):
    """
    Full-sequence causal SDPA reference.
    Q: (B, nh, S, d_qk), K/V: (B, nkv, S, d)
    """

    b, nh, S, d_qk = Q.shape
    _, nkv, _, d_v = V.shape
    # Expand KV to match Q heads
    head_rep = nh // nkv
    K_exp = K.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_qk)
    V_exp = V.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_v)
    # Use PyTorch’s builtin causal attention
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_exp, V_exp, attn_mask=None, scale=scale, is_causal=is_causal
    )


def run_flash_mla_prefill_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention=False,
    block_size=ttnn.TILE_SIZE,
):
    # Log the test parameters
    logger.debug(f"Running FlashMLA Prefill with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length: {seq_len}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"KV LoRA Rank: {kv_lora_rank}")
    logger.debug(f"Dimensionality of RoPE: {d_rope}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")

    # Paged attention configuration
    paged_attention_cfg = None
    if use_paged_attention:
        assert seq_len % block_size == 0, f"Sequence length must be a multiple of {block_size=} for paged attention."

        max_num_blocks = seq_len // block_size * batch
        paged_attention_cfg = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S, D)
    v = k[..., :kv_lora_rank]  # (B, H, S, D)
    ######################
    ### TT Setup
    #######################

    # Page-related setup
    tt_k_torch = k
    tt_page_table = None
    if paged_attention_cfg:
        page_table = page_table_setup(batch, paged_attention_cfg)
        tt_k_torch = to_paged_cache(
            k,
            page_table,
            paged_attention_cfg,
        )
        tt_k_torch_og = from_paged_cache(
            tt_k_torch,
            page_table,
            paged_attention_cfg,
        )
        assert torch.all(tt_k_torch_og == k), "Paged cache conversion for K failed."

        tt_page_table = ttnn.from_torch(
            page_table,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    q_chunk_size = padded_num_heads
    k_chunk_size = 128

    scale = (kv_lora_rank + d_rope) ** -0.5

    max_start_idx = seq_len // 2

    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

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

    tt_q = ttnn.from_torch(
        q,  # (B, H, S, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        tt_k_torch,  # (B, H, S, D)
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ##########################
    ### FlashMLA Prefill
    ##########################
    if tt_page_table:
        tt_out = ttnn.transformer.chunked_flash_mla_prefill(
            tt_q,
            tt_k,
            kv_lora_rank,
            tt_page_table,
            chunk_start_idx=0,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
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
    tt_back = ttnn.to_torch(tt_out)  # now (B, H_padded, S_padded, D)
    # slice out the padded heads and sequence length; no permute needed
    tt_out_torch = tt_back[:, :nh, :seq_len, :]  # (B, nh, S, D)

    ########################
    ### Validation
    ########################
    out_t = scaled_dot_product_attention_reference(
        q,
        k,
        v,
        scale,
        is_causal=True,
    )

    pcc_threshold = 0.99
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.98

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_t, pcc_threshold)
    logger.debug(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        (2, 1024, 128, 1, 512, 64),
        (2, 8 * 1024, 8, 1, 128, 64),
        (2, 4 * 1024, 64, 1, 256, 0),
        (2, 4 * 1024, 64, 1, 32, 64),
        (8, 4 * 1024, 8, 1, 128, 32),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat8_b, ttnn.bfloat4_b),
    ],
)
@pytest.mark.parametrize(
    "use_paged_attention",
    [
        # False,
        True,
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        32,
        128,
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention,
    block_size,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        use_paged_attention,
        block_size,
    )
