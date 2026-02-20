# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
import pytest

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


def run_flash_mla_prefill_impl(
    device,
    batch,
    seq_len_q,
    seq_len_kv,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    v_head_dim,
    q_chunk_first,
    v_chunk_first,
    q_chunk_second,
    v_chunk_second,
):
    # Log the test parameters
    logger.debug(f"Running FlashMLA Prefill with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length (Q): {seq_len_q}")
    logger.debug(f"Sequence Length (KV): {seq_len_kv}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"KV LoRA Rank: {kv_lora_rank}")
    logger.debug(f"Dimensionality of RoPE: {d_rope}")
    logger.debug(f"V Head Dim: {v_head_dim}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")
    logger.debug(f"Query Chunk Size First: {q_chunk_first}")
    logger.debug(f"Value Chunk Size First: {v_chunk_first}")
    logger.debug(f"Query Chunk Size Second: {q_chunk_second}")
    logger.debug(f"Value Chunk Size Second: {v_chunk_second}")

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len_q, kv_lora_rank + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, 1, seq_len_kv, kv_lora_rank + d_rope).float()  # (B, H, S, D)
    v = k[..., :kv_lora_rank]  # (B, H, S, D)
    v_out = torch.randn(batch, nh, kv_lora_rank, v_head_dim).float()
    ######################
    ### TT Setup
    #######################

    tt_k_torch = k

    # Workaround since regular SDPA doesnt allow causality if k and v have different seq_lengths.
    # Need to add this support as part of ring attention work somehow, but for now, pretend it isn't causal.
    is_causal = seq_len_q == seq_len_kv

    scale = (kv_lora_rank + d_rope) ** -0.5
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_first,
        k_chunk_size=v_chunk_first,
        exp_approx_mode=False,
    )

    sdpa_program_config_second = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_second,
        k_chunk_size=v_chunk_second,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    tt_q = ttnn.from_torch(
        q,  # (B, H, S, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v_latent = ttnn.from_torch(
        v,  # (B, H, S, D)
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
    tt_v_out = ttnn.from_torch(
        v_out,  # (B, H, D, D_out)
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ##########################
    ### FlashMLA Prefill
    ##########################

    # K, V in latent space
    # post-multiply sdpa output with W_v
    tt_flash_mla_prefill_out = ttnn.transformer.flash_mla_prefill(
        tt_q,
        tt_k,
        head_dim_v=kv_lora_rank,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        attn_mask=None,
        is_causal=is_causal,
    )
    tt_out = ttnn.linear(tt_flash_mla_prefill_out, tt_v_out)

    # Premultiply V by W_v
    # workaround for #37416
    tt_v_latent_post_repeat = ttnn.repeat(tt_v_latent, [1, nh, 1, 1])
    tt_v_embedding = ttnn.linear(tt_v_latent_post_repeat, tt_v_out, dtype=dtype)

    tt_new_sdpa_out = ttnn.transformer.flash_mla_prefill(
        tt_q,
        tt_k,
        tt_v_embedding,
        scale=scale,
        program_config=sdpa_program_config_second,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        is_causal=is_causal,
        attn_mask=None,
    )
    ref_tt_impl_out_torch = ttnn.to_torch(tt_out)
    new_tt_impl_out_torch = ttnn.to_torch(tt_new_sdpa_out)

    pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(ref_tt_impl_out_torch, new_tt_impl_out_torch, pcc_threshold)
    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, nh, nkv, kv_lora_rank, d_rope, v_head_dim, q_chunk_first, v_chunk_first, q_chunk_second, v_chunk_second",
    [
        # cases from deepseek
        # TP=8, SP=1, seq_len = 1k
        (1, 1024, 1024, 16, 16, 512, 64, 128, 32, 32, 32, 32),
        # TP=8, SP=1, seq_len = 4k
        (1, 4096, 4096, 16, 16, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=1, seq_len = 1k
        (1, 1024, 1024, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=1, seq_len = 4k
        (1, 4096, 4096, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=32, seq_len = 16k
        (1, 512, 16384, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=32, seq_len = 32k
        (1, 1024, 32768, 32, 32, 512, 64, 128, 32, 32, 32, 32),
        # TP=4, SP=32, seq_len = 128k
        (1, 4096, 131072, 32, 32, 512, 64, 128, 128, 256, 128, 512),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len_q,
    seq_len_kv,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    v_head_dim,
    q_chunk_first,
    v_chunk_first,
    q_chunk_second,
    v_chunk_second,
    function_level_defaults,
    reset_seeds,
):
    if device.arch() == ttnn.device.Arch.WORMHOLE_B0:
        if seq_len_kv == 131072:
            pytest.skip("Skip WH test due to pcc failure. Should properly set k/v chunk sizes for this case.")
        q_chunk_first = 32
        v_chunk_first = 32
        q_chunk_second = 32
        v_chunk_second = 32
        logger.info("Using wormhole device, setting chunk sizes to 32")

    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len_q,
        seq_len_kv,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        v_head_dim,
        q_chunk_first,
        v_chunk_first,
        q_chunk_second,
        v_chunk_second,
    )
