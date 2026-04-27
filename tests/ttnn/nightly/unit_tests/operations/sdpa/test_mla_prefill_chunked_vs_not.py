# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose_and_pcc,
)
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    nearest_n,
    nearest_pow_2,
    page_table_setup,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


def run_flash_mla_prefill_chunked_vs_nonchunked(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    num_chunks,
    block_size,
):
    """
    Compare `ttnn.transformer.chunked_flash_mla_prefill` against
    `ttnn.transformer.flash_mla_prefill` by running the non-chunked op over the
    full sequence and the chunked op over `num_chunks` equal-size chunks of the
    same sequence, then comparing the concatenated chunked output to the
    non-chunked output.
    """
    logger.debug(f"Running FlashMLA Prefill Chunked vs Non-chunked with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length: {seq_len}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"KV LoRA Rank: {kv_lora_rank}")
    logger.debug(f"Dimensionality of RoPE: {d_rope}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")
    logger.debug(f"Number of Chunks: {num_chunks}")
    logger.debug(f"Block Size: {block_size}")

    assert seq_len % num_chunks == 0, f"seq_len {seq_len} must be divisible by num_chunks {num_chunks}"
    chunk_size = seq_len // num_chunks

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    q_chunk_size = padded_num_heads
    k_chunk_size = 128

    align = max(q_chunk_size, k_chunk_size)
    assert chunk_size % align == 0, (
        f"chunk_size {chunk_size} must be a multiple of max(q_chunk_size={q_chunk_size}, "
        f"k_chunk_size={k_chunk_size})={align}"
    )
    assert seq_len % block_size == 0, f"seq_len {seq_len} must be divisible by block_size {block_size}"

    ######################
    ### Tensor Setup
    ######################
    tt_q = ttnn.from_torch(
        torch.randn(batch, nh, seq_len, kv_lora_rank + d_rope).float(),
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float(),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    scale = (kv_lora_rank + d_rope) ** -0.5

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

    ######################
    ### Non-chunked reference
    ######################

    tt_nonchunked_out = ttnn.transformer.flash_mla_prefill(
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
    nonchunked_torch = ttnn.to_torch(tt_nonchunked_out)[:, :nh, :seq_len, :]
    ttnn.deallocate(tt_nonchunked_out)

    ######################
    ### Chunked path
    ######################

    max_num_blocks = seq_len // block_size * batch
    paged_attention_cfg = PagedAttentionConfig(
        block_size=block_size,
        max_num_blocks=max_num_blocks,
    )
    tt_page_table = ttnn.from_torch(
        page_table_setup(batch, paged_attention_cfg),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    kvpe_cache = ttnn.zeros(
        [max_num_blocks, nkv, block_size, kv_lora_rank + d_rope],
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    chunked_outputs = []
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size

        tt_k_chunk = tt_k[:, :, chunk_start:chunk_end, :]
        tt_chunk_page_table = tt_page_table[:, chunk_start // block_size : chunk_end // block_size]
        for b in range(batch):
            ttnn.experimental.paged_fill_cache(
                kvpe_cache,
                tt_k_chunk[b : b + 1],
                tt_chunk_page_table,
                batch_idx=b,
            )
        ttnn.deallocate(tt_k_chunk, force=False)
        ttnn.deallocate(tt_chunk_page_table, force=False)

        tt_q_chunk = tt_q[:, :, chunk_start:chunk_end, :]
        tt_chunk_out = ttnn.transformer.chunked_flash_mla_prefill(
            tt_q_chunk,
            kvpe_cache,
            kv_lora_rank,
            tt_page_table,
            chunk_start_idx=chunk_start,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunked_out_torch = ttnn.to_torch(tt_chunk_out)[:, :nh, :chunk_size, :]
        chunked_outputs.append(chunked_out_torch)

        ttnn.deallocate(tt_q_chunk)
        ttnn.deallocate(tt_chunk_out)

    ttnn.deallocate(kvpe_cache)
    chunked_torch = torch.cat(chunked_outputs, dim=2)

    ######################
    ### Compare chunked vs non-chunked
    ######################
    pcc_threshold = 0.999
    out_pass, out_pcc = comp_allclose_and_pcc(nonchunked_torch, chunked_torch, pcc=pcc_threshold)
    logger.debug(f"num_chunks={num_chunks} chunked vs non-chunked PCC: {out_pcc}")

    assert out_pass, (
        f"chunked_flash_mla_prefill (num_chunks={num_chunks}) output mismatch "
        f"vs flash_mla_prefill: PCC {out_pcc} < {pcc_threshold}"
    )


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    [
        (1, 768, 128, 1, 512, 64),
        (4, 768, 16, 1, 512, 64),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype, block_size",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, 64),
    ],
)
@pytest.mark.parametrize(
    "num_chunks",
    [1, 3],
    ids=["1chunk", "3chunks"],
)
def test_chunked_flash_mla_prefill_vs_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    block_size,
    num_chunks,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_chunked_vs_nonchunked(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        num_chunks,
        block_size,
    )
