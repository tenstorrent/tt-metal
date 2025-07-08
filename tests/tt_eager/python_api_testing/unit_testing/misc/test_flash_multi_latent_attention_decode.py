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
from models.utility_functions import nearest_y

import ttnn
from loguru import logger
import pytest


def scaled_dot_product_attention_reference(Q, K, V, start_indices, padded_layer_len, scale, is_causal=True):
    b, nh, _, _ = Q.shape  # b, nh, 1, d
    _, nkv, _, _ = K.shape

    attn_mask = None
    if is_causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        assert False, "Non-causal attention is not supported in this function."

    Q_slice = Q[:, :nh, :, :]  # b, nh, 1, d
    K_slice = K[:, :nkv, :padded_layer_len, :]  # b, nkv, S, d
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    V_slice = V[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, d, S
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S
    out = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d

    return out


def run_flash_mla_decode_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_num_cores,
    q_dtype,
    dtype,
):
    # Can't run too many iters, or run out of L1
    num_iters = 5

    # Log the test parameters
    logger.info(f"Running FlashMLA Decode with parameters: ")
    logger.info(f"Batch: {batch}")
    logger.info(f"Sequence Length: {seq_len}")
    logger.info(f"Number of Heads (Q): {nh}")
    logger.info(f"Number of Heads (KV): {nkv}")
    logger.info(f"KV LoRA Rank: {kv_lora_rank}")
    logger.info(f"Dimensionality of RoPE: {d_rope}")
    logger.info(f"Number of Cores for Q Sharding: {q_num_cores}")
    logger.info(f"Query Data Type: {q_dtype}")
    logger.info(f"Key-Value Data Type: {dtype}")

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, 1, kv_lora_rank + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S, D)
    v = k[..., :kv_lora_rank]  # (B, H, S, D)

    ######################
    ### TT Setup
    #######################

    q_chunk_size = 0  # Not used in decode
    k_chunk_size = 128

    scale = (kv_lora_rank + d_rope) ** -0.5

    max_start_idx = seq_len // 2
    start_indices = np.linspace(0, max_start_idx, batch, dtype=np.int32).tolist() if batch > 1 else [max_start_idx]

    padded_layer_len = nearest_y(max_start_idx + 1, k_chunk_size)

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

    # Set up input tensors
    if q_num_cores < 1:  # DRAM
        q_mem_config = ttnn.DRAM_MEMORY_CONFIG
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        num_cores_x, num_cores_y = device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y
        if q_num_cores > num_cores_x * num_cores_y:
            pytest.skip(
                f"Skipping test with q_num_cores {q_num_cores} > device compute grid size {num_cores_x * num_cores_y}."
            )

        if nkv == 1:
            # Batch + nh shard if nkv == 1
            q_num_cores = min(batch * nh, q_num_cores)  # Limit q_num_cores to batch size
        else:
            # Only batch shard if nkv > 1
            q_num_cores = min(batch, q_num_cores)  # Limit q_num_cores to batch size

        block_height = nearest_y(np.prod(q.shape[:-1]) // q_num_cores, ttnn.TILE_SIZE)

        q_core_grid = ttnn.num_cores_to_corerangeset(
            q_num_cores, device.compute_with_storage_grid_size(), row_wise=True
        )

        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, q.shape[-1]),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        out_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, v.shape[-1]),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

    # GQA only supports DRAM memory config for output
    if nkv > 1:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    tt_q = ttnn.from_torch(
        q.permute(2, 0, 1, 3),  # (B, H, S, D) -> (S, B, H, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
    )
    tt_k = ttnn.from_torch(
        k,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_start_indices = ttnn.from_torch(
        torch.tensor(start_indices),
        device=device,
        dtype=ttnn.int32,
    )

    ##########################
    ### FlashMLA Decode
    ##########################
    logger.info(
        f"Running FlashMLA Decode with TT Q shape: {tt_q.shape}, TT K shape: {tt_k.shape}, head_dim_v: {kv_lora_rank}"
    )

    def run_op():
        tt_out = ttnn.transformer.flash_multi_latent_attention_decode(
            tt_q,
            tt_k,
            head_dim_v=kv_lora_rank,
            cur_pos_tensor=tt_start_indices,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=out_mem_config,
        )

        return tt_out

    tt_outs = []
    for i in range(num_iters):  # Check for program cache
        logger.info(f"Running FlashMLA Decode operation iteration {i + 1}/{num_iters}")
        tt_out = run_op()
        tt_outs.append(tt_out)

        # Increment start indices for the next iteration
        ttnn.plus_one(tt_start_indices)

    ########################
    ### Validation
    ########################
    outs = []
    for _ in range(num_iters):
        out_t = scaled_dot_product_attention_reference(
            q,
            k,
            v,
            start_indices,
            padded_layer_len,
            scale,
        )
        outs.append(out_t)

        start_indices = [x + 1 for x in start_indices]

    pcc_threshold = 0.99
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.91
    if dtype == ttnn.bfloat8_b:
        pcc_threshold = 0.98

    for i, (tt_out, out_t) in enumerate(zip(tt_outs, outs)):
        tt_out_torch = ttnn.to_torch(tt_out)[..., :nh, :].permute(1, 2, 0, 3)  # (S, B, H, D) -> (B, H, S, D)

        out_pass, out_pcc = comp_pcc(tt_out_torch, out_t, pcc_threshold)
        logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"

    # Check program cache entries
    num_program_cache_entries = device.num_program_cache_entries()

    # FlashMLA + PlusOne
    assert num_program_cache_entries == 2, f"Expected 2 program cache entries, got {num_program_cache_entries}."


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope, q_num_cores",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope, number of cores to shard q on
    [
        (2, 1024, 128, 1, 512, 64, 8),  # DeepSeek V3 TG full DP
        (2, 1024, 128, 1, 256, 64, 16),
        (2, 1024, 128, 1, 256, 64, 32),
        (2, 1024, 128, 1, 256, 64, 64),
        (8, 1024, 128, 1, 256, 64, 64),
        (8, 1024, 16, 1, 256, 64, 64),
        (8, 1024, 48, 1, 128, 64, 16),
        (2, 1024, 8, 1, 128, 64, 0),
        (2, 1024, 64, 1, 256, 0, 0),
        (2, 1024, 64, 1, 32, 64, 0),
        (8, 1024, 8, 1, 128, 32, 0),
        (16, 1024, 8, 1, 128, 32, 0),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat8_b, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
        (ttnn.bfloat8_b, ttnn.bfloat4_b),
    ],
)
def test_flash_mla_decode(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_num_cores,
    q_dtype,
    dtype,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_decode_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_num_cores,
        q_dtype,
        dtype,
    )


@pytest.mark.parametrize(
    "batch",
    [
        1,  # Single batch
        2,  # Multiple batches
        8,  # Even larger batch size
    ],
)
@pytest.mark.parametrize(
    "seq_len",
    [
        1 * 1024,  # Long sequence length
    ],
)
@pytest.mark.parametrize(
    "nh",
    [
        16,
        32,
        128,
    ],
)
@pytest.mark.parametrize(
    "nkv",
    [
        1,
        8,
        16,
    ],
)
@pytest.mark.parametrize(
    "kv_lora_rank",
    [
        64,
        512,
    ],
)
@pytest.mark.parametrize(
    "d_rope",
    [
        0,
        32,
        128,
    ],
)
@pytest.mark.parametrize(
    "q_num_cores",
    [
        0,  # No sharding
        8,  # Shard across 8 cores
        64,  # Shard across all cores
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_decode_stress(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_num_cores,
    q_dtype,
    dtype,
    function_level_defaults,
    reset_seeds,
):
    # If GQA (nkv > 1), then only batch shard
    # If nkv == 1, then batch * nh shard, unless q_num_cores is 0 (ie, in DRAM)
    num_sharding_cores = batch if nkv > 1 else max(batch, q_num_cores)
    if nh * (kv_lora_rank + d_rope) * nkv / num_sharding_cores >= 8 * 1024:  # found experimentally
        pytest.skip(
            f"Skipping test with large values, due to memory constraints. Got {nh=}, {kv_lora_rank=}, {nkv=}, {batch=}, {q_num_cores=}, {d_rope=}"
        )

    if batch * nh < ttnn.TILE_SIZE and q_num_cores > 0:
        pytest.skip("Skipping test with small batch and nh with q_num_cores > 0.")

    effective_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    if nkv > 1 and nkv % (effective_num_cores / batch) != 0:
        pytest.skip(
            f"Skipping test with nkv {nkv} not divisible by effective_num_cores {effective_num_cores} / batch {batch}."
        )

    run_flash_mla_decode_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_num_cores,
        q_dtype,
        dtype,
    )
