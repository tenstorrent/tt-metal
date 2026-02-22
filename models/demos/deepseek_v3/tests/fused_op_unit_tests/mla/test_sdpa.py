# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Flash MLA SDPA decode with DeepSeek V3 shapes and settings from mla1d.py.

Tests paged_flash_multi_latent_attention_decode using the exact shapes,
memory configs, and compute kernel configs that MLA1D.decode_model_config generates.
"""


import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import (
    page_table_setup,
    scaled_dot_product_attention_reference,
    to_paged_cache,
)

# DeepSeek V3 MLA dimensions
NUM_HEADS = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
V_HEAD_DIM = 128
KVPE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
NKV = 1  # MLA always has nkv=1


def build_decode_flash_mla_config(device, batch, num_heads, seq_len, k_chunk_size=256):
    """Build the flash_mla config dict and Q memory config matching mla1d.py decode_model_config.

    Reproduces the SDPA config from MLA1D.decode_model_config lines 496-549.

    P1: max_cores_per_head_batch is sized to the compile-time chunk count so the
        factory never allocates idle K-split workers.
    P2: two program configs are built (short/long context) and stored under
        program_config_short / program_config_long.  _fwd_decode_flash_mla picks
        the right one at runtime based on max(cur_pos).
    """
    grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size.x * grid_size.y

    # Q sharding: match mla1d.py lines 514-527
    q_num_cores = min(batch * num_heads, num_cores)
    block_height = nearest_y((batch * num_heads) // q_num_cores, ttnn.TILE_SIZE)
    block_width = KVPE_DIM

    q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)

    q_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, block_width),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    # Output sharding: match mla1d.py lines 528-533
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(block_height, KV_LORA_RANK),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    # P1: compute max K-chunks for the configured sequence length so
    # max_cores_per_head_batch never exceeds the number of actual chunks.
    # Sweep results (HiFi4, batch=4, QPF=4):
    #   seq=128  kcs=128: mcphb=1 optimal (27.3µs), K-split adds REDUCE overhead for 1 chunk
    #   seq=1024 kcs=256: mcphb=2 optimal (90.3µs), balances parallelism vs REDUCE sync
    max_start_idx = seq_len // 2
    padded_layer_len = nearest_y(max_start_idx + 1, k_chunk_size)
    k_num_chunks_max = max(1, padded_layer_len // k_chunk_size)

    # mcphb=1 for single chunk (no benefit from K-split), mcphb=2 for multi-chunk
    max_cores_long = min(2, k_num_chunks_max)

    # Short-context config — no K-split, every core is a standalone reducer
    sdpa_program_config_short = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    )

    # Long-context config — K-split across 2 workers per group
    sdpa_program_config_long = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores_long,
    )

    # Compute kernel config: match mla1d.py lines 507-512
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Scale: match mla1d.py lines 535-538 (without rope_factor adjustment for simplicity)
    scale = QK_HEAD_DIM**-0.5

    # flash_mla config dict: both program configs stored; _fwd_decode_flash_mla selects at runtime
    flash_mla_config = {
        "head_dim_v": KV_LORA_RANK,
        "scale": scale,
        "program_config_short": sdpa_program_config_short,
        "program_config_long": sdpa_program_config_long,
        "compute_kernel_config": compute_kernel_config,
        "memory_config": out_mem_config,
        "k_chunk_size": k_chunk_size,
    }

    return flash_mla_config, q_mem_config


@pytest.mark.parametrize(
    "batch, seq_len",
    [
        (4, 128),  # DeepSeek V3 per-device batch, short seq (chain active)
        (4, 1024),  # DeepSeek V3 per-device batch, long seq (chain disabled, K-split)
    ],
)
@pytest.mark.parametrize("block_size", [32])
def test_mla1d_sdpa_decode(
    device,
    batch,
    seq_len,
    block_size,
    function_level_defaults,
    reset_seeds,
):
    """Test paged Flash MLA decode with DeepSeek V3 shapes from mla1d.py.

    Calls MLA1D._fwd_decode_flash_mla which invokes
    ttnn.transformer.paged_flash_multi_latent_attention_decode
    with the same config mla1d.py decode_model_config produces.

    Shapes (per-device, matching mla1d comments):
        Q:      [1, batch, 128, 576]  height sharded
        Cache:  [max_num_blocks, 1, block_size, 576]  DRAM bfloat8_b
        Output: [1, batch, 128, 512]  height sharded
    """
    nh = NUM_HEADS

    # Build SDPA config matching mla1d.py (P1/P2: sized to seq_len, two configs)
    # Use k_chunk_size=128 for short seqs to avoid padding beyond K tensor length
    kcs = 128 if seq_len <= 256 else 256
    flash_mla_config, q_mem_config = build_decode_flash_mla_config(device, batch, nh, seq_len, k_chunk_size=kcs)
    cfg = {"flash_mla": flash_mla_config}
    scale = flash_mla_config["scale"]

    # Paged attention setup
    assert seq_len % block_size == 0
    max_num_blocks = seq_len // block_size * batch
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    # Torch reference tensors
    q = torch.randn(batch, nh, 1, KVPE_DIM).float()
    k = torch.randn(batch, NKV, seq_len, KVPE_DIM).float()
    v = k[..., :KV_LORA_RANK]

    # Paged cache conversion
    page_table = page_table_setup(batch, paged_cfg)
    tt_k_paged = to_paged_cache(k, page_table, paged_cfg)

    # Position indices (spread across sequence)
    max_start_idx = seq_len // 2
    start_indices = np.linspace(0, max_start_idx, batch, dtype=np.int32).tolist() if batch > 1 else [max_start_idx]
    padded_layer_len = nearest_y(max_start_idx + 1, flash_mla_config["k_chunk_size"])

    # TT tensors
    tt_q = ttnn.from_torch(
        q.permute(2, 0, 1, 3),  # (B, H, S=1, D) -> (S=1, B, H, D) for decode
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
    )

    kvpe_cache = ttnn.from_torch(
        tt_k_paged,  # (max_num_blocks, nkv=1, block_size, KVPE_DIM)
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_page_table = ttnn.from_torch(
        page_table,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    position_idxs = ttnn.from_torch(
        torch.tensor(start_indices),
        device=device,
        dtype=ttnn.int32,
    )

    # Run through MLA1D._fwd_decode_flash_mla (the actual mla1d.py code)
    attn_out = MLA1D._fwd_decode_flash_mla(
        tt_q, kvpe_cache, tt_page_table, position_idxs, cfg, max_cur_pos=max_start_idx
    )

    # Torch reference
    ref_out = scaled_dot_product_attention_reference(q, k, v, start_indices, padded_layer_len, scale)

    # Compare: output is (S=1, B_padded, H_padded, D) -> (B, H, S, D)
    tt_out_torch = ttnn.to_torch(attn_out)[..., :nh, :].permute(1, 2, 0, 3)

    out_pass, out_pcc = comp_pcc(tt_out_torch, ref_out, 0.98)
    logger.info(f"Decode SDPA PCC: {out_pcc}")
    assert out_pass, f"Decode SDPA output mismatch: PCC {out_pcc} < 0.98"
