# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal repro for paged_scaled_dot_product_attention_decode crash on
Blackhole with batch=32, num_kv_heads=8.

Bug: the core allocation math in sdpa_decode_program_factory.cpp does bad
integer division when num_cores_available doesn't divide evenly by B:

    num_cores_available = 110  (Blackhole 10x11 grid)
    B = 32
    num_kv_heads = 8

    num_cores_per_batch_uncapped = 110 / 32 = 3
    num_heads_per_core = ceil(8 / 3) = 3    <-- but 8 % 3 != 0
    num_cores_per_head = max(1, 3 / 8) = 1
    num_cores_per_batch = 1 * 8 / 3 = 2     <-- integer truncation
    num_active_cores = 1 * 8 * 32 / 3 = 85  <-- not evenly divisible
    num_reducer_cores = 8 * 32 / 3 = 85

    But the output loop iterates num_active_cores and indexes by
    core_id / num_cores_per_batch, which can exceed B when the division
    doesn't evenly partition heads across cores.

Shapes from Llama-3.2-3B decode graph (graph_3_ttnn.mlir line 300):
    Q:          1x32x32x64  bf16  DRAM interleaved tile
    K cache:    454x8x32x64 bf16  DRAM interleaved tile
    V cache:    454x8x32x64 bf16  DRAM interleaved tile
    page_table: 32x4        int32 DRAM interleaved row-major
    cur_pos:    32           int32 DRAM interleaved row-major

Run: python perf_debug/repro_sdpa_decode_batch32.py
"""

import math
import torch
import ttnn

# Model parameters matching Llama-3.2-3B decode
BATCH = 32
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 32
MAX_BLOCKS_PER_SEQ = 4  # page_table width from IR: 32x4
NUM_BLOCKS = MAX_BLOCKS_PER_SEQ * BATCH  # 128 blocks needed, but IR shows 454


def main():
    device = ttnn.open_device(device_id=0)

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    print(f"Device grid: {grid.x}x{grid.y} = {num_cores} cores")
    print(f"B={BATCH}, num_kv_heads={NUM_KV_HEADS}")

    # Reproduce the C++ core allocation math to show the bug
    num_cores_per_batch_uncapped = num_cores // BATCH
    num_heads_per_core = max(1, math.ceil(NUM_KV_HEADS / num_cores_per_batch_uncapped))
    num_cores_per_head = max(1, num_cores_per_batch_uncapped // NUM_KV_HEADS)
    num_cores_per_batch = num_cores_per_head * NUM_KV_HEADS // num_heads_per_core
    num_active_cores = num_cores_per_head * NUM_KV_HEADS * BATCH // num_heads_per_core
    num_reducer_cores = NUM_KV_HEADS * BATCH // num_heads_per_core

    print(f"\nCore allocation math:")
    print(f"  num_cores_per_batch_uncapped = {num_cores} / {BATCH} = {num_cores_per_batch_uncapped}")
    print(f"  num_heads_per_core = ceil({NUM_KV_HEADS} / {num_cores_per_batch_uncapped}) = {num_heads_per_core}")
    print(f"  num_cores_per_head = max(1, {num_cores_per_batch_uncapped} / {NUM_KV_HEADS}) = {num_cores_per_head}")
    print(f"  num_cores_per_batch = {num_cores_per_head} * {NUM_KV_HEADS} / {num_heads_per_core} = {num_cores_per_batch}")
    print(f"  num_active_cores = {num_cores_per_head} * {NUM_KV_HEADS} * {BATCH} / {num_heads_per_core} = {num_active_cores}")
    print(f"  num_reducer_cores = {NUM_KV_HEADS} * {BATCH} / {num_heads_per_core} = {num_reducer_cores}")

    if NUM_KV_HEADS % num_heads_per_core != 0:
        print(f"\n  BUG: num_kv_heads ({NUM_KV_HEADS}) % num_heads_per_core ({num_heads_per_core}) != 0")
        print(f"  This causes uneven head-to-core mapping and output loop overflow")

    if num_active_cores > num_cores:
        print(f"  BUG: num_active_cores ({num_active_cores}) > num_cores_available ({num_cores})")

    # Use more blocks than strictly needed (matching IR's 454 blocks)
    total_blocks = 454

    # Q: [1, B, num_q_heads, head_dim] -- padded to tile in dim 2
    q_pt = torch.randn(1, BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)

    # K/V cache: [total_blocks, num_kv_heads, block_size, head_dim]
    k_pt = torch.randn(total_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    v_pt = torch.randn(total_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)

    # Page table: [B, max_blocks_per_seq] with valid block indices
    page_table_pt = torch.stack([
        torch.randperm(total_blocks, dtype=torch.int32)[:MAX_BLOCKS_PER_SEQ]
        for _ in range(BATCH)
    ])

    # Current positions: each sequence at position 1 (early in decode)
    cur_pos_pt = torch.ones(BATCH, dtype=torch.int32)

    print(f"\nTensor shapes:")
    print(f"  Q:          {list(q_pt.shape)}")
    print(f"  K cache:    {list(k_pt.shape)}")
    print(f"  V cache:    {list(v_pt.shape)}")
    print(f"  page_table: {list(page_table_pt.shape)}")
    print(f"  cur_pos:    {list(cur_pos_pt.shape)}")

    dram = ttnn.DRAM_MEMORY_CONFIG

    tt_q = ttnn.as_tensor(q_pt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_k = ttnn.as_tensor(k_pt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v_pt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_page_table = ttnn.Tensor(page_table_pt, ttnn.int32).to(device)
    tt_cur_pos = ttnn.Tensor(cur_pos_pt, ttnn.int32).to(device)

    print("\nCalling paged_scaled_dot_product_attention_decode...")
    print("Expected: crash or assertion due to core allocation math overflow")

    try:
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            tt_page_table,
            cur_pos_tensor=tt_cur_pos,
            scale=HEAD_DIM**-0.5,
        )
        result = ttnn.to_torch(out)
        print(f"Output shape: {list(result.shape)} -- no crash (unexpected)")
    except Exception as e:
        print(f"FAILED as expected: {e}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
