# SPDX-License-Identifier: Apache-2.0
"""Paged SDPA decode calibration sweep (revamp T2.8), issued as tt_transformers issues it.

Op: ttnn.transformer.paged_scaled_dot_product_attention_decode (attention.py:860-871 on origin/main) with
program config ttnn.SDPAProgramConfig(grid (8,8) unless DEC_GRID, exp_approx_mode False, q/k chunk 0 = auto)
(model_config.py get_attn_sdpa_decode_program_config) and compute config compute_kernel_config_hifi2 (HiFi2,
math_approx True, fp32_dest_acc_en True, packer_l1_acc True). Paged K/V cache [max_num_blocks, nkv, block 32, d]
in DRAM, page table [B, blocks_per_user] int32 (random permutation, as the demo), Q [1, B, nh, d] bf16 TILE.
Env: DEC_B (32), DEC_POS (comma list of cache positions), DEC_GRID (8x8), DEC_KV_DTYPE (bfp8_b|bfloat16),
DEC_NH 32, DEC_NKV 8, DEC_D 128, DEC_BLOCK 32, DEC_MAXSEQ (default 2*max pos, >= 8192), DEC_ITERS 3.
One process per DEC_POS list; each position is DEC_ITERS invocations.
"""
import os

import torch


def test_decode_sweep(device):
    import ttnn

    b = int(os.environ.get("DEC_B", "32"))
    nh = int(os.environ.get("DEC_NH", "32")); nkv = int(os.environ.get("DEC_NKV", "8")); d = int(os.environ.get("DEC_D", "128"))
    block = int(os.environ.get("DEC_BLOCK", "32"))
    positions = [int(x) for x in os.environ.get("DEC_POS", "1024").split(",")]
    max_seq = int(os.environ.get("DEC_MAXSEQ", str(max(8192, 2 * max(positions)))))
    iters = int(os.environ.get("DEC_ITERS", "3"))
    kv_dtype = {"bfp8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}[os.environ.get("DEC_KV_DTYPE", "bfp8_b")]
    grid_env = os.environ.get("DEC_GRID", "8x8")
    gx, gy = [int(v) for v in grid_env.split("x")]
    blocks_per_user = max_seq // block
    max_num_blocks = b * blocks_per_user
    print(f"\n[decode_sweep] b={b} nh={nh} nkv={nkv} d={d} block={block} max_seq={max_seq} blocks={max_num_blocks} "
          f"kv_dtype={os.environ.get('DEC_KV_DTYPE', 'bfp8_b')} grid={gx}x{gy} positions={positions} iters={iters}", flush=True)
    torch.manual_seed(0)
    k_paged = torch.randn(max_num_blocks, nkv, block, d).bfloat16().float()
    v_paged = torch.randn(max_num_blocks, nkv, block, d).bfloat16().float()
    k_tt = ttnn.as_tensor(k_paged, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_tt = ttnn.as_tensor(v_paged, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    page_table = torch.randperm(max_num_blocks, dtype=torch.int32).reshape(b, blocks_per_user)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)
    q = torch.randn(1, b, nh, d).bfloat16().float()
    q_tt = ttnn.Tensor(q, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    scale = 1.0 / (d ** 0.5)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True)
    for pos in positions:
        cur_pos_tt = ttnn.Tensor(torch.tensor([pos for _ in range(b)], dtype=torch.int32), ttnn.int32).to(device)
        for it in range(iters):
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, k_tt, v_tt, page_table_tensor=page_table_tt, cur_pos_tensor=cur_pos_tt, scale=scale,
                program_config=pc, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(device)
            out.deallocate()
            print(f"[decode_sweep] pos={pos} iter {it} done", flush=True)
        cur_pos_tt.deallocate()
    for t in (q_tt, k_tt, v_tt, page_table_tt):
        t.deallocate()
