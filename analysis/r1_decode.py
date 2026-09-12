# SPDX-License-Identifier: Apache-2.0
"""R1 non-paged sdpa_decode, the form analysis/decode_latency_sweep.py measured (the model's non-paged constants):
Q [1, b, nh, d] bf16 TILE DRAM; K, V [b, nkv, cache_len, d] in DEC_KV_DTYPE (bfloat16 default, bfp8_b option) DRAM;
cur_pos = cache_len - 1 for every user (attend the full cache); program config q_chunk = padded nh, k_chunk 128 (32 below
cache 512), exp_approx_mode False, full grid; compute HiFi2, math_approx False, fp32_dest_acc_en False, packer_l1_acc False;
output DRAM. Env: DEC_B (8), DEC_NH 32, DEC_NKV 8, DEC_D 128, DEC_KV_DTYPE, DEC_POS (comma list of cache lengths), DEC_ITERS 3.
Device timing through tracy (this file does no host timing).
"""
import os

import torch


def test_r1_decode(device):
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    b = int(os.environ.get("DEC_B", "8")); nh = int(os.environ.get("DEC_NH", "32")); nkv = int(os.environ.get("DEC_NKV", "8"))
    d = int(os.environ.get("DEC_D", "128")); iters = int(os.environ.get("DEC_ITERS", "3"))
    kv_dtype = {"bfp8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}[os.environ.get("DEC_KV_DTYPE", "bfloat16")]
    seqs = [int(x) for x in os.environ.get("DEC_POS", "1024").split(",")]
    padded_nh = ((nh + 31) // 32) * 32
    grid = device.compute_with_storage_grid_size(); grid_size = (grid.x, grid.y)
    scale = d ** -0.5
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                          fp32_dest_acc_en=False, packer_l1_acc=False)
    dram = ttnn.DRAM_MEMORY_CONFIG
    print(f"\n[r1_decode] b={b} nh={nh} nkv={nkv} d={d} kv_dtype={os.environ.get('DEC_KV_DTYPE', 'bfloat16')} grid={grid_size} seqs={seqs} iters={iters}", flush=True)
    for s in seqs:
        k_chunk = 128 if s >= 512 else 32
        K = fa_rand(b, nkv, s, d); V = fa_rand(b, nkv, s, d); Q = fa_rand(1, b, nh, d)
        tt_K = ttnn.as_tensor(K, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        tt_V = ttnn.as_tensor(V, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        cur_pos = ttnn.Tensor(torch.tensor([s - 1 for _ in range(b)]), ttnn.int32).to(device)
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid_size, q_chunk_size=padded_nh, k_chunk_size=k_chunk, exp_approx_mode=False)
        for it in range(iters):
            o = ttnn.transformer.scaled_dot_product_attention_decode(tt_Q, tt_K, tt_V, cur_pos_tensor=cur_pos, scale=scale,
                                                                     program_config=pc, compute_kernel_config=ck, memory_config=dram)
            ttnn.synchronize_device(device); o.deallocate(); print(f"[r1_decode] cache={s} iter {it} done", flush=True)
        for t in (tt_Q, tt_K, tt_V, cur_pos):
            t.deallocate()
