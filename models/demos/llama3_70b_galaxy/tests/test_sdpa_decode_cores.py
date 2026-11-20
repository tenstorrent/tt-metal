# Paged decode SDPA at the TP32 BH per-chip decode shape (8 users, 1 KV head / 8 Q heads, head_dim 128, bf8 KV,
# 64-token pages) on one Blackhole chip, with the model's sub-core-grid layout: Q and output height-sharded on the
# first 8 SDPA cores. Compares core counts (48 = today's sub_core_grids; 64/88 add worker columns 7-10).
#   SDPA_CORES=64 SDPA_POS=4000 pytest models/demos/llama3_70b_galaxy/tests/test_sdpa_decode_cores.py -s
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

B, NKV, NQ, D, BLOCK = 8, 1, 8, 128, 64


def _grid(n):
    sub48 = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    wide = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(10, 9)),
        ]
    )
    return ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), n, sub48 if n <= 48 else wide, row_wise=True
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 50_000_000}], indirect=True)
def test_sdpa_decode_cores(device, device_params):
    n_cores = int(os.environ.get("SDPA_CORES", "48"))
    pos = int(os.environ.get("SDPA_POS", "4000"))
    max_seq = ((pos // BLOCK) + 2) * BLOCK
    torch.manual_seed(0)
    nblk = max_seq // BLOCK
    k_u = torch.randn(B, NKV, max_seq, D).bfloat16().float()
    v_u = torch.randn(B, NKV, max_seq, D).bfloat16().float()
    page_table = torch.randperm(B * nblk, dtype=torch.int32).reshape(B, nblk)

    def paged(t):
        out = torch.zeros(B * nblk, NKV, BLOCK, D)
        for b in range(B):
            for vb in range(nblk):
                out[int(page_table[b, vb])] = t[b, :, vb * BLOCK : (vb + 1) * BLOCK, :]
        return out

    k_tt = ttnn.from_torch(paged(k_u), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    v_tt = ttnn.from_torch(paged(v_u), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    pt_tt = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32)
    cur_pos = torch.full((B,), pos, dtype=torch.int32)
    cp_tt = ttnn.from_torch(cur_pos, device=device, dtype=ttnn.int32)

    grid = _grid(n_cores)
    first8 = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), B, grid, row_wise=True)
    shard_cfg = ttnn.create_sharded_memory_config(
        shape=(32, D), core_grid=first8, strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
    )
    q = torch.randn(1, B, NQ, D).bfloat16().float()
    q_tt = ttnn.from_torch(
        torch.nn.functional.pad(q, (0, 0, 0, 32 - NQ)),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=shard_cfg,
    )
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 6) if n_cores == 48 else (n_cores // 8, 8),
        sub_core_grids=grid,
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )

    def run():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            k_tt,
            v_tt,
            cur_pos_tensor=cp_tt,
            page_table_tensor=pt_tt,
            scale=D**-0.5,
            program_config=cfg,
            compute_kernel_config=ckc,
            memory_config=shard_cfg,
        )

    out = run()
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out)[:, :, :NQ, :].float()
    ref = torch.zeros(1, B, NQ, D)
    for b in range(B):
        s = (q[0, b] @ k_u[b, 0, : pos + 1].T) * D**-0.5
        ref[0, b] = torch.softmax(s, -1) @ v_u[b, 0, : pos + 1]
    _, pcc = comp_pcc(ref, got.reshape(ref.shape), 0.99)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [run() for _ in range(20)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / 100 * 1e6
    ttnn.release_trace(device, tid)
    logger.info(f"SDPADEC cores={n_cores} pos={pos}: {us:.1f} us pcc={pcc:.4f}")
