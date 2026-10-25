# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""R0 screen: is post_combine_reduce (PCR) per-core bound?

PCR splits work only by 32-token chunks, so the production call (640 tokens/chip) runs on 20 cores (247 us in the
L78 profile, ~100 GB/s total). If the time stays flat as tokens (and so cores) grow, the op is per-core bound and a
split over (chunk, emb segment) onto the full grid is worth ~4-5x. Single chip, GLM shapes (emb 6144, top-8,
256 experts), trace-replay timing. Local-expert fraction: 1/4 of the experts (one TP column's share, ~2 of 8 slots
active per token, as on the 8x4 mesh) and all-local (8 of 8).
Lines start with "PCR_R0 ".
"""

import time

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config

EMB = GLM53Config.EMB_SIZE
TOPK = GLM53Config.NUM_EXPERTS_PER_TOKEN
N_EXPERTS = GLM53Config.NUM_ROUTED_EXPERTS
N_INNER, N_REPLAY = 8, 20


def log(*a):
    print("PCR_R0", *a, flush=True)


def _time(device, fn):
    out = fn()
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(N_INNER):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(N_REPLAY):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) / (N_REPLAY * N_INNER) * 1e6)
    ttnn.release_trace(device, tid)
    return sorted(ts)[1]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole")
@pytest.mark.timeout(0)
def test_pcr_r0(device):
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    log(f"grid {grid.x}x{grid.y}")
    rm = dict(layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)
    for local_frac in (4, 1):
        table = torch.full((N_EXPERTS,), -1, dtype=torch.int32)
        local = torch.randperm(N_EXPERTS)[: N_EXPERTS // local_frac]
        table[local] = torch.randint(0, 8, (local.numel(),), dtype=torch.int32)
        table_tt = ttnn.from_torch(table, **rm)
        for tokens in (640, 1280, 2560, 3200):
            combine = torch.randn(1, tokens, TOPK, EMB).to(torch.bfloat16)
            weights = torch.rand(1, tokens, TOPK, 1).to(torch.bfloat16)
            idx = torch.stack([torch.randperm(N_EXPERTS)[:TOPK] for _ in range(tokens)]).unsqueeze(0)
            active = (table[idx.long()] >= 0).float().sum().item() / tokens
            c_tt = ttnn.from_torch(combine, **rm)
            w_tt = ttnn.from_torch(weights, **rm)
            i_tt = ttnn.from_torch(idx.to(torch.uint16), dtype=ttnn.uint16, **rm)

            def run():
                return ttnn.experimental.deepseek_prefill.post_combine_reduce(
                    c_tt, w_tt, i_tt, table_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

            us = _time(device, run)
            got = ttnn.to_torch(run()).float().reshape(tokens, EMB)
            mask = (table[idx.long()] >= 0).to(torch.float32).reshape(1, tokens, TOPK, 1)
            ref = (combine.float() * weights.float() * mask).sum(dim=2).reshape(tokens, EMB)
            pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
            cores = min(tokens // 32, grid.x * grid.y)
            read_mb = tokens * active * EMB * 2 / 1e6
            log(
                f"local=1/{local_frac} tokens={tokens} cores={cores} active_slots/token={active:.2f} "
                f"time={us:.1f} us read={read_mb:.1f} MB -> {read_mb / us * 1e3:.0f} GB/s, per-core "
                f"{read_mb / us * 1e3 / cores:.1f} GB/s, pcc={pcc:.5f}"
            )
            for t in (c_tt, w_tt, i_tt):
                ttnn.deallocate(t)
