# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PLE key+value merged matmul (K=2560, N=1600, bf16) through ``mm_stream2.linear`` vs the stock DRAM-sharded program: torch.equal
(int16 view) of row 0 on random inputs, then traced slope us of both."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import traced_wall_us
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mm_stream2 as m2

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="set QWEN38_FUSED_DEVICE_TEST=1"
)

RESULTS = Path(os.environ.get("X60_RESULTS", "/tmp/mm2_plekv.jsonl"))
SEEDS = [int(s) for s in os.environ.get("X60_SEEDS", "0,1,2,3,4,5").split(",")]
K, N, CORES, WORKERS = 2560, 1600, 8, 1  # stock activation cores and workers per bank of the PLE kv matmul


def act_for(device, act_cfg, seed):
    g = torch.Generator().manual_seed(900 + seed)
    a = torch.zeros(1, 1, 1, K)
    a[:, :, 0, :] = torch.randn(K, generator=g) * 0.5 * (1 + 0.3 * seed)
    return ttnn.to_memory_config(
        ttnn.from_torch(a.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device), act_cfg
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_plekv_vs_stock(mesh_device):
    device = mesh_device
    os.environ["QWEN38_MM_STREAM2"] = "1"
    act_cfg, prog = dm.dram_sharded_matmul_configs(device, K, N, num_cores=CORES, num_workers_per_dram_bank=WORKERS)
    wcfg = dm.dram_sharded_weight_memory_config(device, K, N, num_workers_per_dram_bank=WORKERS)
    ccfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, dm.dense_math_fidelity_name(ttnn.bfloat16)),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    kw = dict(memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, program_config=prog, compute_kernel_config=ccfg)
    res = []
    for wseed in (500, 501):
        g = torch.Generator().manual_seed(wseed)
        w = (torch.randn(1, 1, K, N, generator=g) * 0.02).to(torch.bfloat16)
        weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=wcfg)
        before = m2.CALLS.get((K, N), 0)
        bad_total = 0
        for seed in SEEDS:
            act = act_for(device, act_cfg, seed)
            ref = ttnn.linear(act, weight, **kw)
            ref_t = ttnn.to_torch(ttnn.to_memory_config(ref, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, N)[:1]
            got = m2.linear(act, weight, **kw)
            got_t = ttnn.to_torch(ttnn.to_memory_config(got, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, N)[:1]
            bad = int((ref_t.view(torch.int16) != got_t.view(torch.int16)).sum())
            bad_total += bad
            logger.info(f"plekv wseed {wseed} seed {seed}: differ {bad} of {N}")
            for t in (act, ref, got):
                ttnn.deallocate(t)
        streamed = m2.CALLS.get((K, N), 0) - before
        assert streamed >= len(SEEDS) - 1, f"streamed program ran {streamed} times"
        res.append(bad_total)
        if wseed == 500:
            act = act_for(device, act_cfg, 0)
            stock = lambda: ttnn.deallocate(ttnn.linear(act, weight, **kw))  # noqa: E731
            mine = lambda: ttnn.deallocate(m2.linear(act, weight, **kw))  # noqa: E731
            us_s = [traced_wall_us(device, stock, replays=20)[0] for _ in range(3)]
            us_m = [traced_wall_us(device, mine, replays=20)[0] for _ in range(3)]
            row = dict(stock_us=round(sorted(us_s)[1], 2), mm2_us=round(sorted(us_m)[1], 2))
            ttnn.deallocate(act)
        ttnn.deallocate(weight)
    row["differ"] = res
    row["exact"] = sum(res) == 0
    logger.info("ROW " + json.dumps(row))
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    assert row["exact"]
