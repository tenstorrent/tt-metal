# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x58: more DRAM readers per bank for the decode linears on TODAY's cached weight layout (the bank tile count must not change),
against today's configuration (``decode_matmul.MM_WIDE_BLOCKS`` blocks, today's readers): bitwise on row 0 of the valid columns
with the output L1 region sentinel-filled before every run (a freed sentinel tensor of the output's config: the op's output
lands on the same addresses) and a fresh activation per seed, traced slope us (4-op vs 12-op traces, as x24).
Run on one die (1x1 mesh, MESH_DEVICE=P150x8 only for the import)."""

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm

BF8, BF16 = ttnn.bfloat8_b, ttnn.bfloat16
RESULTS = Path(os.environ.get("X58_RESULTS", "/tmp/x58.jsonl"))
SEEDS = int(os.environ.get("X58_SEEDS", "8"))
REPLAYS = int(os.environ.get("X58_REPLAYS", "200"))
# name: K, N, dtype, today's activation cores / readers, candidate reader counts (the bank tile count must stay)
SHAPES = {
    "qsa_proj": dict(k=2560, n=2304, dtype=BF8, cores=8, today=1, cands=[2, 3]),
    "router": dict(k=2560, n=512, dtype=BF16, cores=5, today=1, cands=[2, 3]),
    "gate_up": dict(k=2560, n=352, dtype=BF8, cores=5, today=1, cands=[2, 3]),
    "gdn_in": dict(k=2560, n=2112, dtype=BF8, cores=8, today=2, cands=[3, 4, 5]),
    "out_proj": dict(k=768, n=2560, dtype=BF8, cores=8, today=2, cands=[3, 4, 5]),
    "lm8000": dict(k=2560, n=8000, dtype=BF8, cores=40, today=2, cands=[3, 4]),
    "lm7040": dict(k=2560, n=7040, dtype=BF8, cores=40, today=2, cands=[3, 4]),
}
ONLY = [s for s in os.environ.get("X58_OPS", "qsa_proj,router,gate_up").split(",") if s]
REAL = {
    "router": "*/moe-small/**/mesh-1x8-physical-1-5-4-0-2-3-7-6/backbone/layer-09/router_replicated_dram_sharded_dtype_BFLOAT16_layout_TILE.tensorbin",
    "gate_up": "*/moe-small/**/mesh-1x8-physical-1-0-2-3-7-6-4-5/backbone/layer-00/shared_gate_up_scalar_dram_sharded.bf8b_dtype_BFLOAT8_B_layout_TILE.tensorbin",
}


def bank_tiles(banks, n, readers):
    return math.ceil(n / (32 * banks * readers)) * readers


def configs(device, spec, readers):
    k, n = spec["k"], spec["n"]
    kt = k // 32
    cores = spec["cores"]
    k_per_core = kt // cores
    block = next(w for w in range(8, 0, -1) if k_per_core % w == 0)
    wide = dm.MM_WIDE_BLOCKS.get((k, n))
    if wide:
        block = wide
    if readers != spec["today"] and os.environ.get("X58_BLOCK"):
        block = int(os.environ["X58_BLOCK"])
    grid = ttnn.CoreGrid(x=cores, y=1) if cores <= 8 else ttnn.CoreGrid(x=8, y=cores // 8)
    act_cfg = ttnn.create_sharded_memory_config(
        (32, k // cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tiles = bank_tiles(8, n, spec["today"])
    per_core_n = math.ceil(n / (32 * cores)) if readers == 1 else tiles
    prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=block,
        per_core_M=1,
        per_core_N=per_core_n,
        fused_activation=None,
        **({} if readers == 1 else {"num_workers_per_dram_bank": readers}),
    )
    return act_cfg, prog


def weight_config(spec):
    tiles = bank_tiles(8, spec["n"], spec["today"])
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (spec["k"], tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
    )


def traced(device, call, replays):
    def capture(n):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(n):
            call()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid

    def wall(tid):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        best = 1e30
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            best = min(best, time.perf_counter() - t0)
        return best / replays * 1e6

    s, l = capture(4), capture(12)
    try:
        ws, wl = wall(s), wall(l)
    finally:
        ttnn.release_trace(device, s)
        ttnn.release_trace(device, l)
    return (wl - ws) / 8


def run_checked(device, act, weight, prog, ccfg, n):
    """The op with its output region sentinel-filled first; row 0 of the n valid columns."""
    probe = ttnn.linear(
        act, weight, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, program_config=prog, compute_kernel_config=ccfg
    )
    cfg, shape = probe.memory_config(), tuple(probe.shape)
    ttnn.deallocate(probe)
    sent = ttnn.from_torch(
        torch.full(shape, 3.0e30).to(torch.bfloat16),
        dtype=BF16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    ttnn.deallocate(sent)  # the op's output lands on the sentinel's addresses
    out = ttnn.linear(
        act, weight, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, program_config=prog, compute_kernel_config=ccfg
    )
    assert out.memory_config() == cfg
    host = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1, out.shape[-1])[:1, :n]
    return out, host, cfg


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("op", ONLY)
@pytest.mark.parametrize("real", [False, True], ids=["random", "real"])
def test_readers(mesh_device, op, real):
    device = mesh_device
    spec = SHAPES[op]
    if real and op not in REAL:
        pytest.skip("no real cached weight for this shape")
    k, n, dtype = spec["k"], spec["n"], spec["dtype"]
    ccfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, dm.dense_math_fidelity_name(dtype)),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    today = spec["today"]
    cands = [r for r in spec["cands"] if bank_tiles(8, n, r) == bank_tiles(8, n, today)]
    logger.info(
        f"{op}: today's readers {today}, tiles/bank {bank_tiles(8, n, today)}, candidates with the same layout: {cands}"
    )
    wcfg = weight_config(spec)
    if real:
        import glob

        path = glob.glob("/home/ttuser/models/qwen38_cache/caches/c32768-vllm/components/" + REAL[op], recursive=True)[
            0
        ]
        part = ttnn.get_device_tensors(ttnn.load_tensor(path))[0]
        weight = ttnn.to_device(part, device, memory_config=part.memory_config())
    else:
        g = torch.Generator().manual_seed(123)
        weight = ttnn.from_torch(
            (torch.randn(1, 1, k, n, generator=g) * 0.02).to(torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=wcfg,
        )
    act_cfg0, prog0 = configs(device, spec, today)
    equal = {r: True for r in cands}
    for seed in range(SEEDS):
        g = torch.Generator().manual_seed(7000 + seed)
        a = (torch.randn(1, 1, 1, k, generator=g) * (0.3 + 0.1 * seed)).to(torch.bfloat16)
        act_dram = ttnn.from_torch(
            a, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        act0 = ttnn.to_memory_config(act_dram, act_cfg0)
        out0, ref, _ = run_checked(device, act0, weight, prog0, ccfg, n)
        assert torch.isfinite(ref.float()).all()
        for r in cands:
            act_cfg, prog = configs(device, spec, r)
            act = ttnn.to_memory_config(act_dram, act_cfg)
            out, got, _ = run_checked(device, act, weight, prog, ccfg, n)
            eq = torch.equal(ref.view(torch.int16), got.view(torch.int16))
            equal[r] &= eq
            ttnn.deallocate(out)
            ttnn.deallocate(act)
        ttnn.deallocate(out0)
        ttnn.deallocate(act0)
        ttnn.deallocate(act_dram)
    row = dict(op=op, real=real, today=today, equal=equal, seeds=SEEDS)
    if not real:
        act_dram = ttnn.from_torch(
            torch.randn(1, 1, 1, k).to(torch.bfloat16),
            dtype=BF16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        times = {}
        for r in [today, *cands]:
            act_cfg, prog = configs(device, spec, r)
            act = ttnn.to_memory_config(act_dram, act_cfg)

            def call():
                o = ttnn.linear(
                    act,
                    weight,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=prog,
                    compute_kernel_config=ccfg,
                )
                ttnn.deallocate(o)

            times[r] = round(traced(device, call, REPLAYS), 2)
        row["us"] = times
    logger.info(json.dumps(row))
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    assert all(equal.values()), equal
