# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the SE engine's gate / up inner loop on one core, compute only (x and W resident in L1, nothing moves):
cycles a tile multiply by sub-block shape (MT x pw), fidelity, W dtype - se_engine3's call pattern (matmul_block a K
tile a call, KB a step, all of K in fp32 DEST, dst_full_sync)."""

import os
import statistics

import torch
from loguru import logger

import ttnn

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

KB, STEPS, REPS = 14, 16, 8  # (a step's K tiles; steps a pass - K 224; passes)
_KEY = "DEVICE KERNEL DURATION [ns]"

_K = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
constexpr uint32_t MT = get_compile_time_arg_val(0), PW = get_compile_time_arg_val(1), KB = get_compile_time_arg_val(2);
constexpr uint32_t STEPS = get_compile_time_arg_val(3), REPS = get_compile_time_arg_val(4);
constexpr uint32_t XIN = 0, WIN = 1, OUT = 16;
void kernel_main() {
    cb_reserve_back(XIN, MT * KB);
    cb_push_back(XIN, MT * KB);
    cb_reserve_back(WIN, KB * PW);
    cb_push_back(WIN, KB * PW);
    compute_kernel_hw_startup<SrcOrder::Reverse>(XIN, WIN, OUT);
    for (uint32_t r = 0; r < REPS; ++r) {
        tile_regs_acquire();
        matmul_block_init(XIN, WIN, 0, PW, MT, KB);
        for (uint32_t st = 0; st < STEPS; ++st) {
            cb_wait_front(XIN, MT * KB);
            cb_wait_front(WIN, KB * PW);
            for (uint32_t kk = 0; kk < KB; ++kk) matmul_block(XIN, WIN, kk, kk * PW, 0, false, PW, MT, KB);
        }
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
}
"""


def _one_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded(dev, t, dtype):
    h, w = t.shape
    mc = ttnn.create_sharded_memory_config(
        shape=(h, w),
        core_grid=_one_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def _ns(dev):
    ttnn.ReadDeviceProfiler(dev)
    tot = None
    for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                tot = (tot or 0) + float(e.duration)
    return tot


CASES = [  # (MT, pw, fidelity, W dtype)
    (2, 4, "HiFi2", "bf8"),
    (2, 4, "LoFi", "bf8"),
    (2, 4, "HiFi2", "bf16"),
    (4, 2, "HiFi2", "bf8"),
    (1, 8, "HiFi2", "bf8"),
    (8, 1, "HiFi2", "bf8"),
    (2, 2, "HiFi2", "bf8"),
    (1, 4, "HiFi2", "bf8"),
    (2, 4, "HiFi4", "bf8"),
]


def test_gu_mm(device):
    torch.manual_seed(0)
    rows = []
    for MT, PW, fid, wd in CASES:
        x = _sharded(device, torch.randn(MT * 32, KB * 32), ttnn.bfloat16)
        w = _sharded(device, torch.randn(KB * 32, PW * 32) * 0.02, {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[wd])
        o = _sharded(device, torch.zeros(MT * 32, PW * 32), ttnn.bfloat16)
        comp = ttnn.ComputeConfigDescriptor()
        comp.math_fidelity = getattr(ttnn.MathFidelity, fid)
        comp.fp32_dest_acc_en = True
        comp.dst_full_sync_en = True
        k = ttnn.KernelDescriptor(
            kernel_source=_K,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_one_core(),
            compile_time_args=[MT, PW, KB, STEPS, REPS],
            config=comp,
        )
        cbs = [ttnn.cb_descriptor_from_sharded_tensor(i, t) for i, t in ((0, x), (1, w), (16, o))]
        desc = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
        for _ in range(2):
            ttnn.generic_op([x, w, o], desc)
        ttnn.synchronize_device(device)
        _ns(device)
        s = []
        for _ in range(5):
            ttnn.generic_op([x, w, o], desc)
            ttnn.synchronize_device(device)
            s.append(_ns(device))
        ns = statistics.median(s)
        tiles = REPS * STEPS * KB * MT * PW
        cyc = ns * 1.35 / tiles
        rows.append((MT, PW, fid, wd, ns / 1e3, cyc))
        logger.info(f"GUMM {MT}x{PW} {fid} W {wd}: {ns / 1e3:.1f} us, {cyc:.1f} cycles a tile multiply ({tiles} tiles)")
        for t in (x, w, o):
            ttnn.deallocate(t)
    logger.info("GUMM table:\n" + "\n".join(f"  {m}x{p} {f:5s} {d:4s} {c:6.1f} cyc/tile" for m, p, f, d, _, c in rows))
