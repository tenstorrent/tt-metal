# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch: LoFi matmul throughput with 32 x 16 in1 tiles (output 32 x 16) vs 32 x 32, one core, compute only.

The inputs are pushed once (garbage in L1) and matmul_block runs ITER times over a [RT x KT] in0 block (32 x 32 tiles)
times a [KT x CT] in1 block, fp32 DST, no pack (unpack + math only; TM_PACK=1 packs each block into a ring popped by a
writer). Two ITER counts: the difference is the steady-state time, so launch / setup cancel. Reported per output tile
of the run's shape and per 32 x 32 x 32 MAC block (32768 MACs; the LoFi peak 2048 MACs / cycle = 16 cycles)."""
import os

import pytest
from loguru import logger

import ttnn

_KEY = "DEVICE KERNEL DURATION [ns]"

_READ = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t N0 = get_compile_time_arg_val(0), N1 = get_compile_time_arg_val(1);
    cb_reserve_back(0, N0);
    cb_push_back(0, N0);
    cb_reserve_back(1, N1);
    cb_push_back(1, N1);
}
"""

_WRITE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t ITER = get_compile_time_arg_val(0), NOUT = get_compile_time_arg_val(1);
    for (uint32_t i = 0; i < ITER; ++i) {
        cb_wait_front(16, NOUT);
        cb_pop_front(16, NOUT);
    }
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
constexpr uint32_t CT = get_compile_time_arg_val(0), RT = get_compile_time_arg_val(1), KT = get_compile_time_arg_val(2);
constexpr uint32_t ITER = get_compile_time_arg_val(3), DO_PACK = get_compile_time_arg_val(4);
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);  // (matmul: in1 -> SrcA, in0 -> SrcB)
    matmul_block_init(0, 1, 0, CT, RT, KT);
    cb_wait_front(0, RT * KT);
    cb_wait_front(1, KT * CT);
    for (uint32_t i = 0; i < ITER; ++i) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KT; ++k) {
            matmul_block(0, 1, k, k * CT, 0, false, CT, RT, KT);
        }
        tile_regs_commit();
        tile_regs_wait();
        if constexpr (DO_PACK) {
            cb_reserve_back(16, CT * RT);
            for (uint32_t t = 0; t < CT * RT; ++t) pack_tile(t, 16);
            cb_push_back(16, CT * RT);
        }
        tile_regs_release();
    }
}
"""


def _dur(mesh_device):
    ttnn.ReadDeviceProfiler(mesh_device)
    us = [
        float(p.program_analyses_results[_KEY].duration) / 1e3
        for progs in (ttnn.get_latest_programs_perf_data() or {}).values()
        for p in progs
        if _KEY in (p.program_analyses_results or {})
    ]
    return us[-1] if us else float("nan")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_tiny_mm(mesh_device):
    core = ttnn.CoreCoord(0, 0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    ghz = float(os.environ.get("TM_GHZ", "1.35"))
    pack = int(os.environ.get("TM_PACK", "0"))
    in0_dt = getattr(ttnn, os.environ.get("TM_IN0", "bfloat8_b"))
    in1_dt = getattr(ttnn, os.environ.get("TM_IN1", "bfloat4_b"))
    fid = getattr(ttnn.MathFidelity, os.environ.get("TM_FID", "LoFi"))
    full = os.environ.get("TM_FULL", "1") == "1"
    shapes = [
        tuple(int(v) for v in s.split("x"))
        for s in os.environ.get("TM_SHAPES", "1x2x8,1x4x8,1x8x8,2x2x8,2x4x8").split(",")
    ]
    iters = (int(os.environ.get("TM_ITER_LO", "200")), int(os.environ.get("TM_ITER_HI", "1200")))
    # TM_TILES: output tile shapes th x tw (in0 th x 32, in1 32 x tw)
    for th, tw in [
        tuple(int(v) for v in t.split("x")) for t in os.environ.get("TM_TILES", "32x32,32x16,16x32").split(",")
    ]:
        t_in0, t_in1, t_out = ttnn.Tile((th, 32)), ttnn.Tile((32, tw)), ttnn.Tile((th, tw))
        b0, b1, bo = t_in0.get_tile_size(in0_dt), t_in1.get_tile_size(in1_dt), t_out.get_tile_size(ttnn.bfloat16)
        for rt, ct, kt in shapes:
            res = {}
            for it in iters:
                pd = ttnn.ProgramDescriptor()
                fmt = lambda i, dt, page, tile: [
                    ttnn.CBFormatDescriptor(i, dt, page, ttnn.TileDescriptor(tile))
                ]  # noqa: E731
                pd.cbs = [
                    ttnn.CBDescriptor(
                        total_size=rt * kt * b0, core_ranges=crs, format_descriptors=fmt(0, in0_dt, b0, t_in0)
                    ),
                    ttnn.CBDescriptor(
                        total_size=kt * ct * b1, core_ranges=crs, format_descriptors=fmt(1, in1_dt, b1, t_in1)
                    ),
                    ttnn.CBDescriptor(
                        total_size=2 * rt * ct * bo,
                        core_ranges=crs,
                        format_descriptors=fmt(16, ttnn.bfloat16, bo, t_out),
                    ),
                ]
                comp = ttnn.ComputeConfigDescriptor()
                comp.math_fidelity = fid
                comp.fp32_dest_acc_en = True
                comp.dst_full_sync_en = full
                comp.math_approx_mode = False
                dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
                    processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
                )
                rt_empty = ttnn.RuntimeArgs()
                rt_empty[0][0] = []
                ks = [
                    ttnn.KernelDescriptor(
                        kernel_source=_READ,
                        source_type=st,
                        core_ranges=crs,
                        compile_time_args=[rt * kt, kt * ct],
                        runtime_args=rt_empty,
                        config=dm("RISCV_0", ttnn.NOC.NOC_0),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=_COMPUTE,
                        source_type=st,
                        core_ranges=crs,
                        compile_time_args=[ct, rt, kt, it, pack],
                        runtime_args=rt_empty,
                        config=comp,
                    ),
                ]
                if pack:
                    ks.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_WRITE,
                            source_type=st,
                            core_ranges=crs,
                            compile_time_args=[it, rt * ct],
                            runtime_args=rt_empty,
                            config=dm("RISCV_1", ttnn.NOC.NOC_1),
                        )
                    )
                pd.kernels = ks
                md = ttnn.MeshProgramDescriptor()
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = pd
                dummy = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
                )
                best = None
                for _ in range(3):
                    _dur(mesh_device)
                    ttnn.generic_op([dummy], md)
                    ttnn.synchronize_device(mesh_device)
                    d = _dur(mesh_device)
                    best = d if best is None or d < best else best
                res[it] = best
                ttnn.deallocate(dummy)
            d_us = res[iters[1]] - res[iters[0]]
            n_out = (iters[1] - iters[0]) * kt * ct * rt  # output-tile x K-tile products
            cyc = d_us * 1e3 * ghz / n_out
            macs = th * 32 * tw  # per product: [th x 32] x [32 x tw]
            cyc32 = cyc * (32 * 32 * 32) / macs
            logger.info(
                f"TINYMM out {th}x{tw} in1 {in1_dt} in0 {in0_dt} rt {rt} ct {ct} kt {kt} pack {pack}: {cyc:6.2f} cycles per "
                f"tile product = {cyc32:6.2f} per 32x32x32 ({16 / cyc32 * 100:5.1f}% of LoFi peak)  "
                f"[{res[iters[0]]:.1f} / {res[iters[1]]:.1f} us]"
            )
