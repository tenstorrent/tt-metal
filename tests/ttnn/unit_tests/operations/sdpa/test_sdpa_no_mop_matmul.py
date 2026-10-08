# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wormhole no-MOP matmul (api/compute/experimental/matmul_custom.h) against the MOP matmul and an FP64 reference.

The SDPA recipes run their QK and PV matmuls through the no-MOP replay path at LoFi, HiFi2 and HiFi4,
plus the PHASES / INNER_HALF variants of the fused max step. Each case here runs one rt x ct x kt block
on one core with FP32 DEST and FP32 output, and checks that:
- the no-MOP result is bitwise equal to the MOP matmul_block result at the same fidelity;
- the error against FP64 falls with fidelity (a stuck fidelity phase would repeat phase 0);
- PHASES=1 on a high-fidelity image equals LoFi, and INNER_HALF equals the full replay when in0's
  columns 16-31 of every tile are zero;
- the math thread's replay image survives the pack thread recording the fast exp's program.
"""

import pytest
import torch

import ttnn

# The Wormhole no-MOP LLK (tt_llk_wormhole_b0 llk_math_matmul_custom_no_mop.h); validated on Wormhole silicon only.
pytestmark = pytest.mark.skipif(ttnn.get_arch_name() != "wormhole_b0", reason="Wormhole no-MOP matmul LLK coverage")

MODE_MOP, MODE_NO_MOP, MODE_NO_MOP_PHASES, MODE_NO_MOP_PACK_EXP_INIT = 0, 1, 2, 3

COMPUTE_SOURCE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/exp.h"

void kernel_main() {
    constexpr uint32_t in0 = 0, in1 = 1, out = 16;
    constexpr uint32_t rt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t mode = get_compile_time_arg_val(3);
    constexpr bool transpose = get_compile_time_arg_val(4);
    constexpr int phases = get_compile_time_arg_val(5);
    constexpr bool inner_half = get_compile_time_arg_val(6);
    constexpr uint32_t reps = get_compile_time_arg_val(7);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    if constexpr (mode == 0) {
        matmul_block_init(in0, in1, transpose, ct, rt, kt);
    } else {
        mm_no_mop_init_short(in0, in1, transpose, ct, rt, kt);
    }
    cb_wait_front(in0, rt * kt);
    cb_wait_front(in1, kt * ct);
    cb_reserve_back(out, rt * ct);
    for (uint32_t rep = 0; rep < reps; rep++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; k++) {
            if constexpr (mode == 0) {
                matmul_block(in0, in1, k, k * ct, 0, transpose, ct, rt, kt);
            } else if constexpr (mode == 1 || mode == 3) {
                matmul_block_no_mop(in0, in1, k, k * ct, 0, transpose, ct, rt, kt);
            } else {
                UNPACK((llk_unpack_AB_matmul(in0, in1, k, k * ct, ct, rt, kt)));
                MATH((llk_math_matmul_no_mop<MATH_FIDELITY, MM_THROTTLE, phases, inner_half>(in0, in1, 0, ct, rt)));
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        if constexpr (mode == 3) {
            // PACK records the fast exp's replay program (slots 0-31) between two no-MOP matmuls.
            if (rep == 0) {
                exp_packthread_tile_init<true, 0x3F800000, InputClamping::None>();
            }
        }
        if (rep + 1 == reps) {
            for (uint32_t i = 0; i < rt * ct; i++) {
                pack_tile(i, out);
            }
        }
        tile_regs_release();
    }
    cb_push_back(out, rt * ct);
}
"""

READER_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t n0 = get_compile_time_arg_val(0);
    constexpr uint32_t n1 = get_compile_time_arg_val(1);
    cb_reserve_back(0, n0);
    cb_push_back(0, n0);
    cb_reserve_back(1, n1);
    cb_push_back(1, n1);
}
"""


def _sharded(device, t, dtype, core):
    h, w = t.shape[-2:]
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core, (h, w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


def run_block(device, a, b, fidelity, mode, transpose=False, phases=0, inner_half=False, reps=1):
    """out = a @ (b with every 32x32 tile transposed if transpose) via one block matmul on core (0,0)."""
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rt, kt = a.shape[0] // 32, a.shape[1] // 32
    ct = b.shape[1] // 32
    ta = _sharded(device, a, ttnn.bfloat16, core)
    tb = _sharded(device, b, ttnn.bfloat16, core)
    tout = _sharded(device, torch.zeros(rt * 32, ct * 32), ttnn.float32, core)
    cbs = []
    for idx, t, dt in ((0, ta, ttnn.bfloat16), (1, tb, ttnn.bfloat16), (16, tout, ttnn.float32)):
        cb = ttnn.cb_descriptor_from_sharded_tensor(idx, t)
        cb.format_descriptors = [ttnn.CBFormatDescriptor(idx, dt, 2048 if dt == ttnn.bfloat16 else 4096)]
        cbs.append(cb)
    reader = ttnn.KernelDescriptor(
        kernel_source=READER_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[rt * kt, kt * ct],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core,
        compile_time_args=[rt, ct, kt, mode, int(transpose), phases, int(inner_half), reps],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
    )
    pd = ttnn.ProgramDescriptor(cbs=cbs, kernels=[reader, compute])
    mesh_pd = ttnn.MeshProgramDescriptor()
    coord = ttnn.MeshCoordinate(0, 0)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd
    ttnn.generic_op([ta, tb, tout], mesh_pd)
    return ttnn.to_torch(tout).float()


def reference(a, b, transpose):
    b = b.double()
    if transpose:
        kt, ct = b.shape[0] // 32, b.shape[1] // 32
        b = b.reshape(kt, 32, ct, 32).transpose(1, 3).reshape(kt * 32, ct * 32)
    return a.double() @ b


def rel_l2(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


FIDELITIES = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4]
# (rt, ct, kt): t_dim 1 and > 1 on both reuse sides, odd t_dim, kt accumulation; out <= 4 FP32 DEST tiles.
SHAPES = [(1, 1, 1), (1, 1, 3), (1, 4, 2), (4, 1, 2), (2, 2, 2), (3, 1, 1), (1, 3, 2)]


@pytest.mark.parametrize("transpose", [False, True], ids=["nt", "t"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_no_mop_matches_mop(device, shape, transpose):
    rt, ct, kt = shape
    torch.manual_seed(1234)
    a = torch.randn(rt * 32, kt * 32).bfloat16()
    b = torch.randn(kt * 32, ct * 32).bfloat16()
    ref = reference(a.float(), b.float(), transpose)
    errs = []
    for fid in FIDELITIES:
        mop = run_block(device, a, b, fid, MODE_MOP, transpose)
        nomop = run_block(device, a, b, fid, MODE_NO_MOP, transpose)
        e_mop, e_nomop = rel_l2(mop, ref), rel_l2(nomop, ref)
        print(f"{shape} t={transpose} {fid}: rel-L2 mop {e_mop:.3e} no-mop {e_nomop:.3e}")
        assert torch.equal(mop, nomop), f"{fid}: no-MOP differs from MOP (max abs {(mop - nomop).abs().max()})"
        errs.append(e_nomop)
    # Each extra phase must improve accuracy (a phase that did not advance would repeat phase 0). HiFi3/HiFi4 on
    # BF16 inputs are limited by the FPU's internal accumulation (rel-L2 about 3e-4 on Wormhole).
    assert errs[0] > errs[1] > errs[3], errs
    assert errs[3] < 1e-3, errs


@pytest.mark.parametrize("fid", [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4], ids=["HiFi2", "HiFi4"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (2, 2, 1), (1, 4, 1), (4, 1, 1), (2, 2, 2)], ids=lambda s: "x".join(map(str, s))
)
def test_no_mop_phases_inner_half(device, shape, fid):
    rt, ct, kt = shape
    torch.manual_seed(42)
    a = torch.randn(rt * 32, kt * 32).bfloat16()
    b = torch.randn(kt * 32, ct * 32).bfloat16()
    lofi = run_block(device, a, b, ttnn.MathFidelity.LoFi, MODE_MOP)
    one_phase = run_block(device, a, b, fid, MODE_NO_MOP_PHASES, phases=1)
    assert torch.equal(lofi, one_phase), "PHASES=1 must equal LoFi"
    full = run_block(device, a, b, fid, MODE_MOP)
    default_phases = run_block(device, a, b, fid, MODE_NO_MOP_PHASES, phases=int(fid.value))
    assert torch.equal(full, default_phases), "PHASES=all must equal the full matmul"
    # INNER_HALF: only inner indices 0-15 of every tile contribute.
    a_half = a.clone().reshape(rt * 32, kt, 32)
    a_half[:, :, 16:] = 0
    a_half = a_half.reshape(rt * 32, kt * 32)
    full_lofi = run_block(device, a_half, b, ttnn.MathFidelity.LoFi, MODE_MOP)
    half_1 = run_block(device, a_half, b, fid, MODE_NO_MOP_PHASES, phases=1, inner_half=True)
    assert torch.equal(full_lofi, half_1), "INNER_HALF + PHASES=1 must equal LoFi on half-zero in0"
    # Repeated calls must leave the counters ready for the next call (two kt steps and several reps).
    half_rep = run_block(device, a_half, b, fid, MODE_NO_MOP_PHASES, phases=1, inner_half=True, reps=3)
    assert torch.equal(full_lofi, half_rep)


# At most four output tiles: FP32 DEST holds four tiles per half, and these cases run two acquires.
@pytest.mark.parametrize("shape", [(2, 2, 2), (1, 4, 1)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("fid", [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2], ids=["LoFi", "HiFi2"])
def test_no_mop_replay_survives_pack_exp_init(device, fid, shape):
    """The pack-thread exp init records its own replay program; the math thread's matmul image must survive it."""
    rt, ct, kt = shape
    torch.manual_seed(7)
    a = torch.randn(rt * 32, kt * 32).bfloat16()
    b = torch.randn(kt * 32, ct * 32).bfloat16()
    expected = run_block(device, a, b, fid, MODE_MOP)
    assert torch.equal(expected, run_block(device, a, b, fid, MODE_NO_MOP, reps=2)), "control"
    actual = run_block(device, a, b, fid, MODE_NO_MOP_PACK_EXP_INIT, reps=2)
    assert torch.equal(expected, actual)
