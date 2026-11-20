# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS2 lever D (V41_MHC_FUSED_TAIL=1): the tail of ``V41HyperConnection.mixes`` -- affine, sigmoid, the comb softmax and the
20 Sinkhorn iterations, ~90 ttnn programs on a [1, 1, S_l, 32] fp32 row -- as ONE generic_op program, written to be
BYTE-IDENTICAL to the ttnn ops it replaces: every step calls the LLK those ops compile to (read from their JIT defines on
host 31: binary_ng SFPU ``mul_binary_tile`` / ``add_binary_tile<NearestEven>`` / ``div_binary_tile`` on operands copied to
dest in UnpackToDestFp32, ``sigmoid_tile<RC, 0>``, ``exp_tile<0>``, ``clamp_tile(-60, 60)``, the matmul's
``matmul_block`` (1 x 1 x 1) at HiFi4 + fp32 dest), and every intermediate is packed to an fp32 CB exactly as the separate
programs pack it to an fp32 tensor. The row operands (SV, BV, COMB_MASK, ONE_COL, EPS_COMB) are uploaded as full tiles with
row 0 replicated -- what binary_ng's reader does (fill_tile_with_first_row) -- and the scalars (hc_eps, 2.0) as filled tiles,
what binary_ng's scalar path does. One core per token tile."""

from __future__ import annotations

import struct

import torch

import ttnn

N_SFPU_CONSTS = 7  # SV, BV, COMB_MASK, ONE_COL, EPS_COMB, EPS (scalar fill), TWO (scalar fill)
N_MM_CONSTS = 3  # R_SOFT, R_AUG, C_AUG

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    const uint32_t mix_addr = get_arg_val<uint32_t>(0);
    const uint32_t consts_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile = get_arg_val<uint32_t>(2);
    constexpr auto mix_args = TensorAccessorArgs<0>();
    constexpr auto consts_args = TensorAccessorArgs<mix_args.next_compile_time_args_offset()>();
    const auto s_mix = TensorAccessor(mix_args, mix_addr);
    const auto s_c = TensorAccessor(consts_args, consts_addr);
    Noc noc;
    CircularBuffer cb_mix(0), cb_cs(1), cb_cm(2);
    for (uint32_t i = 0; i < 7; ++i) {
        cb_cs.reserve_back(1);
        noc.async_read(s_c, cb_cs, 4096, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_cs.push_back(1);
    }
    for (uint32_t i = 0; i < 3; ++i) {
        cb_cm.reserve_back(1);
        noc.async_read(s_c, cb_cm, 4096, {.page_id = 7 + i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_cm.push_back(1);
    }
    cb_mix.reserve_back(1);
    noc.async_read(s_mix, cb_mix, 4096, {.page_id = tile}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_mix.push_back(1);
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    const uint32_t pre_addr = get_arg_val<uint32_t>(0);
    const uint32_t post_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_arg_val<uint32_t>(2);
    const uint32_t tile = get_arg_val<uint32_t>(3);
    constexpr auto pre_args = TensorAccessorArgs<0>();
    constexpr auto post_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    const auto s_pre = TensorAccessor(pre_args, pre_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_x = TensorAccessor(x_args, x_addr);
    Noc noc;
    CircularBuffer cbp(16), cbq(17), cbx(18);
    cbp.wait_front(1);
    noc.async_write(cbp, s_pre, 4096, {}, {.page_id = tile});
    noc.async_write_barrier();
    cbp.pop_front(1);
    cbq.wait_front(1);
    noc.async_write(cbq, s_post, 4096, {}, {.page_id = tile});
    noc.async_write_barrier();
    cbq.pop_front(1);
    cbx.wait_front(1);
    noc.async_write(cbx, s_x, 4096, {}, {.page_id = tile});
    noc.async_write_barrier();
    cbx.pop_front(1);
}
"""

# CBs: 0 mix (UTD) | 1 sfpu consts x7 (UTD, resident) | 2 matmul consts x3 (resident) | 3 A, 4 AFF, 5 SIG, 6 XS, 8 RS (UTD)
#      7 XM (matmul in0) | 16 pre, 17 post, 18 X (out)
COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t F_MIX = 0, F_CS = 1, F_CM = 2, F_A = 3, F_AFF = 4, F_SIG = 5, F_XS = 6, F_XM = 7, F_RS = 8;
constexpr uint32_t F_PRE = 16, F_POST = 17, F_XO = 18;
constexpr uint32_t F_SV = 0, F_BV = 1, F_CMASK = 2, F_ONECOL = 3, F_EPSCOMB = 4, F_EPS = 5, F_TWO = 6;  // tiles in F_CS
constexpr uint32_t F_RSOFT = 0, F_RAUG = 1, F_CAUG = 2;                                       // tiles in F_CM
constexpr uint32_t OP_ADD = 0, OP_MUL = 1, OP_DIV = 2;
constexpr uint32_t U_SIG = 0, U_EXP = 1, U_CLAMP = 2;

// pack dest[0] to cb (and to cb2 when cb2 != 0xFF)
FORCE_INLINE void pack_out(uint32_t cb, uint32_t cb2) {
    tile_regs_commit();
    tile_regs_wait();
    {
        CircularBuffer c(cb);
        c.reserve_back(1);
        pack_tile(0, cb);
        c.push_back(1);
    }
    if (cb2 != 0xFF) {
        CircularBuffer c2(cb2);
        c2.reserve_back(1);
        pack_tile(0, cb2);
        c2.push_back(1);
    }
    tile_regs_release();
}

// out = a (cba[ia]) OP b (cbb[ib]) as binary_ng's SFPU kernel: copy lhs to dest 0, rhs to dest 1, op(0, 1, 0).
// pop_a / pop_b: pop the front tile of that CB after (resident const CBs are never popped).
FORCE_INLINE void bin(uint32_t op, uint32_t cba, uint32_t ia, bool pop_a, uint32_t cbb, uint32_t ib, bool pop_b,
                      uint32_t cbo, uint32_t cbo2) {
    CircularBuffer ca(cba), cb(cbb);
    ca.wait_front(ia + 1);
    cb.wait_front(ib + 1);
    tile_regs_acquire();
    copy_init(cba);
    copy_tile(cba, ia, 0);
    reconfig_data_format_srca(cba, cbb);
    copy_init(cbb);
    copy_tile(cbb, ib, 1);
    if (op == OP_ADD) {
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
    } else if (op == OP_MUL) {
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    } else {
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
    }
    reconfig_data_format_srca(cbb, cba);
    pack_out(cbo, cbo2);
    if (pop_a) ca.pop_front(1);
    if (pop_b) cb.pop_front(1);
}

// out = f(a) as the unary SFPU kernel: copy to dest 0, init + func
FORCE_INLINE void un(uint32_t u, uint32_t cba, bool pop_a, uint32_t cbo) {
    CircularBuffer ca(cba);
    ca.wait_front(1);
    tile_regs_acquire();
    copy_init(cba);
    copy_tile(cba, 0, 0);
    if (u == U_SIG) {
        sigmoid_tile_init<0u>();
        sigmoid_tile<VectorMode::RC, 0u>(0);
    } else if (u == U_EXP) {
        exp_tile_init<0u>();
        exp_tile<0u>(0);
    } else {
        clamp_tile_init();
        clamp_tile(0, 3262119936u, 1114636288u);  // -60.0f, 60.0f
    }
    pack_out(cbo, 0xFF);
    if (pop_a) ca.pop_front(1);
}

// F_RS = F_XM[0] @ F_CM[k] as the matmul kernel (1 x 1 x 1 block), F_XM popped
FORCE_INLINE void mm(uint32_t k) {
    CircularBuffer cx(F_XM), cm(F_CM);
    cx.wait_front(1);
    cm.wait_front(3);
    reconfig_data_format(F_CM, F_XM);  // srcA <- in1, srcB <- in0
    matmul_block_init(F_XM, F_CM, 0, 1, 1, 1);
    tile_regs_acquire();
    matmul_block(F_XM, F_CM, 0, k, 0, 0, 1, 1, 1);
    pack_out(F_RS, 0xFF);
    reconfig_data_format(F_XS, F_XS);
    cx.pop_front(1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t iters = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(F_MIX, F_PRE);
    copy_init(F_MIX);
    // aff = mix * F_SV + F_BV ; sig = sigmoid(aff) ; pre = sig + eps ; post = sig * 2
    bin(OP_MUL, F_MIX, 0, true, F_CS, F_SV, false, F_A, 0xFF);
    bin(OP_ADD, F_A, 0, true, F_CS, F_BV, false, F_AFF, 0xFF);
    un(U_SIG, F_AFF, false, F_SIG);
    bin(OP_ADD, F_SIG, 0, false, F_CS, F_EPS, false, F_PRE, 0xFF);
    bin(OP_MUL, F_SIG, 0, true, F_CS, F_TWO, false, F_POST, 0xFF);
    // E = exp(clamp(aff)) * COMB_MASK + ONE_COL
    un(U_CLAMP, F_AFF, true, F_A);
    un(U_EXP, F_A, true, F_AFF);
    bin(OP_MUL, F_AFF, 0, true, F_CS, F_CMASK, false, F_A, 0xFF);
    bin(OP_ADD, F_A, 0, true, F_CS, F_ONECOL, false, F_XS, F_XM);
    // X = E / (E @ R_SOFT) ; X = X + EPS_COMB ; X = X / (X @ C_AUG)
    mm(F_RSOFT);
    bin(OP_DIV, F_XS, 0, true, F_RS, 0, true, F_A, 0xFF);
    bin(OP_ADD, F_A, 0, true, F_CS, F_EPSCOMB, false, F_XS, F_XM);
    mm(F_CAUG);
    if (iters == 1) {
        bin(OP_DIV, F_XS, 0, true, F_RS, 0, true, F_XO, 0xFF);
    } else {
        bin(OP_DIV, F_XS, 0, true, F_RS, 0, true, F_XS, F_XM);
    }
    for (uint32_t i = 0; i + 1 < iters; ++i) {
        mm(F_RAUG);
        bin(OP_DIV, F_XS, 0, true, F_RS, 0, true, F_XS, F_XM);
        mm(F_CAUG);
        if (i + 2 == iters) {
            bin(OP_DIV, F_XS, 0, true, F_RS, 0, true, F_XO, 0xFF);
        } else {
            bin(OP_DIV, F_XS, 0, true, F_RS, 0, true, F_XS, F_XM);
        }
    }
}
"""


def _fbits(x: float) -> float:
    """x rounded to fp32 (the value a python float scalar becomes on device)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


class FusedTail:
    """``(pre, post, X) = tail(mix)`` for one ``V41HyperConnection`` (its host constants), on a mesh or a single device."""

    def __init__(self, device, hc, host_consts: dict, iters: int):
        self.device = device
        self.iters = int(iters)
        T = 32
        rows = [
            host_consts[k].float().reshape(1, T).expand(T, T) for k in ("SV", "BV", "COMB_MASK", "ONE_COL", "EPS_COMB")
        ]
        eps = torch.full((T, T), _fbits(hc.hc_eps))
        two = torch.full((T, T), 2.0)
        mats = [host_consts[k].float().reshape(T, T) for k in ("R_SOFT", "R_AUG", "C_AUG")]
        c = (
            torch.stack(rows + [eps, two] + mats, dim=0)
            .reshape(1, 1, (N_SFPU_CONSTS + N_MM_CONSTS) * T, T)
            .contiguous()
        )
        mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(device, "shape") else None
        self.consts = ttnn.from_torch(
            c,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._programs: dict = {}

    def _program(self, mix, pre, post, X):
        n = int(mix.shape[2]) // 32
        key = (n, mix.buffer_address(), pre.buffer_address(), post.buffer_address(), X.buffer_address())
        if key in self._programs:
            return self._programs[key]
        g = self.device.compute_with_storage_grid_size()
        coords = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(n)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc) for cc in coords])
        page = 32 * 32 * 4

        def cb(idx, pages):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)
                ],
            )

        cbs = [
            cb(0, 1),
            cb(1, N_SFPU_CONSTS),
            cb(2, N_MM_CONSTS),
            cb(3, 2),
            cb(4, 2),
            cb(5, 2),
            cb(6, 2),
            cb(7, 2),
            cb(8, 2),
            cb(16, 1),
            cb(17, 1),
            cb(18, 1),
        ]
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for t, cc in enumerate(coords):
            rd[cc.x][cc.y] = [mix.buffer_address(), self.consts.buffer_address(), t]
            wr[cc.x][cc.y] = [pre.buffer_address(), post.buffer_address(), X.buffer_address(), t]
        utd = ttnn.UnpackToDestMode.UnpackToDestFp32
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (0, 1, 3, 4, 5, 6, 8):
            modes[i] = utd
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        cfg.unpack_to_dest_mode = type(cfg.unpack_to_dest_mode)(
            modes
        )  # ttnn._ttnn.program_descriptor.VectorUnpackToDestMode
        ct_rd = list(ttnn.TensorAccessorArgs(mix).get_compile_time_args()) + list(
            ttnn.TensorAccessorArgs(self.consts).get_compile_time_args()
        )
        ct_wr = []
        for t_ in (pre, post, X):
            ct_wr += list(ttnn.TensorAccessorArgs(t_).get_compile_time_args())
        SRC = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        prog = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=READER,
                    source_type=SRC,
                    core_ranges=cores,
                    compile_time_args=ct_rd,
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=WRITER,
                    source_type=SRC,
                    core_ranges=cores,
                    compile_time_args=ct_wr,
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=COMPUTE,
                    source_type=SRC,
                    core_ranges=cores,
                    compile_time_args=[self.iters],
                    runtime_args=[],
                    config=cfg,
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        self._programs[key] = prog
        return prog

    def __call__(self, mix, out_mc=None):
        mc = out_mc or ttnn.DRAM_MEMORY_CONFIG
        pre, post, X = (
            ttnn.empty(
                list(mix.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mc
            )
            for _ in range(3)
        )
        ttnn.generic_op([mix, self.consts, pre, post, X], self._program(mix, pre, post, X))
        return pre, post, X
