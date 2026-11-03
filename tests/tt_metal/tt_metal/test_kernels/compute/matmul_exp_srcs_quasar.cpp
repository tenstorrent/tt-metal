// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Route 2: FPU op, pack the result to L1, SFPU op on the SFPU thread from L1. The opt-in.
//
//   UNPACK + MATH + PACK:  matmul -> Dest[0] -> pack -> L1 `mid`          (loop over output tiles)
//   TRISC3:                mid -> unpack to SrcS -> exp -> pack -> out    (one call, all tiles)
//
// Compared with route 1 (matmul_exp_dest_quasar.cpp) this costs one extra pack and one extra unpack
// per tile through `mid`, and in exchange the exp leaves the MATH thread: while TRISC3 runs exp on
// tile t, MATH is already accumulating tile t + 1. Worth it only when that overlap pays for the
// L1 round trip; measure before choosing it.
//
// The two halves live in one source file. The matmul loop expands on UNPACK, MATH and PACK; the two
// SFPU lines expand on TRISC3 only (SFPU_SRCS). Same op::Exp, same exp math as route 1.
//
// NOT RUNNABLE YET. TRISC3 has no DataflowBuffer interface (issue #63416), so the SrcS call takes
// the L1 base of `mid` as an address and cannot wait for PACK to fill each entry: with the two halves
// running concurrently it would read `mid` before the matmul wrote it. Once TRISC3 owns
// DataflowBuffers the two SFPU lines become
//     sfpu_init<op::Exp, SfpuReg::SrcS>(mid, out);
//     sfpu_tiles<op::Exp, SfpuReg::SrcS>(mid, out, num_out_tiles);
// with wait_front / pop_front on `mid` and reserve_back / push_back on `out` inside the tile loop,
// and nothing else in this file changes.
//
// Compile-time args: 0 = in0 DFB id, 1 = in1 DFB id, 2 = mid DFB id (matmul output, exp input),
// 3 = Kt, 4 = number of output tiles, 5 = L1 base of mid (16-byte units), 6 = L1 base of out
// (16-byte units), 7 = L1 data format of mid and out, 8 = SrcS register format.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr std::uint32_t in0_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t in1_id = get_compile_time_arg_val(1);
    constexpr std::uint32_t mid_id = get_compile_time_arg_val(2);
    constexpr std::uint32_t Kt = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_out_tiles = get_compile_time_arg_val(4);
    constexpr std::uint32_t mid_addr_16B = get_compile_time_arg_val(5);
    constexpr std::uint32_t out_addr_16B = get_compile_time_arg_val(6);
    constexpr DataFormat l1_format = static_cast<DataFormat>(get_compile_time_arg_val(7));
    constexpr DataFormat srcs_format = static_cast<DataFormat>(get_compile_time_arg_val(8));

    // ---- UNPACK / MATH / PACK: matmul, result to `mid` ----------------------------------------
    DataflowBuffer in0(static_cast<std::uint16_t>(in0_id));
    DataflowBuffer in1(static_cast<std::uint16_t>(in1_id));
    DataflowBuffer mid(static_cast<std::uint16_t>(mid_id));

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_id, in1_id, mid_id);
    matmul_init(in0_id, in1_id);

    for (std::uint32_t t = 0; t < num_out_tiles; ++t) {
        tile_regs_acquire();
        for (std::uint32_t kt = 0; kt < Kt; ++kt) {
            in0.wait_front(1);
            in1.wait_front(1);
            matmul_tiles(in0_id, in1_id, 0, 0, 0);  // FPU: accumulate into Dest[0]
            in0.pop_front(1);
            in1.pop_front(1);
        }
        tile_regs_commit();  // no SFPU op here: MATH goes straight to the next tile

        mid.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, mid_id);  // PACK: Dest[0] -> mid
        mid.push_back(1);
        tile_regs_release();
    }

    // ---- TRISC3: exp on every tile of `mid`, result to `out` ---------------------------------
    sfpu_init<op::Exp, SfpuReg::SrcS>(mid_addr_16B, l1_format, out_addr_16B, l1_format, srcs_format);
    sfpu_tiles<op::Exp, SfpuReg::SrcS>(srcs_format, num_out_tiles);  // unpack -> exp -> pack, per tile
}
