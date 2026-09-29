// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR read stage 2b, one output column tile per core: the normalized row (Kt tiles) times the core's weight column,
// accumulated in the 32-bit dest in K order 0..Kt-1, packed fp32.  With spill > 0 the running partial is packed to
// the fp32 intermediate CB and reloaded through SrcA every `spill` K tiles, as the DRAM-sharded matmul does between
// its K blocks (the intermediate CB carries no UnpackToDestFp32 there, so each reload rounds the partial); spill = 0 is
// the single-pass accumulation of the default matmul program.
// CBs: in0 normalized row (bf16, Kt), in1 weight column (bf16, 2 x blk), interm (fp32, 1), out partial (fp32, 1).
// Compile-time args: 0 Kt, 1 blk (weight tiles per push, divides Kt), 2-4 the cbs (row, weight, partial), 5 spill,
// 6 the intermediate cb.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t c_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t c_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t c_out = get_compile_time_arg_val(4);
    constexpr uint32_t spill = get_compile_time_arg_val(5);
    constexpr uint32_t c_interm = get_compile_time_arg_val(6);

    compute_kernel_hw_startup<SrcOrder::Reverse>(c_in0, c_in1, c_interm);
    matmul_block_init(c_in0, c_in1, 0, 1, 1, 1);
    DataflowBuffer in0(c_in0);
    DataflowBuffer in1(c_in1);
    DataflowBuffer interm(c_interm);
    DataflowBuffer out(c_out);

    {
        FUSED_ZONE("fz_gr_down_c_main");
        in0.wait_front(Kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < Kt; k += blk) {
            in1.wait_front(blk);
            for (uint32_t kk = 0; kk < blk; ++kk) {
                matmul_block(c_in0, c_in1, k + kk, kk, 0, 0, 1, 1, 1);
            }
            in1.pop_front(blk);
            if constexpr (spill > 0) {
                const uint32_t next = k + blk;
                if (next % spill == 0 && next < Kt) {
                    tile_regs_commit();
                    interm.reserve_back(1);
                    tile_regs_wait();
                    pack_tile(0, c_interm);
                    tile_regs_release();
                    interm.push_back(1);
                    tile_regs_acquire();
                    reconfig_data_format_srca(c_in1, c_interm);
                    copy_init(c_interm);
                    interm.wait_front(1);
                    copy_tile(c_interm, 0, 0);
                    interm.pop_front(1);
                    reconfig_data_format_srca(c_interm, c_in1);
                    matmul_block_init(c_in0, c_in1, 0, 1, 1, 1);
                }
            }
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(c_out);
        pack_tile(0, c_out);
        tile_regs_release();
        out.push_back(1);
        in0.pop_front(Kt);
    }
}
