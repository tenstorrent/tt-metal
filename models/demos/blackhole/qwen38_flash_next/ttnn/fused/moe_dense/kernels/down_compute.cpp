// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The shared expert's down linear as a streaming matmul, `n_tiles` output column tiles per core: the intermediate row
// (k_tiles tiles, multicast into cb_in0 by the eltwise cores) times each of this core's weight columns, accumulated in
// the 32-bit dest in K order 0..k_tiles-1 and packed bf16.  With spill > 0 the running partial is packed to the fp32
// intermediate CB and reloaded through SrcA every `spill` K tiles, as the DRAM-sharded matmul does between its K
// blocks (its intermediate CB has no UnpackToDestFp32, so each reload rounds the partial).
// gr_read/kernels/down_compute.cpp with an output-tile loop.
// CBs: in0 intermediate row (bf16, k_tiles), in1 weight column tiles (the weight's format, n_tiles x k_tiles pushed
// tiles_per_push at a time), interm (fp32, 1), out (bf16, 2). Compile-time args: 0 k_tiles (the builder's K_TILES),
// 1 tiles_per_push (DOWN_BLK, divides k_tiles), 2-4 the cbs (row, weight, out), 5 spill, 6 the intermediate cb,
// 7 n_tiles (output tiles per core).

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t k_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_push = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t spill = get_compile_time_arg_val(5);
    constexpr uint32_t cb_interm = get_compile_time_arg_val(6);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(7);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_interm);
    matmul_block_init(cb_in0, cb_in1, 0, 1, 1, 1);
    DataflowBuffer in0(cb_in0);
    DataflowBuffer in1(cb_in1);
    DataflowBuffer intermediate(cb_interm);
    DataflowBuffer out(cb_out);

    {
        in0.wait_front(k_tiles);
        for (uint32_t n = 0; n < n_tiles; ++n) {
            if constexpr (spill > 0) {
                pack_reconfig_data_format(
                    cb_interm);  // the previous tile's final pack left the packer on the output format
            }
            tile_regs_acquire();
            for (uint32_t k = 0; k < k_tiles; k += tiles_per_push) {
                in1.wait_front(tiles_per_push);
                for (uint32_t kk = 0; kk < tiles_per_push; ++kk) {
                    matmul_block(cb_in0, cb_in1, k + kk, kk, 0, 0, 1, 1, 1);
                }
                in1.pop_front(tiles_per_push);
                if constexpr (spill > 0) {
                    const uint32_t next = k + tiles_per_push;
                    if (next % spill == 0 && next < k_tiles) {
                        tile_regs_commit();
                        intermediate.reserve_back(1);
                        tile_regs_wait();
                        pack_tile(0, cb_interm);
                        tile_regs_release();
                        intermediate.push_back(1);
                        tile_regs_acquire();
                        reconfig_data_format_srca(cb_in1, cb_interm);
                        copy_init(cb_interm);
                        intermediate.wait_front(1);
                        copy_tile(cb_interm, 0, 0);
                        intermediate.pop_front(1);
                        reconfig_data_format_srca(cb_interm, cb_in1);
                        matmul_block_init(cb_in0, cb_in1, 0, 1, 1, 1);
                    }
                }
            }
            tile_regs_commit();
            out.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            out.push_back(1);
        }
        in0.pop_front(k_tiles);
    }
}
