// SPDX-License-Identifier: Apache-2.0
// Custom Metalium compute kernel: y = x * x (elementwise square) via FPU mul_tiles.
// cpp rung attempt for the eltwise BinaryNg (relu^2 square) op.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    CircularBuffer buff_in(tt::CBIndex::c_0);
    CircularBuffer buff_out(tt::CBIndex::c_16);
    const uint32_t in_id = tt::CBIndex::c_0;
    const uint32_t out_id = tt::CBIndex::c_16;

    binary_op_init_common(in_id, in_id, out_id);
    mul_tiles_init(in_id, in_id);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        for (uint32_t i = 0; i < per_core_block_dim; ++i) {
            buff_in.wait_front(1);
            buff_out.reserve_back(1);

            tile_regs_acquire();
            mul_tiles(in_id, in_id, 0, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, out_id);
            tile_regs_release();

            buff_out.push_back(1);
            buff_in.pop_front(1);
        }
    }
}
