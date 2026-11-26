// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/softmax_k.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t num_rows = get_compile_time_arg_val(0);
    constexpr std::uint32_t k = get_compile_time_arg_val(1);

    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer output(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    input.wait_front(1);
    output.reserve_back(1);

    copy_init(tt::CBIndex::c_0);
    softmax_k_tile_init();
    tile_regs_acquire();
    copy_tile(tt::CBIndex::c_0, 0, 0);
    softmax_k_tile<num_rows>(0, k);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_16);
    tile_regs_release();

    input.pop_front(1);
    output.push_back(1);
}
