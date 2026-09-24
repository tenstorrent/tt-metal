// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Column AVG of each input tile through the public Compute API (sfpu_reduce), the same
// copy_tile -> sfpu_reduce_init -> sfpu_reduce sequence the ttnn reduce helper uses for its SFPU path.

namespace {

constexpr std::uint32_t cb_input = 0;
constexpr std::uint32_t cb_output = 16;

}  // namespace

void kernel_main() {
    const std::uint32_t num_tiles = get_arg_val<std::uint32_t>(0);

    unary_op_init_common(cb_input, cb_output);

    // The input CB is backed by a sharded tensor that is already in L1; make it visible.
    cb_reserve_back(cb_input, num_tiles);
    cb_push_back(cb_input, num_tiles);

    for (std::uint32_t tile = 0; tile < num_tiles; ++tile) {
        tile_regs_acquire();
        cb_wait_front(cb_input, 1);
        copy_tile(cb_input, 0, 0);

        sfpu_reduce_init<PoolType::AVG, REDUCE_FORMAT>();
        sfpu_reduce<PoolType::AVG, REDUCE_FORMAT, ReduceDim::REDUCE_COL>(0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_output, 1);
        pack_tile(0, cb_output);
        cb_push_back(cb_output, 1);

        cb_pop_front(cb_input, 1);
        tile_regs_release();
    }
}
