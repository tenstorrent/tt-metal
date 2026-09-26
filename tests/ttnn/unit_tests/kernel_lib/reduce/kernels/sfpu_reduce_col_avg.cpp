// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Column AVG of each input tile through the public Compute API (sfpu_reduce), the same
// copy_tile -> sfpu_reduce_init -> sfpu_reduce sequence the ttnn reduce helper uses for its SFPU path.

namespace {

constexpr std::uint32_t cb_input_id = 0;
constexpr std::uint32_t cb_output_id = 16;

}  // namespace

void kernel_main() {
    const std::uint32_t num_tiles = get_arg_val<std::uint32_t>(0);

    CircularBuffer cb_input(cb_input_id);
    CircularBuffer cb_output(cb_output_id);

    compute_kernel_hw_startup(cb_input.get_cb_id(), cb_output.get_cb_id());
    copy_init(cb_input.get_cb_id());

    // The input CB is backed by a sharded tensor that is already in L1; make it visible.
    cb_input.reserve_back(num_tiles);
    cb_input.push_back(num_tiles);

    for (std::uint32_t tile = 0; tile < num_tiles; ++tile) {
        tile_regs_acquire();
        cb_input.wait_front(1);
        copy_tile(cb_input.get_cb_id(), 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        sfpu_reduce_init<PoolType::AVG, REDUCE_FORMAT>();
        sfpu_reduce<PoolType::AVG, REDUCE_FORMAT, ReduceDim::REDUCE_COL>(0 /*idst*/);

        tile_regs_commit();
        tile_regs_wait();

        cb_output.reserve_back(1);
        pack_tile(0 /*ifrom_dst*/, cb_output.get_cb_id());
        cb_output.push_back(1);

        cb_input.pop_front(1);
        tile_regs_release();
    }
}
