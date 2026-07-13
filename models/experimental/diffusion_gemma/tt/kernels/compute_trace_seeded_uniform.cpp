// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t seed_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb = get_compile_time_arg_val(1);

    const uint32_t seed_offset = get_arg_val<uint32_t>(0);
    const uint32_t core_index = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t from_bits = get_arg_val<uint32_t>(3);
    const uint32_t to_bits = get_arg_val<uint32_t>(4);

    CircularBuffer cb_seed(seed_cb);
    CircularBuffer cb_output(output_cb);
    cb_seed.wait_front(1);
    const uint32_t base_seed = cb_seed.read_tile_value(/*tile_index=*/0, /*element_offset=*/0);
    cb_seed.pop_front(1);

    union {
        float f;
        uint32_t u;
    } from, to, scale;
    from.u = from_bits;
    to.u = to_bits;
    scale.f = to.f - from.f;

    init_sfpu(output_cb, output_cb);
    rand_tile_init(base_seed + seed_offset + core_index);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        cb_output.reserve_back(1);
        tile_regs_acquire();
        rand_tile(0, from.u, scale.u);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output_cb, 0);
        tile_regs_release();
        cb_output.push_back(1);
    }
}
