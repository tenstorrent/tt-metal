// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/csa_index_remap.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t count = get_compile_time_arg_val(0);
    constexpr std::uint32_t block = 3;
    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer output(tt::CBIndex::c_16);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);

    for (std::uint32_t tile = 0; tile < count; tile += block) {
        input.wait_front(block);
        output.reserve_back(block);
        tile_regs_acquire();
        for (std::uint32_t i = 0; i < block; ++i) {
            copy_tile(tt::CBIndex::c_0, i, i);
        }
        // A fused operation may have used the SFPU address modifier. The public
        // initializer must restore it without relying on a private test prelude.
#ifdef TRISC_MATH
        ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 32}}.set(ADDR_MOD_7);
#endif
        csa_index_remap_init();
        csa_index_remap<256>(1);
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t i = 0; i < block; ++i) {
            pack_tile(i, tt::CBIndex::c_16);
        }
        tile_regs_release();
        input.pop_front(block);
        output.push_back(block);
    }
}
