// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

/** Race repro through the metal runtime. Each batch holds 8 A tiles then 8 B tiles in one CB.
 * Per DST section: copy A slots 0..3, math-only delay, copy A slots 4..5, then add A+B on slots 6..7.
 * Compile argument 1 is the delay in cycles; argument 2 adds an unpack drain before add_init.
 */
void kernel_main() {
    constexpr uint32_t batches = get_compile_time_arg_val(0);
    constexpr uint32_t delay_cycles = get_compile_time_arg_val(1);
    constexpr bool unpack_stall = get_compile_time_arg_val(2);
    constexpr uint32_t slots = 8;
    constexpr uint32_t copies_before = 4;
    constexpr uint32_t copies_after = 6;
    constexpr uint32_t input = tt::CBIndex::c_0;
    constexpr uint32_t output = tt::CBIndex::c_16;
    CircularBuffer in(input);
    CircularBuffer out(output);
    compute_kernel_hw_startup(input, output);
    for (uint32_t batch = 0; batch < batches; ++batch) {
        in.wait_front(2 * slots);
        out.reserve_back(slots);
        tile_regs_acquire();
        copy_init(input);
        for (uint32_t slot = 0; slot < copies_before; ++slot) {
            copy_tile(input, slot, slot);
        }
        if constexpr (delay_cycles > 0) {
            MATH((tensix_sync()));
            MATH((ckernel::wait(delay_cycles)));
        }
        for (uint32_t slot = copies_before; slot < copies_after; ++slot) {
            copy_tile(input, slot, slot);
        }
        if constexpr (unpack_stall) {
            UNPACK((TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK)));
        }
        add_init(input, input);
        for (uint32_t slot = copies_after; slot < slots; ++slot) {
            add_tiles(input, input, slot, slots + slot, slot);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t slot = 0; slot < slots; ++slot) {
            pack_tile(slot, output);
        }
        tile_regs_release();
        in.pop_front(2 * slots);
        out.push_back(slots);
    }
}
