// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/experimental/block_max8.h"
#include "api/compute/experimental/pack_rows_to_addr.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

/** Exercise the public unary API and compact row pack with eight occupied DST slots.
 * Other slots use ordinary tile packing after restoring pack state. Thirteen
 * acquisitions visit masking boundaries and repeatedly reuse both DST halves.
 * The host supplies geometry as compile arguments and (DST index, valid count)
 * pairs as runtime arguments, shared with the golden calculation.
 * Dirty ADDR_MOD_7 before each init to verify that fused-op state is reset.
 */
void kernel_main() {
    constexpr uint32_t count = get_compile_time_arg_val(0);
    constexpr uint32_t slots = get_compile_time_arg_val(1);
    constexpr uint32_t result_rows = get_compile_time_arg_val(2);
    constexpr uint32_t input = tt::CBIndex::c_0;
    constexpr uint32_t output = tt::CBIndex::c_16;
    CircularBuffer in(input);
    CircularBuffer out(output);
    compute_kernel_hw_startup(input, output);
    copy_init(input);
    for (uint32_t tile = 0; tile < count; tile += slots) {
        const uint32_t batch = tile / slots;
        const uint32_t dst = get_arg_val<uint32_t>(2 * batch);
        const uint32_t valid_scores = get_arg_val<uint32_t>(2 * batch + 1);
        in.wait_front(slots);
        out.reserve_back(slots);
        tile_regs_acquire();
        for (uint32_t slot = 0; slot < slots; ++slot) {
            copy_tile(input, slot, slot);
        }
        MATH((addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 32}}.set(ADDR_MOD_7)));
        block_max8_init();
        block_max8(dst, valid_scores);
        tile_regs_commit();
        tile_regs_wait();
        pack_rows_to_addr_init(result_rows);
        PACK((pack_rows_to_addr(
            dst, get_local_cb_interface(output).fifo_wr_ptr + dst * get_local_cb_interface(output).fifo_page_size)));
        pack_rows_to_addr_uninit();
        pack_init(output);
        for (uint32_t slot = 0; slot < slots; ++slot) {
            if (slot != dst) {
                pack_tile<true>(slot, output, slot);
            }
        }
        tile_regs_release();
        in.pop_front(slots);
        out.push_back(slots);
    }
}
