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
 */
void kernel_main() {
    constexpr uint32_t count = get_compile_time_arg_val(0);
    constexpr uint32_t slots = 8;
    constexpr uint32_t valid_counts[] = {0, 1, 7, 8, 9, 15, 16, 17, 511, 512, 513, 1023, 1024};
    constexpr uint32_t dst_indices[] = {0, 3, 7};
    constexpr uint32_t input = tt::CBIndex::c_0;
    constexpr uint32_t output = tt::CBIndex::c_16;
    CircularBuffer in(input);
    CircularBuffer out(output);
    compute_kernel_hw_startup(input, output);
    copy_init(input);
    for (uint32_t tile = 0; tile < count; tile += slots) {
        const uint32_t batch = tile / slots;
        const uint32_t dst = dst_indices[batch % 3];
        in.wait_front(slots);
        out.reserve_back(slots);
        tile_regs_acquire();
        for (uint32_t slot = 0; slot < slots; ++slot) {
            copy_tile(input, slot, slot);
        }
        block_max8_init();
        block_max8(dst, valid_counts[batch]);
        tile_regs_commit();
        tile_regs_wait();
        pack_rows_to_addr_init(8);
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
