// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/hadamard.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t count = get_compile_time_arg_val(0);
    constexpr bool normalize = get_compile_time_arg_val(1) != 0;
    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer weights(tt::CBIndex::c_1);
    CircularBuffer copy_input(tt::CBIndex::c_2);
    CircularBuffer output(tt::CBIndex::c_16);
    CircularBuffer copy_output(tt::CBIndex::c_17);

    compute_kernel_hw_startup(tt::CBIndex::c_2, tt::CBIndex::c_17);
    // The reader preloads all inputs and H16 copies before streaming c_2.
    // Keep the first H16 tile at a fixed address throughout the kernel.
    weights.wait_front(count);
    for (std::uint32_t tile = 0; tile < count; ++tile) {
        copy_input.wait_front(1);
        copy_output.reserve_back(1);
        reconfig_full_operand<SrcOrder::Regular>(tt::CBIndex::c_2, tt::CBIndex::c_2);
        copy_init(tt::CBIndex::c_2);
        pack_reconfig_data_format<true>(tt::CBIndex::c_17);
        tile_regs_acquire();
        copy_tile(tt::CBIndex::c_2, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_17);
        tile_regs_release();
        copy_input.pop_front(1);
        copy_output.push_back(1);

        input.wait_front(1);
        output.reserve_back(1);
        tile_regs_acquire();
        // Hadamard needs a clean tile, including its scratch face. Only clear
        // the acquired tile; PACK may still be draining the other DEST half.
        fill_tile_init();
        fill_tile(0, 0.0f);
#if defined(TRISC_MATH) || defined(TRISC_PACK)
        const auto saved_dest_offset = ckernel::dest_offset_id;
#endif
#ifdef TRISC_PACK
        const auto saved_pack_tile_ptr = pack_sync_tile_dst_ptr;
#endif
        hadamard_h128_init_short<normalize>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
#if defined(TRISC_MATH) || defined(TRISC_PACK)
        LLK_ASSERT(ckernel::dest_offset_id == saved_dest_offset, "Hadamard short init changed the DEST section");
#endif
#ifdef TRISC_PACK
        LLK_ASSERT(pack_sync_tile_dst_ptr == saved_pack_tile_ptr, "Hadamard short init reset the pack tile pointer");
#endif
        hadamard_h128_tile<normalize>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();
        input.pop_front(1);
        output.push_back(1);
        // The next iteration switches back to full-tile BF16 copy and reuses
        // DEST, exercising both the entry and exit of the Hadamard micro-op.
    }
    weights.pop_front(count);
}
