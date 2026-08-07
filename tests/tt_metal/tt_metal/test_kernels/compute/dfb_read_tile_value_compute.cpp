// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::in);
    unary_op_init_common(dfb.get_id(), dfb.get_id());

    // Keep both entries at the front so tile_index 1 exercises fifo_page_size stride.
    // Each TRISC thread writes the same mailbox-broadcast results to its own L1 slot so
    // the host can verify UNPACK/MATH/PACK all received the same values.
    constexpr uint32_t k_num_results = 7;
    uint32_t results[k_num_results] = {};

    dfb.wait_front(num_entries_per_consumer);

    results[0] = dfb.read_tile_value<uint32_t>(0, 0);
    results[1] = dfb.read_tile_value<uint32_t>(0, 1);
    results[2] = dfb.read_tile_value<uint32_t>(1, 0);
    results[3] = dfb.read_tile_value<uint32_t>(1, 1);

    const uint32_t tile_addr = dfb.get_tile_address(1);
    results[4] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
    results[5] = static_cast<uint32_t>(dfb.read_tile_value<uint16_t>(1, 0));
    results[6] = static_cast<uint32_t>(dfb.read_tile_value<uint16_t>(1, 1));

    tile_regs_acquire();
    for (uint32_t i = 0; i < num_entries_per_consumer; ++i) {
        copy_tile(dfb.get_id(), 0, 0);  // dummy copy to avoid UNPACK wait -> pop trap on Quasar
        dfb.pop_front(1);
    }
    tile_regs_release();

#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
#ifdef ARCH_QUASAR
    const uint32_t result_l1_ptr_addr = result_l1_addr + MEM_L1_UNCACHED_BASE;
#else
    const uint32_t result_l1_ptr_addr = result_l1_addr;
#endif
    volatile tt_l1_ptr uint32_t* const out =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_ptr_addr);
#if defined(TRISC_UNPACK)
    constexpr uint32_t slot_base = 0 * k_num_results;
#elif defined(TRISC_MATH)
    constexpr uint32_t slot_base = 1 * k_num_results;
#elif defined(TRISC_PACK)
    constexpr uint32_t slot_base = 2 * k_num_results;
#endif
    for (uint32_t i = 0; i < k_num_results; ++i) {
        out[slot_base + i] = results[i];
    }
#endif

    dfb.finish();
}
