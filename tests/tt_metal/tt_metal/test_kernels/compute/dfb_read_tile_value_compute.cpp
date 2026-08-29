// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DM → Tensix consumer drain (1Sx1S), with read_tile_value / get_tile_address
// after each wait_front(1). Quasar also builds TRISC_ISOLATE_SFPU (TRISC3); that
// thread must not enter the mailbox APIs (UNPACK only writes Math+Pack), so the
// peeks are gated to UNPACK/MATH/PACK only.

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::in);

    // HW requires a copy (unpack) between wait_front and pop_front. No output DFB —
    // configure unpack+pack against the input and discard the copied tile.
    unary_op_init_common(dfb.get_id(), dfb.get_id());

    // Per drain iteration (tile_index 0 = current front):
    //   [0] read_tile_value<uint32_t>(0, 0)
    //   [1] read_tile_value<uint32_t>(0, 1)
    //   [2] *get_tile_address(0) as uint32_t
    constexpr uint32_t k_results_per_entry = 3;
    uint32_t results[num_entries_per_consumer * k_results_per_entry] = {};

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        acquire_dst();
        dfb.wait_front(1);

#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
        {
            const uint32_t base = tile_id * k_results_per_entry;
            results[base + 0] = dfb.read_tile_value<uint32_t>(0, 0);
            results[base + 1] = dfb.read_tile_value<uint32_t>(0, 1);
            const uint32_t tile_addr = dfb.get_tile_address(0);
            results[base + 2] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
        }
#endif

        copy_tile(dfb.get_id(), 0, 0);
        dfb.pop_front(1);
        release_dst();
    }
    dfb.finish();

#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
#ifdef ARCH_QUASAR
    const uint32_t result_l1_ptr_addr = result_l1_addr + MEM_L1_UNCACHED_BASE;
#else
    const uint32_t result_l1_ptr_addr = result_l1_addr;
#endif
    volatile tt_l1_ptr uint32_t* const out =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_ptr_addr);
#if defined(TRISC_UNPACK)
    constexpr uint32_t slot_base = 0 * (num_entries_per_consumer * k_results_per_entry);
#elif defined(TRISC_MATH)
    constexpr uint32_t slot_base = 1 * (num_entries_per_consumer * k_results_per_entry);
#elif defined(TRISC_PACK)
    constexpr uint32_t slot_base = 2 * (num_entries_per_consumer * k_results_per_entry);
#endif
    for (uint32_t i = 0; i < num_entries_per_consumer * k_results_per_entry; ++i) {
        out[slot_base + i] = results[i];
    }
#endif
}
