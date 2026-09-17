// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"  // dummy_unpack (TEN-4746)
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::in);
    // copy_init not needed: drain uses dummy_unpack (UNPACR_NOP), not copy_tile.
    compute_kernel_hw_startup(dfb.get_id(), dfb.get_id());

    // Keep both entries at the front so tile_index 1 exercises fifo_page_size stride.
    // Each TRISC thread writes the same mailbox-broadcast results to its own L1 slot so
    // the host can verify UNPACK/MATH/PACK all received the same values.
    constexpr uint32_t k_num_results = 7;
    uint32_t results[k_num_results] = {};

    dfb.wait_front(num_entries_per_consumer);

#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
    // Peek L1 right after wait_front (all three TRISCs call get_tile_address for the mailbox handshake).
    {
        const uint32_t tile0_addr = dfb.get_tile_address(0);
        const uint32_t tile1_addr = dfb.get_tile_address(1);
        volatile tt_l1_ptr uint32_t* const tile0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile0_addr);
        volatile tt_l1_ptr uint32_t* const tile1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile1_addr);
        DPRINT(
            "after wait_front tile0=0x{:x} w0=0x{:x} w1=0x{:x} tile1=0x{:x} w0=0x{:x} w1=0x{:x}\n",
            tile0_addr,
            tile0[0],
            tile0[1],
            tile1_addr,
            tile1[0],
            tile1[1]);
#ifdef ARCH_QUASAR
        volatile tt_l1_ptr uint32_t* const tile0_uc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile0_addr + MEM_L1_UNCACHED_BASE);
        volatile tt_l1_ptr uint32_t* const tile1_uc =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile1_addr + MEM_L1_UNCACHED_BASE);
        DPRINT(
            "after wait_front uncached tile0=0x{:x} w0=0x{:x} w1=0x{:x} tile1=0x{:x} w0=0x{:x} w1=0x{:x}\n",
            tile0_addr + MEM_L1_UNCACHED_BASE,
            tile0_uc[0],
            tile0_uc[1],
            tile1_addr + MEM_L1_UNCACHED_BASE,
            tile1_uc[0],
            tile1_uc[1]);
#endif
    }
#endif

    results[0] = dfb.read_tile_value<uint32_t>(0, 0);
    results[1] = dfb.read_tile_value<uint32_t>(0, 1);
    results[2] = dfb.read_tile_value<uint32_t>(1, 0);
    results[3] = dfb.read_tile_value<uint32_t>(1, 1);

    const uint32_t tile_addr = dfb.get_tile_address(1);
#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_ISOLATE_SFPU)
    DPRINT(
        "after get_tile_address r0=0x{:x} r1=0x{:x} r2=0x{:x} r3=0x{:x} "
        "tile1_addr=0x{:x}\n",
        results[0],
        results[1],
        results[2],
        results[3],
        tile_addr);
#endif

    results[4] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
    results[5] = static_cast<uint32_t>(dfb.read_tile_value<uint16_t>(1, 0));
    results[6] = static_cast<uint32_t>(dfb.read_tile_value<uint16_t>(1, 1));

#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_ISOLATE_SFPU)
    DPRINT(
        "after read_tile_value r0=0x{:x} r1=0x{:x} r2=0x{:x} r3=0x{:x} r4=0x{:x} r5=0x{:x} r6=0x{:x} "
        "tile1_addr=0x{:x}\n",
        results[0],
        results[1],
        results[2],
        results[3],
        results[4],
        results[5],
        results[6],
        tile_addr);
#endif

    // Drain without tile_regs_acquire/copy_tile/release: that path left PACK's Tensix busy so
    // firmware tensix_sync() after kernel_main hung with MEM_READ_NO_RESPONSE. TEN-4746 still
    // requires a real UNPACR between wait_front and pop_front — use UNPACR_NOP via dummy_unpack.
    for (uint32_t i = 0; i < num_entries_per_consumer; ++i) {
        dummy_unpack(dfb.get_id());
        dfb.pop_front(1);
    }

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
