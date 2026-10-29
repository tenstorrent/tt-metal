// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"          // for dummy_pack (TEN-4746 no-write pack ordering)
#include "api/compute/tile_move_copy.h"  // for dummy_unpack (TEN-4746 unpack pop ordering)
#include "api/kernel_thread_globals.h"   // for get_my_thread_id
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr std::uint32_t words_per_entry = get_arg(args::words_per_entry);
    // Which TRISC the host put on each side of the self-looped DFB: trisc2 packs and trisc0 unpacks
    // by default; trisc3 can take over either side or both. That thread does the RISC-side
    // increments below.
    constexpr bool pack_on_trisc3 = get_arg(args::pack_on_trisc3) != 0;
    constexpr bool unpack_on_trisc3 = get_arg(args::unpack_on_trisc3) != 0;
    constexpr bool i_produce = ckernel::TRISC_ID == (pack_on_trisc3 ? 3u : 2u);
    constexpr bool i_consume = ckernel::TRISC_ID == (unpack_on_trisc3 ? 3u : 0u);
    // Entry address of every push, then of every pop, for the host to compare (one row per Neo).
    volatile tt_l1_ptr std::uint32_t* const push_trace = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(
        get_arg(args::trace_l1_address) + get_my_thread_id() * 2 * entries_per_neo * sizeof(std::uint32_t));
    volatile tt_l1_ptr std::uint32_t* const pop_trace = push_trace + entries_per_neo;

    // Both PRODUCER ("out") and CONSUMER ("in") bindings on this kernel reference
    // the same self-looped DFB, so dfb::out and dfb::in resolve to the same ID.
    DataflowBuffer dfb(dfb::out);

    // dummy_pack's PACR_STRIDE validates a pack-partition bd_table entry; compute_kernel_hw_startup
    // is what runs llk_pack_init and programs that entry. copy_init is not needed: dummy_unpack is
    // UNPACR_NOP and does not fetch a descriptor.
    compute_kernel_hw_startup(dfb::out, dfb::out);

    for (std::uint32_t i = 0; i < entries_per_neo; i++) {
        // Producer: wait for free space, increment the entry in place, post the credit.
        dfb.reserve_back(1);
        // TEN-4746: issue a no-write PACR after reserve_back to order the later push_back. The pack
        // thread waits for it below before directly incrementing the reserved L1 entry. trisc3 has no
        // pack op; its reserve_back already blocked the RISC until the space was free.
        dummy_pack(dfb::out);
        if constexpr (i_produce) {
            ckernel::tensix_sync();
            const std::uint32_t entry_addr = dfb.get_write_ptr() << 4;
            push_trace[i] = entry_addr;
            volatile std::uint32_t* entry = reinterpret_cast<volatile std::uint32_t*>(entry_addr);
            for (std::uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
            asm volatile("fence" ::: "memory");  // the increments land before push_back posts the credit
        }
        dfb.push_back(1);

        dfb.wait_front(1);
        // TEN-4746: a real UNPACR must sit between wait_front and pop_front. dummy_unpack also
        // gates the unpacker on WAIT_TILES; tensix_sync then blocks this RISC until that UNPACR
        // retires, so the scalar increments below cannot race an unwritten slot. trisc3 has no
        // unpack op; its wait_front already blocked the RISC until the entry was posted.
        dummy_unpack(dfb::out);
        if constexpr (i_consume) {
            ckernel::tensix_sync();
            const std::uint32_t entry_addr = dfb.get_read_ptr() << 4;
            pop_trace[i] = entry_addr;
            volatile std::uint32_t* entry = reinterpret_cast<volatile std::uint32_t*>(entry_addr);
            for (std::uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
            asm volatile("fence" ::: "memory");  // the increments land before pop_front frees the slot
        }
        dfb.pop_front(1);
    }

    dfb.finish();
}
