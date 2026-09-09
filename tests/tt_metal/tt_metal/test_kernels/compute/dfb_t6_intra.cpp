// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"          // for dummy_pack (TEN-4746 no-write pack ordering)
#include "api/compute/tile_move_copy.h"  // for dummy_unpack (TEN-4746 unpack pop ordering)
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr std::uint32_t words_per_entry = get_arg(args::words_per_entry);

    // Both PRODUCER ("out") and CONSUMER ("in") bindings on this kernel reference
    // the same self-looped DFB, so dfb::out and dfb::in resolve to the same ID.
    DataflowBuffer dfb(dfb::out);

#ifdef UCK_CHLKC_UNPACK
    std::uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    // dummy_pack's PACR_STRIDE validates a pack-partition bd_table entry; compute_kernel_hw_startup
    // is what runs llk_pack_init and programs that entry. copy_init is not needed: dummy_unpack is
    // UNPACR_NOP and does not fetch a descriptor.
    compute_kernel_hw_startup(dfb::out, dfb::out);

    for (std::uint32_t i = 0; i < entries_per_neo; i++) {
        // Pack TRISC: wait for free space, increment entry in-place, post credit.
        dfb.reserve_back(1);
        // TEN-4746: the pack thread wrote L1 directly (no PACR) since reserve_back, so push_back would
        // trip the pack-side ordering guard. A no-write dummy pack issues a real PACR to order the push
        // after the reserve without clobbering the manual increments above.
        dummy_pack(dfb::out);
#ifdef UCK_CHLKC_PACK
        {
            ckernel::tensix_sync();
            volatile std::uint32_t* entry = reinterpret_cast<volatile std::uint32_t*>(dfb.get_write_ptr() << 4);
            for (std::uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
        }
#endif
        dfb.push_back(1);

        dfb.wait_front(1);
        // TEN-4746: a real UNPACR must sit between wait_front and pop_front. dummy_unpack also
        // gates the unpacker on WAIT_TILES; tensix_sync then blocks this RISC until that UNPACR
        // retires, so the scalar increments below cannot race an unwritten slot.
        dummy_unpack(dfb::out);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            ckernel::tensix_sync();
            volatile std::uint32_t* entry = reinterpret_cast<volatile std::uint32_t*>(dfb.get_read_ptr() << 4);
            for (std::uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
        }
#endif
        dfb.pop_front(1);
    }

    dfb.finish();
}
