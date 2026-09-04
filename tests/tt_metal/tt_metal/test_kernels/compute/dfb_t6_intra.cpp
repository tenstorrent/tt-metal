// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
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

    compute_kernel_hw_startup(dfb::out, dfb::out);
    copy_init(dfb::out);

    for (std::uint32_t i = 0; i < entries_per_neo; i++) {
        // Pack TRISC: wait for free space, increment entry in-place, post credit.
        dfb.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            volatile std::uint32_t* entry = reinterpret_cast<volatile std::uint32_t*>(dfb.get_write_ptr() << 4);
            for (std::uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
            // Publish these scalar stores before the credit that releases the entry to unpack.
            asm volatile("fence w, w" ::: "memory");

            // TEN-4746: this producer fills the entry with scalar stores, so without this there is
            // no PACR between reserve_back's WAIT_FREE and push_back's PUSH_TILES, and the WAIT_FREE
            // is free to resolve before the slot really is. Issue a real pack purely for its PACR.
            //
            // It has to name this same dfb -- llk_pack disarms the guard for the dfb it is handed,
            // and the constraint likewise wants a TDMA on the same buffer -- so it is aimed at the
            // scratch slot at the end of this Neo's ring instead of the entry being pushed.
            //
            // out_of_order_output resolves the target to wr_entry_idx + output_tile_index with no
            // wrap. On iteration i this Neo's cursor sits at base + i*stride and its scratch slot is
            // base + (ring_entries_per_neo - 1)*stride, so the difference below is what lands on it.
            //
            // DEST holds whatever the previous iteration's copy_tile left; that is deliberate. The
            // packed bytes are never read, so no MATH->PACK dest handshake is taken out for them.
            pack_tile<true /* out_of_order_output */>(
                0, dfb::out, neo_stride_entries * ((ring_entries_per_neo - 1) - i));
        }
#endif
        // TEN-4746: the pack thread wrote L1 directly (no PACR) since reserve_back, so push_back would
        // trip the pack-side ordering guard. A no-write dummy pack issues a real PACR to order the push
        // after the reserve without clobbering the manual increments above.
        dummy_pack(dfb::out);
        dfb.push_back(1);

        acquire_dst();
        dfb.wait_front(1);
        copy_tile(dfb::out, 0, 0);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            volatile std::uint32_t* entry = reinterpret_cast<volatile std::uint32_t*>(dfb.get_read_ptr() << 4);
            for (std::uint32_t w = 0; w < words_per_entry; w++) {
                entry[w] += 1;
            }
            // Publish these before the entry is handed back to pack.
            asm volatile("fence w, w" ::: "memory");
        }
#endif
        dfb.pop_front(1);
        release_dst();
    }

    dfb.finish();
}
