// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side consumer for the single-DFB matrix
// sweep (DM → DFB → TRISC case).
//
// This kernel adds Tensix-as-consumer coverage that the DM-side consumer
// (dfb_consumer_2_0.cpp) doesn't reach. The DM consumer drains the DFB and
// NoC-writes data to DRAM; this kernel drains DFB credits on the Tensix side
// and calls finish(), exercising the Tensix wait_front / unpack / pop_front /
// finish path without coupling the test to a NoC write back-half. Per HW spec an
// unpacker op must sit between wait_front and pop_front; dummy_unpack supplies
// one without reading the ring or writing dest (there is no output DFB).
//
// Because there is no output DFB, the payload can't be checked through DRAM the
// way the DM consumer's is. Instead the UNPACK thread folds each entry it drains
// into an FNV-1a digest and reports it to an L1 scratch region, so the host can
// verify both the bytes and the delivery order of every entry. Digests are laid
// out as [consumer_idx][drain_index], one uint32_t each; the host seeds the
// region with a sentinel so a slot the kernel never reached is distinguishable
// from a wrong value.
//
// Flow per test invocation:
//   1. DM producer kernel writes data into the DFB L1 ring (NoC read from DRAM).
//   2. This kernel does wait_front + dummy_unpack + digest + pop_front for
//      num_entries_per_consumer iterations, then dfb.finish().
//   3. Host reads the digest region and compares against the input pages it
//      expects this consumer to have been handed, in order.
//
// Bindings (set by host KernelSpec):
//   dfb::in — CONSUMER (host binds the same DFB the DM producer pushes to).

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::in);

    // HW requires an unpacker op between wait_front and pop_front (TEN-4746).
    // dummy_unpack supplies that op (UNPACR_NOP): it does not fetch a buffer descriptor, so
    // copy_init is not needed. compute_kernel_hw_startup is still required for the dest
    // handshake that acquire_dst / release_dst use; those balance MATH<->PACK without a
    // pack_tile (packing into the input ring would race the producer). There is no output
    // DFB, so both operands are the same input id.
    compute_kernel_hw_startup(dfb.get_id(), dfb.get_id());

#ifdef UCK_CHLKC_UNPACK
    // UNPACK owns the read cursor, so it is the only thread that can address the
    // entry at the front of this Neo's tile counter (MATH has no fifo state at all
    // and PACK holds the write cursor). One UNPACK thread per Neo, so the Neo's
    // thread id keys its slice of the digest region.
    const uint32_t words_per_entry = dfb.get_entry_size() / sizeof(uint32_t);

    volatile tt_l1_ptr uint32_t* const digests = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        result_l1_addr + get_my_thread_id() * num_entries_per_consumer * sizeof(uint32_t));
#endif

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        acquire_dst();
        dfb.wait_front(1);
        ckernel::dummy_unpack(dfb.get_id());
#ifdef UCK_CHLKC_UNPACK
        {
            // The digest has to be taken after the unpack, not before. wait_front does not block this
            // RISC: it lowers to a single TT_WAIT_TILES pushed into the Tensix instruction buffer (see
            // llk_wait_tiles) which stalls the *unpacker*, while the scalar stream runs straight past
            // it. A read placed before the UNPACR is therefore ungated and races the producer -- the
            // hazard TEN-4746 states as "the wait can resolve before tiles are available".
            //
            // The UNPACR that dummy_unpack issues is gated by that stall, so it cannot execute until
            // this entry has actually arrived; tensix_sync() then blocks until the backend is idle,
            // i.e. until that UNPACR has completed. Only then is the entry known to be in L1. Syncing
            // before the unpack is not enough: it would only drain the *previous* iteration's UNPACR,
            // which says nothing about this entry (and leaves drain index 0 unprotected entirely).
            // dummy_unpack reads nothing from L1, so it neither supplies nor disturbs the bytes hashed
            // here -- it serves only to prove the entry landed.
            // get_read_ptr() is in 16B units on both arches, hence the << 4 (cf. dfb_t6_intra_2_0.cpp).
            ckernel::tensix_sync();
            const volatile tt_l1_ptr uint32_t* const entry =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>((dfb.get_read_ptr() << 4));
            uint32_t digest = 2166136261u;  // FNV-1a offset basis
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                digest = (digest ^ entry[w]) * 16777619u;  // FNV-1a prime
            }
            digests[tile_id] = digest;
        }
#endif
        dfb.pop_front(1);
        release_dst();
    }
    dfb.finish();
}
