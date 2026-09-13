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
//   2. This kernel does wait_front(share) + dummy_unpack + one digest per tile of
//      the share + pop_front(share) until num_entries_per_consumer tiles are
//      drained, then dfb.finish(). The share is what the ring dictates for this
//      hart (get_consume_share): a whole block on a BLOCKED consumer, its part of
//      each producer block when only the producer is BLOCKED, else 1.
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

    // One op is this hart's whole share: the block on a BLOCKED ring (contiguous), its part of
    // each block when only the producers are BLOCKED (spaced `stride` tiles apart), else 1 tile.
    // MATH has no fifo state, so its share reads as 1; its loop therefore takes one tile per
    // iteration while UNPACK/PACK take `share` -- the per-tile acquire_dst/release_dst below keeps
    // the MATH<->PACK handshake count equal (num_entries_per_consumer) on every thread.
#ifdef ARCH_QUASAR
    const uint32_t share = dfb.get_consume_share();
#else
    const uint32_t share = 1;
#endif

#ifdef UCK_CHLKC_UNPACK
    // UNPACK owns the read cursor, so it is the only thread that can address the
    // entry at the front of this Neo's tile counter (MATH has no fifo state at all
    // and PACK holds the write cursor). One UNPACK thread per Neo, so the Neo's
    // thread id keys its slice of the digest region.
    const uint32_t entry_bytes = dfb.get_entry_size();
    const uint32_t words_per_entry = entry_bytes / sizeof(uint32_t);
    // Spacing between consecutive tiles of one share: 1 (contiguous) on a BLOCKED consumer, the
    // wire stride when only the producer is BLOCKED (entry j of the share is at bookmark +
    // j * stride, the same walk the implicit-sync NoC path takes). A share never straddles the
    // ring end: a BLOCKED share is one block and a strided share sits inside one producer block.
#ifdef ARCH_QUASAR
    const uint32_t stride_tiles = dfb.get_consume_stride_tiles();
#else
    const uint32_t stride_tiles = 1;
#endif

    // Host sizes this region with dfb_tensix_digest_region_bytes(num_consumers,
    // num_entries_per_consumer) using the same CTA compiled into this kernel.
    volatile tt_l1_ptr uint32_t* const digests = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        result_l1_addr + get_my_thread_id() * num_entries_per_consumer * sizeof(uint32_t));
#endif

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id += share) {
        dfb.wait_front(share);
        for (uint32_t j = 0; j < share; ++j) {
            acquire_dst();
            // One UNPACR per tile of the share (a no-op on MATH/PACK); the first already orders the
            // pop_front below after the wait_front above (TEN-4746 guard), the rest keep this kernel
            // shaped exactly like the 1-tile-per-op path when share == 1.
            ckernel::dummy_unpack(dfb.get_id());
            release_dst();
        }
#ifdef UCK_CHLKC_UNPACK
        {
            // The digest has to be taken after the unpack, not before. wait_front lowers to a
            // TT_WAIT_TILES that stalls the *unpacker* (llk_wait_tiles now also polls the SYNC busy
            // bit before returning to this RISC, but the L1 read below still relies on the ordering
            // proof rather than on that poll). The UNPACRs that dummy_unpack issues are gated by that
            // stall, so they cannot execute until this share has actually arrived; tensix_sync() then
            // blocks until the backend is idle, i.e. until those UNPACRs have completed. Only then are
            // the share's entries known to be in L1. Syncing before the unpack is not enough: it would
            // only drain the *previous* iteration's UNPACRs, which says nothing about this share (and
            // leaves drain index 0 unprotected entirely). dummy_unpack reads nothing from L1, so it
            // neither supplies nor disturbs the bytes hashed here -- it serves only to prove the
            // entries landed (TEN-4746: "the wait can resolve before tiles are available").
            // get_read_ptr() is in 16B units on both arches, hence the << 4 (cf. dfb_t6_intra_2_0.cpp).
            ckernel::tensix_sync();
            const uint32_t share_base = dfb.get_read_ptr() << 4;
            for (uint32_t j = 0; j < share; ++j) {
                const volatile tt_l1_ptr uint32_t* const entry =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(share_base + j * stride_tiles * entry_bytes);
                uint32_t digest = 2166136261u;  // FNV-1a offset basis
                for (uint32_t w = 0; w < words_per_entry; ++w) {
                    digest = (digest ^ entry[w]) * 16777619u;  // FNV-1a prime
                }
                digests[tile_id + j] = digest;
            }
        }
#endif
        dfb.pop_front(share);
    }
    dfb.finish();
}
