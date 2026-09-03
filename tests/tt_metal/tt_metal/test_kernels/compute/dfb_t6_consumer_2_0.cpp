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
// and calls finish(), exercising the Tensix wait_front / copy_tile / pop_front /
// finish path without coupling the test to a NoC write back-half. Per HW spec a
// copy (unpack) instruction must sit between wait_front and pop_front, so each
// entry is copied into the math dest register and then discarded (there is no
// output DFB).
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
//   2. This kernel does wait_front + digest + copy_tile + pop_front for
//      num_entries_per_consumer iterations, then dfb.finish().
//   3. Host reads the digest region and compares against the input pages it
//      expects this consumer to have been handed, in order.
//
// Bindings (set by host KernelSpec):
//   dfb::in — CONSUMER (host binds the same DFB the DM producer pushes to).

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::in);

    // HW requires a copy (unpack) instruction between wait_front and pop_front.
    // This kernel has no output DFB, so configure unpack + pack hw against the
    // single input DFB (matches the production compute_kernel_hw_startup(in, out);
    // copy_init(in) shape; this is also what programs the buffer-descriptor table the UNPACR
    // reads) and discard the copied tile. copy_tile reads the entry into the
    // math dest register without mutating the DFB L1 ring. The acquire_dst /
    // release_dst pair alone balances the MATH<->PACK dest handshake (no
    // pack_tile required, and packing into the input ring would race the producer).
    compute_kernel_hw_startup(dfb.get_id(), dfb.get_id());
    copy_init(dfb.get_id());

#ifdef UCK_CHLKC_UNPACK
    // UNPACK owns the read cursor, so it is the only thread that can address the
    // entry at the front of this Neo's tile counter (MATH has no fifo state at all
    // and PACK holds the write cursor). One UNPACK thread per Neo, so the Neo's
    // thread id keys its slice of the digest region.
    const uint32_t words_per_entry = dfb.get_entry_size() / sizeof(uint32_t);
    // The producer NoC-writes the ring, so read it through the uncached alias on
    // Quasar rather than risk a stale cached line (same reason dfb_l1_uncached_*_ptr
    // exists for TRISC reads of DM-written DFB config). MEM_L1_BASE is 0, so the
    // alias is just a constant bias. WH/BH TRISC L1 access is not cached.
    volatile tt_l1_ptr uint32_t* const digests = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        result_l1_addr + get_my_thread_id() * num_entries_per_consumer * sizeof(uint32_t));
#endif

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        acquire_dst();
        dfb.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        {
            // Safe between wait_front and pop_front: the entry is credited to us and
            // the producer cannot reclaim the slot until we pop. get_read_ptr() is in
            // 16B units on both arches, hence the << 4 (cf. dfb_t6_intra_2_0.cpp).
            const volatile tt_l1_ptr uint32_t* const entry =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>((dfb.get_read_ptr() << 4) + l1_uncached_bias);
            uint32_t digest = 2166136261u;  // FNV-1a offset basis
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                digest = (digest ^ entry[w]) * 16777619u;  // FNV-1a prime
            }
            digests[tile_id] = digest;
        }
#endif
        copy_tile(dfb.get_id(), 0, 0);
        dfb.pop_front(1);
        release_dst();
    }
    dfb.finish();
}
