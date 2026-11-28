// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data-movement kernel for the NOC line sweep test (tests/tt_metal/tt_metal/noc/test_noc_line_sweep.cpp).
//
// Moves a block of L1 data between this core and a list of peer cores over one NOC, either by writing this core's
// payload to every peer (PUSH) or by reading every peer's payload into this core (PULL). Runs on BRISC/NCRISC and
// on ERISC (active or idle Ethernet cores) with the NOC selected via the kernel config. All configuration is passed
// through runtime args so a single compiled binary is reused for every step of the sweep.
//
// Runtime args:
//   0: mode            0 = PUSH: write my local payload into one slot on every peer
//                      1 = PULL: read each peer's payload into one slot in my L1
//   1: local_addr      PUSH: L1 address of my payload.  PULL: base of my slot region
//   2: transfer_bytes  bytes per (active, peer) pair; slot k starts at slot base + k * transfer_bytes
//   3: chunk_bytes     bytes per NOC transaction (0 = transfer_bytes in a single call)
//   4: num_iters       repeat the whole transfer this many times (same data; stresses the path)
//   5: num_peers
//   6..: (kind, a, b, remote_base, slot) per peer:
//                      kind 0: a core at virtual NOC coordinates (a, b); remote_base is an L1 address.
//                      kind 1: DRAM bank a (b unused), addressed through the bank-id tables exactly as an
//                              interleaved-buffer access is; remote_base is the buffer address, the bank offset is
//                              added by the bank lookup.
//                      remote_base: PUSH: the peer's slot region. PULL: the peer's payload.
//                      slot: PUSH: the slot written on the peer. PULL: the local slot the peer's payload lands in.
//
// The kernel never waits on data arriving from another core, so a corrupted payload can never turn into a hang:
// the host reads every slot back after the program finishes and does the comparison.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(0);
    const uint32_t local_addr = get_arg_val<uint32_t>(1);
    const uint32_t transfer_bytes = get_arg_val<uint32_t>(2);
    uint32_t chunk_bytes = get_arg_val<uint32_t>(3);
    const uint32_t num_iters = get_arg_val<uint32_t>(4);
    const uint32_t num_peers = get_arg_val<uint32_t>(5);
    constexpr uint32_t peer_args_base = 6;

    if (chunk_bytes == 0 || chunk_bytes > transfer_bytes) {
        chunk_bytes = transfer_bytes;
    }

    for (uint32_t it = 0; it < num_iters; it++) {
        for (uint32_t p = 0; p < num_peers; p++) {
            const uint32_t kind = get_arg_val<uint32_t>(peer_args_base + 5 * p);
            const uint32_t peer_a = get_arg_val<uint32_t>(peer_args_base + 5 * p + 1);
            const uint32_t peer_b = get_arg_val<uint32_t>(peer_args_base + 5 * p + 2);
            const uint32_t remote_addr = get_arg_val<uint32_t>(peer_args_base + 5 * p + 3);
            const uint32_t slot_off = get_arg_val<uint32_t>(peer_args_base + 5 * p + 4) * transfer_bytes;
            auto peer_addr = [&](uint32_t addr) -> uint64_t {
                return kind == 1 ? get_noc_addr_from_bank_id<true>(peer_a, addr, noc_index)
                                 : get_noc_addr(peer_a, peer_b, addr, noc_index);
            };
            if (mode == 0) {
                // PUSH: my payload -> my slot on the peer
                for (uint32_t off = 0; off < transfer_bytes; off += chunk_bytes) {
                    const uint32_t len = (transfer_bytes - off) < chunk_bytes ? (transfer_bytes - off) : chunk_bytes;
                    noc_async_write(local_addr + off, peer_addr(remote_addr + slot_off + off), len, noc_index);
                }
            } else {
                // PULL: peer payload -> the peer's slot in my L1
                const uint32_t slot = local_addr + slot_off;
                for (uint32_t off = 0; off < transfer_bytes; off += chunk_bytes) {
                    const uint32_t len = (transfer_bytes - off) < chunk_bytes ? (transfer_bytes - off) : chunk_bytes;
                    noc_async_read(peer_addr(remote_addr + off), slot + off, len, noc_index);
                }
            }
        }
        if (mode == 0) {
            noc_async_write_barrier(noc_index);
        } else {
            noc_async_read_barrier(noc_index);
        }
    }
    // Make sure nothing is left in flight on this NOC before the kernel exits.
    noc_async_full_barrier(noc_index);
}
