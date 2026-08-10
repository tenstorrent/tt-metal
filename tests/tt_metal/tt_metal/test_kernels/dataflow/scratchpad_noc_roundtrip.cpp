// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Scratchpad as a NoC endpoint (data movement).
//
// A Scratchpad is node-local L1, so noc_traits_t<Scratchpad<T>> lets it serve as the local endpoint
// of a NoC transaction: the destination of a read, or the source of a write. This kernel exercises
// both directions against a *remote* core (never a self-NoC loopback), at a non-zero offset within
// the scratchpad:
//
//   1. Zero the whole scratchpad.
//   2. NoC read : remote L1 [src_addr]                -> scratchpad [offset, offset + transfer)
//   3. NoC write: scratchpad [offset, offset + transfer) -> remote L1 [dst_addr]
//   4. Report the scratchpad's base address to a host-known local L1 address.
//
// The host seeds the pattern at the remote src_addr and checks it round-tripped to dst_addr. That
// alone would still pass if BOTH directions ignored offset_bytes, so the host additionally reads the
// scratchpad's own L1 via the reported base and requires the region outside [offset, offset +
// transfer) to still be zero. That is what actually pins down offset resolution.
//
// `Scratchpad` and the `scratch::pad` token are provided by the auto-generated kernel_bindings
// header (genfiles emits `#include "api/scratchpad.h"` plus the `scratch::` namespace when a kernel
// has a scratchpad binding), so no manual scratchpad include is needed here.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

// Must match the host-side constants in test_scratchpad_hw.cpp.
constexpr uint32_t kOffsetBytes = 32;
constexpr uint32_t kTransferBytes = 64;

void kernel_main() {
    // Remote core hosting the source and destination L1 buffers, and its two addresses.
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);
    const uint32_t src_addr = get_arg(args::src_addr);
    const uint32_t dst_addr = get_arg(args::dst_addr);
    // Host-known local L1 address to report the scratchpad's base address into.
    const uintptr_t report_addr = get_arg(args::report_addr);

    Scratchpad<uint32_t> pad(scratch::pad);

    // Zero the whole scratchpad, so any byte the host later finds non-zero was put there by the NoC
    // read -- and every byte outside the transfer window is expected to still be zero.
    const uint32_t n = pad.size();
    for (uint32_t i = 0; i < n; i++) {
        pad[i] = 0;
    }

    Noc noc;
    UnicastEndpoint remote;

    // Scratchpad as NoC read DESTINATION (its local-L1 address comes from noc_traits_t::dst_addr).
    noc.async_read(
        remote,
        pad,
        kTransferBytes,
        {
            .noc_x = remote_noc_x,
            .noc_y = remote_noc_y,
            .addr = src_addr,
        },
        {.offset_bytes = kOffsetBytes});
    noc.async_read_barrier();

    // Scratchpad as NoC write SOURCE (its local-L1 address comes from noc_traits_t::src_addr).
    noc.async_write(
        pad,
        remote,
        kTransferBytes,
        {.offset_bytes = kOffsetBytes},
        {
            .noc_x = remote_noc_x,
            .noc_y = remote_noc_y,
            .addr = dst_addr,
        });
    noc.async_write_barrier();

    // Report where the framework put the scratchpad, so the host can inspect its L1 directly. A plain
    // volatile L1 write is host-visible via ReadFromDeviceL1 after the (blocking) enqueue completes.
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = pad.get_base_address();
}
