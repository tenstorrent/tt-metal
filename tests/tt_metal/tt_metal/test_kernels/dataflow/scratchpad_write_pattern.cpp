// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 kernel-scratchpad happy-path (data movement).
//
// Proves two things end-to-end on real Gen1 (WH/BH) hardware:
//   (a) the scratchpad is a real, writable, node-local L1 region, and
//   (b) the framework delivered the scratchpad's framework-allocated L1 base address to the kernel
//       via the binding token (the CRTA word the Scratchpad ctor reads).
//
// The kernel writes a known pattern into the scratchpad, then reports the scratchpad's base address
// (Scratchpad::get_base_address()) to a host-known fixed L1 location passed as a named RTA. The host
// reads that reported address, then reads the scratchpad's L1 at that address and checks the
// pattern. If the framework had shipped a 0 / stale base address, either get_base_address() would be
// wrong (so the host reads the wrong L1) or the pattern wouldn't land where the host looks — both
// fail the test.
//
// `Scratchpad` and the `scratch::pad` token are provided by the auto-generated kernel_bindings
// header (genfiles emits `#include "api/scratchpad.h"` plus the `scratch::` namespace
// when a kernel has a scratchpad binding), so no manual scratchpad include is needed here.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Host-known L1 address to report the scratchpad's base address into (named per-node RTA).
    uintptr_t report_addr = get_arg(args::report_addr);

    // Construct the scratchpad from the codegen-emitted accessor. Element type is the kernel
    // author's choice (uint32_t here); size()/size_in_bytes() come from the accessor's compile-time size.
    auto s = Scratchpad<uint32_t>(scratch::pad);

    // (a) Write a known pattern across the whole scratchpad. The base value is non-trivial so a
    // wrong base address is unlikely to coincidentally read back the expected pattern.
    const uint32_t n = s.size();
    for (uint32_t i = 0; i < n; i++) {
        s[i] = 0xC0DE0000u + i;
    }

    // (b) Report the scratchpad's base address (as delivered by the framework via the token's CRTA
    // word) to the host-known L1 location. A plain volatile L1 write is host-visible via
    // ReadFromDeviceL1 after the (blocking) enqueue completes — the same pattern test_add_two_ints
    // and test_single_dm_l1_write rely on for Gen1.
    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = s.get_base_address();
}
