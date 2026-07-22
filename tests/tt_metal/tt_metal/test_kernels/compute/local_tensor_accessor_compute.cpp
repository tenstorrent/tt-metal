// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 LocalTensorAccessor compile + token-wiring proof (compute side).
//
// Binds a tensor and constructs a LocalTensorAccessor<uint32_t> from the binding token on a compute
// (TRISC) kernel — the path that previously could not compile, because the generated header pulled in
// tensor_accessor.h, whose NOC helpers require NOC_INDEX (which compute kernel builds intentionally do
// not define). LocalTensorAccessor carries no NOC machinery, so it compiles here.
//
// The accessor is constructed on all three TRISC threads (UNPACK/MATH/PACK) — the strongest compile
// proof — and its full surface (get_bank_base_address / local_mem / operator[]) is exercised so every
// method is instantiated. The PACK thread deposits the reported values into a host-known L1 report
// buffer (named RTA); the host reads that L1 after launch.
//
// The host checks that the reported base address equals the bound tensor's address — proving the
// binding token reached the compute kernel and resolved to the correct local L1 shard address. The
// local_mem().get_unsafe_ptr() / operator[] reports resolve to the same address (address-of avoids an
// actual load).

#include <cstdint>

#include "api/compute/common.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);

    // Construct on every TRISC thread (the binding's base-address CRTA is broadcast to all three).
    LocalTensorAccessor<uint32_t> acc(tensor::local_t);

    const uint32_t base_address = acc.get_bank_base_address();
    // Instantiate the element-access surface too, so it is compile-proven on TRISC. Both resolve to the
    // base address; address-of avoids an actual L1 load/store.
    const uint32_t via_unsafe_ptr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(acc.local_mem().get_unsafe_ptr()));
    const uint32_t via_subscript = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&acc[0]));

    // Also exercise the legacy (raw base-address) ctor: compile-proves it on TRISC and confirms it
    // agrees with the token-built accessor.
    LocalTensorAccessor<uint32_t> acc_legacy(base_address);
    const uint32_t legacy_base = acc_legacy.get_bank_base_address();

    // Single writer so UNPACK/MATH/PACK do not race the report slot.
#ifdef TRISC_PACK
    volatile tt_l1_ptr uint32_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    out_ptr[0] = base_address;
    out_ptr[1] = via_unsafe_ptr;
    out_ptr[2] = via_subscript;
    out_ptr[3] = legacy_base;
#endif
}
