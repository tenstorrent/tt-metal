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
// method is instantiated. The PACK thread deposits the reported values into each output DFB entry; a
// DM consumer carries them to DRAM for host verification.
//
// The host checks that the reported base address equals the bound tensor's address — proving the
// binding token reached the compute kernel and resolved to the correct local L1 shard address. The
// local_mem().get_unsafe_ptr() / operator[] reports resolve to the same address (address-of avoids an
// actual load).

#include <cstdint>

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // CTAs — compile-time constants.
    constexpr uint32_t entry_size = get_arg(args::entry_size);
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);

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

    DataflowBuffer dfb_out(dfb::out_dfb);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_out.reserve_back(1);  // implementation gates this to PACK only

        // PACK is the only TRISC that populates the L1 slot. On tt-1xx TRISCs, fifo_wr_ptr is a
        // 16-byte unit address; shift left 4 to get a byte address (mirrors named_args_loopback_compute).
#ifdef TRISC_PACK
        {
            volatile tt_l1_ptr uint32_t* out_ptr = (volatile tt_l1_ptr uint32_t*)(dfb_out.get_write_ptr() << 4);
            out_ptr[0] = base_address;
            out_ptr[1] = via_unsafe_ptr;
            out_ptr[2] = via_subscript;
            out_ptr[3] = legacy_base;
            const uint32_t words = entry_size / sizeof(uint32_t);
            for (uint32_t w = 4; w < words; ++w) {
                out_ptr[w] = 0;
            }
        }
#endif

        dfb_out.push_back(1);  // implementation gates this to PACK only
    }
}
