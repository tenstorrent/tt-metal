// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 named-args producer (compute side).
// Acts as the producer of an output DFB: every entry it pushes contains the
// XOR sum of every named arg + vararg the kernel was given, deposited into the
// first uint32_t of the entry's L1 slot. A DM consumer carries those entries
// to a DRAM buffer the host can verify byte-for-byte.
//
// Companion test on the COMPUTE compile path (TRISC_UNPACK / TRISC_MATH /
// TRISC_PACK) for the existing dataflow named_args_loopback pair, which only
// covers the BRISC/NCRISC compile path. The named-args surface reaches a
// compute kernel via a different include chain (compute_kernel_api.h →
// api/compute/common.h) than DM (api/dataflow/dataflow_api.h), and
// experimental/kernel_args.h must work in both contexts.
//
// Exercises the Metal 2.0 kernel-args feature surface:
//   args::magic        — named CTA (compile-time)
//   args::entry_size   — named CTA (compile-time)
//   args::num_tiles    — named CRTA (broadcast at runtime)
//   args::input_offset — named RTA (per-node at runtime)
//   get_vararg(0..1)   — two RTA varargs
//   get_common_vararg(0) — one CRTA vararg
//
// Verification: the host arranges all six values so their XOR equals a known
// target. The output DRAM should contain that target in word 0 of every
// entry, with the rest zeroed. A wrong offset on any accessor → wrong sum →
// wrong word in DRAM output → test fails. (No tile compute pipeline is
// involved — the kernel writes raw bytes to the output DFB's L1 slot, which
// avoids Float16_b lossy conversion and inter-TRISC sync issues that aren't
// what this test is trying to cover.)

#include <cstdint>

#include "api/compute/common.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // CTAs — compile-time constants, so the constexpr form is legal here.
    constexpr uint32_t magic = get_arg(args::magic);
    constexpr uint32_t entry_size = get_arg(args::entry_size);

    // CRTA (broadcast at runtime)
    const uint32_t num_tiles = get_arg(args::num_tiles);

    // RTA (per-node at runtime)
    const uint32_t input_offset = get_arg(args::input_offset);

    // XOR every named arg + every vararg — exercises all six accessor paths.
    const uint32_t vararg_xor =
        magic ^ entry_size ^ num_tiles ^ input_offset ^ get_vararg(0) ^ get_vararg(1) ^ get_common_vararg(0);

    experimental::DataflowBuffer dfb_out(dfb::out_dfb);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_out.reserve_back(1);  // implementation gates this to PACK only

        // PACK is the only TRISC that needs to populate the L1 slot — the
        // others reach reserve_back / push_back as no-ops via the impl gates.
        // On tt-1xx TRISCs, fifo_wr_ptr is stored as a 16-byte unit address;
        // shift left 4 to get a byte address for L1 access.
#ifdef TRISC_PACK
        {
            volatile tt_l1_ptr uint32_t* out_ptr = (volatile tt_l1_ptr uint32_t*)(dfb_out.get_write_ptr() << 4);
            out_ptr[0] = vararg_xor;
            const uint32_t words = entry_size / sizeof(uint32_t);
            for (uint32_t w = 1; w < words; ++w) {
                out_ptr[w] = 0;
            }
        }
#endif

        dfb_out.push_back(1);  // implementation gates this to PACK only
    }
}
