// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: TT_KERNEL named-args producer (compute side), authored in the "1st world
// arguments" syntax.
//
// The TT_KERNEL counterpart to named_args_loopback_compute.cpp. Instead of a hand-written
// kernel_main() with get_arg(args::...) calls, the arguments ARE the entry's parameters: CTAs
// as template parameters, RTAs/CRTAs as function parameters. genfiles generates the
// kernel_main() shim. This is the test that proves the shim is generated on the COMPUTE (TRISC)
// compile path — not just data movement.
//
// Args (no varargs — the TT_KERNEL syntax doesn't express them):
//   magic        — CTA (template parameter, compile-time)
//   entry_size   — CTA (template parameter, compile-time)
//   input_offset — RTA (function parameter, per-node)
//   num_tiles    — CRTA (function parameter, broadcast)
//
// Verification: writes magic ^ entry_size ^ input_offset ^ num_tiles into the first uint32_t of
// every out_dfb entry (rest zeroed). The host arranges the values so that XOR equals a known
// target; a wrong arg binding → wrong sum → wrong DRAM word → test fails. (No tile pipeline —
// raw L1 writes from PACK, same as the named_args_loopback_compute baseline.)

#include <cstdint>

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"  // provides TT_KERNEL, get_arg, the args:: / dfb:: accessors

template <uint32_t magic, uint32_t entry_size>  // CTAs (compile-time)
TT_KERNEL void loopback_compute(
    uint32_t input_offset,  // RTA (per-node)
    uint32_t num_tiles) {   // CRTA (broadcast)
    const uint32_t sum = magic ^ entry_size ^ input_offset ^ num_tiles;

    DataflowBuffer dfb_out(dfb::out_dfb);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_out.reserve_back(1);  // implementation gates this to PACK only

        // PACK is the only TRISC that needs to populate the L1 slot — the others reach
        // reserve_back / push_back as no-ops via the impl gates. On tt-1xx TRISCs, fifo_wr_ptr is
        // stored as a 16-byte unit address; shift left 4 to get a byte address for L1 access.
#ifdef TRISC_PACK
        {
            volatile tt_l1_ptr uint32_t* out_ptr = (volatile tt_l1_ptr uint32_t*)(dfb_out.get_write_ptr() << 4);
            out_ptr[0] = sum;
            const uint32_t words = entry_size / sizeof(uint32_t);
            for (uint32_t w = 1; w < words; ++w) {
                out_ptr[w] = 0;
            }
        }
#endif

        dfb_out.push_back(1);  // implementation gates this to PACK only
    }
}

// No kernel_main() here — genfiles generates it from this signature (the compute path now wires
// the shim, same as data movement). run_kernel() calls that generated kernel_main().
