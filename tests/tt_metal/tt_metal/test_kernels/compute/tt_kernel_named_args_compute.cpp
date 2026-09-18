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
//   report_addr  — RTA; host-allocated L1 word for the XOR result
//   num_tiles    — CRTA (function parameter, broadcast)
//
// Verification: PACK writes magic ^ entry_size ^ input_offset ^ num_tiles into report_addr. The
// host arranges the values so that XOR equals a known target; a wrong arg binding → wrong sum →
// test fails.

#include <cstdint>

#include "api/compute/common.h"
#include "experimental/kernel_args.h"  // provides TT_KERNEL, get_arg, the args:: accessors

template <uint32_t magic, uint32_t entry_size>  // CTAs (compile-time)
TT_KERNEL void loopback_compute(
    uint32_t input_offset,  // RTA (per-node)
    uint32_t report_addr,   // RTA (host L1 report word)
    uint32_t num_tiles) {   // CRTA (broadcast)
    const uint32_t sum = magic ^ entry_size ^ input_offset ^ num_tiles;

#ifdef TRISC_PACK
    volatile tt_l1_ptr uint32_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    out_ptr[0] = sum;
#endif
}

// No kernel_main() here — genfiles generates it from this signature (the compute path now wires
// the shim, same as data movement). run_kernel() calls that generated kernel_main().
