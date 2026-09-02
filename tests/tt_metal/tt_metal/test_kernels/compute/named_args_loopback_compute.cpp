// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 named-args producer (compute side).
// Companion test on the COMPUTE compile path (TRISC_UNPACK / TRISC_MATH / TRISC_PACK) for the
// existing dataflow named_args_loopback pair, which only covers the BRISC/NCRISC compile path.
// The named-args surface reaches a compute kernel via a different include chain
// (compute_kernel_api.h → api/compute/common.h) than DM (api/dataflow/dataflow_api.h), and
// experimental/kernel_args.h must work in both contexts.
//
// Exercises the Metal 2.0 kernel-args feature surface:
//   args::magic        — named CTA (compile-time)
//   args::entry_size   — named CTA (compile-time)
//   args::num_tiles    — named CRTA (broadcast at runtime)
//   args::input_offset — named RTA (per-node at runtime)
//   args::report_addr  — named RTA; host-allocated L1 word for the XOR result
//   get_vararg(0..1)   — two RTA varargs
//   get_common_vararg(0) — one CRTA vararg
//
// Verification: the host arranges all six values so their XOR equals a known target. PACK writes
// that XOR into report_addr; the host reads that L1 word after LaunchProgram. A wrong offset on any
// accessor → wrong sum → test fails.

#include <cstdint>

#include "api/compute/common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t magic = get_arg(args::magic);
    constexpr uint32_t entry_size = get_arg(args::entry_size);

    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t input_offset = get_arg(args::input_offset);
    const uint32_t report_addr = get_arg(args::report_addr);

    const uint32_t vararg_xor =
        magic ^ entry_size ^ num_tiles ^ input_offset ^ get_vararg(0) ^ get_vararg(1) ^ get_common_vararg(0);

#ifdef TRISC_PACK
    volatile tt_l1_ptr uint32_t* out_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    out_ptr[0] = vararg_xor;
#endif
}
