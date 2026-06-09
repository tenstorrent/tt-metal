// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1 (first porting step) of the "kernel arguments as function & template parameters"
// work (see tech_reports/NamedKernelArgs/kernel_args_as_parameters.md).
//
// The kernel is now authored in the TARGET form: a TT_KERNEL-marked entry whose template
// parameters are the compile-time args (CTAs) and whose function parameters are the runtime
// args (RTAs/CRTAs). The signature parser (jit_build/kernel_signature_parser) finds the
// TT_KERNEL marker before JIT compile and logs the extracted CTA / runtime arg names.
//
// The kernel_main() at the bottom is a TEMPORARY hand-written shim. It is exactly what
// write_kernel_main_shim() will generate in the next step; once codegen lands, this hand-shim
// is deleted and the firmware-called kernel_main() becomes generated. The DPRINT output is
// unchanged from the Phase-0 baseline, so we can diff it across porting steps.
//
// NOTE: In Phase 1 the TT_KERNEL marker only needs to be a distinctive token for the
// tokenizer; it expands to FORCE_INLINE so the user entry folds into the generated
// kernel_main() with no call indirection. The [[tt::kernel_main]] attribute from the design
// doc is deferred to Phase 2 (its AST parser needs it, and it requires -Wno-attributes since
// kernels build with -Wall -Werror). The macro will move to a shared kernel header then.

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

#ifndef TT_KERNEL
#define TT_KERNEL FORCE_INLINE
#endif

template <uint32_t block_h, uint32_t block_w, uint32_t untilize>  // CTAs (compile-time)
TT_KERNEL void named_kernel_args_kernel(
    uint32_t src_addr,    // RTA
    uint32_t dst_addr,    // RTA
    uint32_t num_tiles,   // RTA
    uint32_t scaler,      // CRTA
    uint32_t sem_addr) {  // CRTA
    DPRINT("[named_kernel_args] CTA  block_h={} block_w={} untilize={}\n", block_h, block_w, untilize);
    DPRINT("[named_kernel_args] RTA  src_addr={:x} dst_addr={:x} num_tiles={}\n", src_addr, dst_addr, num_tiles);
    DPRINT("[named_kernel_args] CRTA scaler={:x} sem_addr={:x}\n", scaler, sem_addr);
}

// TEMPORARY hand-written shim — write_kernel_main_shim() will generate exactly this.
void kernel_main() {
    named_kernel_args_kernel<get_arg(args::block_h), get_arg(args::block_w), get_arg(args::untilize)>(
        get_arg(args::src_addr),
        get_arg(args::dst_addr),
        get_arg(args::num_tiles),
        get_arg(args::scaler),
        get_arg(args::sem_addr));
}
