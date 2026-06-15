// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1 (first porting step) of the "kernel arguments as function & template parameters"
// work (see tech_reports/NamedKernelArgs/kernel_args_as_parameters.md).
//
// The kernel is authored in the TARGET form: a TT_KERNEL-marked entry whose template
// parameters are the compile-time args (CTAs) and whose function parameters are the runtime
// args (RTAs/CRTAs). The user writes NO kernel_main() and NO get_arg() calls. Before JIT
// compile, the signature parser (jit_build/kernel_signature_parser) extracts the entry's name
// and its CTA / runtime arg names, and genfiles generates the kernel_main() shim that fetches
// each arg by name (get_arg(args::<name>)) and calls this entry. The DPRINT output is identical
// to the Phase-0 baseline.
//
// NOTE: TT_KERNEL is provided by experimental/kernel_args.h (the Metal 2.0 device header), so
// the kernel never defines it. In Phase 1 it expands to FORCE_INLINE, so the user entry folds
// into the generated kernel_main() with no call indirection; the parser keys on the literal
// token, not the expansion. (The [[tt::kernel_main]] attribute from the design doc is deferred
// to Phase 2, which needs -Wno-attributes since kernels build with -Wall -Werror.)

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"  // provides get_arg, the args:: accessors, and TT_KERNEL

template <uint32_t Ht, uint32_t Wt, uint32_t untilize>  // CTAs (compile-time)
TT_KERNEL void named_kernel_args_kernel(
    uint32_t start_tile_id,  // RTA
    uint32_t num_tiles,      // RTA
    uint32_t start_row,      // RTA
    uint32_t scaler) {       // CRTA
    DPRINT("[named_kernel_args] CTA  Ht={} Wt={} untilize={}\n", Ht, Wt, untilize);
    DPRINT(
        "[named_kernel_args] RTA  start_tile_id={} num_tiles={} start_row={}\n", start_tile_id, num_tiles, start_row);
    DPRINT("[named_kernel_args] CRTA scaler={}\n", scaler);
}

// No kernel_main() here — it is generated from this signature by genfiles (see
// generate_kernel_main_shim). The firmware calls that generated kernel_main(), which inlines
// this entry (TT_KERNEL == FORCE_INLINE).
