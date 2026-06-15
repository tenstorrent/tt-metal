// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// LEGACY compute-path counterpart to named_kernel_args_legacy_kernel (the dataflow legacy kernel).
//
// Same schema (3 CTAs, 3 RTAs, 2 CRTAs) and the same hand-written, positional Metal 1.0 style —
// a plain kernel_main() that reads every argument by index — but compiled on the TRISC/compute
// path. This is the "before" against which the TT_KERNEL compute kernel is the "after".
//
//   CTAs  -> get_compile_time_arg_val(i)      (ComputeConfig.compile_args = {block_h, block_w, untilize})
//   RTAs  -> get_arg_val<uint32_t>(i)          (SetRuntimeArgs       = {src_addr, dst_addr, num_tiles})
//   CRTAs -> get_common_arg_val<uint32_t>(i)   (SetCommonRuntimeArgs = {scaler, sem_addr})
//
// The compute kernel source is compiled once per TRISC (UNPACK/MATH/PACK); DPRINT_MATH emits only
// from the MATH TRISC, so the output is one clean set of lines rather than three duplicates.
//
// Run with DPRINT enabled:
//   export TT_METAL_DPRINT_CORES=0,0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/compute/compute_kernel_api.h"  // get_compile_time_arg_val
#include "api/compute/common.h"              // get_arg_val / get_common_arg_val on the compute path

void kernel_main() {
    // Compile-time args (CTAs) — positional, constexpr.
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t untilize = get_compile_time_arg_val(2);

    // Per-core runtime args (RTAs) — positional.
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    // Common runtime args (CRTAs) — positional.
    uint32_t scaler = get_common_arg_val<uint32_t>(0);
    uint32_t sem_addr = get_common_arg_val<uint32_t>(1);

    DPRINT_MATH("[named_kernel_args:compute] CTA  block_h={} block_w={} untilize={}\n", block_h, block_w, untilize);
    DPRINT_MATH(
        "[named_kernel_args:compute] RTA  src_addr={:x} dst_addr={:x} num_tiles={}\n", src_addr, dst_addr, num_tiles);
    DPRINT_MATH("[named_kernel_args:compute] CRTA scaler={:x} sem_addr={:x}\n", scaler, sem_addr);
}
