// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute-path counterpart to the dataflow named_kernel_args_kernel.
//
// Same TT_KERNEL named-arg authoring style as the dataflow kernel — template parameters are the
// compile-time args (CTAs), function parameters are the runtime args (RTAs/CRTAs) — but this
// entry is compiled on the TRISC/compute path instead of BRISC/NCRISC. The signature parser and
// genfiles generate kernel_main() for the compute path exactly as they do for data movement (the
// compute entry symbol that run_kernel() calls is kernel_main() too), so the user still writes no
// kernel_main() and no get_arg() calls.
//
// The named-arg surface reaches a compute kernel through a different include chain than a DM
// kernel (api/compute/common.h via compute_kernel_api.h, rather than api/dataflow/dataflow_api.h);
// experimental/kernel_args.h works in both.
//
// A compute kernel's source is compiled once per TRISC (UNPACK/MATH/PACK). DPRINT_MATH emits only
// from the MATH TRISC, so the output is one clean set of lines rather than three duplicates.
//
// Run with DPRINT enabled to see the output:
//   export TT_METAL_DPRINT_CORES=0,0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/compute/compute_kernel_api.h"
#include "experimental/kernel_args.h"  // provides get_arg, the args:: accessors, and TT_KERNEL

template <uint32_t block_h, uint32_t block_w, uint32_t untilize>  // CTAs (compile-time)
TT_KERNEL void named_kernel_args_compute_kernel(
    uint32_t src_addr,    // RTA
    uint32_t dst_addr,    // RTA
    uint32_t num_tiles,   // RTA
    uint32_t scaler,      // CRTA
    uint32_t sem_addr) {  // CRTA
    DPRINT_MATH("[named_kernel_args:compute] CTA  block_h={} block_w={} untilize={}\n", block_h, block_w, untilize);
    DPRINT_MATH(
        "[named_kernel_args:compute] RTA  src_addr={:x} dst_addr={:x} num_tiles={}\n", src_addr, dst_addr, num_tiles);
    DPRINT_MATH("[named_kernel_args:compute] CRTA scaler={:x} sem_addr={:x}\n", scaler, sem_addr);
}
