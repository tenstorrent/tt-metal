// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase-0 baseline for the "kernel arguments as function & template parameters" work
// (see tech_reports/NamedKernelArgs/kernel_args_as_parameters.md).
//
// This kernel uses the EXISTING Metal 2.0 named-argument infrastructure on main: it reads
// every argument by name via get_arg(args::<name>) and DPRINTs the value. The schema mirrors
// the §2 worked example exactly:
//   - 3 CTAs  (compile-time):     block_h, block_w, untilize
//   - 3 RTAs  (per-core runtime): src_addr, dst_addr, num_tiles
//   - 2 CRTAs (common runtime):   scaler, sem_addr
//
// As Phase 1 lands, this kernel is incrementally ported to the new authoring surface:
//
//   template <uint32_t block_h, uint32_t block_w, uint32_t untilize>      // CTAs
//   TT_KERNEL void named_kernel_args_kernel(uint32_t src_addr,            // RTAs / CRTAs
//                                           uint32_t dst_addr,
//                                           uint32_t num_tiles,
//                                           uint32_t scaler,
//                                           uint32_t sem_addr) { ... }
//
// where the get_arg(args::...) calls below disappear into the generated kernel_main() shim.
// The DPRINT output is the fixture we diff against at each porting step.

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Compile-time args (CTAs): constexpr today; template parameters in the new design.
    constexpr uint32_t block_h = get_arg(args::block_h);
    constexpr uint32_t block_w = get_arg(args::block_w);
    constexpr uint32_t untilize = get_arg(args::untilize);

    // Per-core runtime args (RTAs).
    uint32_t src_addr = get_arg(args::src_addr);
    uint32_t dst_addr = get_arg(args::dst_addr);
    uint32_t num_tiles = get_arg(args::num_tiles);

    // Common runtime args (CRTAs): shared across cores.
    uint32_t scaler = get_arg(args::scaler);
    uint32_t sem_addr = get_arg(args::sem_addr);

    DPRINT("[named_kernel_args] CTA  block_h={} block_w={} untilize={}\n", block_h, block_w, untilize);
    DPRINT("[named_kernel_args] RTA  src_addr={:x} dst_addr={:x} num_tiles={}\n", src_addr, dst_addr, num_tiles);
    DPRINT("[named_kernel_args] CRTA scaler={:x} sem_addr={:x}\n", scaler, sem_addr);
}
