// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// LEGACY counterpart to programming_examples/named_kernel_args (the TT_KERNEL version).
//
// Same kernel, same DPRINT output — but written the classic Metal 1.0 way: a hand-written
// kernel_main() that reads every argument POSITIONALLY by index. No TT_KERNEL marker, no
// generated shim, no named args. This is the "before" against which the TT_KERNEL example is
// the "after"; the DPRINT output of the two is identical.
//
//   CTAs  -> get_compile_time_arg_val(i)   (compile_args = {block_h, block_w, untilize})
//   RTAs  -> get_arg_val<uint32_t>(i)       (SetRuntimeArgs   = {src_addr, dst_addr, num_tiles})
//   CRTAs -> get_common_arg_val<uint32_t>(i) (SetCommonRuntimeArgs = {scaler, sem_addr})
//
// The index of every argument is tracked by hand and must stay in lockstep with the host
// ordering — exactly the fragility the TT_KERNEL/named-arg design removes.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

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

    DPRINT("[named_kernel_args] CTA  block_h={} block_w={} untilize={}\n", block_h, block_w, untilize);
    DPRINT("[named_kernel_args] RTA  src_addr={:x} dst_addr={:x} num_tiles={}\n", src_addr, dst_addr, num_tiles);
    DPRINT("[named_kernel_args] CRTA scaler={:x} sem_addr={:x}\n", scaler, sem_addr);
}
