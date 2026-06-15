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
//   CTAs  -> get_compile_time_arg_val(i)   (compile_args = {Ht, Wt, untilize})
//   RTAs  -> get_arg_val<uint32_t>(i)       (SetRuntimeArgs   = {start_tile_id, num_tiles, start_row})
//   CRTA  -> get_common_arg_val<uint32_t>(i) (SetCommonRuntimeArgs = {scaler})
//
// The index of every argument is tracked by hand and must stay in lockstep with the host
// ordering — exactly the fragility the TT_KERNEL/named-arg design removes.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // Compile-time args (CTAs) — positional, constexpr.
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t untilize = get_compile_time_arg_val(2);

    // Per-core runtime args (RTAs) — positional.
    uint32_t start_tile_id = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);

    // Common runtime arg (CRTA) — positional.
    uint32_t scaler = get_common_arg_val<uint32_t>(0);

    DPRINT("[named_kernel_args] CTA  Ht={} Wt={} untilize={}\n", Ht, Wt, untilize);
    DPRINT(
        "[named_kernel_args] RTA  start_tile_id={} num_tiles={} start_row={}\n", start_tile_id, num_tiles, start_row);
    DPRINT("[named_kernel_args] CRTA scaler={}\n", scaler);
}
