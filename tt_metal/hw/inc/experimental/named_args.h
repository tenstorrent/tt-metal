// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENTAL: Named kernel-args — temporary, Blaze-only.
//
// This header provides the device-side `rt_args::get<>()` accessor template
// that works with the JIT-generated `named_args_generated.h` descriptors
// (`rt_args::Arg` / `rt_args::ArrayArg`).  It is an opt-in header: Blaze
// (and test) kernels must `#include "experimental/named_args.h"` explicitly.
// Core device headers (`dataflow_api.h`, `compute/common.h`) no longer pull
// this template into every kernel's namespace.
//
// This feature will be deleted when Blaze migrates to the Metal 2.0
// `args::` system.  See:
//   tt_metal/api/tt-metalium/experimental/README_named_kernel_args.md

#pragma once

#include "api/rt_arg.h"

// DEPENDENCY NOTE:
// This header requires the following to be visible at the point of inclusion:
//  - `get_arg_val<T>`
//  - `get_common_arg_val<T>`
//  - `FORCE_INLINE`
// On the data-movement path, those come from `api/dataflow/dataflow_api.h`,
// which the firmware wrapper includes before <kernel_includes.hpp>.
// On the TRISC path, those come from `api/compute/common.h`; nothing pulls
// that in before <kernel_includes.hpp>, so we include it here.
#ifdef COMPILE_FOR_TRISC
#include "api/compute/common.h"
#endif

// Unified accessor for named runtime args (works for both Arg and ArrayArg).
// Scalar:  uint32_t n = rt_args::get<ct_args::my_op::num_tiles>();
// Array:   uint32_t a = rt_args::get<ct_args::my_op::worker_sem_addr>(i);
namespace rt_args {
template <auto arg, typename T = uint32_t>
FORCE_INLINE T get(uint32_t i = 0) {
    if constexpr (arg.dispatch == Dispatch::COMMON) {
        return get_common_arg_val<T>(arg.index + i);
    } else {
        return get_arg_val<T>(arg.index + i);
    }
}
}  // namespace rt_args
