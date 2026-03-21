// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Named runtime arg types for device kernels.
//
// Defines rt_args::Dispatch and rt_args::Arg used by the JIT-generated header
// (named_args_generated.h) and by rt_args::get<>() in the API headers.
//
// This file is safe to -include before any API headers because it only
// defines types (no functions that depend on get_arg_val etc.).

#pragma once

#include <cstdint>

namespace rt_args {

enum class Dispatch : uint8_t { COMMON, PER_CORE };

struct Arg {
    uint32_t index;
    Dispatch dispatch;
};

}  // namespace rt_args
