// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Conservative minimum number of runtime args (unique + common combined) guaranteed to be settable on
// any core type. This is a stable floor for callers that want a portable limit; it is NOT the enforced
// hard cap. The actual per-core ceiling is larger and computed by the runtime from the available L1
// kernel-config space for the target core type (see Kernel::validate_runtime_args_size).
constexpr uint32_t max_runtime_args = 341;

}  // namespace tt::tt_metal
