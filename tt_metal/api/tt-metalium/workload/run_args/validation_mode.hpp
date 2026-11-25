// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// How much a run-args function checks its input before it applies it.
enum class ValidationMode : uint8_t {
    // Check every requirement of the function and report each violation as an error.
    FULL,
    // Skip the requirement checks; the caller guarantees the requirements. Checks that protect host memory still run.
    SKIP,
};

}  // namespace tt::tt_metal
