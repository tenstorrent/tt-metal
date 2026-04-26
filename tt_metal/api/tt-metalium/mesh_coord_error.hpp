// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exception types for mesh coordinate operations.
// Use these instead of asserting on exception message substrings in tests.
#pragma once
#include <stdexcept>
#include <string>

namespace tt::tt_metal {

// Thrown when a dimension index is out of bounds.
struct MeshCoordIndexOutOfBounds : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace tt::tt_metal
