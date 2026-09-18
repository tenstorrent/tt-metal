// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

namespace tt::tt_metal::detail {

inline bool per_core_program_size_enabled() { return std::getenv("TT_METAL_PER_CORE_PROGRAM_SIZE") != nullptr; }

}  // namespace tt::tt_metal::detail
