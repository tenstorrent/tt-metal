// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace dprint_detail {

namespace structures {

struct DPrintStringMetadata {
    const char* format_string_ptr;
    const char* file;
    std::uint32_t line;
};

}  // namespace structures

}  // namespace dprint_detail
