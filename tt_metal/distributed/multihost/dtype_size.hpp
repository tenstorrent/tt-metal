// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include "api/tt-metalium/distributed_context.hpp"

namespace tt::tt_metal::distributed::multihost {

// Element size in bytes of a DType. Every DistributedContext implementation validates collective
// buffers against this table, so a call accepted by one implementation is accepted by all of them.
constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::INT8:
        case DType::UINT8:
        case DType::BOOL:
        case DType::BYTE: return 1;
        case DType::INT16:
        case DType::UINT16: return 2;
        case DType::INT32:
        case DType::UINT32:
        case DType::FLOAT32: return 4;
        case DType::INT64:
        case DType::UINT64:
        case DType::FLOAT64:
        case DType::COMPLEX_FLOAT: return 8;
        case DType::COMPLEX_DOUBLE: return 16;
    }
    return 0;
}

}  // namespace tt::tt_metal::distributed::multihost
