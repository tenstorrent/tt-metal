// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include <tt-logger/tt-logger.hpp>
#include <functional>
#include <memory>

#include "device.hpp"
#include <tt_stl/overloaded.hpp>

using namespace tt::tt_metal;

namespace tt::tt_metal::detail {

Buffer& GetBufferObject(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer) {
    return std::visit(
        ttsl::overloaded{
            [](const std::shared_ptr<Buffer>& b) -> Buffer& { return *b; },
            [](Buffer& b) -> Buffer& { return b; },
        },
        buffer);
}

void ValidateBufferRegion(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer, const BufferRegion& region) {
    Buffer& buffer_obj = GetBufferObject(buffer);

    TT_FATAL(
        buffer_obj.impl().is_valid_region(region),
        "Buffer region with offset {} and size {} is invalid.",
        region.offset,
        region.size);
}
}  // namespace tt::tt_metal::detail
