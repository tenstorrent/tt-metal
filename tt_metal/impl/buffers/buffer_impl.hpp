// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>

namespace tt::tt_metal {

namespace detail {

void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer);

template <typename DType>
void WriteToBuffer(Buffer& buffer, const std::vector<DType>& host_buffer) {
    WriteToBuffer(
        buffer,
        ttsl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(DType)));
}
template <typename DType>
void WriteToBuffer(const std::shared_ptr<Buffer>& buffer, const std::vector<DType>& host_buffer) {
    WriteToBuffer(*buffer, host_buffer);
}

void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer);

template <typename DType>
void ReadFromBuffer(Buffer& buffer, std::vector<DType>& host_buffer) {
    auto buffer_size = buffer.size();
    TT_FATAL(buffer_size % sizeof(DType) == 0, "Buffer size is not divisible by dtype size");
    host_buffer.resize(buffer.size() / sizeof(DType));
    ReadFromBuffer(buffer, reinterpret_cast<uint8_t*>(host_buffer.data()));
}
template <typename DType>
void ReadFromBuffer(const std::shared_ptr<Buffer>& buffer, std::vector<DType>& host_buffer) {
    ReadFromBuffer(*buffer, host_buffer);
}

}  // namespace detail

}  // namespace tt::tt_metal
