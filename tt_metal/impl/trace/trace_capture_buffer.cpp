// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trace_capture_buffer.hpp"

#include <cstring>
#include <utility>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

void* TraceCaptureBuffer::issue_queue_reserve(uint32_t cmd_size_B) {
    TT_ASSERT(cmd_size_B % sizeof(uint32_t) == 0, "Trace command of {} B is not a whole number of words", cmd_size_B);
    // Space reserved earlier but not pushed back yet is reused, as in a CQ ring.
    const size_t needed_words = (static_cast<size_t>(write_offset_B_) + cmd_size_B) / sizeof(uint32_t);
    if (data_.size() < needed_words) {
        data_.resize(needed_words);
    }
    return reinterpret_cast<char*>(data_.data()) + write_offset_B_;
}

std::vector<uint32_t> TraceCaptureBuffer::take() && {
    TT_ASSERT(
        data_.size() * sizeof(uint32_t) == write_offset_B_,
        "Trace has {} B reserved but {} B pushed back",
        data_.size() * sizeof(uint32_t),
        write_offset_B_);
    write_offset_B_ = 0;
    return std::move(data_);
}

void TraceCaptureBuffer::cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) {
    TT_ASSERT(
        write_ptr + size_B <= data_.size() * sizeof(uint32_t),
        "Trace write of {} B at offset {} is past the reserved {} B",
        size_B,
        write_ptr,
        data_.size() * sizeof(uint32_t));
    std::memcpy(reinterpret_cast<char*>(data_.data()) + write_ptr, data, size_B);
}

}  // namespace tt::tt_metal
