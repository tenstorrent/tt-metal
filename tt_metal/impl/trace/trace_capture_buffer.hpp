// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "impl/dispatch/command_sink.hpp"

namespace tt::tt_metal {

// Records a command stream on the host while a trace is built. Issue queue addresses are byte offsets into the
// recorded stream, and fetch queue entries are dropped, since trace replay fetches the whole stream at once.
class TraceCaptureBuffer final : public CommandSink {
public:
    void* issue_queue_reserve(uint32_t cmd_size_B) override;

    void cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) override;

    uint32_t issue_queue_write_ptr() const override { return write_offset_B_; }

    void issue_queue_push_back(uint32_t push_size_B) override { write_offset_B_ += push_size_B; }

    void fetch_queue_reserve_back() override {}

    void fetch_queue_write(uint32_t /*command_size_B*/, bool /*stall_prefetcher*/) override {}

    // Ends the capture and returns the recorded commands. Every reserved byte must have been pushed back.
    std::vector<uint32_t> take() &&;

private:
    std::vector<uint32_t> data_;
    uint32_t write_offset_B_ = 0;
};

}  // namespace tt::tt_metal
