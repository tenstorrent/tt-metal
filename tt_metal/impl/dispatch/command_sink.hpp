// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

// Destination for one command queue's prefetcher command stream: the CQ in system memory, or a trace being recorded.
// Writers reserve issue queue space, fill it, push it back, then publish it with a fetch queue entry.
class CommandSink {
public:
    virtual ~CommandSink() = default;

    // Returns a pointer to cmd_size_B bytes of issue queue space starting at issue_queue_write_ptr().
    virtual void* issue_queue_reserve(uint32_t cmd_size_B) = 0;

    // write_ptr comes from issue_queue_write_ptr().
    virtual void cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) = 0;

    virtual uint32_t issue_queue_write_ptr() const = 0;

    virtual void issue_queue_push_back(uint32_t push_size_B) = 0;

    virtual void fetch_queue_reserve_back() = 0;

    virtual void fetch_queue_write(uint32_t command_size_B, bool stall_prefetcher) = 0;

    void publish(uint32_t size_B, bool stall_prefetcher = false) {
        issue_queue_push_back(size_B);
        fetch_queue_reserve_back();
        fetch_queue_write(size_B, stall_prefetcher);
    }

protected:
    CommandSink() = default;
    CommandSink(const CommandSink&) = default;
    CommandSink& operator=(const CommandSink&) = default;
    CommandSink(CommandSink&&) = default;
    CommandSink& operator=(CommandSink&&) = default;
};

}  // namespace tt::tt_metal
