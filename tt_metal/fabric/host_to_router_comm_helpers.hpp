// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_fabric {

class HostChannelCounter {
public:
    HostChannelCounter() = default;
    ~HostChannelCounter() = default;

    HostChannelCounter(uint8_t num_buffer_slots) : num_buffer_slots_(num_buffer_slots) { reset(); }

    inline void reset() {
        counter_ = 0;
        buffer_index_ = 0;
    }

    inline void increment() {
        counter_++;
        buffer_index_ = wrap_increment();
    }

    inline uint8_t get_buffer_index() const { return buffer_index_; }
    inline uint32_t get_counter() const { return counter_; }

private:
    uint32_t counter_;
    uint8_t buffer_index_;
    uint8_t num_buffer_slots_;

    inline uint8_t wrap_increment() { return (buffer_index_ == num_buffer_slots_ - 1) ? 0 : buffer_index_ + 1; }
};

class RouterCommContext {
public:
    RouterCommContext() = default;
    ~RouterCommContext() = default;

    RouterCommContext(uint8_t num_buffer_slots) : local_write_counter_(num_buffer_slots) {
        local_write_counter_.reset();
    }

    HostChannelCounter& get_local_write_counter() { return local_write_counter_; }

    const HostChannelCounter& get_local_write_counter() const { return local_write_counter_; }

private:
    HostChannelCounter local_write_counter_;
};

}  // namespace tt::tt_fabric
