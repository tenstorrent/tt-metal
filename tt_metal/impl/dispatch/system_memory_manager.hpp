// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// needed for private members
#include "system_memory_cq_interface.hpp"
#include <umd/device/chip_helpers/tlb_manager.hpp>  // needed because tt_io.hpp requires needs TLBManager
#include <umd/device/tt_io.hpp>                     // for umd::Writer
#include <umd/device/types/xy_pair.hpp>           // for tt_cxy_pair
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

using ChipId = int;

namespace tt::tt_metal {

class SystemMemoryManager {
public:
    SystemMemoryManager(ChipId device_id, uint8_t num_hw_cqs);

    uint32_t get_next_event(uint8_t cq_id);

    void reset_event_id(uint8_t cq_id);

    void increment_event_id(uint8_t cq_id, uint32_t val);

    void set_last_completed_event(uint8_t cq_id, uint32_t event_id);

    uint32_t get_last_completed_event(uint8_t cq_id);

    void reset(uint8_t cq_id);

    void set_issue_queue_size(uint8_t cq_id, uint32_t issue_queue_size);

    void set_bypass_mode(bool enable, bool clear);

    bool get_bypass_mode();

    std::vector<uint32_t>& get_bypass_data();

    uint32_t get_issue_queue_size(uint8_t cq_id) const;

    uint32_t get_issue_queue_limit(uint8_t cq_id) const;

    uint32_t get_completion_queue_size(uint8_t cq_id) const;

    uint32_t get_completion_queue_limit(uint8_t cq_id) const;

    uint32_t get_issue_queue_write_ptr(uint8_t cq_id) const;

    uint32_t get_completion_queue_read_ptr(uint8_t cq_id) const;

    uint32_t get_completion_queue_read_toggle(uint8_t cq_id) const;

    uint32_t get_cq_size() const;

    ChipId get_device_id() const;

    std::vector<SystemMemoryCQInterface>& get_cq_interfaces();

    void* issue_queue_reserve(uint32_t cmd_size_B, uint8_t cq_id);

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr);

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, uint8_t cq_id);

    uint32_t completion_queue_wait_front(uint8_t cq_id, std::atomic<bool>& exit_condition) const;

    void send_completion_queue_read_ptr(uint8_t cq_id) const;

    void wrap_issue_queue_wr_ptr(uint8_t cq_id);

    void wrap_completion_queue_rd_ptr(uint8_t cq_id);

    void completion_queue_pop_front(uint32_t num_pages_read, uint8_t cq_id);

    void fetch_queue_reserve_back(uint8_t cq_id);

    void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher = false);

private:
    ChipId device_id = 0;
    uint8_t num_hw_cqs = 0;
    std::vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start = nullptr;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size = 0;
    uint32_t channel_offset = 0;
    std::vector<uint32_t> cq_to_event;
    std::vector<uint32_t> cq_to_last_completed_event;
    std::vector<std::mutex> cq_to_event_locks;
    std::vector<tt_cxy_pair> prefetcher_cores;
    std::vector<umd::Writer> prefetch_q_writers;
    std::vector<umd::Writer> completion_q_writers;
    std::vector<uint32_t> prefetch_q_dev_ptrs;
    std::vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable = false;
    std::vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset = 0;
};

}  // namespace tt::tt_metal
