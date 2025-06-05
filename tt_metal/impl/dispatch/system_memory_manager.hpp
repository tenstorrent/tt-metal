// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// needed for private members
#include "system_memory_cq_interface.hpp"
#include <umd/device/chip_helpers/tlb_manager.h>  // needed because tt_io.hpp requires needs TLBManager
#include <umd/device/tt_io.hpp>                   // for tt::Writer
#include <umd/device/tt_xy_pair.h>                // for tt_cxy_pair
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

using chip_id_t = int;

namespace tt::tt_metal {

class SystemMemoryManager {
public:
    SystemMemoryManager(chip_id_t device_id, uint8_t num_hw_cqs);

    uint32_t get_next_event(uint8_t cq_id);

    void reset_event_id(uint8_t cq_id);

    void increment_event_id(uint8_t cq_id, uint32_t val);

    void set_last_completed_event(uint8_t cq_id, uint32_t event_id);

    uint32_t get_last_completed_event(uint8_t cq_id);

    void reset(uint8_t cq_id);

    void set_issue_queue_size(uint8_t cq_id, uint32_t issue_queue_size);

    void set_bypass_mode(const bool enable, const bool clear);

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

    chip_id_t get_device_id() const;

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
    chip_id_t device_id = 0;
    uint8_t num_hw_cqs = 0;
    const std::function<void(uint32_t, uint32_t, uint8_t*)> fast_write_callable;
    std::vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start = nullptr;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size = 0;
    uint32_t channel_offset = 0;
    std::vector<uint32_t> cq_to_event;
    std::vector<uint32_t> cq_to_last_completed_event;
    std::vector<std::mutex> cq_to_event_locks;
    std::vector<tt_cxy_pair> prefetcher_cores;
    std::vector<tt::Writer> prefetch_q_writers;
    std::vector<uint32_t> prefetch_q_dev_ptrs;
    std::vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable = false;
    std::vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset = 0;
};

}  // namespace tt::tt_metal
