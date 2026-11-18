// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// needed for private members
#include "aux_region_allocator.hpp"
#include "system_memory_cq_interface.hpp"
#include <tt-umd/io_window/io_window.hpp>  // for tt::umd::IoWindow
#include <tt-umd/types/xy_pair.hpp>        // for tt_cxy_pair
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using ChipId = int;

namespace tt::tt_metal {

class Buffer;
class MetalContext;
class MetalEnvImpl;

class SystemMemoryManager {
public:
    // Create a SystemMemoryManager for accessing system memory accessible by the given device
    // The referenced MetalEnvImpl and MetalContext must outlive this SystemMemoryManager.
    SystemMemoryManager(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs);

    // Per CQ, the last event issued to and completed by this device; the mesh command queue allocates the IDs.
    void set_last_issued_event(uint8_t cq_id, uint32_t event_id);

    uint32_t get_last_issued_event(uint8_t cq_id) const;

    void set_last_completed_event(uint8_t cq_id, uint32_t event_id);

    uint32_t get_last_completed_event(uint8_t cq_id) const;

    // Sets both counters, skipping the ordering check in set_last_completed_event. The CQ must be idle.
    void reset_events(uint8_t cq_id, uint32_t event_id);

    void reset(uint8_t cq_id);

    void set_bypass_mode(bool enable, bool clear);

    bool get_bypass_mode() const;

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

    MetalEnvImpl& env_impl() const { return env_; }

    MetalContext& metal_context() const { return ctx_; }

    std::vector<SystemMemoryCQInterface>& get_cq_interfaces();

    void* issue_queue_reserve(uint32_t cmd_size_B, uint8_t cq_id);

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr);

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, uint8_t cq_id);

    uint32_t completion_queue_wait_front(uint8_t cq_id, std::atomic<bool>& exit_condition) const;

    void* get_completion_queue_ptr(uint8_t cq_id) const;

    void wrap_issue_queue_wr_ptr(uint8_t cq_id);

    void completion_queue_pop_front(uint32_t num_pages_read, uint8_t cq_id);

    void fetch_queue_reserve_back(uint8_t cq_id);

    void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher = false);

    bool is_dram_backed() const { return is_dram_backed_; }

    uint32_t get_dram_region_base_addr() const;

    uint32_t get_dram_region_bank_id() const;

    AuxRegionAllocator& aux_region_allocator() { return aux_region_allocator_; }

private:
    bool is_mock_device() const { return is_mock_device_; }

    void init_dispatch_core_interfaces(uint8_t num_hw_cqs, uint16_t channel);

    void send_completion_queue_read_ptr(uint8_t cq_id) const;

    MetalEnvImpl& env_;
    MetalContext& ctx_;
    const bool is_mock_device_;
    const bool is_dram_backed_;
    const uint32_t cq_alignment_;  // DRAM alignment if DRAM-backed, host alignment otherwise
    ChipId device_id = 0;
    std::vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start = nullptr;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size = 0;
    uint32_t channel_offset = 0;
    struct CqEventProgress {
        std::atomic<uint32_t> last_issued{0};
        std::atomic<uint32_t> last_completed{0};
    };
    std::unique_ptr<CqEventProgress[]> cq_events_;
    std::vector<tt_cxy_pair> prefetcher_cores;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> prefetch_q_windows;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> completion_q_windows;
    std::vector<uint32_t> prefetch_q_dev_ptrs;
    std::vector<uint32_t> prefetch_q_dev_fences;
    // Phase bit value for the current lap of each fetch queue, used when the prefetcher polls it through its cache.
    std::vector<uint32_t> prefetch_q_phases;

    bool bypass_enable = false;
    std::vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset = 0;

    std::unique_ptr<char[]> dram_region_staging_buffer;

    // Empty on mock devices.
    AuxRegionAllocator aux_region_allocator_;
};

}  // namespace tt::tt_metal
