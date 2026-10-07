// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// needed for private members
#include "system_memory_cq_interface.hpp"
#include <umd/device/io_window/io_window.hpp>     // for tt::umd::IoWindow
#include <umd/device/types/xy_pair.hpp>           // for tt_cxy_pair
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>
#include <hostdevcommon/common_values.hpp>  // DISPATCH_MAX_MESSAGE_ENTRIES
#include "impl/context/context_types.hpp"

using ChipId = int;

namespace tt::tt_metal {

class Buffer;
class MetalContext;

// True when the host can't pin D2H memory (no 64-bit PCIe addressing and no IOMMU).
bool d2h_uses_hugepage_fallback(const MetalContext& ctx);

class SystemMemoryManager {
public:
    // Create a SystemMemoryManager for accessing system memory accessible by the given device
    // TODO: context_id will be removed in favor of directly passing around the MetalContext and MetalEnv reference.
    SystemMemoryManager(ContextId context_id, ChipId device_id, uint8_t num_hw_cqs);

    uint32_t get_next_event(uint8_t cq_id);

    uint32_t get_last_event(uint8_t cq_id);

    void reset_event_id(uint8_t cq_id);

    void increment_event_id(uint8_t cq_id, uint32_t val);

    void set_last_completed_event(uint8_t cq_id, uint32_t event_id);

    uint32_t get_last_completed_event(uint8_t cq_id);

    uint32_t get_current_event(uint8_t cq_id);

    void reset(uint8_t cq_id);

    void set_issue_queue_size(uint8_t cq_id, uint32_t issue_queue_size);

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

    ContextId get_context_id() const;

    std::vector<SystemMemoryCQInterface>& get_cq_interfaces();

    void* issue_queue_reserve(uint32_t cmd_size_B, uint8_t cq_id);

    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr);

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, uint8_t cq_id);

    uint32_t completion_queue_wait_front(uint8_t cq_id, std::atomic<bool>& exit_condition) const;

    void send_completion_queue_read_ptr(uint8_t cq_id) const;

    void* get_completion_queue_ptr(uint8_t cq_id) const;

    void wrap_issue_queue_wr_ptr(uint8_t cq_id);

    void wrap_completion_queue_rd_ptr(uint8_t cq_id);

    void completion_queue_pop_front(uint32_t num_pages_read, uint8_t cq_id);

    void fetch_queue_reserve_back(uint8_t cq_id);

    void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher = false);

    // Both CQs on the device must be idle when this is called.
    void set_current_and_last_completed_event(
        uint8_t cq_id, uint32_t current_event_id, uint32_t last_completed_event_id);

    bool is_dram_backed() const;

    uint32_t get_dram_region_base_addr() const;

    uint32_t get_dram_region_bank_id() const;

    std::pair<void*, uint32_t> allocate_region(uint32_t size);

    uint32_t get_channel_offset() const { return channel_offset; }

    // Host mirror of one worker GO message slot (one per sub-device sync index) on this chip. It lives here rather
    // than in a MeshDevice because a parent mesh and its submeshes drive the same chips' dispatchers and workers:
    // the state it mirrors is physical, so every MeshDevice using this chip must see the same copy.
    struct WorkerGoMirror {
        // CQ whose dispatcher coords are in the workers' go_msg_t word 1 and whose dispatcher go counter matches the
        // workers. A CQ must emit CQ_DISPATCH_SET_GO_SIGNAL_NOC_ADDR (which reseeds its counter from go_count) before
        // its first go whenever this is not its id.
        std::optional<uint8_t> noc_addr_cq_id;
        // The workers' go count for this slot: advances with every go a dispatcher sends to it. The reseed baseline.
        uint8_t go_count = 0;
    };

    WorkerGoMirror& worker_go_mirror(uint32_t sync_index) { return worker_go_mirrors_.at(sync_index); }

    // Forget which CQ configured each slot. With rebaselined_slots, a RESET_READ_PTR go was also sent to the first
    // rebaselined_slots slots, leaving their go count at 1; the remaining slots were cleared to 0.
    void reset_worker_go_mirrors(std::optional<uint32_t> rebaselined_slots);

private:
    bool is_mock_device() const;

    void init_dispatch_core_interfaces(uint8_t num_hw_cqs, uint16_t channel);

    ContextId context_id;
    ChipId device_id = 0;
    std::vector<uint32_t> completion_byte_addrs;
    char* cq_sysmem_start = nullptr;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    uint32_t cq_size = 0;
    uint32_t channel_offset = 0;
    std::vector<uint32_t> cq_to_event;
    std::vector<uint32_t> cq_to_last_completed_event;
    mutable std::vector<std::mutex> cq_to_event_locks;
    std::vector<tt_cxy_pair> prefetcher_cores;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> prefetch_q_windows;
    std::vector<std::unique_ptr<tt::umd::IoWindow>> completion_q_windows;
    std::vector<uint32_t> prefetch_q_dev_ptrs;
    std::vector<uint32_t> prefetch_q_dev_fences;

    bool bypass_enable = false;
    std::vector<uint32_t> bypass_buffer;
    uint32_t bypass_buffer_write_offset = 0;

    std::unique_ptr<char[]> dram_region_staging_buffer;

    // Bump-allocated tail of CQ sysmem (after all per-CQ issue/completion buffers). Device and host
    // addresses are returned together by allocate_region() for PCIe/D2H consumers.
    uint32_t free_region_start_ = 0;
    uint32_t free_region_size_ = 0;
    uint32_t free_region_bump_ = 0;
    char* free_region_host_ptr_ = nullptr;

    std::array<WorkerGoMirror, DISPATCH_MAX_MESSAGE_ENTRIES> worker_go_mirrors_{};
};

}  // namespace tt::tt_metal
