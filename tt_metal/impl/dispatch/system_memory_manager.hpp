// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// needed for private members
#include "aux_region_allocator.hpp"
#include "command_sink.hpp"
#include "cq_backing.hpp"
#include "system_memory_cq_interface.hpp"
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

    bool is_dram_backed() const { return backing_->is_dram_backed(); }

    CqBacking& backing() const { return *backing_; }

    AuxRegionAllocator& aux_region_allocator() { return backing_->aux_region_allocator(); }

private:
    void send_completion_queue_read_ptr(uint8_t cq_id) const;

    MetalEnvImpl& env_;
    MetalContext& ctx_;
    ChipId device_id = 0;
    std::unique_ptr<CqBacking> backing_;
    std::vector<SystemMemoryCQInterface> cq_interfaces;
    struct CqEventProgress {
        std::atomic<uint32_t> last_issued{0};
        std::atomic<uint32_t> last_completed{0};
    };
    std::unique_ptr<CqEventProgress[]> cq_events_;
};

class CqCommandSink final : public CommandSink {
public:
    CqCommandSink(SystemMemoryManager& manager, uint8_t cq_id) : manager_(manager), cq_id_(cq_id) {}

    void* issue_queue_reserve(uint32_t cmd_size_B) override { return manager_.issue_queue_reserve(cmd_size_B, cq_id_); }

    void cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) override {
        manager_.cq_write(data, size_B, write_ptr);
    }

    uint32_t issue_queue_write_ptr() const override { return manager_.get_issue_queue_write_ptr(cq_id_); }

    void issue_queue_push_back(uint32_t push_size_B) override { manager_.issue_queue_push_back(push_size_B, cq_id_); }

    void fetch_queue_reserve_back() override { manager_.fetch_queue_reserve_back(cq_id_); }

    void fetch_queue_write(uint32_t command_size_B, bool stall_prefetcher) override {
        manager_.fetch_queue_write(command_size_B, cq_id_, stall_prefetcher);
    }

private:
    SystemMemoryManager& manager_;
    uint8_t cq_id_;
};

}  // namespace tt::tt_metal
