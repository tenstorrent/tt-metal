// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// needed for private members
#include "aux_region_allocator.hpp"
#include "cq_backing.hpp"
#include "host_cq.hpp"
#include "system_memory_cq_interface.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/device_types.hpp>  // for tt::ChipId
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

class Buffer;
class MetalContext;
class MetalEnvImpl;

// The host side of a device's fast dispatch command queues: one HostCq per hardware CQ, sharing a single CqBacking.
class SystemMemoryManager {
public:
    // Create a SystemMemoryManager for accessing system memory accessible by the given device
    // The referenced MetalEnvImpl and MetalContext must outlive this SystemMemoryManager.
    SystemMemoryManager(MetalEnvImpl& env, MetalContext& ctx, ChipId device_id, uint8_t num_hw_cqs);

    HostCq& cq(uint8_t cq_id) {
        TT_ASSERT(cq_id < cqs_.size(), "CQ {} is out of range: device {} has {} CQs", cq_id, device_id_, cqs_.size());
        return *cqs_[cq_id];
    }
    const HostCq& cq(uint8_t cq_id) const {
        TT_ASSERT(cq_id < cqs_.size(), "CQ {} is out of range: device {} has {} CQs", cq_id, device_id_, cqs_.size());
        return *cqs_[cq_id];
    }

    uint8_t num_cqs() const { return static_cast<uint8_t>(cqs_.size()); }

    void set_last_issued_event(uint8_t cq_id, uint32_t event_id) { cq(cq_id).set_last_issued_event(event_id); }

    uint32_t get_last_issued_event(uint8_t cq_id) const { return cq(cq_id).last_issued_event(); }

    void set_last_completed_event(uint8_t cq_id, uint32_t event_id) { cq(cq_id).set_last_completed_event(event_id); }

    uint32_t get_last_completed_event(uint8_t cq_id) const { return cq(cq_id).last_completed_event(); }

    void reset_events(uint8_t cq_id, uint32_t event_id) { cq(cq_id).reset_events(event_id); }

    void reset(uint8_t cq_id) { cq(cq_id).reset(); }

    uint32_t get_issue_queue_size(uint8_t cq_id) const { return cq(cq_id).issue_queue_size(); }

    uint32_t get_issue_queue_limit(uint8_t cq_id) const { return cq(cq_id).issue_queue_limit(); }

    uint32_t get_completion_queue_size(uint8_t cq_id) const { return cq(cq_id).completion_queue_size(); }

    uint32_t get_completion_queue_limit(uint8_t cq_id) const { return cq(cq_id).completion_queue_limit(); }

    uint32_t get_issue_queue_write_ptr(uint8_t cq_id) const { return cq(cq_id).issue_queue_write_ptr(); }

    uint32_t get_completion_queue_read_ptr(uint8_t cq_id) const { return cq(cq_id).completion_queue_read_ptr(); }

    uint32_t get_completion_queue_read_toggle(uint8_t cq_id) const { return cq(cq_id).completion_queue_read_toggle(); }

    uint32_t get_cq_size() const { return backing_->cq_size(); }

    ChipId get_device_id() const { return device_id_; }

    MetalEnvImpl& env_impl() const { return env_; }

    MetalContext& metal_context() const { return ctx_; }

    void* issue_queue_reserve(uint32_t cmd_size_B, uint8_t cq_id) { return cq(cq_id).issue_queue_reserve(cmd_size_B); }

    // write_ptr is an issue queue address from get_issue_queue_write_ptr() of any of this device's CQs.
    void cq_write(const void* data, uint32_t size_in_bytes, uint32_t write_ptr) {
        backing_->copy_to_issue_queue(backing_->host_ptr(write_ptr), data, size_in_bytes);
    }

    // TODO: RENAME issue_queue_stride ?
    void issue_queue_push_back(uint32_t push_size_B, uint8_t cq_id) { cq(cq_id).issue_queue_push_back(push_size_B); }

    uint32_t completion_queue_wait_front(uint8_t cq_id, std::atomic<bool>& exit_condition) const {
        return cq(cq_id).completion_queue_wait_front(exit_condition);
    }

    void* get_completion_queue_ptr(uint8_t cq_id) const { return cq(cq_id).completion_queue_ptr(); }

    void wrap_issue_queue_wr_ptr(uint8_t cq_id) { cq(cq_id).wrap_issue_queue_wr_ptr(); }

    void completion_queue_pop_front(uint32_t num_pages_read, uint8_t cq_id) {
        cq(cq_id).completion_queue_pop_front(num_pages_read);
    }

    void fetch_queue_reserve_back(uint8_t cq_id) { cq(cq_id).fetch_queue_reserve_back(); }

    void fetch_queue_write(uint32_t command_size_B, uint8_t cq_id, bool stall_prefetcher = false) {
        cq(cq_id).fetch_queue_write(command_size_B, stall_prefetcher);
    }

    bool is_dram_backed() const { return backing_->is_dram_backed(); }

    CqBacking& backing() const { return *backing_; }

    AuxRegionAllocator& aux_region_allocator() { return backing_->aux_region_allocator(); }

private:
    MetalEnvImpl& env_;
    MetalContext& ctx_;
    ChipId device_id_ = 0;
    std::unique_ptr<CqBacking> backing_;
    std::vector<std::unique_ptr<HostCq>> cqs_;
};

}  // namespace tt::tt_metal
