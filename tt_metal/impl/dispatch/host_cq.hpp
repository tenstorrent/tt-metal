// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>

#include "command_sink.hpp"
#include "cq_backing.hpp"
#include "system_memory_cq_interface.hpp"

namespace tt::tt_metal {

class IDevice;

// One hardware command queue of a device, seen from the host. It owns the ring state of the issue and completion
// queues and how far the host has got with events, and writes through the device's CqBacking.
class HostCq final : public CommandSink {
public:
    // The backing must outlive this object.
    HostCq(CqBacking& backing, uint8_t cq_id, const SystemMemoryCQInterface& cq_interface);

    uint8_t id() const { return cq_id_; }

    CqBacking& backing() const { return backing_; }

    const SystemMemoryCQInterface& cq_interface() const { return cq_interface_; }

    // Issue queue.
    void* issue_queue_reserve(uint32_t cmd_size_B) override;

    void cq_write(const void* data, uint32_t size_B, uint32_t write_ptr) override;

    uint32_t issue_queue_write_ptr() const override { return cq_interface_.issue_fifo_wr_ptr << 4; }

    void issue_queue_push_back(uint32_t push_size_B) override;

    void fetch_queue_reserve_back() override { backing_.fetch_queue_reserve_back(cq_id_); }

    void fetch_queue_write(uint32_t command_size_B, bool stall_prefetcher) override {
        backing_.fetch_queue_write(command_size_B, cq_id_, stall_prefetcher);
    }

    void wrap_issue_queue_wr_ptr();

    uint32_t issue_queue_size() const { return cq_interface_.issue_fifo_size << 4; }

    uint32_t issue_queue_limit() const { return cq_interface_.issue_fifo_limit << 4; }

    // Completion queue.
    uint32_t completion_queue_wait_front(std::atomic<bool>& exit_condition) const {
        return backing_.completion_queue_wait_front(cq_id_, cq_interface_, exit_condition);
    }

    void completion_queue_pop_front(uint32_t num_pages_read);

    // Host pointer to the start of the completion queue, synced from the device where the backing needs that.
    void* completion_queue_ptr() const;

    uint32_t completion_queue_size() const { return cq_interface_.completion_fifo_size << 4; }

    uint32_t completion_queue_limit() const { return cq_interface_.completion_fifo_limit << 4; }

    uint32_t completion_queue_read_ptr() const { return cq_interface_.completion_fifo_rd_ptr << 4; }

    uint32_t completion_queue_read_toggle() const { return cq_interface_.completion_fifo_rd_toggle; }

    // Returns both queues to their starting positions.
    void reset();

    // The last event issued to and completed by this CQ; the mesh command queue allocates the IDs.
    void set_last_issued_event(uint32_t event_id) { last_issued_event_.store(event_id, std::memory_order_relaxed); }

    uint32_t last_issued_event() const { return last_issued_event_.load(std::memory_order_relaxed); }

    void set_last_completed_event(uint32_t event_id);

    uint32_t last_completed_event() const { return last_completed_event_.load(std::memory_order_acquire); }

    // Sets both counters, skipping the ordering check in set_last_completed_event. The CQ must be idle.
    void reset_events(uint32_t event_id);

private:
    // Start of the issue queue, where the write pointer wraps to, in 16 B words.
    uint32_t issue_queue_base_16B() const { return (cq_interface_.cq_start + cq_interface_.offset) >> 4; }

    void send_completion_queue_read_ptr() const;

    CqBacking& backing_;
    const uint8_t cq_id_;
    SystemMemoryCQInterface cq_interface_;
    std::atomic<uint32_t> last_issued_event_{0};
    std::atomic<uint32_t> last_completed_event_{0};
};

// The device's CQ cq_id. The device must be a single device, not a MeshDevice.
HostCq& host_cq(IDevice& device, uint8_t cq_id);

}  // namespace tt::tt_metal
