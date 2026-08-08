// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "context/context_types.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_device.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_service.hpp"

namespace tt::tt_metal {

class IDevice;
class Program;
class DataCollector;

namespace distributed {
class D2HSocket;
class MeshDevice;
}  // namespace distributed

using experimental::ProgramRealtimeRecord;

// The record producer for one MeshDevice: owns the per-device sockets, the record ring, and the receiver thread behind
// them, and publishes what it drains to whatever consumers the service has attached.
class RealtimeProfilerReceiver : public ProgramRecordProducer {
public:
    // Null when no local device passed the eligibility gate; a constructed receiver always has devices to drain.
    static std::unique_ptr<RealtimeProfilerReceiver> create(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~RealtimeProfilerReceiver();

    RealtimeProfilerReceiver(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver& operator=(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver(RealtimeProfilerReceiver&&) = delete;
    RealtimeProfilerReceiver& operator=(RealtimeProfilerReceiver&&) = delete;

    // Idempotent.
    void shutdown();

    size_t max_batch_records() const override;
    RealtimeProfilerRecordRing::Reader make_reader() override;
    void wait_until_no_readers() override;

    // All-time peak D2H FIFO usage.
    uint32_t peak_fifo_pages() const { return peak_fifo_pages_.load(std::memory_order_relaxed); }
    // High-water since the previous call: reading it clears it, so exactly one caller can use it.
    uint32_t take_peak_fifo_pages() { return peak_fifo_pages_since_report_.exchange(0, std::memory_order_relaxed); }
    uint32_t host_fifo_capacity_pages() const;
    uint64_t num_published_records() const { return num_published_records_.load(std::memory_order_relaxed); }
    uint64_t num_published_batches() const { return num_published_batches_.load(std::memory_order_relaxed); }
    // end_timestamp < start_timestamp (torn dispatch_s → BRISC handoff).
    uint64_t num_inverted_timestamp_records() const {
        return num_inverted_timestamp_records_.load(std::memory_order_relaxed);
    }
    // map_record failed: timestamps predate every retained probe (sync/history problem).
    uint64_t num_unmappable_records() const { return num_unmappable_records_.load(std::memory_order_relaxed); }
    uint32_t read_ring_full_wait_count();
    size_t num_active_devices() const { return devices_.size(); }

private:
    RealtimeProfilerReceiver(ContextId context_id, std::vector<RealtimeProfilerDevice> devices);

    void note_fifo_depth(uint32_t available);
    // Decodes `pages` and publishes them, placed against the probe history. `batch` is the caller's scratch, reused so
    // this never allocates.
    void publish_pages(
        RealtimeProfilerDevice& dev_state, std::span<const uint32_t> pages, std::vector<ProgramRealtimeRecord>& batch);

    // Receiver thread body.
    void run();
    uint64_t run_loop(std::vector<uint32_t>& page_buf);
    uint64_t drain_on_shutdown(std::vector<uint32_t>& page_buf);
    uint32_t drain_all_devices(std::chrono::steady_clock::time_point now, std::vector<uint32_t>& page_buf);
    // Reads, does a sync probe, then publishes.
    // Returns number of pages read.
    uint32_t drain_device_pages(RealtimeProfilerDevice& dev_state, std::vector<uint32_t>& page_buf);

    const DataCollector* data_collector_ = nullptr;
    RealtimeProfilerService* realtime_profiler_service_ = nullptr;

    std::vector<RealtimeProfilerDevice> devices_;
    RealtimeProfilerRecordRing ring_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};

    // Diagnostics
    std::atomic<uint32_t> peak_fifo_pages_{0};               // all-time peak D2H FIFO occupancy
    std::atomic<uint32_t> peak_fifo_pages_since_report_{0};  // peak since take_peak_fifo_pages()
    uint32_t fifo_pages_window_max_ = 0;                     // peak since the last Tracy plot sample
    // Worst clock-sync error among records published since last plot; nullopt when the window saw none.
    std::optional<std::chrono::nanoseconds> sync_error_window_max_;
    std::atomic<uint64_t> num_published_records_{0};  // records published to the ring
    std::atomic<uint64_t> num_published_batches_{0};  // batches published to the ring
    std::atomic<uint64_t> num_inverted_timestamp_records_{0};
    std::atomic<uint64_t> num_unmappable_records_{0};

    std::chrono::steady_clock::time_point last_inverted_timestamp_warn_;
    std::chrono::steady_clock::time_point last_unmappable_warn_;

    std::vector<ProgramRealtimeRecord> publish_batch_;
};

}  // namespace tt::tt_metal
