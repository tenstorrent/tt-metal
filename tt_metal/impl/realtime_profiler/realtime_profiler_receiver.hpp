// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "context/context_types.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_device.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_service.hpp"

namespace tt::tt_metal {

class DataCollector;

namespace distributed {
class MeshDevice;
}  // namespace distributed

using experimental::ProgramRealtimeRecord;

// The record producer for one MeshDevice: owns the per-device sockets, the probe scheduler, the record ring, and the
// receiver thread behind them, and publishes what it drains to whatever consumers the service has attached.
class RealtimeProfilerReceiver {
public:
    // Null when no local device could run the profiler (eligibility gate, clock or AICLK bring-up,
    // or socket creation); a constructed receiver always has devices to drain.
    static std::unique_ptr<RealtimeProfilerReceiver> create(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~RealtimeProfilerReceiver();

    RealtimeProfilerReceiver(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver& operator=(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver(RealtimeProfilerReceiver&&) = delete;
    RealtimeProfilerReceiver& operator=(RealtimeProfilerReceiver&&) = delete;

    // Idempotent.
    void shutdown();

    size_t max_batch_records() const;

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
    uint32_t read_ring_full_wait_count();
    size_t num_active_devices() const { return devices_.size(); }
    // Largest gap between consecutive clock probes on any device since the previous call; reading
    // clears it. Gap pairs beyond kDvfsCertificateWindowBudget cost chords their certificate.
    uint64_t take_peak_probe_gap_ns();
    uint64_t num_chords_finalized() const;
    uint64_t num_chords_certified() const;
    // Records published with a fallback-tier bound (uncertified chord or history envelope). Never
    // dropped; tests bound this to a tiny fraction of published records.
    uint64_t num_records_on_uncertified_chords() const;
    // Clock probes rejected as implausible (garbage PCIe reads) plus probes the view discarded
    // as non-monotone at ingest.
    uint64_t num_rejected_probes() const;
    // Held-back records published early because the pending ring was full (probe outage deeper
    // than a FIFO fill). Published with fallback-quality bounds, never dropped.
    uint64_t num_holdback_evictions() const { return num_holdback_evictions_.load(std::memory_order_relaxed); }
    // Worst full receiver-loop iteration and iteration count since the previous call; reading
    // clears them. The loop period is what FIFO occupancy scales with.
    std::array<uint64_t, 2> take_loop_stats() {
        return {
            peak_loop_ns_.exchange(0, std::memory_order_relaxed), loop_count_.exchange(0, std::memory_order_relaxed)};
    }

private:
    RealtimeProfilerReceiver(
        ContextId context_id,
        std::vector<RealtimeProfilerDevice> devices,
        std::unique_ptr<ProbeScheduler> probe_scheduler,
        ProbeScheduler::Demand probe_demand);

    void note_fifo_depth(uint32_t available);
    // Publishes the device's held-back records, then decodes `pages`, publishing records whose chord bounds are final
    // and holding the rest back for the next probe. `batch` is the caller's scratch, reused so this never allocates.
    // Returns the number of records published.
    size_t publish_pages(
        RealtimeProfilerDevice& dev_state, std::span<const uint32_t> pages, std::vector<ProgramRealtimeRecord>& batch);

    // Receiver thread body.
    void run();
    uint64_t run_loop(std::vector<uint32_t>& page_buf);
    uint64_t drain_on_shutdown(std::vector<uint32_t>& page_buf);
    uint32_t drain_all_devices(std::vector<uint32_t>& page_buf);
    // Reads, ingests queued probes, then publishes.
    // Returns number of pages read.
    uint32_t drain_device_pages(RealtimeProfilerDevice& dev_state, std::vector<uint32_t>& page_buf);

    const DataCollector* data_collector_ = nullptr;
    RealtimeProfilerService* realtime_profiler_service_ = nullptr;

    std::vector<RealtimeProfilerDevice> devices_;
    // The probe cadence, running since before this receiver existed; the receiver never touches
    // a clock register, so a blocked clock read can delay probes but never draining — and
    // receiver health never delays probes. Stopped by shutdown() after the final drain.
    std::unique_ptr<ProbeScheduler> probe_scheduler_;
    // Held while records are deliverable; released by the consumer gate, re-acquired with the
    // first consumer, and the shutdown drain scopes its own.
    std::optional<ProbeScheduler::Demand> probe_demand_;
    RealtimeProfilerRecordRing ring_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};
    // Receiver thread only: whether records are deliverable; gates probing and draining-vs-discarding.
    bool consumers_active_ = true;

    std::atomic<uint32_t> peak_fifo_pages_{0};
    std::atomic<uint32_t> peak_fifo_pages_since_report_{0};
    uint32_t fifo_pages_window_max_ = 0;  // peak since the last Tracy plot sample
    // Worst clock-sync error among records published since last plot; nullopt when the window saw none.
    std::optional<std::chrono::nanoseconds> sync_error_window_max_;
    std::atomic<uint64_t> num_published_records_{0};
    std::atomic<uint64_t> num_published_batches_{0};
    std::atomic<uint64_t> peak_loop_ns_{0};
    std::atomic<uint64_t> loop_count_{0};

    std::atomic<uint64_t> num_inverted_timestamp_records_{0};
    std::atomic<uint64_t> num_holdback_evictions_{0};

    std::chrono::steady_clock::time_point last_inverted_timestamp_warn_;
    std::chrono::steady_clock::time_point last_eviction_warn_;

    std::vector<ProgramRealtimeRecord> publish_batch_;
};

}  // namespace tt::tt_metal
