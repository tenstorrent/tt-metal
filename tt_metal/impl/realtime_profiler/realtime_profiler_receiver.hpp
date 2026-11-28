// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "context/context_types.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_device.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_service.hpp"

namespace tt::tt_metal {

class DataCollector;

namespace distributed {
class MeshDevice;
}  // namespace distributed

// The record producer for one MeshDevice: owns the per-device state, the record ring, the receiver thread that drains
// the device sockets into the ring, and the host-device clock mapping. Consumers of the ring live in the
// RealtimeProfilerService; while it has none, the receiver acks the sockets unread instead of draining them.
class RealtimeProfilerReceiver {
public:
    // Null when no local device passed the eligibility gate; a constructed receiver always has devices to drain.
    static std::unique_ptr<RealtimeProfilerReceiver> create(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    // Asks the device kernels to exit, joins the receiver thread, detaches from the service and removes its devices
    // from the Tracy consumer.
    ~RealtimeProfilerReceiver();

    RealtimeProfilerReceiver(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver& operator=(const RealtimeProfilerReceiver&) = delete;

    // RT-profiler diagnostics
    uint32_t peak_fifo_pages() const { return peak_fifo_pages_.load(std::memory_order_relaxed); }
    uint32_t host_fifo_capacity_pages() const;
    uint64_t num_published_records() const { return num_published_records_.load(std::memory_order_relaxed); }
    uint64_t num_published_batches() const { return num_published_batches_.load(std::memory_order_relaxed); }
    // Records dropped because their end time preceded their start time.
    uint64_t num_end_before_start_records() const {
        return num_end_before_start_records_.load(std::memory_order_relaxed);
    }
    uint32_t ring_full_wait_count() const;  // reads device L1
    // Peak count of dispatch_s waits for a free record slot (the BRISC fell a full ring behind); reads device L1.
    uint32_t record_ring_full_wait_count() const;
    // Dispatch-stall markers received from all devices (host-side counts; no device read).
    uint64_t dispatch_stall_events() const { return dispatch_stall_events_.load(std::memory_order_relaxed); }
    uint64_t dispatch_stall_cycles() const { return dispatch_stall_cycles_.load(std::memory_order_relaxed); }

    // Testing only: while paused, the receiver thread stops reading the D2H socket, so the device-side profiler
    // path backs up until dispatch_s waits for record slots. Records are held, not dropped; unpausing drains
    // them. Do not wait on the device (e.g. Finish) while paused: once the path backs up, dispatch waits for the
    // receiver. Unpause before closing the device.
    void pause_receiver_for_testing(bool paused) { receiver_paused_.store(paused, std::memory_order_release); }
    size_t num_active_devices() const { return devices_.size(); }

private:
    void record_anchor_time(std::chrono::steady_clock::duration took);
    RealtimeProfilerReceiver(ContextId context_id, std::vector<RealtimeProfilerDevice> devices);

    // Receiver thread entry point: drain every device socket, re-anchor the clock mapping when due, and
    // publish decoded records to the ring for consumer threads to read.
    void run_receiver();
    void run_receiver_loop();
    // Acks every device's pending pages without reading them.
    void discard_pending_pages();
    // Tracks the FIFO high-water marks and warns once per device when its FIFO fills.
    void note_fifo_depth(RealtimeProfilerDevice& dev_state, uint32_t available);
    // Gives the device profiler this device's clock mapping, so worker zones line up with real-time records.
    void publish_device_profiler_sync_anchor(const RealtimeProfilerDevice& dev_state);
    // One pass over every device. Returns the most pages any device's FIFO held.
    uint32_t drain_all_devices();
    // Reads what the device's FIFO holds, up to one read's worth, and publishes it. Returns how many pages the FIFO
    // held.
    uint32_t drain_device_pages(RealtimeProfilerDevice& dev_state);
    // Count and surface one dispatch-stall marker (log on the first per device, Tracy zone + message each time).
    void report_dispatch_stall(RealtimeProfilerDevice& dev_state, uint64_t stall_end_timestamp, uint32_t stall_cycles);
    // Decodes the program records in the first `num_pages` pages of page_buf_ and publishes them to the ring.
    void publish_pages(RealtimeProfilerDevice& dev_state, uint32_t num_pages);

    ContextId context_id_;
    DataCollector* data_collector_ = nullptr;
    RealtimeProfilerService* service_ = nullptr;

    std::vector<RealtimeProfilerDevice> devices_;
    RealtimeProfilerRecordRing ring_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};
    // Receiver thread scratch: the pages of one read and the records decoded from them.
    std::vector<uint32_t> page_buf_;
    std::vector<experimental::ProgramRealtimeRecord> record_buf_;

    // Receiver diagnostics
    std::atomic<uint32_t> peak_fifo_pages_{0};        // all-time peak D2H FIFO usage
    uint32_t windowed_peak_fifo_pages_ = 0;           // for plotting in Tracy; gets reset each plot sample
    // The time one round of clock anchors takes, over all devices; written by the receiver thread, read after it joins.
    uint64_t anchor_count_ = 0;
    uint64_t anchor_total_ns_ = 0;
    uint64_t anchor_max_ns_ = 0;
    std::atomic<uint64_t> num_published_records_{0};  // count of records published to the ring
    std::atomic<uint64_t> num_published_batches_{0};  // count of batches published to the ring
    std::atomic<uint64_t> dispatch_stall_events_{0};  // dispatch-stall markers received
    std::atomic<uint64_t> dispatch_stall_cycles_{0};  // device cycles those stalls lasted
    std::atomic<bool> receiver_paused_{false};        // see pause_receiver_for_testing()
    // Records dropped for ending before they started.
    std::atomic<uint64_t> num_end_before_start_records_{0};
};

}  // namespace tt::tt_metal
