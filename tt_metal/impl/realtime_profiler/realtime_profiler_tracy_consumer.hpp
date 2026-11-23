// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <tracy/TracyTTDevice.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#endif

namespace tt::tt_metal {

// Sends real-time profiler records to Tracy while a Tracy client is connected. The process-wide service registers one
// of these as a consumer, and unregisters it once Tracy's connect window has passed with no client. Every Tracy call is
// made on that consumer's thread, so each device's context and its zones go through one lock-free queue. Zones are
// placed in host time with the device's current clock mapping, so a context is never calibrated after it is created.
class RealtimeProfilerTracyConsumer {
public:
    RealtimeProfilerTracyConsumer() = default;

    RealtimeProfilerTracyConsumer(const RealtimeProfilerTracyConsumer&) = delete;
    RealtimeProfilerTracyConsumer& operator=(const RealtimeProfilerTracyConsumer&) = delete;

    // Consumer thread: a batch of records. False once Tracy will take no more, which is when its connect window
    // has passed with no client.
    bool on_records(const experimental::ProgramRealtimeRecordBatch& batch);

    // Device events. Callable from any thread; each one is applied on the consumer thread before its next batch.
    void add_device(uint32_t chip_id, int64_t anchor_host, uint64_t anchor_device, double frequency);
    void remove_device(uint32_t chip_id);
    // Replaces the device's clock mapping; later zones are placed with it.
    void calibrate_device(uint32_t chip_id, int64_t host_time, uint64_t device_timestamp, double frequency);
    // A zone on the device's "Dispatch stall" lane, plus a rate-limited error message.
    void push_dispatch_stall_marker(uint32_t chip_id, uint64_t stall_end_timestamp, uint32_t stall_cycles);

private:
    struct Event {
        enum class Kind : uint8_t { Add, Remove, Calibrate, Stall };
        Kind kind;
        uint32_t chip_id = 0;
        int64_t host_time = 0;
        uint64_t device_timestamp = 0;
        uint32_t stall_cycles = 0;
        double frequency = 0.0;
    };
    struct Chip {
        TracyTTCtx ctx = nullptr;
        uint32_t programs_lane = 0;
        uint32_t stall_lane = 0;
        // The context counts ns from context_host. Device time anchor_device is host time anchor_host, and the device
        // clock runs at `frequency` cycles per ns.
        int64_t context_host = 0;
        int64_t anchor_host = 0;
        uint64_t anchor_device = 0;
        double frequency = 0.0;
        // Where the last zone on each lane ended, in ns on the context. A new mapping never moves a zone before it.
        uint64_t programs_end = 0;
        uint64_t stall_end = 0;
        // Dispatch-stall messages are limited to one per kStallMessageInterval; the rest are counted here.
        std::chrono::steady_clock::time_point last_stall_message;
        uint64_t suppressed_stalls = 0;
        uint64_t suppressed_stall_cycles = 0;
    };
    void enqueue(const Event& event);
    void apply_events();
    void apply_add(const Event& event);
    void apply_remove(const Event& event);
    void apply_stall(Chip& chip, const Event& event);
    void flush_stall_message(uint32_t chip_id, Chip& chip);
    void handle_record(const experimental::ProgramRealtimeRecord& record);
#if defined(TRACY_ENABLE)
    // The source location that names a program's zones: its kernel names, with the kernel paths as the file.
    const tracy::SourceLocationData* program_srcloc(const experimental::ProgramRealtimeRecord& record);
#endif
    // The device timestamp in ns on the chip's context; 0 if it is before the context's start.
    uint64_t context_ns(const Chip& chip, uint64_t device_timestamp) const;

    // Written by any thread, drained by the consumer thread. events_pending_ keeps the consumer thread off the lock
    // while nothing is queued.
    std::mutex events_mutex_;
    std::vector<Event> events_;
    std::atomic<bool> events_pending_{false};

    // Everything below is touched only by the consumer thread.
    std::vector<Chip> chips_;  // indexed by chip id
    double ns_per_host_tick_ = 0.0;
#if defined(TRACY_ENABLE)
    struct Program {
        std::string name;
        std::string sources;
        tracy::SourceLocationData srcloc;
    };
    // Tracy keeps a zone's source location by address, so one is kept for good per distinct list of kernel sources.
    // Keyed by a record's kernel_sources storage, which the data collector keeps one of per list; the map's nodes do
    // not move.
    std::unordered_map<const std::string_view*, Program> programs_;
#endif
};

}  // namespace tt::tt_metal
