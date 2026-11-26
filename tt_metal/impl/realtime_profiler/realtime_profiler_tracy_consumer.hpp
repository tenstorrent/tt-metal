// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
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
// made on that consumer's thread, so each device's context and its zones go through one lock-free queue.
//
// Zones are placed in host time, and they run one to two anchor periods behind the device. A device's clock rate
// changes with load, so a zone is not placed with a rate guessed from the past. It is held until anchors exist on
// both sides of it, then placed on a monotone cubic through the anchors, which follows the changing rate smoothly.
// A segment's curve is fixed once the anchor after its far end has arrived, so a zone that has been drawn is never
// moved. The curve is continuous and increasing, so consecutive zones can neither overlap nor leave a gap that the
// anchors alone caused. Other consumers get the raw timestamps.
class RealtimeProfilerTracyConsumer {
public:
    RealtimeProfilerTracyConsumer() = default;

    RealtimeProfilerTracyConsumer(const RealtimeProfilerTracyConsumer&) = delete;
    RealtimeProfilerTracyConsumer& operator=(const RealtimeProfilerTracyConsumer&) = delete;

    // Consumer thread: a batch of records. False once Tracy will take no more, which is when its connect window
    // has passed with no client.
    bool on_records(const experimental::ProgramRealtimeRecordBatch& batch);

    // Consumer thread, when it has no records: applies queued device events and emits the zones that new anchors
    // have made placeable.
    void on_idle();

    // Blocks until the consumer thread has applied every event queued so far, or the timeout passes. The service must
    // wake its consumers first.
    void wait_for_events(std::chrono::milliseconds timeout);

    // Device events. Callable from any thread; each one is applied on the consumer thread before its next batch.
    void add_device(uint32_t chip_id, int64_t anchor_host, uint64_t anchor_device, double frequency);
    void remove_device(uint32_t chip_id);
    // Adds an anchor to the device's clock mapping. Zones are placed once anchors bracket them.
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
    // A point on a device's clock mapping: device wall clock cycles and Tracy host ticks, each relative to the
    // chip's first anchor. `slope` is the curve's slope there in host ticks per cycle, set once the anchor is final.
    struct Anchor {
        double wall = 0.0;
        double host = 0.0;
        double slope = 0.0;
    };
    struct PendingStall {
        uint64_t device_timestamp = 0;
        uint32_t stall_cycles = 0;
    };
    struct Chip {
        TracyTTCtx ctx = nullptr;
        uint32_t programs_lane = 0;
        uint32_t stall_lane = 0;
        // The context counts ns from context_host.
        int64_t context_host = 0;
        // The first anchor, which the others are relative to.
        int64_t base_host = 0;
        uint64_t base_wall = 0;
        // The newest anchors as measured, oldest first. One becomes final when the anchor after it arrives.
        std::vector<Anchor> raw;
        // The final anchors, oldest first, newest last. Zones are placed on the curve through them.
        std::vector<Anchor> smooth;
        // Host ticks per device cycle from the nominal frequency, for zones before any anchor is final.
        double host_per_wall = 0.0;
        // Records and stalls waiting for a final anchor at or after their end.
        std::deque<experimental::ProgramRealtimeRecord> pending_programs;
        std::deque<PendingStall> pending_stalls;
        // Where the last zone on each lane ended, in ns on the context. A zone never starts before it.
        uint64_t programs_end = 0;
        uint64_t stall_end = 0;
        // Dispatch-stall messages are limited to one per kStallMessageInterval; the rest are counted here.
        std::chrono::steady_clock::time_point last_stall_message;
        uint64_t suppressed_stalls = 0;
        uint64_t suppressed_stall_cycles = 0;
        // The device clock's frequency in cycles per ns, for stall durations.
        double frequency = 0.0;
    };
    void enqueue(const Event& event);
    void apply_events();
    void apply_add(const Event& event);
    void apply_remove(const Event& event);
    void add_anchor(Chip& chip, int64_t host_time, uint64_t device_timestamp);
    // Makes raw anchor `index` final: fixes the curve's slope at it from its neighbours.
    void finalize_anchor(Chip& chip, size_t index);
    // Emits the chip's pending zones that smoothed anchors bracket, or all of them when `all` is set.
    void emit_pending(Chip& chip, bool all);
    // Drops the oldest final anchors that no waiting zone can still need.
    void trim_anchors(Chip& chip);
    void emit_program(Chip& chip, const experimental::ProgramRealtimeRecord& record);
    void emit_stall(Chip& chip, const PendingStall& stall);
    void flush_stall_message(uint32_t chip_id, Chip& chip);
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
    // How many events have been queued, and how many of those the consumer thread has applied.
    uint64_t events_queued_ = 0;
    std::atomic<uint64_t> events_applied_{0};

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
