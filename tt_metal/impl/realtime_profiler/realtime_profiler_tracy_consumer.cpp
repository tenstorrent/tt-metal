// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "realtime_profiler_tracy_consumer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>

#if defined(TRACY_ENABLE)
#include <client/TracyProfiler.hpp>
#include <tracy/Tracy.hpp>
#endif

namespace tt::tt_metal {

void RealtimeProfilerTracyConsumer::add_device(
    uint32_t chip_id, int64_t anchor_host, uint64_t anchor_device, double frequency) {
    enqueue(
        {.kind = Event::Kind::Add,
         .chip_id = chip_id,
         .host_time = anchor_host,
         .device_timestamp = anchor_device,
         .frequency = frequency});
}

void RealtimeProfilerTracyConsumer::remove_device(uint32_t chip_id) {
    enqueue({.kind = Event::Kind::Remove, .chip_id = chip_id});
}

void RealtimeProfilerTracyConsumer::calibrate_device(
    uint32_t chip_id, int64_t host_time, uint64_t device_timestamp, double frequency) {
    enqueue(
        {.kind = Event::Kind::Calibrate,
         .chip_id = chip_id,
         .host_time = host_time,
         .device_timestamp = device_timestamp,
         .frequency = frequency});
}

void RealtimeProfilerTracyConsumer::push_dispatch_stall_marker(
    uint32_t chip_id, uint64_t stall_end_timestamp, uint32_t stall_cycles) {
    enqueue(
        {.kind = Event::Kind::Stall,
         .chip_id = chip_id,
         .device_timestamp = stall_end_timestamp,
         .stall_cycles = stall_cycles});
}

#if defined(TRACY_ENABLE)

namespace {

// Synthetic core for the real-time profiler's Tracy lanes: programs go on BRISC, dispatch stalls on NONE.
constexpr uint32_t kRealtimeProfilerCore_X = 100;
constexpr uint32_t kRealtimeProfilerCore_Y = 100;
constexpr auto kStallMessageInterval = std::chrono::milliseconds(100);

uint32_t lane_thread_id(uint32_t chip_id, tracy::RiscType risc) {
    tracy::TTDeviceMarker lane{};
    lane.chip_id = chip_id;
    lane.core_x = kRealtimeProfilerCore_X;
    lane.core_y = kRealtimeProfilerCore_Y;
    lane.risc = risc;
    return lane.get_thread_id();
}

// The name Kernel::name() gives the kernel: its source file's stem. An inline-source kernel's entry is its source
// text, not a path.
std::string kernel_name(std::string_view source) {
    if (source.find('\n') != std::string_view::npos) {
        return "Kernel_Source_Code";
    }
    return std::filesystem::path(source).stem().string();
}

constexpr tracy::SourceLocationData kStallSrcloc{
    "DISPATCH STALL: RT profiler record ring full", "realtime_profiler", "realtime_profiler", 0, tracy::Color::Red};

}  // namespace

void RealtimeProfilerTracyConsumer::enqueue(const Event& event) {
    // Past its connect window Tracy takes nothing more. Before a client connects it keeps whatever it is given, so a
    // stall is not queued until then.
    if (tracy::GetProfiler().IsEmitSuppressed() ||
        (event.kind == Event::Kind::Stall && !tracy::GetProfiler().IsConnected())) {
        return;
    }
    std::lock_guard lock(events_mutex_);
    events_.push_back(event);
    events_queued_++;
    events_pending_.store(true, std::memory_order_release);
}

bool RealtimeProfilerTracyConsumer::on_records(const experimental::ProgramRealtimeRecordBatch& batch) {
    if (tracy::GetProfiler().IsEmitSuppressed()) {
        return false;
    }
    if (events_pending_.load(std::memory_order_acquire)) {
        apply_events();
    }
    if (!tracy::GetProfiler().IsConnected()) {
        return true;
    }
    for (const auto& record : batch.records) {
        Chip& chip = chips_[record.chip_id];
        if (chip.ctx != nullptr) {
            chip.pending_programs.push_back(record);
        }
    }
    for (Chip& chip : chips_) {
        emit_pending(chip, false);
    }
    return true;
}

void RealtimeProfilerTracyConsumer::on_idle() {
    if (tracy::GetProfiler().IsEmitSuppressed()) {
        return;
    }
    if (events_pending_.load(std::memory_order_acquire)) {
        apply_events();
    }
    if (!tracy::GetProfiler().IsConnected()) {
        return;
    }
    for (Chip& chip : chips_) {
        emit_pending(chip, false);
    }
}

void RealtimeProfilerTracyConsumer::apply_events() {
    std::vector<Event> events;
    uint64_t queued = 0;
    {
        std::lock_guard lock(events_mutex_);
        events.swap(events_);
        queued = events_queued_;
        events_pending_.store(false, std::memory_order_release);
    }
    for (const Event& event : events) {
        switch (event.kind) {
            case Event::Kind::Add: apply_add(event); break;
            case Event::Kind::Remove: apply_remove(event); break;
            case Event::Kind::Calibrate: {
                Chip& chip = chips_[event.chip_id];
                if (chip.ctx != nullptr) {
                    chip.clock.add_anchor(event.host_time, event.device_timestamp);
                }
                break;
            }
            case Event::Kind::Stall: {
                Chip& chip = chips_[event.chip_id];
                if (chip.ctx != nullptr) {
                    chip.pending_stalls.push_back({event.device_timestamp, event.stall_cycles});
                }
                break;
            }
        }
    }
    events_applied_.store(queued, std::memory_order_release);
}

void RealtimeProfilerTracyConsumer::wait_for_events(std::chrono::milliseconds timeout) {
    uint64_t queued = 0;
    {
        std::lock_guard lock(events_mutex_);
        queued = events_queued_;
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (events_applied_.load(std::memory_order_acquire) < queued && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RealtimeProfilerTracyConsumer::apply_add(const Event& event) {
    if (event.chip_id >= chips_.size()) {
        chips_.resize(event.chip_id + 1);
    }
    Chip& chip = chips_[event.chip_id];
    ns_per_host_tick_ = TracyGetTimerMul();
    chip = Chip{};
    chip.frequency = event.frequency;
    // Host ticks per device cycle, from the nominal frequency until anchors measure it.
    chip.clock.start(event.host_time, event.device_timestamp, 1.0 / (event.frequency * ns_per_host_tick_));
    chip.context_host = event.host_time;
    chip.ctx = TracyTTContext();
    // Time 0 on the context is host time context_host, and the context counts in ns.
    TracyTTContextPopulateCalibratedLockfree(chip.ctx, event.host_time, 0.0, 1.0);
    const std::string name = fmt::format("Device {}:", event.chip_id);
    TracyTTContextNameLockfree(chip.ctx, name.c_str(), name.size());

    chip.programs_lane = lane_thread_id(event.chip_id, tracy::RiscType::BRISC);
    chip.stall_lane = lane_thread_id(event.chip_id, tracy::RiscType::NONE);
    // The GUI labels a lane with the thread name registered for its id.
    tracy::SetThreadName(chip.programs_lane, "Programs");
    tracy::SetThreadName(chip.stall_lane, "Dispatch stall");
}

void RealtimeProfilerTracyConsumer::apply_remove(const Event& event) {
    Chip& chip = chips_[event.chip_id];
    if (chip.ctx == nullptr) {
        return;
    }
    // No later anchor is coming, so finalize the last one and place what is left.
    chip.clock.finish();
    emit_pending(chip, true);
    flush_stall_message(event.chip_id, chip);
    TracyTTDestroy(chip.ctx);
    chip = Chip{};
}

void RealtimeProfilerTracyConsumer::emit_stall(Chip& chip, const PendingStall& stall) {
    const uint64_t stall_end = context_ns(chip, stall.device_timestamp);
    const uint64_t stall_ns = std::llround(static_cast<double>(stall.stall_cycles) / chip.frequency);
    const uint64_t start = std::max(stall_end - std::min(stall_ns, stall_end), chip.stall_end);
    chip.stall_end = std::max(stall_end, start);
    TracyTTPushZone(chip.ctx, &kStallSrcloc, chip.stall_lane, start, chip.stall_end);

    chip.suppressed_stalls++;
    chip.suppressed_stall_cycles += stall.stall_cycles;
    const auto now = std::chrono::steady_clock::now();
    if (now - chip.last_stall_message >= kStallMessageInterval) {
        flush_stall_message(static_cast<uint32_t>(&chip - chips_.data()), chip);
        chip.last_stall_message = now;
    }
}

void RealtimeProfilerTracyConsumer::flush_stall_message(uint32_t chip_id, Chip& chip) {
    if (chip.suppressed_stalls == 0) {
        return;
    }
    const std::string text = fmt::format(
        "Device {}: dispatch stalled {} time(s), {:.2f} us, waiting for the real-time profiler (record ring full); "
        "profiler records are delayed, not lost",
        chip_id,
        chip.suppressed_stalls,
        static_cast<double>(chip.suppressed_stall_cycles) / (chip.frequency * 1000.0));
    TracyLogString(tracy::MessageSeverity::Error, 0xFF0000, TRACY_CALLSTACK, text.size(), text.c_str());
    chip.suppressed_stalls = 0;
    chip.suppressed_stall_cycles = 0;
}

void RealtimeProfilerTracyConsumer::emit_program(Chip& chip, const experimental::ProgramRealtimeRecord& record) {
    const uint64_t start = std::max(context_ns(chip, record.start_timestamp), chip.programs_end);
    chip.programs_end = std::max(context_ns(chip, record.end_timestamp), start);
    TracyTTPushZone(chip.ctx, program_srcloc(record), chip.programs_lane, start, chip.programs_end);
    char runtime_id[32];
    const auto text = fmt::format_to_n(runtime_id, sizeof(runtime_id), "Runtime ID: {}", record.runtime_id);
    TracyTTPushZoneText(chip.ctx, chip.programs_lane, runtime_id, text.size);
}

void RealtimeProfilerTracyConsumer::emit_pending(Chip& chip, bool all) {
    if (chip.ctx == nullptr) {
        return;
    }
    // A zone is placeable once a final anchor sits at or after its end, and an earlier one starts the segment.
    const auto placeable = [&](uint64_t end_timestamp) { return all || chip.clock.covers(end_timestamp); };
    while (!chip.pending_programs.empty() && placeable(chip.pending_programs.front().end_timestamp)) {
        emit_program(chip, chip.pending_programs.front());
        chip.pending_programs.pop_front();
    }
    while (!chip.pending_stalls.empty() && placeable(chip.pending_stalls.front().device_timestamp)) {
        emit_stall(chip, chip.pending_stalls.front());
        chip.pending_stalls.pop_front();
    }
    trim_anchors(chip);
}

void RealtimeProfilerTracyConsumer::trim_anchors(Chip& chip) {
    // Zones of a chip come in order, so the oldest one still waiting is the only one that decides which anchors stay.
    std::optional<uint64_t> oldest_waiting;
    if (!chip.pending_programs.empty()) {
        oldest_waiting = chip.pending_programs.front().start_timestamp;
    }
    if (!chip.pending_stalls.empty()) {
        const uint64_t stall = chip.pending_stalls.front().device_timestamp;
        oldest_waiting = oldest_waiting ? std::min(*oldest_waiting, stall) : stall;
    }
    chip.clock.trim(oldest_waiting);
}

const tracy::SourceLocationData* RealtimeProfilerTracyConsumer::program_srcloc(
    const experimental::ProgramRealtimeRecord& record) {
    const auto [it, fresh] = programs_.try_emplace(record.kernel_sources.data());
    Program& program = it->second;
    if (fresh) {
        // The zone is named after the program's first kernel, with a count of its other kernels. A program can build
        // several kernels from one file; that name counts once.
        std::vector<std::string> names;
        for (const std::string_view source : record.kernel_sources) {
            if (!program.sources.empty()) {
                program.sources += ",\n";
            }
            program.sources += source;
            std::string name = kernel_name(source);
            if (std::ranges::find(names, name) == names.end()) {
                names.push_back(std::move(name));
            }
        }
        if (program.sources.empty()) {
            program.sources = "realtime_profiler";
        }
        program.name = names.empty() ? "Program" : names.front();
        if (names.size() > 1) {
            program.name += fmt::format(" +{}", names.size() - 1);
        }
        program.srcloc = {program.name.c_str(), "realtime_profiler", program.sources.c_str(), 0, 0};
    }
    return &program.srcloc;
}

uint64_t RealtimeProfilerTracyConsumer::context_ns(const Chip& chip, uint64_t device_timestamp) const {
    const double host = chip.clock.host_ticks(device_timestamp);
    const int64_t ns =
        std::llround((static_cast<double>(chip.clock.base_host() - chip.context_host) + host) * ns_per_host_tick_);
    return static_cast<uint64_t>(std::max<int64_t>(ns, 0));
}

#else

void RealtimeProfilerTracyConsumer::enqueue(const Event&) {}
bool RealtimeProfilerTracyConsumer::on_records(const experimental::ProgramRealtimeRecordBatch&) { return false; }
void RealtimeProfilerTracyConsumer::on_idle() {}
void RealtimeProfilerTracyConsumer::wait_for_events(std::chrono::milliseconds) {}

#endif

}  // namespace tt::tt_metal
