// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "perf_debug_profiler_tracy_handler.hpp"

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <string>
#include <utility>

#include <hostdevcommon/profiler_common.h>

#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>
#endif

namespace tt::tt_metal {

#if defined(TRACY_ENABLE)
namespace {

// Bytes per NoC word on this part. The NIU counters count WORDS, and the device reports the authoritative
// scale in out[129] (NOC_WORD_BYTES straight out of noc_parameters.h) -- which the per-run log block already
// uses. This constant exists only because noc_parameters.h is a device header and cannot be included here;
// if the two ever disagree the log block is right and this is wrong.
constexpr double kNocWordBytes = 64.0;

// Plot names must outlive the capture: the server queries the CLIENT to dereference the pointer, so a
// std::string that goes out of scope leaves the server resolving freed memory. These are interned once per
// (core, series) and deliberately never freed -- there are at most 6 drainers x 4 series + 1 aggregate.
const char* intern_plot_name(uint64_t key, const std::string& text) {
    static std::mutex mu;
    static std::unordered_map<uint64_t, const char*> names;
    std::lock_guard<std::mutex> g(mu);
    auto it = names.find(key);
    if (it != names.end()) {
        return it->second;
    }
    char* p = new char[text.size() + 1];
    std::memcpy(p, text.c_str(), text.size() + 1);
    names.emplace(key, p);
    return p;
}

}  // namespace
#endif

PerfDebugTracyHandler::PerfDebugTracyHandler() = default;

PerfDebugTracyHandler::~PerfDebugTracyHandler() {
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(TRACY_ENABLE)
    if (orphan_end_count_ > 0) {
        log_debug(
            tt::LogMetal,
            "[perf-debug profiler] dropped {} orphan ZONE_ENDs across {} lanes (capture-boundary straddles / "
            "over-run); each would SEGV tracy-capture",
            orphan_end_count_,
            orphan_lanes_.size());
    }
    for (auto& entry : tracy_contexts_) {
        TracyTTDestroy(entry.second);
    }
    tracy_contexts_.clear();
#endif
}

void PerfDebugTracyHandler::AddDevice(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] int64_t host_start,
    [[maybe_unused]] double first_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    // Record the chip anchor; per-core contexts are created (Populated with this) in PreCreateContexts
    // / GetOrCreateContext. A single per-chip context would collapse every core's RISCs into one row.
    chip_anchors_[chip_id] = ChipAnchor{host_start, first_timestamp, frequency};
#endif
}

TracyTTCtx PerfDebugTracyHandler::GetOrCreateContext(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t core_x,
    [[maybe_unused]] uint32_t core_y,
    [[maybe_unused]] const std::string& name) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t key = ContextKey(chip_id, core_x, core_y);
    if (auto it = tracy_contexts_.find(key); it != tracy_contexts_.end()) {
        return it->second;
    }
    auto ait = chip_anchors_.find(chip_id);
    if (ait == chip_anchors_.end()) {
        return nullptr;  // device was never AddDevice'd
    }
    const ChipAnchor& a = ait->second;
    ZoneScopedNC("ctx-create", 0xD35400);  // orange: creating a Tracy GPU context (GpuNewContext+Populate+name)
                                           // -- one per core, all lazily on the first batch -> startup spike
    TracyTTCtx ctx = TracyTTContext();
    // Calibrated variant: marks the context calibrated (calibrationMod=1.0, no calibration events) so the
    // Tracy GUI does NOT show a per-context "Drift (ns/s)/Auto" control under every core. Timestamps are
    // host-rebased, so the anchor mapping is exact and no drift correction is wanted.
    TracyTTContextPopulateCalibrated(ctx, a.host_start, a.first_timestamp, a.frequency);
    TracyTTContextName(ctx, name.c_str(), name.size());
    tracy_contexts_[key] = ctx;
    return ctx;
#else
    return nullptr;
#endif
}

void PerfDebugTracyHandler::PreCreateContexts(
    [[maybe_unused]] uint32_t chip_id, [[maybe_unused]] const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0) {
#if defined(TRACY_ENABLE)
    for (const auto& [cx, cy] : worker_noc0) {
        GetOrCreateContext(chip_id, cx, cy, fmt::format("Device: {} Physical ({},{})", chip_id, cx, cy));
    }
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] Device {}: pre-created {} per-core Tracy contexts (off the drain hot path)",
        chip_id,
        worker_noc0.size());
#endif
}

void PerfDebugTracyHandler::HandleWorkerZone([[maybe_unused]] const perf_debug::WorkerZonePacket& zone) {
#if defined(TRACY_ENABLE)
    // if (!tracy::GetProfiler().IsConnected()) {
    //     return;
    // }
    TracyTTCtx ctx = GetOrCreateContext(
        zone.chip_id,
        zone.core_noc0_x,
        zone.core_noc0_y,
        fmt::format("Device: {} Physical ({},{})", zone.chip_id, zone.core_noc0_x, zone.core_noc0_y));
    if (!ctx) {
        return;
    }

    static constexpr tracy::RiscType kRisc[5] = {
        tracy::RiscType::BRISC,
        tracy::RiscType::NCRISC,
        tracy::RiscType::TRISC_0,
        tracy::RiscType::TRISC_1,
        tracy::RiscType::TRISC_2};

    tracy::TTDeviceMarker marker;
    marker.chip_id = zone.chip_id;
    marker.core_x = zone.core_noc0_x;
    marker.core_y = zone.core_noc0_y;
    // per-hart lane labels; Tensix RISCs map the 0..4 index through kRisc.
    marker.risc = kRisc[zone.risc % 5];
    // MUST be set: PushStartMarker/PushEndMarker derive the zone's gpuTime from this field alone
    // (round(marker.timestamp / m_frequency)). Leaving it default (INVALID_NUM) gave every zone on every
    // core the same constant gpuTime, so all durations came out 0 or negative and the GUI nested them
    // arbitrarily -- the "wrong parent/child, inconsistent durations" symptom.
    marker.timestamp = zone.timestamp;
    marker.runtime_host_id = zone.timer_id;
    marker.marker_type = zone.is_start ? tracy::TTDeviceMarkerType::ZONE_START : tracy::TTDeviceMarkerType::ZONE_END;
    marker.marker_name = zone.name.empty() ? fmt::format("Zone_{}", zone.timer_id) : std::string(zone.name);
    marker.file = "kernel_profiler";
    marker.line = 0;
    marker.color = zone.color;

    // Mirror Tracy's per-lane GPU zone stack depth; drop an unmatched ZONE_END (would pop an empty
    // stack -> SEGV in tracy-capture). A never-opened lane's first END is a benign capture-start
    // straddle (its START predates the drain); an extra END after balanced traffic is a pairing bug.
    const uint64_t lane_key = (ContextKey(zone.chip_id, zone.core_noc0_x, zone.core_noc0_y) << 3) | (zone.risc & 0x7);
    // lane_depth_ / orphan_* are SHARED across the (multiple) socket-drain threads that call this. Guard the
    // read-modify-write: without the lock, concurrent inserts from two drain threads rehash the map and
    // corrupt an UNRELATED lane's depth, which then spuriously trips the orphan-END drop below and loses a
    // burst of real ZONE_ENDs -> a deep unclosed-zone staircase on a random single lane (rare, intermittent).
    // Release before the Tracy push so pushes stay concurrent (a lane is single-threaded, so push order is
    // preserved regardless; Tracy's serial queue is itself thread-safe).
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (zone.is_start) {
            lane_depth_[lane_key]++;
        } else {
            auto it = lane_depth_.find(lane_key);
            const int32_t depth = (it == lane_depth_.end()) ? 0 : it->second;
            if (depth <= 0) {
                ++orphan_end_count_;
                orphan_lanes_.insert(lane_key);
                return;  // orphan END -> drop (lock_guard releases on return)
            }
            --it->second;
        }
    }

    // (no per-marker zone here: it would emit one CPU zone per device marker -> millions, doubling Tracy load
    // and distorting the measurement. The steady per-marker push cost is captured by the per-batch tracy-emit
    // zone's duration; the startup spike is the ctx-create children.)
    if (zone.is_start) {
        TracyTTPushStartMarker(ctx, marker);
    } else {
        TracyTTPushEndMarker(ctx, marker);
    }
#endif
}

void PerfDebugTracyHandler::HandleWorkerEvent([[maybe_unused]] const perf_debug::WorkerEventPacket& event) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetOrCreateContext(
        event.chip_id,
        event.core_noc0_x,
        event.core_noc0_y,
        fmt::format("Device: {} Physical ({},{})", event.chip_id, event.core_noc0_x, event.core_noc0_y));
    if (!ctx) {
        return;
    }

    static constexpr tracy::RiscType kRisc[5] = {
        tracy::RiscType::BRISC,
        tracy::RiscType::NCRISC,
        tracy::RiscType::TRISC_0,
        tracy::RiscType::TRISC_1,
        tracy::RiscType::TRISC_2};

    tracy::TTDeviceMarker marker;
    marker.chip_id = event.chip_id;
    marker.core_x = event.core_noc0_x;
    marker.core_y = event.core_noc0_y;
    marker.risc = kRisc[event.risc % 5];
    marker.timestamp = event.timestamp;
    marker.runtime_host_id = event.runtime_host_id;
    // Three distinct kinds, by ID provenance and payload -- not one type with a size of zero:
    //   RUNTIME_EVENT: runtime id, no name exists (DeviceRuntimeEvent)
    //   DATA:          compile-time tag + payload (DeviceData)
    //   FLAG:          compile-time tag, no payload (DeviceFlag)
    if (event.runtime_id) {
        marker.marker_type = tracy::TTDeviceMarkerType::RUNTIME_EVENT;
    } else if (event.num_values != 0) {
        marker.marker_type = tracy::TTDeviceMarkerType::DATA;
    } else {
        marker.marker_type = tracy::TTDeviceMarkerType::FLAG;
    }
    // A runtime event has no source location and so no harvested name; show the id rather than a blank.
    marker.marker_name = event.name.empty() ? fmt::format("Event_{}", event.id) : std::string(event.name);
    marker.file = "kernel_profiler";
    marker.line = 0;

    // The first two uint64s ride the marker's dedicated fields (the Tracy tooltip prints them as
    // Data / Data high); any beyond that go into the metadata map so nothing is silently dropped.
    if (event.num_values > 0) {
        marker.data = event.values[0];
    }
    if (event.num_values > 1) {
        marker.data_high = event.values[1];
    }
#ifdef TRACY_TT_HAS_FULL_DEPS
    for (uint32_t i = 2; i < event.num_values; i++) {
        marker.meta_data[fmt::format("value{}", i)] = event.values[i];
    }
#endif

    TracyTTPushMarker(ctx, marker);

    // ---- NoC-FOOTPRINT per-sweep sample -> Tracy PLOTS, on the DEVICE timebase ------------------------
    //
    // The whole point is the timestamp. A plot stamped at decode time would land milliseconds to the RIGHT of
    // the drainer zones it explains (the mover's ring tail alone trails the last worker zone by 2.5-2.9 ms),
    // so PlotDataAt takes the instant explicitly and we convert the device tick into the TSC domain the
    // server will map. That conversion is the exact inverse of the device-ZONE mapping, which is why a sample
    // lands on the same pixel column as the DRISC-SWEEP zone it came from:
    //
    //   a zone displays at   ConvertGpuTime(tgpu) = round(ts/freq) - anchor_ns + TscTime(host_start)
    //   a plot displays at   TscTime(tsc) = (tsc - baseTime) * timerMul
    //   equate them  =>      tsc = (round(ts/freq) - anchor_ns) / timerMul + host_start
    //
    // and the server-private baseTime cancels, which is what makes this computable client-side at all.
    if (event.id == kernel_profiler::SPSC_DATA_ID_NOCFP && event.num_values >= 2) {
        ChipAnchor a{};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto ait = chip_anchors_.find(event.chip_id);
            if (ait == chip_anchors_.end() || ait->second.frequency == 0.0) {
                return;  // no anchor yet: a sample with no mapping is worse than no sample
            }
            a = ait->second;
        }
        const double timer_mul = tracy::Profiler::GetTimerMul();
        if (timer_mul == 0.0) {
            return;
        }
        const double dev_ns = static_cast<double>(event.timestamp) / a.frequency;
        const double anchor_ns = a.first_timestamp / a.frequency;
        const int64_t tsc =
            static_cast<int64_t>((dev_ns - anchor_ns) / timer_mul) + static_cast<int64_t>(a.host_start);

        // Payload order is the shared contract in profiler_common.h (SpscNocFpWord). The consumer packed each
        // pair hi-word first, so values[0] = rd_words<<32 | rd_txns and values[1] = wr_words<<32 | wr_txns.
        const uint64_t rd_words = event.values[0] >> 32, rd_txns = event.values[0] & 0xFFFFFFFFu;
        const uint64_t wr_words = event.values[1] >> 32, wr_txns = event.values[1] & 0xFFFFFFFFu;
        const double rd_kb = static_cast<double>(rd_words) * kNocWordBytes / 1024.0;
        const double wr_kb = static_cast<double>(wr_words) * kNocWordBytes / 1024.0;

        const uint64_t ck = ContextKey(event.chip_id, event.core_noc0_x, event.core_noc0_y);
        const std::string who = fmt::format("DRISC {}-{}", event.core_noc0_x, event.core_noc0_y);
        tracy::Profiler::PlotDataAt(intern_plot_name(ck * 8 + 0, who + " NoC rd KB/sweep"), rd_kb, tsc);
        tracy::Profiler::PlotDataAt(intern_plot_name(ck * 8 + 1, who + " NoC wr KB/sweep"), wr_kb, tsc);
        tracy::Profiler::PlotDataAt(
            intern_plot_name(ck * 8 + 2, who + " NoC rd txns/sweep"), static_cast<int64_t>(rd_txns), tsc);
        tracy::Profiler::PlotDataAt(
            intern_plot_name(ck * 8 + 3, who + " NoC wr txns/sweep"), static_cast<int64_t>(wr_txns), tsc);

        // DERIVED AGGREGATE: the sum over drainers of each one's most recent per-sweep bytes. Stated plainly
        // because it is not a measurement -- the six drainers sweep independently, so there is no instant at
        // which all six sampled together. It answers "roughly how much NoC is the profiler moving right now"
        // and must not be read as a synchronous total.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nocfp_last_kb_[ck] = rd_kb + wr_kb;
            double total = 0.0;
            for (const auto& [k, v] : nocfp_last_kb_) {
                total += v;
            }
            tracy::Profiler::PlotDataAt(intern_plot_name(1, "DRISC all NoC KB/sweep (sum of latest)"), total, tsc);
        }
    }
#endif
}

}  // namespace tt::tt_metal
