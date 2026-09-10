// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tracy/TracyTTDevice.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

class Service;

// The built-in Tracy sink: registers for every record type and pushes each record onto Tracy's device timeline, one
// context per (chip, core). Constructing it registers; destroying it unregisters. Everything after construction
// runs on the callback's thread.
class TracySink {
public:
    explicit TracySink(Service& service);
    ~TracySink();
    TracySink(const TracySink&) = delete;
    TracySink& operator=(const TracySink&) = delete;

private:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::RecordType::All>;
    using Core = experimental::streaming_profiler::Core;
    struct Lane {
        TracyTTCtx ctx = nullptr;
        uint32_t thread = 0;
        uint32_t risc = 0;
    };
    // Each RISC's timeline row is created and named on first use.
    struct CoreEntry {
        TracyTTCtx ctx = nullptr;
        std::array<uint32_t, 5> thread{};
        uint8_t named = 0;
    };
    struct SrclocEntry {
        const char* name = nullptr;
        uint64_t key = 0;
        const void* srcloc = nullptr;
    };

    void on_batch(const Batch& batch, uint64_t capture);
    // Records are placed by their steady_clock time through a continuous piecewise-linear map onto Tracy's timeline:
    // a fresh segment per capture, then one per second whose slope is the two clocks' rate ratio measured over the
    // whole baseline since construction. Continuity keeps order and containment exact across segments.
    struct Probe {
        int64_t steady_ns;
        int64_t tracy_ns;
    };
    struct Segment {
        int64_t steady_ns;  // from here on
        int64_t tracy_ns;   // the map's value here
        double slope;       // timeline ns per steady_clock ns
    };
    Probe probe() const;
    double slope_since_base(const Probe& p) const;
    void start_capture_map();
    void refine_map();
    int64_t to_timeline(int64_t steady_ns) const;
    Lane lane(const Core& core);
    const void* srcloc(std::string_view name, uint32_t color, uint32_t risc);
    const void* srcloc_slow(std::string_view name, uint32_t color, uint32_t risc);
    void push_zone(const Core& core, std::string_view name, int64_t start_ns, int64_t end_ns, uint32_t color);
    void push_marker(
        const Core& core,
        std::string_view name,
        int64_t timestamp_ns,
        uint32_t runtime_id,
        std::span<const uint64_t> values);
    // Device<->device sync plots: at each PP_CLOCK sample, plot the correction (ns) at the sample's device time
    // on a per-chip line. LOCAL samples (3 us) feed the local-only plot, LINK samples (1 ms) the linked plot; the
    // gap between the two curves on a non-root chip is the cross-chip error the linked sync removes.
    void plot_clock(uint32_t dev, uint32_t kind, uint64_t device_ticks);
    const char* plot_name(uint32_t chip, bool linked);
    // Plots are emitted at capture end, not during decode: at decode time the correction is not yet solved
    // (lookup returns 0) and the timeline map has no segments (points land at raw, hours-off timestamps).
    void emit_plots();

    Service& service_;
    ConsumerHandle handle_ = 0;
    uint64_t capture_ = 0;      // the capture the map holds for; a new one starts a new map
    int64_t anchor_tracy_ = 0;  // Tracy timer at construction; every context's cpuTime
    Probe base_{};              // taken at construction; every slope is measured against it
    std::vector<Segment> segments_;
    int64_t next_refine_ns_ = 0;
    uint64_t lane_key_ = ~uint64_t{0};
    Lane lane_hit_;
    std::unordered_map<uint64_t, CoreEntry> cores_;
    // Keyed by the name's address: a callback's name strings never move or die while it lives.
    std::vector<SrclocEntry> srcloc_table_;  // open addressing, power-of-two size, at most half full
    size_t srcloc_count_ = 0;
    std::unordered_map<std::string, const void*> srclocs_;
    std::vector<DeviceClock> clocks_;  // per device index, for mapping a clock sample's device time to the timeline
    std::vector<DeviceClock> eth_clocks_;  // per device index, the idle-eth wall anchor for PP_CLOCK samples
    std::unordered_map<uint32_t, std::string> plot_local_, plot_linked_;  // persistent per-chip plot names for Tracy
    struct PlotSample {
        uint32_t dev;
        uint32_t kind;
        uint64_t ts;
    };
    std::vector<PlotSample> plot_samples_;  // accumulated during the capture, drained in emit_plots()
};

}  // namespace tt::tt_metal::streaming_profiler
