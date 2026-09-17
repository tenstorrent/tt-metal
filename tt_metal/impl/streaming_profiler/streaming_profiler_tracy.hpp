// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tracy/TracyTTDevice.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

class Service;

// The built-in Tracy sink: registers for every record type and pushes each record onto Tracy's device timeline, one
// context per (chip, core). Constructing it registers; destroying it unregisters. Everything after construction
// runs on the callback's thread.
//
// Record host times are host_clock, the TSC scaled: a record's timeline position is its TSC count through Tracy's
// calibrated multiplier.
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

    void on_batch(const Batch& batch);
    // TT_METAL_STREAMING_PROFILER_TRACY_PLOTS_ONLY: the trace carries the clock and d2d sync plots and no records,
    // so an hours-long stress capture stays a few MB.
    const bool plots_only_ = std::getenv("TT_METAL_STREAMING_PROFILER_TRACY_PLOTS_ONLY") != nullptr;
    void emit_zone(const experimental::streaming_profiler::Zone& z);
    void emit_data(const experimental::streaming_profiler::TimestampedData& d);
    void emit_event(const experimental::streaming_profiler::Event& e);
    // A TSC tick as a GPU-context timestamp: nanoseconds from the contexts' origin.
    int64_t timeline_ns(int64_t tsc) const;
    Lane lane(const Core& core);
    const void* srcloc(std::string_view name, uint32_t color, uint32_t risc);
    const void* srcloc_slow(std::string_view name, uint32_t color, uint32_t risc);
    void push_zone(const Core& core, std::string_view name, int64_t start_tsc, int64_t end_tsc, uint32_t color);
    void push_marker(
        const Core& core, std::string_view name, int64_t tsc, uint32_t runtime_id, std::span<const uint64_t> values);
    struct FreqPoint {
        int64_t tsc;
        double ghz;
    };
    std::vector<FreqPoint> compute_frequency(size_t begin, size_t end) const;
    const char* intern_name(const std::string& name);
    // Plots are emitted at capture end, not during decode: at decode time the placement is not yet solved (a lookup
    // returns 0) and the points would land at the origin.
    void emit_plots();
    void plot_point(const char* name, double value, int64_t tsc);

    Service& service_;
    ConsumerHandle handle_ = 0;
    // Read only from the Tracy-enabled paths below, so it is unused in a build without Tracy.
    [[maybe_unused]] int64_t anchor_tracy_ = 0;  // Tracy timer at construction; every context's cpuTime
    // The GPU contexts' origin sits this far before anchor_tracy_, so a record from device bring-up keeps its real
    // place instead of falling off the front.
    int64_t origin_margin_ns_ = 0;
    // Records the origin still could not hold: clamped to the origin and counted; nonzero means a defect upstream,
    // never expected in a healthy capture.
    uint64_t clamped_zones_ = 0, clamped_markers_ = 0, clamped_plot_points_ = 0;
    uint64_t lane_key_ = ~uint64_t{0};
    Lane lane_hit_;
    std::unordered_map<uint64_t, CoreEntry> cores_;
    // Keyed by the name's address: a callback's name strings never move or die while it lives.
    std::vector<SrclocEntry> srcloc_table_;     // open addressing, power-of-two size, at most half full
    [[maybe_unused]] size_t srcloc_count_ = 0;  // ditto: only the Tracy-enabled srcloc path touches it
    std::unordered_map<std::string, const void*> srclocs_;
    std::vector<uint32_t> chips_;  // per device index, the chip id the placement is keyed by
    uint32_t root_dev_ = 0;
    struct PlotSample {
        uint32_t dev;
        uint32_t kind;
        uint32_t core;  // eth core index on the device: one refclk counter per stream
        uint64_t ts;    // the eth tile's wall tick
        uint64_t value;  // the refclk reading
    };
    std::vector<PlotSample> plot_samples_;
    std::unordered_set<std::string> plot_names_;
};

}  // namespace tt::tt_metal::streaming_profiler
