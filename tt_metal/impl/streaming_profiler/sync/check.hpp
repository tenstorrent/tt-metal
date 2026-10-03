// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "impl/streaming_profiler/sync/engine.hpp"

namespace tt::tt_metal::streaming_profiler {

// Measures the synced timeline against an independent reference: the ruler's refclk readings on each chip, placed
// through the clock map, against the held-out link rounds solved onto the root. Logs the chip-to-chip error and each
// chip's AICLK over the capture at finish().
class SyncCheck {
public:
    struct Reading {
        double refclk;
        int64_t wall_eighths;
        float weight = 1;
    };
    struct Round {
        size_t link;
        RoundPoint point;
    };
    struct ChipBatch {
        std::vector<Reading> readings;
        std::optional<double> offset;
        std::vector<PlotPoint> aiclk;  // GHz, one point per instant on a line
    };
    struct Batch {
        explicit Batch(size_t devices) : chips(devices) {}
        std::vector<Round> rounds;
        std::vector<ChipBatch> chips;
    };
    struct PlotSeries {
        std::string name;
        std::vector<PlotPoint> points;
    };

    SyncCheck(const CaptureContext& ctx, const ClockMap& map);
    SyncCheck(const SyncCheck&) = delete;
    SyncCheck& operator=(const SyncCheck&) = delete;

    // The batch the caller fills until submit().
    Batch& batch() { return staged_; }
    // Hands the batch to the check's thread and leaves it empty.
    void submit();
    // Stops the check's thread and logs the report.
    void finish();
    // The worst error per millisecond, pooled and per chip, as Tracy plot series.
    std::vector<PlotSeries> plots() const;

private:
    // AICLK weighted by how long each value held.
    struct ClockStats {
        static constexpr double kBinMhz = 50.0;
        std::optional<PlotPoint> last;
        double span = 0.0, sum = 0.0, sum_squares = 0.0;
        double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
        uint64_t changes = 0;
        std::map<int, double> by_bin;
        void add(double root, double mhz);
    };
    // A step's line needs its whole span of rounds, except once the capture is finishing.
    enum class Steps { Complete, All };
    struct LinkRef {
        std::deque<RoundPoint> rounds;
        std::map<int64_t, LineFit> lines_by_step;
        std::optional<int64_t> next_step;
        void fit_steps(Steps steps);
    };
    struct Sample {
        double root;
        float error_ns;
        float weight;
    };
    struct WorstByMs {
        int64_t first = 0;
        std::deque<float> ns;
        void add(double root, double error_ns);
    };
    struct Chip {
        std::deque<Reading> waiting;
        std::optional<double> last_refclk;
        std::deque<Sample> placed;
        uint64_t popped = 0;
        uint64_t no_reference = 0, no_node = 0;
        std::optional<double> offset;
        ErrorStats error;
        WorstByMs worst;
        ClockStats clock;
        const Sample& at(uint64_t index) const { return placed[index - popped]; }
        uint64_t end() const { return popped + placed.size(); }
    };
    // Cursors into the two chips' samples: chip a's next sample to pair, and chip b's sample where the search for its
    // nearest starts.
    struct Pair {
        uint32_t chip_a = 0, chip_b = 0;
        uint64_t next_a = 0, last_b = 0;
        ErrorStats error;
    };
    enum class Ref { Ready, Wait, None };
    // Each chip's refclk onto the root's at one step.
    using StepTransforms = std::vector<std::optional<RootTransform>>;

    int64_t sender_step(size_t link_index, double root) const;
    double root_step(uint32_t dev, double refclk) const;
    std::pair<Ref, const StepTransforms*> transforms_at(int64_t step);
    // The root refclk of a reading at `refclk` on `dev`, from the link reference.
    std::pair<Ref, double> reference(uint32_t dev, double refclk);
    bool place(uint32_t dev, ClockMap::Reader& reader);
    bool pair_up(double until);
    void prune();
    void run(std::stop_token stop);
    void report() const;

    const CaptureContext& ctx_;
    const ClockMap& map_;
    std::vector<LinkRef> links_;
    std::vector<Chip> chips_;
    std::vector<Pair> pairs_;
    std::map<int64_t, StepTransforms> transforms_by_step_;
    ErrorHistogram<16, 256> pooled_;
    WorstByMs worst_;
    bool finishing_seen_ = false;
    Batch staged_;
    std::mutex mu_;
    Batch submitted_;
    std::jthread worker_;
};

}  // namespace tt::tt_metal::streaming_profiler
