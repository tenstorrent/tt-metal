// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/least_squares.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

static_assert(1'000'000'000 % kernel_profiler::kEthRefclkHz == 0);
inline constexpr int64_t kNsPerRefclk = 1'000'000'000 / kernel_profiler::kEthRefclkHz;
inline constexpr double kRefclkTicksPerMs = kernel_profiler::kEthRefclkHz / 1e3;

// The tracker reports wall clocks, and wall ticks per refclk tick, in eighths of a wall tick.
inline constexpr int kWallEighthBits = 3;
inline constexpr int64_t kWallEighths = int64_t{1} << kWallEighthBits;
constexpr int64_t whole_ticks(int64_t eighths) { return eighths >> kWallEighthBits; }
constexpr int64_t nearest_tick(int64_t eighths) { return (eighths + kWallEighths / 2) >> kWallEighthBits; }
constexpr double tick_fraction(int64_t eighths) {
    return static_cast<double>(eighths & (kWallEighths - 1)) / kWallEighths;
}
constexpr double eighths_as_ticks(int64_t eighths) { return static_cast<double>(eighths) / kWallEighths; }

// One link round in the sender's refclk ticks: the midpoint of its stamps, and the receiver's offset from it there.
struct RoundPoint {
    double mid = 0.0, offset = 0.0;
};

// A chip's refclk onto the root's.
struct RootTransform {
    double scale = 1.0, shift = 0.0;
    double operator()(double refclk) const { return scale * refclk + shift; }
};

// A point of a Tracy plot, at a root refclk tick.
struct PlotPoint {
    double root = 0.0, value = 0.0;
};

// Weighted errors in ns, with the worst one's magnitude and root refclk tick.
struct ErrorStats {
    double weight_sum = 0.0, sum = 0.0, worst = 0.0;
    double worst_root = 0.0;
    void add(double ns, double root, double weight);
    // Requires weight_sum > 0.
    double mean() const { return sum / weight_sum; }
};

// Weighted errors in ns, in bins of 1 / BinsPerNs ns over [-RangeNs, RangeNs) ns, with the worst one's magnitude.
template <int BinsPerNs, int RangeNs>
struct ErrorHistogram {
    static constexpr int kCentre = RangeNs * BinsPerNs;
    double weight_sum = 0.0, worst = 0.0;
    std::array<double, 2 * kCentre> bins{};
    void add(double ns, double weight) {
        const int bin = static_cast<int>(std::floor(ns * BinsPerNs)) + kCentre;
        if (bin >= 0 && bin < 2 * kCentre) {
            bins[bin] += weight;
        }
        weight_sum += weight;
        worst = std::max(worst, std::abs(ns));
    }
    // The centre of the bin that holds the |error| quantile, or the worst error if that is out of range. Requires
    // weight_sum > 0.
    double abs_quantile(double quantile) const {
        double covered = 0.0;
        for (int i = 0; i < kCentre; i++) {
            covered += bins[kCentre + i] + bins[kCentre - 1 - i];
            if (covered >= quantile * weight_sum) {
                return (i + 0.5) / BinsPerNs;
            }
        }
        return worst;
    }
};

// Chains each link's line (null for a link with none) onto the root chip; a chip no line reaches has none.
std::vector<std::optional<RootTransform>> compose_on_root(
    const CaptureContext& ctx, std::span<const LineFit* const> lines);

class SyncCheck;

// Maps each chip's wall clock onto the root chip's refclk and publishes the result into the ClockMap. The links are
// solved refclk against refclk, so DVFS on either chip cannot enter them.
class SyncEngine {
public:
    // Starts a capture on `ctx`, whose every device has a path over its links to the root (device index 0), publishing
    // into `map`.
    SyncEngine(const CaptureContext& ctx, ClockMap& map);
    ~SyncEngine();
    SyncEngine(const SyncEngine&) = delete;
    SyncEngine& operator=(const SyncEngine&) = delete;

    void on_record(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& rec);
    // Publishes what the batch's records added to the map, and returns whether anything was.
    bool on_batch_end();
    void on_capture_end();

private:
    struct Instant {
        int64_t refclk = 0;
        int64_t wall_eighths = 0;
        uint32_t wall_per_refclk_eighths = 0;
        double wall() const { return eighths_as_ticks(wall_eighths); }
        int64_t wall_tick() const { return nearest_tick(wall_eighths); }
        double wall_per_refclk() const { return eighths_as_ticks(wall_per_refclk_eighths); }
    };
    // Values are offsets from the chip's bases, so no double holds a count since power-on.
    struct Chip {
        std::optional<int64_t> refclk_base, wall_base;
        std::deque<Instant> instants;
        size_t published = 0;
        const char* aiclk_plot = nullptr;
    };
    struct Round {
        std::array<std::optional<double>, 4> stamps;
        std::optional<double>& operator[](kernel_profiler::SyncRole role) { return stamps[static_cast<size_t>(role)]; }
        bool complete() const {
            return std::ranges::all_of(stamps, [](const auto& stamp) { return stamp.has_value(); });
        }
    };
    struct Link {
        std::map<uint32_t, Round> pending;
        std::vector<RoundPoint> rounds;
        std::optional<LineFit> line;
        double solved_at = -std::numeric_limits<double>::infinity();
        // Per full solve window: the rounds' scatter about the line and the line's error at the window centre, in ns.
        ErrorHistogram<64, 16> scatter_ns, centre_ns;
    };
    struct CoreRef {
        uint32_t dev = 0, core = 0;
        auto operator<=>(const CoreRef&) const = default;
    };
    // A solve waits for a full window of rounds, except at capture end, which takes whatever a link has.
    enum class Window { Full, Partial };

    void on_stamp(uint32_t dev, uint32_t core, const kernel_profiler::SyncLinkRecord& record);
    void solve(Link& link, Window window);
    // Logs, over the capture's links, the worst link's median round scatter and fit error, and the worst window's.
    void report_precision() const;
    // Logs how far apart parallel links between the same two chips put the chips' refclk offset.
    void report_parallel_links() const;
    const std::vector<std::optional<RootTransform>>& transforms();
    bool publish_dev(uint32_t dev);
    void plot(const char* name, std::span<const PlotPoint> series);

    ClockMap& map_;
    ClockMap::Reader reader_;
    const CaptureContext& ctx_;
    std::vector<Chip> chips_;
    std::vector<Link> links_;
    std::map<CoreRef, size_t> link_of_;
    std::vector<std::optional<RootTransform>> to_root_;
    bool links_moved_ = true;
    std::vector<PlotPoint> aiclk_;
    std::unique_ptr<SyncCheck> check_;
};

}  // namespace tt::tt_metal::streaming_profiler
