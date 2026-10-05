// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/least_squares.hpp"

namespace tt::tt_metal::streaming_profiler {

inline constexpr int64_t kNsPerRefclk = kernel_profiler::kNsPerRefclkTick;
inline constexpr double kRefclkTicksPerMs = kernel_profiler::kEthRefclkHz / 1e3;

// The number of eighths in a wall tick. Each chip's wall-clock core, which measures its wall clock against its refclk,
// reports wall clocks and wall ticks per refclk tick in eighths.
inline constexpr int64_t kWallEighths = int64_t{1} << kernel_profiler::kWallEighthBits;
constexpr int64_t whole_ticks(int64_t eighths) { return eighths >> kernel_profiler::kWallEighthBits; }
constexpr double tick_fraction(int64_t eighths) {
    return static_cast<double>(eighths & (kWallEighths - 1)) / kWallEighths;
}
constexpr double eighths_as_ticks(int64_t eighths) { return static_cast<double>(eighths) / kWallEighths; }

// One link round. mid is the midpoint of the transmitter's egress and ingress stamps, in its refclk ticks, and offset
// is the receiver's refclk minus the transmitter's at that instant.
struct RoundPoint {
    double mid = 0.0, offset = 0.0;
};

// Maps a chip's refclk onto the root chip's refclk.
struct RootTransform {
    double scale = 1.0, shift = 0.0;
    double operator()(double refclk) const { return scale * refclk + shift; }
};

// A point of a Tracy plot, placed at a root refclk tick.
struct PlotPoint {
    double root = 0.0, value = 0.0;
};

// A histogram of weighted errors in ns, in bins 1 / BinsPerNs ns wide over [-RangeNs, RangeNs), that also tracks the
// worst error's magnitude.
template <int BinsPerNs, int RangeNs>
struct ErrorHistogram {
    static constexpr int kCentre = RangeNs * BinsPerNs;
    double weight_sum = 0.0, worst = 0.0;
    std::array<double, 2 * kCentre> bins{};
    void add(double ns, double weight) {
        const double bin = std::floor(ns * BinsPerNs) + kCentre;
        if (bin >= 0 && bin < 2 * kCentre) {
            bins[static_cast<size_t>(bin)] += weight;
        }
        weight_sum += weight;
        worst = std::max(worst, std::abs(ns));
    }
    // Returns the centre of the bin that holds the |error| quantile, or the worst error if the quantile is out of
    // range. Requires weight_sum > 0.
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

// Solves each chip's refclk-to-root transform by least squares from `lines`, which holds each link's line fitted to its
// RoundPoints, or null for a link without one. A chip that no line reaches gets no transform.
std::vector<std::optional<RootTransform>> compose_on_root(
    const CaptureContext& ctx, std::span<const LineFit* const> lines);

// A fit waits for a full window of points, except at capture end, where it uses whatever points it has.
enum class Window { Full, Partial };

class SyncCheck;

// One host read of the root chip's refclk. It holds the host TSC at the read's midpoint, the read's round trip and the
// refclk count.
struct RefclkRead {
    int64_t mid = 0, rtt = 0;
    uint64_t refclk = 0;
};
inline constexpr uint32_t kRefclkBurstReads = 1000;
using RefclkBurst = std::array<RefclkRead, kRefclkBurstReads>;
// How often the host reads a burst of the root chip's refclk. On an 8-chip LoudBox, a line fitted over 1 s of bursts
// and extrapolated 10 ms ahead misses the next burst's line by 0.07 ns rms (0.6 ns max).
inline constexpr auto kRefclkBurstPeriod = std::chrono::milliseconds(10);

// Builds a capture's ClockMap from its clock readings. It maps each chip's wall clock onto the root chip's refclk, and
// the root chip's refclk onto the host TSC.
class ClockSolver {
public:
    // Starts a capture on `ctx` whose map starts at `bases`. Every device in `ctx` must have a path over its links to
    // the root (device index 0).
    ClockSolver(const CaptureContext& ctx, const ClockBases& bases);
    ~ClockSolver();
    ClockSolver(const ClockSolver&) = delete;
    ClockSolver& operator=(const ClockSolver&) = delete;

    const ClockMap& map() const { return map_; }

    void on_record(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& record);
    // Publishes what the batch's records added to the map, and returns whether anything was published.
    bool on_batch_end();
    // Fits a burst of host reads of the root refclk into the host series. Returns whether the series has a line yet.
    bool on_refclk_burst(const RefclkBurst& burst);
    void on_capture_end();

private:
    // The AICLK plot measures a ramp instant's AICLK back to an earlier instant at least kAiclkSpanTicks of refclk
    // away, and each chip keeps its kAiclkKeptInstants newest instants for the next batch. A ramp instant, taken while
    // AICLK changes, has no rate, and its position is off by about half a cycle, so a rate measured over 64 refclk
    // ticks (1.28 us, about as long as a PLL step) is within about 1 MHz. Ramp instants are about 16 ticks apart, so 8
    // kept instants reach back that far.
    static constexpr int64_t kAiclkSpanTicks = 64;
    static constexpr size_t kAiclkKeptInstants = 8;
    // One clock point from the chip's wall-clock core. It holds a refclk tick and the wall clock there, and while AICLK
    // is steady also the wall clock's rate.
    struct Instant {
        int64_t refclk = 0;
        int64_t wall_eighths = 0;
        uint32_t wall_per_refclk_eighths = 0;
        double wall() const { return eighths_as_ticks(wall_eighths); }
        int64_t wall_tick() const { return (wall_eighths + kWallEighths / 2) >> kernel_profiler::kWallEighthBits; }
        double wall_per_refclk() const { return eighths_as_ticks(wall_per_refclk_eighths); }
    };
    // The solver's state for one chip. Instants count from refclk_base and wall_base, because counts from power-on are
    // too large to convert to double without losing precision.
    struct Chip {
        std::optional<int64_t> refclk_base, wall_base, last_refclk;
        std::deque<Instant> instants;
        size_t published = 0;
        const char* aiclk_plot = nullptr;
        std::deque<PlotPoint> aiclk_pending;                              // not yet in the mean plot
        std::optional<double> aiclk;                                      // as of the newest point in the mean plot
        double aiclk_through = -std::numeric_limits<double>::infinity();  // root time of the newest point published
    };
    struct Round {
        std::array<std::optional<double>, static_cast<size_t>(kernel_profiler::SyncRole::ReturnIngress) + 1> stamps;
        std::optional<double>& operator[](kernel_profiler::SyncRole role) { return stamps[static_cast<size_t>(role)]; }
        bool complete() const {
            return std::ranges::all_of(stamps, [](const auto& stamp) { return stamp.has_value(); });
        }
    };
    struct LinkState {
        std::map<uint32_t, Round> pending;
        std::vector<RoundPoint> rounds;
        std::optional<LineFit> line;
        double solved_at = -std::numeric_limits<double>::infinity();
        // scatter_ns collects each full solve window's rms scatter of rounds about its line, in ns. worst_centre_ns is
        // the largest window fit error, which is the scatter / sqrt(rounds).
        ErrorHistogram<64, 16> scatter_ns;
        double worst_centre_ns = 0.0;
    };

    struct BurstPoint {
        double tsc, refclk;
    };

    void on_stamp(uint32_t dev, uint32_t core, const kernel_profiler::SyncLinkRecord& record);
    void solve(LinkState& state, Window window);
    // Logs the worst link's median round scatter over the capture, and the worst single window's fit error.
    void report_precision() const;
    // Logs how much parallel links between the same two chips disagree about the chips' refclk offset.
    void report_parallel_links() const;
    const std::vector<std::optional<RootTransform>>& transforms();
    bool publish_dev(uint32_t dev);
    void plot(const char* name, std::span<const PlotPoint> series);
    // Plots the chips' mean AICLK at each published AICLK point up to root time `until`, once every chip has one.
    void plot_aiclk_mean(double until);

    const ClockBases bases_;
    ClockMap map_;
    ClockMap::Reader reader_;
    const CaptureContext& ctx_;
    std::vector<Chip> chips_;
    std::vector<LinkState> links_;
    // The link of each port, by the port's device and core.
    std::map<std::pair<uint32_t, uint32_t>, size_t> link_of_;
    std::vector<std::optional<RootTransform>> to_root_;
    bool links_moved_ = true;
    std::vector<PlotPoint> aiclk_;
    // The batch's AICLK points for the mean plot. A steady instant gives its own rate, as in aiclk_. A ramp instant
    // gives the mean rate over at least kAiclkSpanTicks back, which never reaches past the last steady instant.
    std::vector<PlotPoint> aiclk_estimates_;
    const char* aiclk_mean_plot_ = nullptr;
    double aiclk_sum_ = 0.0;
    size_t aiclk_known_ = 0;
    std::unique_ptr<SyncCheck> check_;
    std::deque<BurstPoint> burst_points_;
    std::optional<HostNode> host_node_;
};

}  // namespace tt::tt_metal::streaming_profiler
