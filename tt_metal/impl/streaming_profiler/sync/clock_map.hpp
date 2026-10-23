// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>

#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal::streaming_profiler {

// Returns the host TSC now. Every host time in the profiler is in TSC ticks.
int64_t tsc_now() noexcept;
// Returns the host TSC's ns per tick. The first call blocks while it measures the rate.
double ns_per_tsc_tick();

// Rounds x to the nearest integer, ties to even.
inline int64_t round_nearest(double x) noexcept { return static_cast<int64_t>(std::nearbyint(x)); }

// Returns the count nearest `near` whose low word is `lo`.
constexpr int64_t widen(int64_t near, uint32_t lo) {
    return near + static_cast<int32_t>(lo - static_cast<uint32_t>(near));
}

// A node of a piecewise-linear map from Key to a double. At key `at` the map's value is `value` and its slope is
// `tangent`.
template <typename Key>
struct ClockNode {
    Key at{};
    double value = 0.0;
    double tangent = 0.0;
};
using SyncNode = ClockNode<int64_t>;
using HostNode = ClockNode<double>;

struct ClockBases {
    int64_t root_refclk = 0, tsc = 0;
};

// The most nodes a series holds, at most 1.5 GB at 24 B per node. A chip adds a node once per millisecond while its
// AICLK is steady, so its series holds about 18.6 hours of steady clock. The host series and SteadyClock add one per
// refclk burst, at most every 10 ms, so they hold at least 7.8 days.
inline constexpr uint32_t kSeriesNodes = 1u << 26;

// A piecewise-linear map through its nodes, written by one thread and read lock-free by any thread. No later node
// changes what keys up to `cover` map to.
template <typename Key>
struct ClockSeries {
    using Node = ClockNode<Key>;
    BroadcastRing<Node, SlotBacking::OnFirstWrite> nodes{kSeriesNodes};
    Node last{.at = std::numeric_limits<Key>::lowest()};  // the writer's own copy
    std::atomic<Key> cover{std::numeric_limits<Key>::lowest()};

    void append(const Node& node);
    // Makes keys up to `until` final. `until` must be at least `cover`.
    void extend(Key until) { cover.store(until, std::memory_order_release); }
};
extern template struct ClockSeries<int64_t>;
extern template struct ClockSeries<double>;

// One linear piece of a series, over keys [from, to]. A reader keeps the last one it found, so most lookups skip the
// search.
template <typename Key>
struct Segment {
    Key from = std::numeric_limits<Key>::max();
    Key to = std::numeric_limits<Key>::lowest();
    Key origin{};
    double value = 0.0;
    double slope = 0.0;
    uint64_t hint = 0;  // the index of the first node after `from`, where the next search starts
    bool holds(Key key) const noexcept { return key >= from && key <= to; }
    // One fused multiply-add, as place_host4() computes it, so the scalar and vector paths round alike.
    double at(Key key) const noexcept { return std::fma(slope, static_cast<double>(key - origin), value); }
};

// Each chip's series maps its wall ticks onto the root chip's refclk, and the host series maps root refclk ticks onto
// the host TSC. One thread writes, and readers never lock or block it.
class ClockMap {
public:
    // The segments one thread last looked up in a ClockMap, one per chip series and one for the host series. Only that
    // thread uses it.
    class Reader {
    public:
        Reader() = default;

    private:
        friend class ClockMap;
        Reader(size_t devices, int64_t tsc_base) : chips_(devices), tsc_base_(tsc_base) {}
        std::vector<Segment<int64_t>> chips_;
        Segment<double> host_;
        int64_t tsc_base_ = 0;
    };

    ClockMap(size_t devices, ClockBases bases);
    Reader reader() const;

    // Appends `node` to the chip's series. A node at or before the series' last node is dropped.
    void append(uint32_t dev, SyncNode node);
    // Appends `node` to the host series and makes keys up to `until` final along its tangent.
    void append_host(HostNode node, double until);
    // Makes every wall tick of the chip's series final.
    void finish(uint32_t dev);

    // Returns the root refclk of the chip's wall tick plus `tick_fraction`, or nullopt before the chip's series has a
    // node.
    std::optional<double> place_root(Reader& reader, uint32_t dev, int64_t wall, double tick_fraction) const noexcept;
    int64_t place_host(Reader& reader, uint32_t dev, int64_t wall) const {
        const Segment<int64_t>& chip = reader.chips_[dev];
        if (chip.holds(wall)) [[likely]] {
            const double root = chip.at(wall);
            if (reader.host_.holds(root)) [[likely]] {
                return reader.tsc_base_ + round_nearest(reader.host_.at(root));
            }
        }
        return place_slow(reader, dev, wall);
    }
    // Returns place_host() of each of four wall ticks. When the reader's segments hold all four and every value
    // converted between int64 and double is below 2^51 in magnitude, the four take place_host()'s operations side by
    // side.
    simde__m256i place_host4(Reader& reader, uint32_t dev, simde__m256i wall) const {
        const Segment<int64_t>& chip = reader.chips_[dev];
        const Segment<double>& host = reader.host_;
        // Adding 2^52 + 2^51 and reinterpreting the bits converts between int64 and double exactly below 2^51.
        constexpr int64_t kExactBelow = int64_t{1} << 51;
        const simde__m256i magic_bits = simde_mm256_set1_epi64x(0x4338000000000000);
        const simde__m256d magic = simde_mm256_castsi256_pd(magic_bits);
        const simde__m256i key = simde_mm256_sub_epi64(wall, simde_mm256_set1_epi64x(chip.origin));
        const simde__m256i biased = simde_mm256_add_epi64(key, simde_mm256_set1_epi64x(kExactBelow));
        const simde__m256i missed = simde_mm256_or_si256(
            simde_mm256_or_si256(
                simde_mm256_cmpgt_epi64(simde_mm256_set1_epi64x(chip.from), wall),
                simde_mm256_cmpgt_epi64(wall, simde_mm256_set1_epi64x(chip.to))),
            simde_mm256_or_si256(
                simde_mm256_cmpgt_epi64(simde_mm256_setzero_si256(), biased),
                simde_mm256_cmpgt_epi64(biased, simde_mm256_set1_epi64x(2 * kExactBelow - 1))));
        const simde__m256d root = simde_mm256_fmadd_pd(
            simde_mm256_set1_pd(chip.slope),
            simde_mm256_sub_pd(simde_mm256_castsi256_pd(simde_mm256_add_epi64(key, magic_bits)), magic),
            simde_mm256_set1_pd(chip.value));
        const simde__m256d tsc = simde_mm256_round_pd(
            simde_mm256_fmadd_pd(
                simde_mm256_set1_pd(host.slope),
                simde_mm256_sub_pd(root, simde_mm256_set1_pd(host.origin)),
                simde_mm256_set1_pd(host.value)),
            SIMDE_MM_FROUND_CUR_DIRECTION | SIMDE_MM_FROUND_NO_EXC);
        const simde__m256d held = simde_mm256_and_pd(
            simde_mm256_and_pd(
                simde_mm256_cmp_pd(root, simde_mm256_set1_pd(host.from), SIMDE_CMP_GE_OQ),
                simde_mm256_cmp_pd(root, simde_mm256_set1_pd(host.to), SIMDE_CMP_LE_OQ)),
            simde_mm256_cmp_pd(
                simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0), tsc),
                simde_mm256_set1_pd(static_cast<double>(kExactBelow)),
                SIMDE_CMP_LT_OQ));
        if (simde_mm256_movemask_pd(held) == 0xF && simde_mm256_testz_si256(missed, missed)) [[likely]] {
            return simde_mm256_add_epi64(
                simde_mm256_sub_epi64(simde_mm256_castpd_si256(simde_mm256_add_pd(tsc, magic)), magic_bits),
                simde_mm256_set1_epi64x(reader.tsc_base_));
        }
        alignas(32) int64_t walls[4];
        simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(walls), wall);
        return simde_mm256_setr_epi64x(
            place_host(reader, dev, walls[0]),
            place_host(reader, dev, walls[1]),
            place_host(reader, dev, walls[2]),
            place_host(reader, dev, walls[3]));
    }
    // Returns the host TSC of a root refclk tick, or nullopt before the host series has a node.
    std::optional<int64_t> place_tsc(Reader& reader, double root) const noexcept;
    bool has_host_nodes() const noexcept;
    // Returns the chip's newest wall tick its series is final up to, or the lowest int64 before it has one. The clock sync
    // moves it on for as long as the capture runs.
    int64_t covered(uint32_t dev) const noexcept { return chips_[dev].cover.load(std::memory_order_acquire); }
    // Returns whether no later node of either series can move the host time of the chip's wall tick.
    bool is_final(Reader& reader, uint32_t dev, int64_t wall) const noexcept;

private:
    [[gnu::cold]] int64_t place_slow(Reader& reader, uint32_t dev, int64_t wall) const;

    std::deque<ClockSeries<int64_t>> chips_;
    ClockSeries<double> host_;
    const int64_t tsc_base_;
};

// Maps host TSC to steady_clock ns. Appends take a lock, and reads never do. There is one SteadyClock per process,
// because the API caches a segment per thread, not per capture.
class SteadyClock {
public:
    // Appends a pair of the host TSC and CLOCK_MONOTONIC read together.
    void sample();
    // Returns the segment that maps `tsc` to ns from base_ns(), or nullopt before the first sample.
    std::optional<Segment<int64_t>> segment_at(int64_t tsc) const noexcept;
    // The CLOCK_MONOTONIC ns of the first sample. Call it only once segment_at() has returned a segment.
    int64_t base_ns() const noexcept;

private:
    friend class Service;
    SteadyClock() = default;

    std::mutex append_mu_;
    ClockSeries<int64_t> steady_;
    std::optional<int64_t> base_ns_;
};

}  // namespace tt::tt_metal::streaming_profiler
