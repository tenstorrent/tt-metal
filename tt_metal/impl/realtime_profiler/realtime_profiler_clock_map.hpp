// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <vector>

namespace tt::tt_metal {

// Maps one device's wall clock (cycles) to host clock ticks from a series of anchors, each a reading of both clocks
// taken together. The device clock's rate changes with load, so the mapping is a monotone cubic (Fritsch and Carlson)
// through the anchors: increasing and continuous, so zones placed on it can never overlap, and a segment's curve is
// fixed once the anchor after its far end has arrived, so a placed zone is never moved. The first anchor is the origin;
// everything is relative to it.
class DeviceClockMap {
public:
    // Starts the map at its first anchor. `nominal_host_per_cycle` maps zones before any anchor is final.
    void start(int64_t host, uint64_t wall, double nominal_host_per_cycle) {
        *this = DeviceClockMap{};
        base_host_ = host;
        base_wall_ = wall;
        host_per_cycle_ = nominal_host_per_cycle;
        raw_.push_back({});
    }

    // Adds a measured anchor. A reading that does not move both clocks forward is a bad one and is ignored.
    void add_anchor(int64_t host, uint64_t wall) {
        if (raw_.empty()) {
            return;
        }
        const Anchor anchor{relative_wall(wall), static_cast<double>(host - base_host_)};
        if (anchor.wall <= raw_.back().wall || anchor.host <= raw_.back().host) {
            return;
        }
        raw_.push_back(anchor);
        // The anchor before this one now has a neighbour on each side.
        finalize(raw_.size() - 2);
        if (raw_.size() > kRawAnchorsKept) {
            raw_.erase(raw_.begin());
        }
    }

    // Makes the newest anchor final, once no later one is coming.
    void finish() {
        if (!raw_.empty()) {
            finalize(raw_.size() - 1);
        }
    }

    // Drops the oldest final anchors that no zone at or after `oldest_waiting` (a device timestamp) can need. Zones of
    // a chip come in order, so the oldest one still waiting is the only one that matters. No zone waiting: keep just
    // the last segment.
    void trim(std::optional<uint64_t> oldest_waiting) {
        const double oldest = oldest_waiting ? relative_wall(*oldest_waiting) : std::numeric_limits<double>::infinity();
        size_t drop = 0;
        while (smooth_.size() - drop > 2 && smooth_[drop + 1].wall <= oldest) {
            ++drop;
        }
        smooth_.erase(smooth_.begin(), smooth_.begin() + static_cast<std::ptrdiff_t>(drop));
    }

    // True when a final anchor sits at or after the timestamp and an earlier one starts the segment, so the timestamp
    // is interpolated and no later anchor can change where it lands.
    [[nodiscard]] bool covers(uint64_t device_timestamp) const {
        return smooth_.size() >= 2 && relative_wall(device_timestamp) <= smooth_.back().wall;
    }

    // The timestamp in host ticks after the first anchor.
    [[nodiscard]] double host_ticks(uint64_t device_timestamp) const {
        const double wall = relative_wall(device_timestamp);
        if (smooth_.empty()) {
            return wall * host_per_cycle_;
        }
        if (wall <= smooth_.front().wall) {
            return smooth_.front().host + (wall - smooth_.front().wall) * smooth_.front().slope;
        }
        if (wall >= smooth_.back().wall) {
            return smooth_.back().host + (wall - smooth_.back().wall) * smooth_.back().slope;
        }
        // Between two final anchors: the cubic Hermite curve through them with their slopes.
        const auto upper = std::ranges::upper_bound(smooth_, wall, {}, &Anchor::wall);
        const Anchor& hi = *upper;
        const Anchor& lo = *std::prev(upper);
        const double span = hi.wall - lo.wall;
        const double t = (wall - lo.wall) / span;
        const double t2 = t * t;
        const double t3 = t2 * t;
        return (2.0 * t3 - 3.0 * t2 + 1.0) * lo.host + (t3 - 2.0 * t2 + t) * span * lo.slope +
               (-2.0 * t3 + 3.0 * t2) * hi.host + (t3 - t2) * span * hi.slope;
    }

    [[nodiscard]] int64_t base_host() const { return base_host_; }
    [[nodiscard]] uint64_t base_wall() const { return base_wall_; }
    [[nodiscard]] size_t num_final_anchors() const { return smooth_.size(); }

private:
    // A point on the mapping: device cycles and host ticks, each relative to the first anchor. `slope` is the curve's
    // slope there in host ticks per cycle, set once the anchor is final.
    struct Anchor {
        double wall = 0.0;
        double host = 0.0;
        double slope = 0.0;
    };
    static constexpr size_t kRawAnchorsKept = 8;

    [[nodiscard]] double relative_wall(uint64_t wall) const {
        return static_cast<double>(static_cast<int64_t>(wall - base_wall_));
    }

    // Fixes the curve's slope at raw anchor `index`: the weighted harmonic mean of the slopes of the two segments that
    // meet there, or the one segment's slope at an end. It keeps every segment increasing.
    void finalize(size_t index) {
        const auto segment_slope = [&](size_t from) {
            return (raw_[from + 1].host - raw_[from].host) / (raw_[from + 1].wall - raw_[from].wall);
        };
        Anchor result = raw_[index];
        const bool has_prev = index > 0;
        const bool has_next = index + 1 < raw_.size();
        if (has_prev && has_next) {
            const double prev_span = raw_[index].wall - raw_[index - 1].wall;
            const double next_span = raw_[index + 1].wall - raw_[index].wall;
            const double w1 = 2.0 * next_span + prev_span;
            const double w2 = next_span + 2.0 * prev_span;
            result.slope = (w1 + w2) / (w1 / segment_slope(index - 1) + w2 / segment_slope(index));
        } else if (has_next) {
            result.slope = segment_slope(index);
        } else if (has_prev) {
            result.slope = segment_slope(index - 1);
        } else {
            result.slope = host_per_cycle_;
        }
        smooth_.push_back(result);
    }

    int64_t base_host_ = 0;
    uint64_t base_wall_ = 0;
    // Host ticks per device cycle from the nominal frequency, for zones before any anchor is final.
    double host_per_cycle_ = 0.0;
    // The newest anchors as measured, oldest first. One becomes final when the anchor after it arrives.
    std::vector<Anchor> raw_;
    // The final anchors, oldest first. Zones are placed on the curve through them.
    std::vector<Anchor> smooth_;
};

}  // namespace tt::tt_metal
