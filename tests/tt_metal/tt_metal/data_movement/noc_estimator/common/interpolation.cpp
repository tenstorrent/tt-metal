// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interpolation.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace tt::noc_estimator::common {

// Quadratic interpolation for better accuracy when possible
double interpolate_latency(const LatencyData& data, const std::vector<uint32_t>& sizes, uint32_t transaction_size) {
    const std::size_t count = std::min(sizes.size(), data.latencies.size());
    if (count == 0) {
        return 0.0;
    }

    // Find bounds for interpolation
    std::size_t lower_idx = 0;
    std::size_t upper_idx = 0;

    for (std::size_t i = 0; i < count; i++) {
        if (sizes[i] <= transaction_size) {
            lower_idx = i;
        }
        if (sizes[i] >= transaction_size) {
            upper_idx = i;
            break;
        }
        upper_idx = i;
    }

    if (lower_idx == upper_idx || sizes[lower_idx] == sizes[upper_idx]) {
        return data.latencies[lower_idx];
    }

    if (count >= 3) {
        std::size_t i0 = 0;
        std::size_t i1 = 1;
        std::size_t i2 = 2;
        if (lower_idx == 0) {
            i0 = 0;
            i1 = 1;
            i2 = 2;
        } else if (upper_idx >= count - 1) {
            i0 = count - 3;
            i1 = count - 2;
            i2 = count - 1;
        } else {
            i0 = lower_idx - 1;
            i1 = lower_idx;
            i2 = upper_idx;
        }

        const double x0 = static_cast<double>(sizes[i0]);
        const double x1 = static_cast<double>(sizes[i1]);
        const double x2 = static_cast<double>(sizes[i2]);
        const double x = static_cast<double>(transaction_size);
        const double denom0 = (x0 - x1) * (x0 - x2);
        const double denom1 = (x1 - x0) * (x1 - x2);
        const double denom2 = (x2 - x0) * (x2 - x1);
        if (denom0 != 0.0 && denom1 != 0.0 && denom2 != 0.0) {
            const double y0 = data.latencies[i0];
            const double y1 = data.latencies[i1];
            const double y2 = data.latencies[i2];
            const double l0 = (x - x1) * (x - x2) / denom0;
            const double l1 = (x - x0) * (x - x2) / denom1;
            const double l2 = (x - x0) * (x - x1) / denom2;
            return (y0 * l0) + (y1 * l1) + (y2 * l2);
        }
    }

    const double t = static_cast<double>(transaction_size - sizes[lower_idx]) /
                     static_cast<double>(sizes[upper_idx] - sizes[lower_idx]);

    return data.latencies[lower_idx] + (t * (data.latencies[upper_idx] - data.latencies[lower_idx]));
}

struct InterpolationBounds {
    std::vector<uint32_t> lower;
    std::vector<uint32_t> upper;
};

static InterpolationBounds find_bounds(const GroupKey& target, const std::map<GroupKey, LatencyData>& entries) {
    InterpolationBounds bounds;
    auto target_values = NumericFields::extract(target);
    std::size_t n = NumericFields::count;

    bounds.lower.resize(n, 0);
    bounds.upper.resize(n, std::numeric_limits<uint32_t>::max());

    std::vector<std::set<uint32_t>> available_values(n);

    for (const auto& [key, _] : entries) {
        if (!key.matches_non_numeric(target)) {
            continue;
        }

        auto values = NumericFields::extract(key);
        for (std::size_t i = 0; i < n; i++) {
            available_values[i].insert(values[i]);
        }
    }

    for (std::size_t i = 0; i < n; i++) {
        uint32_t target_val = target_values[i];
        const auto& vals = available_values[i];

        if (vals.empty()) {
            continue;
        }

        auto it_upper = vals.upper_bound(target_val);
        if (it_upper != vals.begin()) {
            bounds.lower[i] = *prev(it_upper);
        } else {
            bounds.lower[i] = *vals.begin();
        }

        auto it_lb = vals.lower_bound(target_val);
        if (it_lb != vals.end()) {
            bounds.upper[i] = *it_lb;
        } else {
            bounds.upper[i] = *vals.rbegin();
        }
    }

    return bounds;
}

double interpolate_latency_nd(
    const GroupKey& key,
    uint32_t transaction_size,
    const std::vector<uint32_t>& sizes,
    const std::map<GroupKey, LatencyData>& entries) {
    InterpolationBounds bounds = find_bounds(key, entries);
    auto target_values = NumericFields::extract(key);
    std::size_t n = NumericFields::count;

    std::vector<double> weights(n, 0.0);
    for (std::size_t i = 0; i < n; i++) {
        if (bounds.lower[i] == bounds.upper[i]) {
            weights[i] = 0.0;
        } else {
            weights[i] = static_cast<double>(target_values[i] - bounds.lower[i]) /
                         static_cast<double>(bounds.upper[i] - bounds.lower[i]);
        }
    }

    std::size_t num_corners = 1 << n;
    double result = 0.0;

    for (std::size_t corner = 0; corner < num_corners; corner++) {
        std::vector<uint32_t> corner_values(n);
        double corner_weight = 1.0;

        for (std::size_t i = 0; i < n; i++) {
            bool use_upper = (corner >> i) & 1;
            corner_values[i] = use_upper ? bounds.upper[i] : bounds.lower[i];
            corner_weight *= use_upper ? weights[i] : (1.0 - weights[i]);
        }

        GroupKey corner_key = NumericFields::with_values(key, corner_values);
        auto it = entries.find(corner_key);
        if (it != entries.end()) {
            result += corner_weight * interpolate_latency(it->second, sizes, transaction_size);
        }
    }

    return result;
}

bool has_matching_data(const GroupKey& key, const std::map<GroupKey, LatencyData>& entries) {
    for (const auto& [k, _] : entries) {
        if (k.matches_non_numeric(key)) {
            return true;
        }
    }
    return false;
}

// Only allow relaxation of linked and same_axis parameters
// Other parameters (memory, mechanism, pattern, arch) must match exactly
static const char* RELAX_PARAM_NAMES[] = {"linked", "same_axis"};
static constexpr std::size_t RELAX_PARAM_COUNT = 2;

// Check if two keys match, ignoring parameters where mask bit is set
// All other parameters must match exactly
static bool matches_with_mask(const GroupKey& a, const GroupKey& b, uint32_t ignore_mask) {
    // Check exact match for non-relaxable parameters
    if (a.memory != b.memory || a.mechanism != b.mechanism || a.pattern != b.pattern || a.arch != b.arch) {
        return false;
    }

    // Check relaxable parameters
    if (!(ignore_mask & (1 << 0)) && a.linked != b.linked) {
        return false;
    }
    if (!(ignore_mask & (1 << 1)) && a.same_axis != b.same_axis) {
        return false;
    }
    return true;
}

double find_with_relaxation(
    const GroupKey& key,
    uint32_t transaction_size,
    const std::vector<uint32_t>& sizes,
    const std::map<GroupKey, LatencyData>& entries,
    std::string& relaxed_param) {
    // Try each relaxation level (progressively ignore more parameters)
    uint32_t ignore_mask = 0;
    for (std::size_t level = 0; level < RELAX_PARAM_COUNT; level++) {
        ignore_mask |= (1 << level);

        for (const auto& [k, data] : entries) {
            if (matches_with_mask(key, k, ignore_mask)) {
                relaxed_param = RELAX_PARAM_NAMES[level];
                GroupKey relaxed_key = k;
                relaxed_key.num_transactions = key.num_transactions;
                relaxed_key.num_subordinates = key.num_subordinates;

                if (entries.contains(relaxed_key)) {
                    return interpolate_latency(entries.at(relaxed_key), sizes, transaction_size);
                }
                return interpolate_latency_nd(relaxed_key, transaction_size, sizes, entries);
            }
        }
    }

    relaxed_param = "none";
    return 0.0;
}

}  // namespace tt::noc_estimator::common
