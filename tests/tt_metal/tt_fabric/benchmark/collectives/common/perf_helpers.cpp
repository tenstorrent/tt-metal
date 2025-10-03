// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "perf_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <tt-metalium/device_pool.hpp>

namespace tt::tt_fabric::bench {

double mean_of(const std::vector<double>& v) {
    if (v.empty()) {
        return 0.0;
    }
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / static_cast<double>(v.size());
}

double stddev_of(const std::vector<double>& v, double m) {
    if (v.size() < 2) {
        return 0.0;
    }
    double acc = 0.0;
    for (double x : v) {
        double d = x - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(v.size() - 1));
}

double percentile(std::vector<double> v, double p_in_0_100) {
    if (v.empty()) {
        return 0.0;
    }
    p_in_0_100 = std::clamp(p_in_0_100, 0.0, 100.0);
    const size_t n = v.size();
    const size_t k = static_cast<size_t>(std::round((p_in_0_100 / 100.0) * (n - 1)));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

PerfStats aggregate_stats(const std::vector<PerfPoint>& pts) {
    PerfStats s{};
    if (pts.empty()) {
        return s;
    }
    s.bytes = pts.front().bytes;
    s.iters = static_cast<int>(pts.size());

    std::vector<double> v_ms;
    v_ms.reserve(pts.size());
    std::vector<double> v_GB_s;
    v_GB_s.reserve(pts.size());
    for (const auto& p : pts) {
        v_ms.push_back(p.ms);
        v_GB_s.push_back(p.GB_s);
    }

    s.mean_ms = mean_of(v_ms);
    s.std_ms = stddev_of(v_ms, s.mean_ms);
    s.p50_ms = percentile(v_ms, 50.0);
    s.p95_ms = percentile(v_ms, 95.0);
    s.min_ms = *std::min_element(v_ms.begin(), v_ms.end());
    s.max_ms = *std::max_element(v_ms.begin(), v_ms.end());
    // throughput stats
    s.mean_GB_s = mean_of(v_GB_s);
    s.std_GB_s = stddev_of(v_GB_s, s.mean_GB_s);
    s.p50_GB_s = percentile(v_GB_s, 50.0);
    s.p10_GB_s = percentile(v_GB_s, 10.0);
    s.min_GB_s = *std::min_element(v_GB_s.begin(), v_GB_s.end());
    s.max_GB_s = *std::max_element(v_GB_s.begin(), v_GB_s.end());
    s.cv_GB_s_pct = (s.mean_GB_s > 0.0) ? (s.std_GB_s / s.mean_GB_s) * 100.0 : 0.0;
    return s;
}

void warmup_once(HelpersFixture* fixture, PerfParams base, int iters) {
    for (int i = 0; i < iters; ++i) {
        (void)run_unicast_once(fixture, base);
    }
}

PerfStats run_repeated(HelpersFixture* fixture, const PerfParams& p, int warmup_iters, int iters) {
    for (int w = 0; w < warmup_iters; ++w) {
        (void)run_unicast_once(fixture, p);
    }
    std::vector<PerfPoint> pts;
    pts.reserve(iters > 0 ? iters : 0);
    for (int i = 0; i < iters; ++i) {
        pts.push_back(run_unicast_once(fixture, p));
    }
    return aggregate_stats(pts);
}

tt::tt_metal::IDevice* find_device_by_id(chip_id_t phys_id) {
    auto devices = tt::DevicePool::instance().get_all_active_devices();
    for (auto* d : devices) {
        if (d->id() == phys_id) {
            return d;
        }
    }
    return nullptr;
}

}  // namespace tt::tt_fabric::bench
