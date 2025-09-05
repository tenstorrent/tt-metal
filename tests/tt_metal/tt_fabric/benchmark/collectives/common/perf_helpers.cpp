// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "perf_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <tt-metalium/device_pool.hpp>
using tt::DevicePool;

namespace tt {
namespace tt_fabric {
namespace fabric_router_tests {

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
    std::vector<double> v_gbps;
    v_gbps.reserve(pts.size());
    for (const auto& p : pts) {
        v_ms.push_back(p.ms);
        v_gbps.push_back(p.gbps);
    }

    s.mean_ms = mean_of(v_ms);
    s.std_ms = stddev_of(v_ms, s.mean_ms);
    s.p50_ms = percentile(v_ms, 50.0);
    s.p95_ms = percentile(v_ms, 95.0);
    s.min_ms = *std::min_element(v_ms.begin(), v_ms.end());
    s.max_ms = *std::max_element(v_ms.begin(), v_ms.end());
    s.mean_gbps = mean_of(v_gbps);
    return s;
}

void warmup_once(BaseFabricFixture* fixture, PerfParams base, int iters) {
    base.tensor_bytes = std::max<uint32_t>(base.page_size, 4 * base.page_size);
    for (int i = 0; i < iters; ++i) {
        (void)RunUnicastConnWithParams(fixture, base);
    }
}

PerfStats run_repeated(BaseFabricFixture* fixture, const PerfParams& p, int warmup_iters, int iters) {
    for (int w = 0; w < warmup_iters; ++w) {
        (void)RunUnicastConnWithParams(fixture, p);
    }
    std::vector<PerfPoint> pts;
    pts.reserve(iters > 0 ? iters : 0);
    for (int i = 0; i < iters; ++i) {
        pts.push_back(RunUnicastConnWithParams(fixture, p));
    }
    return aggregate_stats(pts);
}

tt::tt_metal::IDevice* find_device_by_id(chip_id_t phys_id) {
    auto devices = DevicePool::instance().get_all_active_devices();
    for (auto* d : devices) {
        if (d->id() == phys_id) {
            return d;
        }
    }
    return nullptr;
}

}  // namespace fabric_router_tests
}  // namespace tt_fabric
}  // namespace tt
