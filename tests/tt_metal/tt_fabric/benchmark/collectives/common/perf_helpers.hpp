// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>

#include "fabric_fixture.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt {
namespace tt_fabric {
namespace fabric_router_tests {

// ---- Shared data types ----
struct PerfPoint {
    uint64_t bytes{};
    double sec{};
    double ms{};
    double gbps{};
};

struct PerfParams {
    uint32_t mesh_id = 0;
    chip_id_t src_chip = 0;
    chip_id_t dst_chip = 1;
    bool use_dram_dst = false;
    uint32_t tensor_bytes = 1024 * 1024;
    uint32_t page_size = 4096;
    tt::tt_metal::CoreCoord sender_core{0, 0};
    tt::tt_metal::CoreCoord receiver_core{0, 0};
};

struct PerfStats {
    uint64_t bytes{};
    int iters{};
    double mean_ms{}, std_ms{}, p50_ms{}, p95_ms{}, min_ms{}, max_ms{};
    double mean_gbps{};
};

PerfPoint RunUnicastConnWithParams(BaseFabricFixture* fixture, const PerfParams& p);

// Helpers implemented in perf_helpers.cpp
double mean_of(const std::vector<double>& v);
double stddev_of(const std::vector<double>& v, double mean);
double percentile(std::vector<double> v, double p_in_0_100);

PerfStats aggregate_stats(const std::vector<PerfPoint>& pts);

void warmup_once(BaseFabricFixture* fixture, PerfParams base, int iters = 1);

PerfStats run_repeated(BaseFabricFixture* fixture, const PerfParams& p, int warmup_iters, int iters);

// Utility used by multiple tests
tt::tt_metal::IDevice* find_device_by_id(chip_id_t phys_id);

}  // namespace fabric_router_tests
}  // namespace tt_fabric
}  // namespace tt
