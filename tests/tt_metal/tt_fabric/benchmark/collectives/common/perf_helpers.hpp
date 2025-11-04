// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_fabric::bench {

// API variants for addrgen overload testing
enum class AddrgenApiVariant {
    UnicastWrite,           // fabric_unicast_noc_unicast_write
    UnicastWriteWithState,  // fabric_unicast_noc_unicast_write_with_state
    UnicastWriteSetState    // fabric_unicast_noc_unicast_write_set_state + _with_state
};

using HelpersFixture = ::tt::tt_metal::MeshDeviceFixtureBase;

// ---- Reusable defaults -------
inline constexpr uint32_t kDefaultMeshId = 0;
inline constexpr ChipId kDefaultSrcChip = 0;
inline constexpr ChipId kDefaultDstChip = 1;
inline constexpr bool kDefaultUseDramDst = false;
inline constexpr uint32_t kDefaultTensorBytes = 1u << 20;  // 1 MiB
inline constexpr uint32_t kDefaultPageSize = 4096;         // 4 KiB
inline constexpr tt::tt_metal::CoreCoord kDefaultCore = {0, 0};
inline constexpr uint32_t kDefaultTraceIters = 100;

// ---- Shared data types ----
struct PerfPoint {
    uint64_t bytes{};
    double sec{};
    double ms{};
    double GB_s{};
};

struct PerfParams {
    uint32_t mesh_id = kDefaultMeshId;
    ChipId src_chip = kDefaultSrcChip;
    ChipId dst_chip = kDefaultDstChip;
    bool use_dram_dst = kDefaultUseDramDst;
    uint32_t tensor_bytes = kDefaultTensorBytes;
    uint32_t page_size = kDefaultPageSize;
    tt::tt_metal::CoreCoord sender_core = kDefaultCore;
    tt::tt_metal::CoreCoord receiver_core = kDefaultCore;
    uint32_t trace_iters = kDefaultTraceIters;  // number of enqueues captured per trace
    AddrgenApiVariant api_variant = AddrgenApiVariant::UnicastWrite;  // API variant to test
};

struct PerfStats {
    uint64_t bytes{};
    int iters{};
    // latency (ms)
    double mean_ms{}, std_ms{}, p50_ms{}, p95_ms{}, min_ms{}, max_ms{};
    // throughput (GB/s)
    double mean_GB_s{}, std_GB_s{}, p50_GB_s{}, p10_GB_s{}, min_GB_s{}, max_GB_s{}, cv_GB_s_pct{};
};

PerfPoint run_unicast_once(HelpersFixture* fixture, const PerfParams& p);

// Helpers implemented in perf_helpers.cpp
double mean_of(const std::vector<double>& v);
double stddev_of(const std::vector<double>& v, double mean);
double percentile(std::vector<double> v, double p_in_0_100);

PerfStats aggregate_stats(const std::vector<PerfPoint>& pts);

void warmup_once(HelpersFixture* fixture, PerfParams base, int iters = 1);

PerfStats run_repeated(HelpersFixture* fixture, const PerfParams& p, int warmup_iters, int iters);

// Utility used by multiple tests
tt::tt_metal::IDevice* find_device_by_id(ChipId phys_id);

}  // namespace tt::tt_fabric::bench
