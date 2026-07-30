// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//////////////////////////////////////////////////////////////////////////////////////////
// Host init/teardown microbenchmark (#46305)
//
// Device bring-up and teardown is pure host-side wall-clock work a user pays before/after
// every workload: opening a MeshDevice drives cluster/SoC-descriptor setup, firmware load,
// dispatch-kernel programming and command-queue init; closing it tears all of that down.
// This cost grows with the device topology (single chip -> multichip -> galaxy -> larger
// clusters), yet nothing in CI tracks it today. This benchmark isolates and tracks the
// host init and teardown time per topology so a regression in the open/close path is caught
// over time, completing the runtime-microbenchmark suite tracked in #46305.
//
// Two host-timed families, over a topology axis:
//
//   * host_init     - time MeshDevice::create*(...) (the device open). The complementary
//                     close() runs untimed to return to a closed state for the next sample.
//   * host_teardown - time MeshDevice::close() (the device teardown). The open runs untimed.
//
// Topology axis (a real sweep, enumerated at runtime from the attached cluster so the binary
// runs only what the local/CI machine actually exposes):
//
//   * single_chip - unit mesh on device 0 (always present).
//   * mesh_<N>    - a device open over an N-device mesh, for a log-scaled set of shapes
//                   derived from the system-mesh shape (line sub-meshes 1xc and full-width
//                   rxC bands, plus the full mesh). This lets init/teardown cost be tracked
//                   AS A FUNCTION of topology size: on a single-chip host only single_chip
//                   runs; on a 2-chip n300 single_chip + mesh_2; on a galaxy / larger cluster
//                   single_chip, mesh_2, mesh_4, mesh_8, ... up to the full mesh. Each shape
//                   is probe-opened once at startup and only registered if it actually opens,
//                   so an unsupported shape on a given SKU is skipped rather than erroring the
//                   timed run. Names are device-count keyed so one golden entry per topology
//                   size covers whatever concrete shape a SKU maps that count to.
//
// Timing note: benchmarks use UseManualTime() and time only the open (or close) call; the
// complementary teardown (or setup) and a single warmup open/close run outside the timed
// region so one-time process init does not skew the measured steady-state cost. Repetition
// and iteration counts are deliberately small because a device open on silicon is expensive
// and Google Benchmark re-runs the whole registered function per repetition. The compare
// script consumes the "min" aggregate: a contended host only inflates a sample, so the
// minimum is the most representative measure of true init/teardown cost while a genuine
// regression still raises it.
//////////////////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-logger/tt-logger.hpp>

#include "context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

// Device open on silicon is seconds-scale, so keep the sample counts small: Google Benchmark
// re-runs the whole registered function (a full open+close cycle per iteration) once per
// repetition, and every iteration of both families performs one open and one close.
constexpr uint32_t INIT_ITERATIONS = 3;
constexpr uint32_t INIT_REPETITIONS = 3;
constexpr uint32_t TEARDOWN_ITERATIONS = 3;
constexpr uint32_t TEARDOWN_REPETITIONS = 3;

// Open a MeshDevice for a mesh `shape`. A single-device mesh uses the canonical unit-mesh
// path on device 0; anything larger opens the requested shape over the system mesh.
std::shared_ptr<MeshDevice> open_shape(const MeshShape& shape) {
    if (shape.mesh_size() == 1) {
        constexpr ChipId device_id = 0;
        return MeshDevice::create_unit_mesh(device_id);
    }
    return MeshDevice::create(MeshDeviceConfig(shape));
}

double seconds_between(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

// Build the topology sweep from the system-mesh shape: single chip, a log-scaled set of line
// (1xc) and full-width (rxC) sub-meshes, and the full mesh. Deduplicated by device count so
// each size appears once; ordered ascending. Shapes are candidates only - main() probe-opens
// each and drops any the attached hardware cannot instantiate.
std::vector<std::pair<std::string, MeshShape>> candidate_topologies(const MeshShape& system_shape) {
    std::vector<std::pair<std::string, MeshShape>> candidates;
    std::set<size_t> seen_sizes;

    auto add = [&](const MeshShape& shape) {
        const size_t n = shape.mesh_size();
        if (n == 0 || seen_sizes.count(n) != 0) {
            return;
        }
        seen_sizes.insert(n);
        std::string label = (n == 1) ? "single_chip" : ("mesh_" + std::to_string(n));
        candidates.emplace_back(std::move(label), shape);
    };

    add(MeshShape(1, 1));  // single chip is always a candidate

    if (system_shape.dims() == 2) {
        const auto rows = static_cast<uint32_t>(system_shape[0]);
        const auto cols = static_cast<uint32_t>(system_shape[1]);
        // Line sub-meshes along a row: 1x2, 1x4, 1x8, ... up to the full row width.
        for (uint32_t c = 2; c <= cols; c *= 2) {
            add(MeshShape(1, c));
        }
        // Full-width bands: 2xC, 4xC, ... up to the full mesh (log-scaled to bound runtime).
        for (uint32_t r = 2; r <= rows; r *= 2) {
            add(MeshShape(r, cols));
        }
        add(MeshShape(rows, cols));  // ensure the full mesh is always included
    } else {
        add(system_shape);  // non-2D system: at least sweep single chip vs the full mesh
    }
    return candidates;
}

//////////////////////////////////////////////////////////////////////////////////////////
// host_init: time the device open. The paired close() runs untimed to reset for next sample.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_host_init(benchmark::State& state, const MeshShape& shape) {
    // Warm up one-time process init (cluster/SoC descriptors, firmware image load) so the
    // timed opens measure steady-state re-open cost rather than first-open fixed cost.
    open_shape(shape)->close();

    for ([[maybe_unused]] auto _ : state) {
        auto start = std::chrono::steady_clock::now();
        auto device = open_shape(shape);
        auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(seconds_between(start, end));

        device->close();  // untimed teardown to return to a closed state
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// host_teardown: time the device close. The paired open runs untimed.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_host_teardown(benchmark::State& state, const MeshShape& shape) {
    open_shape(shape)->close();  // warm up one-time process init

    for ([[maybe_unused]] auto _ : state) {
        auto device = open_shape(shape);  // untimed setup

        auto start = std::chrono::steady_clock::now();
        device->close();
        auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(seconds_between(start, end));
    }
}

void InitConfig(benchmark::internal::Benchmark* b) {
    b->Iterations(INIT_ITERATIONS)
        ->Repetitions(INIT_REPETITIONS)
        ->ComputeStatistics("min", [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); })
        ->ReportAggregatesOnly(true)
        ->UseManualTime();
}

void TeardownConfig(benchmark::internal::Benchmark* b) {
    b->Iterations(TEARDOWN_ITERATIONS)
        ->Repetitions(TEARDOWN_REPETITIONS)
        ->ComputeStatistics("min", [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); })
        ->ReportAggregatesOnly(true)
        ->UseManualTime();
}

// Register init + teardown for a topology. Names are device-count keyed so a single golden
// entry per topology size covers whatever concrete shape a given SKU maps that count to.
void register_topology(const std::string& label, const MeshShape& shape) {
    benchmark::RegisterBenchmark("BM_host_init/" + label, [shape](benchmark::State& state) {
        BM_host_init(state, shape);
    })->Apply(InitConfig);
    benchmark::RegisterBenchmark("BM_host_teardown/" + label, [shape](benchmark::State& state) {
        BM_host_teardown(state, shape);
    })->Apply(TeardownConfig);
}

// Query the system-mesh shape, falling back to a single chip if the cluster cannot be read.
MeshShape system_mesh_shape_or_single() {
    try {
        return MetalContext::instance().get_system_mesh().shape();
    } catch (const std::exception& e) {
        log_warning(tt::LogTest, "Could not query system mesh shape ({}); running single_chip only.", e.what());
        return MeshShape(1, 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Enumerate the topology sweep from the attached cluster and register only the shapes that
    // physically open on this machine: probe-open (and immediately close) each candidate, so an
    // unsupported shape on a given SKU is skipped here instead of erroring inside the timed run.
    // A single-chip host registers single_chip alone; multichip / galaxy / larger-cluster SKUs
    // light up the larger mesh_<N> points.
    const std::vector<std::pair<std::string, MeshShape>> candidates =
        candidate_topologies(system_mesh_shape_or_single());
    for (const auto& [label, shape] : candidates) {
        try {
            open_shape(shape)->close();  // probe: confirm the hardware can instantiate this shape
        } catch (const std::exception& e) {
            log_info(tt::LogTest, "Skipping topology {} (open failed: {})", label, e.what());
            continue;
        }
        register_topology(label, shape);
    }

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
