// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar microbenchmark for the DFB kernel-exit write barrier: what does an unconditional
// noc_async_write_barrier() cost compared with a conditional one or a per-txn-id barrier, does it
// cover txn-id writes, does it stall on another DM's traffic, and can one DM barrier on another DM's
// txn-id writes. Results are logged; the asserts only check that the kernels ran and that every
// barrier variant actually covers what it should.
//
//   ./build/test/tt_metal/unit_tests_api --gtest_filter='*DfbWriteBarrierBench*'

#include "dataflow_buffer/dfb_test_common.hpp"
#include "tt_metal/tt_metal/test_kernels/dataflow/dfb_write_barrier_bench_common.h"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {
namespace {

using namespace dfb_wb_bench;

constexpr const char* kVariantNames[NUM_VARIANTS] = {"none", "conditional", "full", "per_trid"};
constexpr const char* kScenarioNames[] = {"S1 idle", "S2 explicit writes", "S3 txn-id writes", "S4 idle, other DM streaming",
                                          "S5 other DM's txn-id writes"};

struct BenchParams {
    uint32_t scenario;
    uint32_t num_writes;
    uint32_t write_bytes = 2048;
    uint32_t iters = 16;
};

struct BenchResult {
    std::vector<uint32_t> words;
    uint32_t word(uint32_t v, uint32_t w) const { return words[v * WORDS_PER_VARIANT + w]; }
};

BenchResult run_bench(distributed::MeshDevice& mesh_device, const BenchParams& p) {
    const uint32_t bytes = p.num_writes * p.write_bytes;
    auto make_l1 = [&](uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            distributed::DeviceLocalBufferConfig{.page_size = size, .buffer_type = BufferType::L1},
            &mesh_device);
    };
    // L1 buffers are allocated at the same address on every worker core, so the destination region
    // is free on the target core whichever one it is.
    auto src_buffer = make_l1(std::max(bytes, 64u));
    auto dst_buffer = make_l1(std::max(bytes, 64u));
    auto result_buffer = make_l1(NUM_RESULT_WORDS * sizeof(uint32_t));

    const CoreCoord core{0, 0};
    // Write to a neighbouring core when there is one, so the writes cross the NoC.
    const CoreCoord dst_logical = mesh_device.compute_with_storage_grid_size().x > 1 ? CoreCoord{1, 0} : core;
    const CoreCoord dst_virtual = mesh_device.worker_core_from_logical_core(dst_logical);
    const CoreRangeSet crs(CoreRange(core, core));
    const experimental::NodeCoord node{core.x, core.y};

    const bool needs_streamer = p.scenario == SCENARIO_COUPLED_IDLE || p.scenario == SCENARIO_CROSS_HART_TRID;
    const std::vector<std::string> rta_names = {
        "scenario", "num_writes", "write_bytes", "iters", "src_addr", "dst_x", "dst_y", "dst_addr", "result_addr"};
    const experimental::KernelSpecName MEASURER{"measurer"};
    const experimental::KernelSpecName STREAMER{"streamer"};

    std::vector<experimental::KernelSpec> kernels = {experimental::KernelSpec{
        .unique_id = MEASURER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_write_barrier_bench_dm.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = rta_names},
        .hw_config = experimental::DataMovementHardwareConfig{},
    }};
    std::vector<experimental::KernelSpecName> wu_kernels = {MEASURER};
    if (needs_streamer) {
        kernels.push_back(experimental::KernelSpec{
            .unique_id = STREAMER,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_write_barrier_bench_streamer_dm.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = experimental::DataMovementHardwareConfig{},
        });
        wu_kernels.push_back(STREAMER);
    }

    experimental::ProgramSpec spec{
        .name = "dfb_write_barrier_bench",
        .kernels = kernels,
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = wu_kernels, .target_nodes = crs}},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    // The streamer writes to a different half of the destination than the measurer would, but the
    // contents never matter here, so both share one source and one destination.
    auto args = experimental::MakeRuntimeArgsForSingleNode(
        node,
        {{"scenario", static_cast<uint32_t>(p.scenario)},
         {"num_writes", p.num_writes},
         {"write_bytes", p.write_bytes},
         {"iters", p.iters},
         {"src_addr", static_cast<uint32_t>(src_buffer->address())},
         {"dst_x", static_cast<uint32_t>(dst_virtual.x)},
         {"dst_y", static_cast<uint32_t>(dst_virtual.y)},
         {"dst_addr", static_cast<uint32_t>(dst_buffer->address())},
         {"result_addr", static_cast<uint32_t>(result_buffer->address())}});
    experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{.kernel = MEASURER, .runtime_arg_values = args}};
    if (needs_streamer) {
        run_args.kernel_run_args.push_back(
            experimental::ProgramRunArgs::KernelRunArgs{.kernel = STREAMER, .runtime_arg_values = args});
    }
    experimental::SetProgramRunArgs(program, run_args);

    const uint32_t result_addr = static_cast<uint32_t>(result_buffer->address());
    std::vector<uint32_t> zeros(NUM_RESULT_WORDS, 0u);
    slow_dispatch::WriteToL1(mesh_device, core, result_addr, zeros);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    BenchResult r;
    slow_dispatch::ReadFromL1(mesh_device, core, result_addr, NUM_RESULT_WORDS * sizeof(uint32_t), r.words);
    return r;
}

void log_bench(const BenchParams& p, const BenchResult& r) {
    log_info(
        tt::LogTest,
        "DFB write barrier bench [{}] num_writes={} write_bytes={} iters={}",
        kScenarioNames[p.scenario],
        p.num_writes,
        p.write_bytes,
        p.iters);
    for (uint32_t v = 0; v < NUM_VARIANTS; v++) {
        const uint32_t iters = r.word(v, R_ITERS);
        log_info(
            tt::LogTest,
            "  {:<12} cycles min={} avg={} max={}  leaked={}/{}  max_outstanding_seen={}",
            kVariantNames[v],
            r.word(v, R_MIN),
            iters ? r.word(v, R_SUM) / iters : 0,
            r.word(v, R_MAX),
            r.word(v, R_LEAKED),
            iters,
            r.word(v, R_MAX_OUTSTANDING));
    }
    if (p.scenario == SCENARIO_CROSS_HART_TRID) {
        log_info(
            tt::LogTest,
            "  streamer's own max outstanding on its txn id right after issuing: {}",
            r.words[STREAMER_SELF_MAX_OUTSTANDING]);
    }
}

BenchResult run_and_log(distributed::MeshDevice& mesh_device, const BenchParams& p) {
    BenchResult r = run_bench(mesh_device, p);
    log_bench(p, r);
    for (uint32_t v = 0; v < NUM_VARIANTS; v++) {
        EXPECT_EQ(r.word(v, R_ITERS), p.iters) << kVariantNames[v] << " did not run";
    }
    return r;
}

}  // namespace

TEST_F(QuasarAnyDispatchMeshDeviceSingleCardFixture, DfbWriteBarrierBenchIdle) {
    run_and_log(this->device(), {.scenario = SCENARIO_IDLE, .num_writes = 0});
}

TEST_F(QuasarAnyDispatchMeshDeviceSingleCardFixture, DfbWriteBarrierBenchExplicitWrites) {
    for (uint32_t n : {1u, 8u}) {
        BenchResult r = run_and_log(this->device(), {.scenario = SCENARIO_EXPLICIT_WRITES, .num_writes = n});
        // A full barrier (conditional or not) must leave no plain writes in flight.
        EXPECT_EQ(r.word(V_CONDITIONAL, R_LEAKED), 0u);
        EXPECT_EQ(r.word(V_FULL, R_LEAKED), 0u);
    }
}

TEST_F(QuasarAnyDispatchMeshDeviceSingleCardFixture, DfbWriteBarrierBenchTridWrites) {
    BenchResult r = run_and_log(this->device(), {.scenario = SCENARIO_TRID_WRITES, .num_writes = 8});
    // The per-txn-id barrier is the one implicit-sync DFB writes rely on today.
    EXPECT_EQ(r.word(V_TRID, R_LEAKED), 0u);
}

TEST_F(QuasarAnyDispatchMeshDeviceSingleCardFixture, DfbWriteBarrierBenchCoupledIdle) {
    run_and_log(this->device(), {.scenario = SCENARIO_COUPLED_IDLE, .num_writes = 8});
}

TEST_F(QuasarAnyDispatchMeshDeviceSingleCardFixture, DfbWriteBarrierBenchCrossHartTrid) {
    run_and_log(this->device(), {.scenario = SCENARIO_CROSS_HART_TRID, .num_writes = 8});
}

}  // namespace tt::tt_metal
