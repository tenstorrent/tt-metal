// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End-to-end compute semaphores (SemScope::COMPUTE_ATOMIC) on Blackhole silicon.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>

#include "jit_build/jit_build_settings.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecHWTest;

// ============================================================================
// Compute semaphore (SemScope::COMPUTE_ATOMIC): the Tensix hardware (Sync Unit) semaphore, index
// UNPACK_OPERAND_SYNC. One kernel (test_compute_semaphore.cpp), three patterns (A, D, E) selected by
// the `pattern` compile-time arg.
// ============================================================================
namespace {

// Pattern D/E ring depth, mirrors the kernel's kDepth, passed to the host as the semaphore's capacity.
constexpr std::uint32_t kSemDepth = 4;
// Report words live here; pattern D's tile regions are fixed offsets above it (mirrors the kernel).
constexpr std::uint32_t kSemReportAddr = 100 * 1024;
const NodeCoord kSemNode{0, 0};

struct ComputeSemaphoreRun {
    std::uint32_t pattern = 0;
    std::uint32_t thread_sel = 0;
    std::uint32_t num_iters = 0;
    std::uint32_t nosync = 0;     // patterns D/E negative control
    std::uint32_t batch = 1;      // pattern E: credits per wait_not_full(batch); up(batch)
    std::uint32_t max_value = 0;  // SemaphoreAdvancedOptions::max_value (0 = default capacity 15)
};

// Build the compute-semaphore program for `run` (one compute kernel, one bound semaphore), run args set.
Program MakeComputeSemaphoreProgram(distributed::MeshDevice& mesh_device, const ComputeSemaphoreRun& run) {
    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/test_compute_semaphore.cpp";
    compute.runtime_arg_schema.runtime_arg_names = {"num_iters", "report_addr"};
    compute.compile_time_args = {
        {"pattern", run.pattern}, {"thread_sel", run.thread_sel}, {"nosync", run.nosync}, {"batch", run.batch}};
    compute.semaphore_bindings.push_back({.semaphore_spec_name = SemaphoreSpecName{"sem"}, .accessor_name = "sem"});

    SemaphoreSpec sem{.unique_id = SemaphoreSpecName{"sem"}, .target_nodes = kSemNode};
    sem.advanced_options.max_value = run.max_value;
    ProgramSpec spec{
        .name = "compute_semaphore",
        .kernels = {compute},
        .semaphores = {sem},
        .work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", kSemNode, {"compute"})},
    };
    Program program = MakeProgramFromSpec(mesh_device, spec);

    ProgramRunArgs args;
    args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"compute"},
        .runtime_arg_values =
            MakeRuntimeArgsForSingleNode(kSemNode, {{"num_iters", run.num_iters}, {"report_addr", kSemReportAddr}}),
    }};
    SetProgramRunArgs(program, args);
    return program;
}

// Zero the report words, launch, return the first `n_report` report words.
std::vector<std::uint32_t> RunComputeSemaphore(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const ComputeSemaphoreRun& run,
    std::uint32_t n_report) {
    Program program = MakeComputeSemaphoreProgram(*mesh_device, run);
    std::vector<std::uint32_t> zero_report(64, 0u);
    slow_dispatch::WriteToL1(*mesh_device, kSemNode, kSemReportAddr, zero_report);
    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> r;
    slow_dispatch::ReadFromL1(*mesh_device, kSemNode, kSemReportAddr, n_report * sizeof(std::uint32_t), r);
    return r;
}

}  // namespace

// PATTERN A: one thread drives set/up/down/wait/wait_min/value through a known value sequence. Run
// once on UNPACK and once on PACK -- the same primitive must work from either thread.
TEST_F(ProgramSpecHWTest, ComputeSemaphoreSelfCheckUnpack) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    const auto r = RunComputeSemaphore(mesh_device, {.pattern = 0, .thread_sel = 0}, 6);
    GTEST_LOG_(INFO) << "UNPACK self-check: pass=" << r[0] << " v(after up5)=" << r[1] << " v(after down3)=" << r[2]
                     << " v(after up1)=" << r[3] << " fail_step=" << r[4];
    EXPECT_EQ(r[1], 5u);
    EXPECT_EQ(r[2], 2u);
    EXPECT_EQ(r[3], 3u);
    EXPECT_EQ(r[0], 1u) << "UNPACK primitive self-check failed at step " << r[4];
}

TEST_F(ProgramSpecHWTest, ComputeSemaphoreSelfCheckPack) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    const auto r = RunComputeSemaphore(mesh_device, {.pattern = 0, .thread_sel = 2}, 6);
    GTEST_LOG_(INFO) << "PACK self-check: pass=" << r[0] << " v(after up5)=" << r[1] << " v(after down3)=" << r[2]
                     << " v(after up1)=" << r[3] << " fail_step=" << r[4];
    EXPECT_EQ(r[1], 5u);
    EXPECT_EQ(r[2], 2u);
    EXPECT_EQ(r[3], 3u);
    EXPECT_EQ(r[0], 1u) << "PACK primitive self-check failed at step " << r[4];
}

// PATTERN D: compute-only two-hop datacopy through a shared L1 ring, 2.0 LLKOperand API, no data-movement
// kernels. The host writes num_tiles distinct 32x32 Float16_b tiles straight into L1 (in), poisons the ring
// (mid) and the output (out), runs the kernel, and checks out == in bit for bit. Region offsets mirror the
// kernel's kDInOffset/kDMidOffset/kDOutOffset. Returns the number of output tiles that differ from input.
namespace {

struct DatacopyResult {
    std::uint32_t mismatched_tiles = 0;
    std::vector<std::uint32_t> report;
};

DatacopyResult RunComputeSemaphoreDatacopy(
    distributed::MeshDevice& mesh_device, std::uint32_t num_tiles, std::uint32_t nosync) {
    constexpr std::uint32_t kMaxTiles = 8;
    constexpr std::uint32_t kDepth = kSemDepth;
    constexpr std::uint32_t kTileWords = 32 * 32 * 2 / 4;
    constexpr std::uint32_t kInOffset = 0x40000;
    constexpr std::uint32_t kMidOffset = kInOffset + kMaxTiles * kTileWords * 4;
    constexpr std::uint32_t kOutOffset = kMidOffset + kDepth * kTileWords * 4;
    if (num_tiles > kMaxTiles) {
        ADD_FAILURE() << "pattern D supports at most " << kMaxTiles << " tiles";
        return {};
    }

    // Capacity = ring depth, so PACK's wait_not_full() holds the packer while all kDepth slots are full.
    Program program = MakeComputeSemaphoreProgram(
        mesh_device, {.pattern = 3, .num_iters = num_tiles, .nosync = nosync, .max_value = kSemDepth});

    // Input: tile t, datum k = bf16 0x4000 + (t << 7) + (k & 0x7F). Normal positive values (exact through a
    // bf16 datacopy) and distinct per tile, so a stale ring read (the previous tile) is detected.
    std::vector<std::uint32_t> in(num_tiles * kTileWords);
    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        for (std::uint32_t w = 0; w < kTileWords; ++w) {
            const std::uint32_t lo = 0x4000u + (t << 7) + ((2 * w) & 0x7Fu);
            const std::uint32_t hi = 0x4000u + (t << 7) + ((2 * w + 1) & 0x7Fu);
            in[t * kTileWords + w] = (hi << 16) | lo;
        }
    }
    std::vector<std::uint32_t> zero_report(64, 0u);
    std::vector<std::uint32_t> poison((kDepth + kMaxTiles) * kTileWords, 0xDEADBEEFu);
    slow_dispatch::WriteToL1(mesh_device, kSemNode, kSemReportAddr, zero_report);
    slow_dispatch::WriteToL1(mesh_device, kSemNode, kSemReportAddr + kInOffset, in);
    slow_dispatch::WriteToL1(mesh_device, kSemNode, kSemReportAddr + kMidOffset, poison);  // covers mid and out
    LaunchProgram(mesh_device, std::move(program));

    DatacopyResult res;
    slow_dispatch::ReadFromL1(mesh_device, kSemNode, kSemReportAddr, 4 * sizeof(std::uint32_t), res.report);
    std::vector<std::uint32_t> out;
    slow_dispatch::ReadFromL1(mesh_device, kSemNode, kSemReportAddr + kOutOffset, num_tiles * kTileWords * 4, out);
    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        if (!std::equal(
                out.begin() + t * kTileWords, out.begin() + (t + 1) * kTileWords, in.begin() + t * kTileWords)) {
            ++res.mismatched_tiles;
        }
    }
    return res;
}

}  // namespace

TEST_F(ProgramSpecHWTest, ComputeSemaphoreDatacopy) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_tiles = 8;
    const auto r = RunComputeSemaphoreDatacopy(*mesh_device, num_tiles, /*nosync=*/0);
    GTEST_LOG_(INFO) << "datacopy: packed=" << r.report[0] << " final_sem=" << r.report[2]
                     << " mismatched_tiles=" << r.mismatched_tiles;
    EXPECT_EQ(r.report[0], num_tiles) << "PACK did not finish every tile";
    EXPECT_EQ(r.report[2], 0u) << "semaphore did not settle to 0 -- up/down unbalanced";
    EXPECT_EQ(r.mismatched_tiles, 0u) << "output != input: UNPACK read the ring before PACK's writes landed";
}

// Negative control: same kernel with wait_not_full/wait_min/down removed. UNPACK unpacks the ring slot
// immediately after the input tile, before PACK has written it, so the output must NOT equal the input. If this passes
// the positive test above is not a detector.
TEST_F(ProgramSpecHWTest, ComputeSemaphoreDatacopyNoSyncControl) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_tiles = 8;
    const auto r = RunComputeSemaphoreDatacopy(*mesh_device, num_tiles, /*nosync=*/1);
    GTEST_LOG_(INFO) << "datacopy no-sync control: packed=" << r.report[0] << " mismatched_tiles=" << r.mismatched_tiles
                     << " / " << num_tiles;
    EXPECT_EQ(r.report[0], num_tiles);
    EXPECT_GT(r.mismatched_tiles, 0u) << "unsynchronized datacopy came out correct -- the positive test cannot "
                                         "distinguish a working semaphore from no semaphore";
}

// PATTERN E: producer back-pressure. PACK gates every up(1) with wait_not_full() against a capacity of
// kSemDepth (the host's max_value); UNPACK consumes slowly and records the highest value it observes.
// The high-water mark must never exceed the capacity, and nothing may be lost.
TEST_F(ProgramSpecHWTest, ComputeSemaphoreBoundedProducer) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_iters = 1024;
    const auto r = RunComputeSemaphore(mesh_device, {.pattern = 4, .num_iters = num_iters, .max_value = kSemDepth}, 6);
    GTEST_LOG_(INFO) << "bounded producer: produced=" << r[0] << " consumed=" << r[1] << " high_water=" << r[2]
                     << " final=" << r[3] << (r[4] ? " [UNPACK TIMEOUT]" : "");
    EXPECT_EQ(r[4], 0u) << "UNPACK timed out -- a post was lost";
    EXPECT_EQ(r[1], num_iters) << "UNPACK did not consume every credit";
    EXPECT_LE(r[2], kSemDepth) << "producer ran past the capacity -- wait_not_full() did not hold it";
    EXPECT_GE(r[2], 2u) << "consumer was never behind by more than one credit -- the test did not exercise "
                           "back-pressure";
    EXPECT_EQ(r[3], 0u) << "semaphore did not settle to 0";
}

// Batched producer: wait_not_full(2); up(2) against the same capacity of kSemDepth. Takes the RISC-poll form
// of wait_not_full (no SEMWAIT condition for "room for n"); the high-water mark must still stay at the
// capacity. A wait that only reserved one slot would let up(2) reach kSemDepth + 1.
TEST_F(ProgramSpecHWTest, ComputeSemaphoreBoundedProducerBatched) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_iters = 1024;
    const auto r =
        RunComputeSemaphore(mesh_device, {.pattern = 4, .num_iters = num_iters, .batch = 2, .max_value = kSemDepth}, 6);
    GTEST_LOG_(INFO) << "bounded producer, batch 2: produced=" << r[0] << " consumed=" << r[1] << " high_water=" << r[2]
                     << " final=" << r[3] << (r[4] ? " [UNPACK TIMEOUT]" : "");
    EXPECT_EQ(r[4], 0u) << "UNPACK timed out -- a post was lost";
    EXPECT_EQ(r[1], num_iters) << "UNPACK did not consume every credit";
    EXPECT_LE(r[2], kSemDepth) << "batched producer ran past the capacity -- wait_not_full(2) reserved too little";
    EXPECT_GE(r[2], 2u) << "consumer was never behind -- the test did not exercise back-pressure";
    EXPECT_EQ(r[3], 0u) << "semaphore did not settle to 0";
}

// Negative control: same run without wait_not_full(). PACK races to the 15-credit hardware ceiling
// (observed high-water mark above the capacity) and the posts beyond it are dropped, so UNPACK cannot
// consume them all. If this passes the positive test above is not a detector.
TEST_F(ProgramSpecHWTest, ComputeSemaphoreBoundedProducerNoWaitControl) {
    auto mesh_device = devices_.at(0);
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole-only";
    }
    constexpr std::uint32_t num_iters = 64;  // small: the consumer's timeout is the expected exit
    const auto r = RunComputeSemaphore(
        mesh_device, {.pattern = 4, .num_iters = num_iters, .nosync = 1, .max_value = kSemDepth}, 6);
    GTEST_LOG_(INFO) << "bounded producer no-wait control: produced=" << r[0] << " consumed=" << r[1]
                     << " high_water=" << r[2] << (r[4] ? " [UNPACK TIMEOUT, expected]" : "");
    EXPECT_GT(r[2], kSemDepth) << "ungated producer never exceeded the capacity -- the positive test cannot "
                                  "tell wait_not_full() from nothing";
    EXPECT_LT(r[1], num_iters) << "every post survived without back-pressure -- saturation was not reached";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
