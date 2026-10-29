// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>

#include "command_queue_fixture.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal::experimental {
namespace test_helpers {

constexpr uint32_t dfb_report_address = 100 * 1024;
constexpr uint32_t dfb_transfers = 8;
constexpr uint32_t dfb_report_words = 2 + dfb_transfers;
constexpr uint32_t dfb_storage_bytes = 64;
constexpr uint32_t dfb_sentinel = 0xDEADBEEF;

ProgramSpec MakeGuardedDFBTransferSpecForTest() {
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_address));
    if (!get_arg(args::enabled)) {
        report[0] = 2;
        return;
    }

    DataflowBuffer buffer(dfb::buffer);
    for (uint32_t entry = 0; entry < get_arg(args::transfers); ++entry) {
        buffer.reserve_back(1);
        auto* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer.get_write_ptr());
        *value = get_arg(args::pattern) + entry;
        buffer.push_back(1);
    }
    report[0] = 1;
}
)"};
    producer.runtime_arg_schema.runtime_arg_names = {"report_address", "enabled", "pattern", "transfers"};
    producer.dfb_bindings = {ProducerOf(DFBSpecName{"buffer"}, "buffer")};
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_address));
    if (!get_arg(args::enabled)) {
        report[1] = 2;
        return;
    }

    DataflowBuffer buffer(dfb::buffer);
    for (uint32_t entry = 0; entry < get_arg(args::transfers); ++entry) {
        buffer.wait_front(1);
        auto* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer.get_read_ptr());
        report[2 + entry] = *value;
        buffer.pop_front(1);
    }
    report[1] = 1;
}
)"};
    consumer.runtime_arg_schema.runtime_arg_names = {"report_address", "enabled", "transfers"};
    consumer.dfb_bindings = {ConsumerOf(DFBSpecName{"buffer"}, "buffer")};
    return ProgramSpec{
        .kernels = {producer, consumer},
        .dataflow_buffers = {MakeMinimalDFB("buffer", dfb_storage_bytes / 2, 2)},
        .work_units = {MakeMinimalWorkUnit("transfer", NodeRange({0, 0}, {1, 0}), {"producer", "consumer"})}};
}

ProgramRunArgs MakeGuardedDFBTransferArgsForTest(uint32_t pattern) {
    ProgramRunArgs::KernelRunArgs producer{.kernel = KernelSpecName{"producer"}};
    ProgramRunArgs::KernelRunArgs consumer{.kernel = KernelSpecName{"consumer"}};
    for (uint32_t x = 0; x < 2; ++x) {
        AddRuntimeArgsForNode(
            producer.runtime_arg_values,
            {x, 0},
            {{"report_address", dfb_report_address},
             {"enabled", x == 0},
             {"pattern", pattern},
             {"transfers", dfb_transfers}});
        AddRuntimeArgsForNode(
            consumer.runtime_arg_values,
            {x, 0},
            {{"report_address", dfb_report_address}, {"enabled", x == 0}, {"transfers", dfb_transfers}});
    }
    ProgramRunArgs args;
    args.kernel_run_args = {producer, consumer};
    return args;
}

void ClearDFBReportsForTest(distributed::MeshDevice& device) {
    std::vector<uint32_t> zeros(dfb_report_words, 0);
    for (uint32_t x = 0; x < 3; ++x) {
        slow_dispatch::WriteToL1(device, {x, 0}, dfb_report_address, zeros);
    }
}

std::vector<uint32_t> ReadDFBReportForTest(distributed::MeshDevice& device, CoreCoord node) {
    std::vector<uint32_t> report;
    slow_dispatch::ReadFromL1(device, node, dfb_report_address, dfb_report_words * sizeof(uint32_t), report);
    return report;
}

std::vector<uint32_t> ExpectedDFBReportForTest(uint32_t pattern, bool enabled = true) {
    std::vector<uint32_t> report(dfb_report_words, 0);
    report[0] = report[1] = enabled ? 1 : 2;
    if (enabled) {
        std::iota(report.begin() + 2, report.end(), pattern);
    }
    return report;
}

std::vector<std::vector<uint32_t>> ExpectedDFBReportsForTest(const std::vector<uint32_t>& patterns) {
    std::vector<std::vector<uint32_t>> reports;
    std::ranges::transform(
        patterns, std::back_inserter(reports), [](uint32_t pattern) { return ExpectedDFBReportForTest(pattern); });
    return reports;
}

uint32_t DFBStorageAddressForTest(const Program& program) {
    return program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("buffer"))->uniform_alloc_addr();
}

void FillExtraDFBStorageForTest(distributed::MeshDevice& device, const Program& program) {
    std::vector<uint32_t> sentinel(dfb_storage_bytes / sizeof(uint32_t), dfb_sentinel);
    slow_dispatch::WriteToL1(device, {2, 0}, DFBStorageAddressForTest(program), sentinel);
}

std::vector<uint32_t> ReadExtraDFBStorageForTest(distributed::MeshDevice& device, const Program& program) {
    std::vector<uint32_t> storage;
    slow_dispatch::ReadFromL1(device, {2, 0}, DFBStorageAddressForTest(program), dfb_storage_bytes, storage);
    return storage;
}

}  // namespace test_helpers
namespace {

using test_helpers::ExpectedDFBReportForTest;
using test_helpers::ExpectedDFBReportsForTest;
using test_helpers::FillExtraDFBStorageForTest;
using test_helpers::MakeGuardedDFBTransferArgsForTest;
using test_helpers::MakeGuardedDFBTransferSpecForTest;
using test_helpers::ReadDFBReportForTest;
using test_helpers::ReadExtraDFBStorageForTest;

template <typename DispatchFixture>
class DFBPlacementHardwareTest : public DispatchFixture {
protected:
    distributed::MeshDevice& fixture_device() { return *this->devices_.at(0); }

    void SetUp() override {
        DispatchFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        if (fixture_device().arch() != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Requires Wormhole hardware";
        }
        if (fixture_device().compute_with_storage_grid_size().x < 3) {
            GTEST_SKIP() << "Requires three worker columns";
        }
        test_helpers::ClearDFBReportsForTest(fixture_device());
    }

    void EnqueueFromTestFixture(distributed::MeshWorkload& workload) {
        if constexpr (std::is_same_v<DispatchFixture, test_helpers::ProgramSpecHWTest>) {
            auto& [range, program] = *workload.get_programs().begin();
            auto completed = LaunchProgram(fixture_device(), std::move(program));
            program = std::move(completed.get_programs().at(range));
        } else {
            distributed::EnqueueMeshWorkload(fixture_device().mesh_command_queue(), workload, true);
        }
    }
};

struct DFBDispatchNames {
    template <typename DispatchFixture>
    static std::string GetName(int) {
        return std::is_same_v<DispatchFixture, test_helpers::ProgramSpecHWTest> ? "SlowDispatch" : "FastDispatch";
    }
};

using DFBDispatchFixtures = ::testing::Types<test_helpers::ProgramSpecHWTest, UnitMeshCQSingleCardFixture>;
TYPED_TEST_SUITE(DFBPlacementHardwareTest, DFBDispatchFixtures, DFBDispatchNames);

TYPED_TEST(DFBPlacementHardwareTest, GuardedAccessSkipsExcludedNode) {
    // Endpoint kernels transfer data on a reserved node and skip DFB access on an excluded node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = MakeGuardedDFBTransferSpecForTest();
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(std::vector<CoreCoord>{{0, 0}, {2, 0}});
    auto workload = MakeMeshWorkloadFromSpec(fixture_device, spec);
    Program& program = workload.get_programs().begin()->second;
    SetProgramRunArgs(program, MakeGuardedDFBTransferArgsForTest(0x100));
    this->EnqueueFromTestFixture(workload);
    EXPECT_EQ(ReadDFBReportForTest(fixture_device, {0, 0}), ExpectedDFBReportForTest(0x100));
    EXPECT_EQ(ReadDFBReportForTest(fixture_device, {1, 0}), ExpectedDFBReportForTest(0, false));
}

TYPED_TEST(DFBPlacementHardwareTest, ReservationOnlyNodeRemainsUntouched) {
    // A node with only a DFB reservation keeps its backing storage and kernel report unchanged.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = MakeGuardedDFBTransferSpecForTest();
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(std::vector<CoreCoord>{{0, 0}, {2, 0}});
    auto workload = MakeMeshWorkloadFromSpec(fixture_device, spec);
    Program& program = workload.get_programs().begin()->second;
    SetProgramRunArgs(program, MakeGuardedDFBTransferArgsForTest(0x100));
    FillExtraDFBStorageForTest(fixture_device, program);
    this->EnqueueFromTestFixture(workload);
    EXPECT_EQ(ReadExtraDFBStorageForTest(fixture_device, program), std::vector<uint32_t>(16, 0xDEADBEEF));
    EXPECT_EQ(ReadDFBReportForTest(fixture_device, {2, 0}), std::vector<uint32_t>(10, 0));
}

TYPED_TEST(DFBPlacementHardwareTest, RepeatedEnqueueTransfersNewData) {
    // Repeated enqueue transfers each new pattern through the same DFB placement override.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = MakeGuardedDFBTransferSpecForTest();
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(std::vector<CoreCoord>{{0, 0}, {2, 0}});
    auto workload = MakeMeshWorkloadFromSpec(fixture_device, spec);
    Program& program = workload.get_programs().begin()->second;
    std::vector<std::vector<uint32_t>> reports;
    for (uint32_t pattern : {0x100u, 0x200u, 0x300u}) {
        SetProgramRunArgs(program, MakeGuardedDFBTransferArgsForTest(pattern));
        this->EnqueueFromTestFixture(workload);
        reports.push_back(ReadDFBReportForTest(fixture_device, {0, 0}));
    }
    EXPECT_EQ(reports, ExpectedDFBReportsForTest({0x100, 0x200, 0x300}));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
