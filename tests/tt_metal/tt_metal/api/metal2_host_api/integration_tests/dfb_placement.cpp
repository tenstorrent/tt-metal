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
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/host_api.hpp>

#include "command_queue_fixture.hpp"
#include "device_fixture.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal::test_helpers {

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

ProgramSpec MakeDifferentDFBMetadataSpecForTest() {
    auto spec = MakeGuardedDFBTransferSpecForTest();
    const KernelSpec::SourceCode source{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <typename Binding>
void transfer(Binding binding) {
    DataflowBuffer buffer(binding);
    auto* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_address));
#ifdef PRODUCER
    report[0] = static_cast<uint32_t>(buffer.get_dataformat());
#else
    report[1] = buffer.get_tile_size();
#endif
    for (uint32_t entry = 0; entry < get_arg(args::transfers); ++entry) {
#ifdef PRODUCER
        buffer.reserve_back(1);
        auto* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer.get_write_ptr());
        *value = get_arg(args::pattern) + entry;
        buffer.push_back(1);
#else
        buffer.wait_front(1);
        auto* value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer.get_read_ptr());
        report[2 + entry] = *value;
        buffer.pop_front(1);
#endif
    }
}

void kernel_main() {
    if (get_arg(args::enabled)) {
        transfer(dfb::left);
    } else {
        transfer(dfb::right);
    }
}
)"};
    spec.dataflow_buffers = {MakeMinimalDFB("left", 1024, 2), MakeMinimalDFB("right", 4096, 2)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].tile_format_metadata = Tile({16, 32});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    spec.dataflow_buffers[1].data_format_metadata = tt::DataFormat::Float32;
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    spec.kernels[0].source = spec.kernels[1].source = source;
    spec.kernels[0].compiler_options.defines["PRODUCER"] = "1";
    spec.kernels[0].dfb_bindings = {ProducerOf(DFBSpecName{"left"}, "left"), ProducerOf(DFBSpecName{"right"}, "right")};
    spec.kernels[1].dfb_bindings = {ConsumerOf(DFBSpecName{"left"}, "left"), ConsumerOf(DFBSpecName{"right"}, "right")};
    return spec;
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
    for (uint32_t x = 0; x < 2; ++x) {
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

std::vector<uint32_t> ExpectedDFBMetadataReportForTest(uint32_t pattern, tt::DataFormat format, uint32_t tile_size) {
    auto report = ExpectedDFBReportForTest(pattern);
    report[0] = static_cast<uint32_t>(format);
    report[1] = tile_size;
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

void FillExcludedDFBStorageForTest(distributed::MeshDevice& device, const Program& program) {
    std::vector<uint32_t> sentinel(dfb_storage_bytes / sizeof(uint32_t), dfb_sentinel);
    slow_dispatch::WriteToL1(device, {1, 0}, DFBStorageAddressForTest(program), sentinel);
}

std::vector<uint32_t> ReadExcludedDFBStorageForTest(distributed::MeshDevice& device, const Program& program) {
    std::vector<uint32_t> storage;
    slow_dispatch::ReadFromL1(device, {1, 0}, DFBStorageAddressForTest(program), dfb_storage_bytes, storage);
    return storage;
}

ProgramSpec MakeQuasarDFBStartupSpecForTest(bool include_consumer_on_excluded_node = true) {
    auto producer = MakeMinimalGen2DMKernel("producer");
    producer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto report = Scratchpad<uint32_t>(scratch::report);
    if (get_arg(args::enabled)) {
        DataflowBuffer buffer(dfb::buffer);
        report[0] = buffer.get_entry_size();
    } else {
        report[0] = 2;
    }
}
)"};
    producer.runtime_arg_schema.runtime_arg_names = {"enabled"};
    producer.dfb_bindings = {ProducerOf(DFBSpecName{"buffer"}, "buffer")};
    producer.scratchpad_bindings = {{ScratchpadSpecName{"report"}, "report"}};
    auto consumer = MakeMinimalGen2ComputeKernel("consumer", 2);
    consumer.source = KernelSpec::SourceCode{"#include \"api/compute/common.h\"\nvoid kernel_main() {}"};
    consumer.dfb_bindings = {AllConsumerOf(DFBSpecName{"buffer"}, "buffer")};
    auto buffer = MakeMinimalDFB("buffer", 2048, 2);
    buffer.data_format_metadata = tt::DataFormat::Float16_b;
    ProgramSpec spec{
        .kernels = {producer, consumer},
        .dataflow_buffers = {buffer},
        .scratchpads = {{.unique_id = ScratchpadSpecName{"report"}, .size_per_node = 64}},
        .work_units = {MakeMinimalWorkUnit("included", NodeCoord{0, 0}, {"producer", "consumer"})}};
    const std::vector<std::string> excluded_kernels = include_consumer_on_excluded_node
                                                          ? std::vector<std::string>{"producer", "consumer"}
                                                          : std::vector<std::string>{"producer"};
    spec.work_units.push_back(MakeMinimalWorkUnit("excluded", NodeCoord{1, 0}, excluded_kernels));
    return spec;
}

ProgramRunArgs MakeQuasarDFBStartupArgsForTest() {
    ProgramRunArgs::KernelRunArgs producer{.kernel = KernelSpecName{"producer"}};
    AddRuntimeArgsForNode(producer.runtime_arg_values, {0, 0}, {{"enabled", 1}});
    AddRuntimeArgsForNode(producer.runtime_arg_values, {1, 0}, {{"enabled", 0}});
    return ProgramRunArgs{.kernel_run_args = {producer}};
}

uint32_t DFBStartupReportAddressForTest(const Program& program) {
    return program.impl().get_kernel_by_spec_name("producer")->scratchpad_binding_handles()[0].allocated_address;
}

void ClearDFBStartupReportsForTest(distributed::MeshDevice& device, const Program& program) {
    std::vector<uint32_t> zeros{0};
    for (uint32_t x = 0; x < 2; ++x) {
        slow_dispatch::WriteToL1(device, {x, 0}, DFBStartupReportAddressForTest(program), zeros);
    }
}

std::vector<uint32_t> ReadDFBStartupReportsForTest(distributed::MeshDevice& device, const Program& program) {
    std::vector<uint32_t> reports;
    for (uint32_t x = 0; x < 2; ++x) {
        std::vector<uint32_t> report;
        slow_dispatch::ReadFromL1(device, {x, 0}, DFBStartupReportAddressForTest(program), sizeof(uint32_t), report);
        reports.push_back(report.at(0));
    }
    return reports;
}

}  // namespace tt::tt_metal::test_helpers

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::ExpectedDFBReportForTest;
using test_helpers::ExpectedDFBReportsForTest;
using test_helpers::FillExcludedDFBStorageForTest;
using test_helpers::MakeGuardedDFBTransferArgsForTest;
using test_helpers::MakeGuardedDFBTransferSpecForTest;
using test_helpers::ReadDFBReportForTest;
using test_helpers::ReadExcludedDFBStorageForTest;

template <typename DispatchFixture>
class DFBPlacementHardwareTest : public DispatchFixture {
protected:
    distributed::MeshDevice& fixture_device() { return *this->devices_.at(0); }

    void SetUp() override {
        DispatchFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        // These tests use Gen1 kernels; Quasar coverage is in QuasarDFBPlacementHardwareTest.
        if (fixture_device().arch() != tt::ARCH::WORMHOLE_B0 && fixture_device().arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Requires Gen1 (Wormhole or Blackhole) hardware";
        }
        if (fixture_device().compute_with_storage_grid_size().x < 2) {
            GTEST_SKIP() << "Requires two worker columns";
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
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto workload = MakeMeshWorkloadFromSpec(fixture_device, spec);
    Program& program = workload.get_programs().begin()->second;
    SetProgramRunArgs(program, MakeGuardedDFBTransferArgsForTest(0x100));
    this->EnqueueFromTestFixture(workload);
    EXPECT_EQ(ReadDFBReportForTest(fixture_device, {0, 0}), ExpectedDFBReportForTest(0x100));
    EXPECT_EQ(ReadDFBReportForTest(fixture_device, {1, 0}), ExpectedDFBReportForTest(0, false));
}

TYPED_TEST(DFBPlacementHardwareTest, DisjointDFBsUseDifferentMetadata) {
    // Shared kernels transfer data with each node's selected DFB format and tile size.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto workload = MakeMeshWorkloadFromSpec(fixture_device, test_helpers::MakeDifferentDFBMetadataSpecForTest());
    Program& program = workload.get_programs().begin()->second;
    SetProgramRunArgs(program, MakeGuardedDFBTransferArgsForTest(0x300));
    this->EnqueueFromTestFixture(workload);
    EXPECT_EQ(
        ReadDFBReportForTest(fixture_device, {0, 0}),
        test_helpers::ExpectedDFBMetadataReportForTest(0x300, tt::DataFormat::Float16_b, 1024));
    EXPECT_EQ(
        ReadDFBReportForTest(fixture_device, {1, 0}),
        test_helpers::ExpectedDFBMetadataReportForTest(0x300, tt::DataFormat::Float32, 4096));
}

TYPED_TEST(DFBPlacementHardwareTest, ExcludedNodeStorageRemainsUntouched) {
    // A kernel on an excluded node leaves memory at the DFB address unchanged.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = MakeGuardedDFBTransferSpecForTest();
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto workload = MakeMeshWorkloadFromSpec(fixture_device, spec);
    Program& program = workload.get_programs().begin()->second;
    SetProgramRunArgs(program, MakeGuardedDFBTransferArgsForTest(0x100));
    FillExcludedDFBStorageForTest(fixture_device, program);
    this->EnqueueFromTestFixture(workload);
    EXPECT_EQ(ReadExcludedDFBStorageForTest(fixture_device, program), std::vector<uint32_t>(16, 0xDEADBEEF));
}

TYPED_TEST(DFBPlacementHardwareTest, RepeatedEnqueueTransfersNewData) {
    // Repeated enqueue transfers each new pattern through the same DFB placement override.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = MakeGuardedDFBTransferSpecForTest();
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
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

class QuasarDFBPlacementHardwareTest : public QuasarMeshDeviceSingleCardFixture {
protected:
    distributed::MeshDevice& fixture_device() { return *this->devices_.at(0); }

    void SetUp() override {
        QuasarMeshDeviceSingleCardFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        if (fixture_device().compute_with_storage_grid_size().x < 2) {
            GTEST_SKIP() << "Requires two worker columns";
        }
    }
};

TEST_F(QuasarDFBPlacementHardwareTest, RemapperStartupCompletesOnExcludedNode) {
    // Kernels start on both nodes when a DFB that needs the remapper reserves storage on one node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeQuasarDFBStartupSpecForTest();
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::ClearDFBStartupReportsForTest(fixture_device, program);
    SetProgramRunArgs(program, test_helpers::MakeQuasarDFBStartupArgsForTest());
    auto completed = LaunchProgram(fixture_device, std::move(program));
    EXPECT_EQ(
        test_helpers::ReadDFBStartupReportsForTest(fixture_device, completed.get_programs().begin()->second),
        (std::vector<uint32_t>{2048, 2}));
}

TEST_F(QuasarDFBPlacementHardwareTest, RemapperStartupCompletesWithoutExcludedConsumer) {
    // An excluded producer node starts without a consumer kernel when it skips DFB interaction.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeQuasarDFBStartupSpecForTest(/*include_consumer_on_excluded_node=*/false);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::ClearDFBStartupReportsForTest(fixture_device, program);
    SetProgramRunArgs(program, test_helpers::MakeQuasarDFBStartupArgsForTest());
    auto completed = LaunchProgram(fixture_device, std::move(program));
    EXPECT_EQ(
        test_helpers::ReadDFBStartupReportsForTest(fixture_device, completed.get_programs().begin()->second),
        (std::vector<uint32_t>{2048, 2}));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
