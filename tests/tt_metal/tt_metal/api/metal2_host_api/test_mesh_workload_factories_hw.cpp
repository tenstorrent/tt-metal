// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-hardware tests for Metal 2.0 MeshWorkload factory APIs (WH/BH).
//
// These tests prove the ProgramSpec -> MakeMeshWorkloadFromSpec(s) -> MeshWorkload enqueue -> verify
// pipeline end-to-end on real Wormhole B0 and Blackhole hardware.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "command_queue_fixture.hpp"
#include "device_fixture.hpp"
#include "multi_device_fixture.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

constexpr uint32_t kNumLaunches = 3;
constexpr uint32_t kScratchpadBytes = 64;       // 16 x uint32_t
constexpr uint32_t kReportAddr = 100 * 1024;    // host-known fixed L1 addr
constexpr uint32_t kPatternBase = 0xC0DE0000u;  // must match the kernel
constexpr uint32_t kDfbEntrySize = 1024;
constexpr uint32_t kTransfers = 8;
constexpr uint32_t kBufferBytes = kDfbEntrySize * kTransfers;
constexpr auto kScratchKernel = "writer";
constexpr auto kLoopbackDfb = "loopback_dfb";

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

class MeshWorkloadFactoryHWTest : public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        auto mesh_device = devices_.at(0);
        IDevice* device = mesh_device->get_devices()[0];
        if (device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

class MeshWorkloadFactorySlowDispatchHWTest : public MeshDeviceFixture {
protected:
    void SetUp() override {
        MeshDeviceFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        auto mesh_device = devices_.at(0);
        IDevice* device = mesh_device->get_devices()[0];
        if (device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

class MeshWorkloadFactory1x2HWTest : public MeshDevice1x2Fixture {
protected:
    void SetUp() override {
        MeshDevice1x2Fixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        IDevice* device = mesh_device_->get_devices().at(0);
        if (device->arch() != tt::ARCH::WORMHOLE_B0 && device->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: test requires Wormhole B0 or Blackhole hardware";
        }
    }
};

ProgramSpec MakeScratchpadSpec(uint32_t scratchpad_bytes = kScratchpadBytes) {
    const KernelSpecName kernel_id{kScratchKernel};
    auto dm_kernel = MakeMinimalGen1DMKernel(kScratchKernel, DataMovementProcessor::RISCV_0);
    dm_kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scratchpad_write_pattern.cpp";
    dm_kernel.runtime_arg_schema = {.runtime_arg_names = {"report_addr"}};
    dm_kernel.scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    return ProgramSpec{
        .name = "scratch",
        .kernels = {dm_kernel},
        .scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratchpad_bytes}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {kernel_id},
            .target_nodes = NodeCoord{0, 0},
        }},
    };
}

ProgramSpec MakeLoopbackSpec() {
    const NodeCoord node{0, 0};

    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 3;

    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 3;

    auto dfb = MakeMinimalDFB(kLoopbackDfb, kDfbEntrySize, /*num_entries=*/2);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{kLoopbackDfb}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{kLoopbackDfb}, "a_dfb_named_bob"));

    return ProgramSpec{
        .name = "loopback",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("main", node, {"producer", "consumer"})},
    };
}

void SetScratchpadArgs(Program& program) {
    ProgramRunArgs params;
    params.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{kScratchKernel},
        .runtime_arg_values = {{NodeCoord{0, 0}, {{"report_addr", kReportAddr}}}},
    }};
    SetProgramRunArgs(program, params);
}

void SetScratchpadArgs(distributed::MeshWorkload& workload) {
    SetScratchpadArgs(workload.get_programs().begin()->second);
}

std::vector<uint32_t> ExpectedScratchpadContents(uint32_t scratchpad_bytes) {
    std::vector<uint32_t> expected(scratchpad_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < expected.size(); i++) {
        expected[i] = kPatternBase + static_cast<uint32_t>(i);
    }
    return expected;
}

IDevice* DeviceAt(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const distributed::MeshCoordinate& coordinate) {
    return mesh_device->get_view().get_devices(distributed::MeshCoordinateRange(coordinate)).at(0);
}

void ClearScratchpadReport(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const distributed::MeshCoordinate& coordinate) {
    std::vector<uint32_t> report(1, 0u);
    detail::WriteToDeviceL1(DeviceAt(mesh_device, coordinate), NodeCoord{0, 0}, kReportAddr, report);
}

void CheckScratchpad(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const distributed::MeshCoordinate& coordinate,
    uint32_t scratchpad_bytes) {
    const NodeCoord node{0, 0};
    IDevice* device = DeviceAt(mesh_device, coordinate);

    std::vector<uint32_t> reported;
    detail::ReadFromDeviceL1(device, node, kReportAddr, sizeof(uint32_t), reported);
    ASSERT_EQ(reported.size(), 1u);
    const uint32_t scratch_base = reported[0];
    EXPECT_NE(scratch_base, 0u) << "Kernel reported a 0 scratchpad base address";

    std::vector<uint32_t> scratch_contents;
    detail::ReadFromDeviceL1(device, node, scratch_base, scratchpad_bytes, scratch_contents);
    EXPECT_EQ(scratch_contents, ExpectedScratchpadContents(scratchpad_bytes));
}

void EnqueueAndCheckScratchpad(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshWorkload& workload) {
    const auto coordinate = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    ClearScratchpadReport(mesh_device, coordinate);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    CheckScratchpad(mesh_device, coordinate, kScratchpadBytes);
}

void SetLoopbackArgs(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<Buffer>& input_buffer,
    const std::shared_ptr<Buffer>& output_buffer,
    uint32_t dfb_num_entries) {
    const NodeCoord node{0, 0};
    Program& program = workload.get_programs().begin()->second;

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              input_buffer->address(),
                              0u,  // bank_id (single-page buffer -> bank 0)
                              kTransfers,
                          }}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              output_buffer->address(),
                              0u,  // bank_id
                              kTransfers,
                          }}},
                },
        },
    };
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{kLoopbackDfb}, .num_entries = dfb_num_entries});
    SetProgramRunArgs(program, params);
}

TEST_F(MeshWorkloadFactoryHWTest, MakeMeshWorkloadFromSpecSupportsRepeatedEnqueue) {
    auto mesh_device = devices_.at(0);

    // Build a mesh-wide workload from one ProgramSpec.
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device, MakeScratchpadSpec());

    // Set runtime args once, then repeatedly enqueue and verify scratchpad access.
    SetScratchpadArgs(workload);
    for (uint32_t launch = 0; launch < kNumLaunches; launch++) {
        EnqueueAndCheckScratchpad(mesh_device, workload);
    }
}

TEST_F(MeshWorkloadFactorySlowDispatchHWTest, FactoryMethodsSupportSlowDispatch) {
    auto mesh_device = devices_.at(0);

    // Exercise the single-spec factory through slow dispatch.
    distributed::MeshWorkload single_workload = MakeMeshWorkloadFromSpec(*mesh_device, MakeScratchpadSpec());
    SetScratchpadArgs(single_workload);
    EnqueueAndCheckScratchpad(mesh_device, single_workload);

    // Exercise the mapped-spec factory through slow dispatch.
    std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec> specs;
    specs.emplace(distributed::MeshCoordinateRange(mesh_device->shape()), MakeScratchpadSpec());
    distributed::MeshWorkload mapped_workload = MakeMeshWorkloadFromSpecs(*mesh_device, specs);
    SetScratchpadArgs(mapped_workload);
    EnqueueAndCheckScratchpad(mesh_device, mapped_workload);
}

TEST_F(MeshWorkloadFactoryHWTest, MakeMeshWorkloadFromSpecSupportsDfbResizeBetweenEnqueues) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // Build a workload with a producer-consumer DFB loopback.
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device, MakeLoopbackSpec());

    InterleavedBufferConfig dram_config{
        .device = device, .size = kBufferBytes, .page_size = kBufferBytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    // Resize the DFB between enqueues and verify each loopback result.
    for (uint32_t dfb_num_entries : {2u, 4u, 6u}) {
        std::vector<uint32_t> input_data(kBufferBytes / sizeof(uint32_t));
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = (dfb_num_entries << 16) + static_cast<uint32_t>(i);
        }
        std::vector<uint32_t> output_data(input_data.size());

        detail::WriteToBuffer(input_buffer, input_data);
        detail::WriteToBuffer(output_buffer, output_data);
        SetLoopbackArgs(workload, input_buffer, output_buffer, dfb_num_entries);

        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

        output_data.clear();
        detail::ReadFromBuffer(output_buffer, output_data);
        EXPECT_EQ(output_data, input_data);
    }
}

TEST_F(MeshWorkloadFactoryHWTest, MakeMeshWorkloadFromSpecsMapOverloadSupportsRepeatedEnqueue) {
    auto mesh_device = devices_.at(0);

    // Build a region-mapped workload through the ProgramSpec map overload.
    std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec> specs;
    specs.emplace(distributed::MeshCoordinateRange(mesh_device->shape()), MakeScratchpadSpec());
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpecs(*mesh_device, specs);
    ASSERT_EQ(workload.get_programs().size(), 1u);

    // Set runtime args once, then repeatedly enqueue and verify scratchpad access.
    SetScratchpadArgs(workload);
    for (uint32_t launch = 0; launch < kNumLaunches; launch++) {
        EnqueueAndCheckScratchpad(mesh_device, workload);
    }
}

TEST_F(MeshWorkloadFactory1x2HWTest, MakeMeshWorkloadFromSpecsRunsDistinctSpecsOnMappedDevices) {
    constexpr uint32_t kLargeScratchpadBytes = 2 * kScratchpadBytes;
    const distributed::MeshCoordinate first{0, 0};
    const distributed::MeshCoordinate second{0, 1};

    // Map ProgramSpecs with observably different scratchpad sizes to separate devices.
    std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec> specs;
    specs.emplace(distributed::MeshCoordinateRange(first), MakeScratchpadSpec(kScratchpadBytes));
    specs.emplace(distributed::MeshCoordinateRange(second), MakeScratchpadSpec(kLargeScratchpadBytes));
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpecs(*mesh_device_, specs);
    ASSERT_EQ(workload.get_programs().size(), 2u);

    // Configure both programs, enqueue once, then verify each device ran its mapped spec.
    for (auto& [_, program] : workload.get_programs()) {
        SetScratchpadArgs(program);
    }
    ClearScratchpadReport(mesh_device_, first);
    ClearScratchpadReport(mesh_device_, second);
    distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);

    CheckScratchpad(mesh_device_, first, kScratchpadBytes);
    CheckScratchpad(mesh_device_, second, kLargeScratchpadBytes);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
