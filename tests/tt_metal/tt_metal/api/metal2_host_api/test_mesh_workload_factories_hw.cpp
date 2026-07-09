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
#include <string>
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
#include "test_helpers.hpp"

namespace tt::tt_metal {
namespace {

constexpr uint32_t kNumLaunches = 3;
constexpr uint32_t kScratchpadBytes = 64;  // 16 x uint32_t
constexpr uint32_t kNumScratchpadElems = kScratchpadBytes / sizeof(uint32_t);
constexpr uint32_t kReportAddr = 100 * 1024;    // host-known fixed L1 addr
constexpr uint32_t kPatternBase = 0xC0DE0000u;  // must match the kernel
constexpr uint32_t kDfbEntrySize = 1024;
constexpr uint32_t kDfbNumTransfers = 8;
constexpr uint32_t kDfbTotalBytes = kDfbEntrySize * kDfbNumTransfers;

using experimental::test_helpers::MakeMinimalDFB;
using experimental::test_helpers::MakeMinimalGen1DMKernel;
using experimental::test_helpers::MakeMinimalWorkUnit;

std::vector<distributed::MeshCoordinate> GetLocalMeshCoordinates(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    std::vector<distributed::MeshCoordinate> coords;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->get_view().get_devices(distributed::MeshCoordinateRange(coord)).empty()) {
            coords.push_back(coord);
        }
    }
    return coords;
}

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

experimental::ProgramSpec MakeScratchpadSpec(const std::string& spec_name, const std::string& kernel_name) {
    const experimental::KernelSpecName kernel_id{kernel_name};
    experimental::KernelSpec dm_kernel{
        .unique_id = kernel_id,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/scratchpad_write_pattern.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"report_addr"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = DataMovementProcessor::RISCV_0,
                    },
            },
    };
    dm_kernel.scratchpad_bindings.push_back(experimental::KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = experimental::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    return experimental::ProgramSpec{
        .name = spec_name,
        .kernels = {dm_kernel},
        .scratchpads = {experimental::ScratchpadSpec{
            .unique_id = experimental::ScratchpadSpecName{"pad"}, .size_per_node = kScratchpadBytes}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {kernel_id},
            .target_nodes = experimental::NodeCoord{0, 0},
        }},
    };
}

experimental::ProgramSpec MakeDfbLoopbackSpec() {
    const experimental::NodeCoord node{0, 0};

    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 3;

    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 3;

    auto dfb = MakeMinimalDFB("loopback_dfb", kDfbEntrySize, /*num_entries=*/2);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(
        experimental::ProducerOf(experimental::DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(
        experimental::ConsumerOf(experimental::DFBSpecName{"loopback_dfb"}, "a_dfb_named_bob"));

    return experimental::ProgramSpec{
        .name = "factory_dfb_resize_loopback",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .work_units =
            std::vector<experimental::WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})},
    };
}

void SetScratchpadArgs(Program& program, const std::string& kernel_name) {
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = experimental::KernelSpecName{kernel_name},
        .runtime_arg_values = {{experimental::NodeCoord{0, 0}, {{"report_addr", kReportAddr}}}},
    }};
    experimental::SetProgramRunArgs(program, params);
}

void SetScratchpadArgs(distributed::MeshWorkload& workload, const std::string& kernel_name) {
    Program& program = workload.get_programs().begin()->second;
    SetScratchpadArgs(program, kernel_name);
}

void EnqueueAndVerifyScratchpadResult(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshWorkload& workload,
    const std::vector<distributed::MeshCoordinate>& coords) {
    const experimental::NodeCoord node{0, 0};
    for (const auto& coord : coords) {
        IDevice* device = mesh_device->get_view().get_devices(distributed::MeshCoordinateRange(coord)).at(0);
        std::vector<uint32_t> zero_report(1, 0u);
        detail::WriteToDeviceL1(device, node, kReportAddr, zero_report);
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    for (const auto& coord : coords) {
        IDevice* device = mesh_device->get_view().get_devices(distributed::MeshCoordinateRange(coord)).at(0);
        std::vector<uint32_t> reported;
        detail::ReadFromDeviceL1(device, node, kReportAddr, sizeof(uint32_t), reported);
        ASSERT_EQ(reported.size(), 1u);
        const uint32_t scratch_base = reported[0];
        EXPECT_NE(scratch_base, 0u) << "Kernel reported a 0 scratchpad base address";

        std::vector<uint32_t> scratch_contents;
        detail::ReadFromDeviceL1(device, node, scratch_base, kScratchpadBytes, scratch_contents);
        ASSERT_EQ(scratch_contents.size(), kNumScratchpadElems);

        std::vector<uint32_t> expected(kNumScratchpadElems);
        for (uint32_t i = 0; i < kNumScratchpadElems; i++) {
            expected[i] = kPatternBase + i;
        }
        EXPECT_EQ(scratch_contents, expected);
    }
}

void VerifyScratchpadResult(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshWorkload& workload) {
    EnqueueAndVerifyScratchpadResult(mesh_device, workload, {distributed::MeshCoordinate(0, 0)});
}

void SetDfbLoopbackArgs(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<Buffer>& input_buffer,
    const std::shared_ptr<Buffer>& output_buffer,
    uint32_t dfb_num_entries) {
    const experimental::NodeCoord node{0, 0};
    Program& program = workload.get_programs().begin()->second;

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"producer"},
            .advanced_options =
                experimental::AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              input_buffer->address(),
                              0u,  // bank_id (single-page buffer -> bank 0)
                              kDfbNumTransfers,
                          }}},
                },
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"consumer"},
            .advanced_options =
                experimental::AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              output_buffer->address(),
                              0u,  // bank_id
                              kDfbNumTransfers,
                          }}},
                },
        },
    };
    params.dfb_run_overrides.push_back(
        {.dfb = experimental::DFBSpecName{"loopback_dfb"}, .num_entries = dfb_num_entries});
    experimental::SetProgramRunArgs(program, params);
}

TEST_F(MeshWorkloadFactoryHWTest, MakeMeshWorkloadFromSpecUsesScratchpad) {
    auto mesh_device = devices_.at(0);

    // -------------------------------------------------------
    // Build one scratchpad ProgramSpec and create a mesh-wide workload.
    // -------------------------------------------------------
    constexpr auto kernel_name = "factory_single_kernel";

    distributed::MeshWorkload workload =
        experimental::MakeMeshWorkloadFromSpec(*mesh_device, MakeScratchpadSpec("factory_single", kernel_name));
    ASSERT_EQ(workload.get_programs().size(), 1u);

    // -------------------------------------------------------
    // Repeatedly set runtime args, dispatch, and verify scratchpad access.
    // -------------------------------------------------------
    for (uint32_t launch = 0; launch < kNumLaunches; launch++) {
        SetScratchpadArgs(workload, kernel_name);
        VerifyScratchpadResult(mesh_device, workload);
    }
}

TEST_F(MeshWorkloadFactorySlowDispatchHWTest, FactoryMethodsUseScratchpad) {
    auto mesh_device = devices_.at(0);

    // -------------------------------------------------------
    // Build one workload with MakeMeshWorkloadFromSpec.
    // -------------------------------------------------------
    constexpr auto single_kernel_name = "factory_slow_single_kernel";
    distributed::MeshWorkload single_workload = experimental::MakeMeshWorkloadFromSpec(
        *mesh_device, MakeScratchpadSpec("factory_slow_single", single_kernel_name));
    ASSERT_EQ(single_workload.get_programs().size(), 1u);

    // -------------------------------------------------------
    // Set runtime args, dispatch, and verify scratchpad access.
    // -------------------------------------------------------
    SetScratchpadArgs(single_workload, single_kernel_name);
    VerifyScratchpadResult(mesh_device, single_workload);

    // -------------------------------------------------------
    // Build one workload with MakeMeshWorkloadFromSpecs.
    // -------------------------------------------------------
    constexpr auto map_kernel_name = "factory_slow_map_kernel";
    std::unordered_map<distributed::MeshCoordinateRange, experimental::ProgramSpec> program_specs;
    program_specs.emplace(
        distributed::MeshCoordinateRange(mesh_device->shape()),
        MakeScratchpadSpec("factory_slow_map", map_kernel_name));
    distributed::MeshWorkload map_workload = experimental::MakeMeshWorkloadFromSpecs(*mesh_device, program_specs);
    ASSERT_EQ(map_workload.get_programs().size(), 1u);

    // -------------------------------------------------------
    // Set runtime args, dispatch, and verify scratchpad access.
    // -------------------------------------------------------
    SetScratchpadArgs(map_workload, map_kernel_name);
    VerifyScratchpadResult(mesh_device, map_workload);
}

TEST_F(MeshWorkloadFactoryHWTest, AddProgramAndCompileAfterInitialProgramBeforeEnqueue) {
    auto mesh_device = devices_.at(0);
    const std::vector<distributed::MeshCoordinate> coords = GetLocalMeshCoordinates(mesh_device);
    if (coords.size() < 2) {
        GTEST_SKIP() << "Skipping: test requires at least two local mesh devices";
    }

    // -------------------------------------------------------
    // Create an initial factory workload on the first device.
    // -------------------------------------------------------
    constexpr auto first_kernel_name = "manual_first_compile_kernel";
    const distributed::MeshCoordinateRange first_range(coords[0]);
    std::unordered_map<distributed::MeshCoordinateRange, experimental::ProgramSpec> program_specs;
    program_specs.emplace(first_range, MakeScratchpadSpec("manual_first_compile", first_kernel_name));
    distributed::MeshWorkload workload = experimental::MakeMeshWorkloadFromSpecs(*mesh_device, program_specs);

    // -------------------------------------------------------
    // Add a second eagerly compiled program before first enqueue.
    // -------------------------------------------------------
    constexpr auto second_kernel_name = "manual_second_compile_kernel";
    const distributed::MeshCoordinateRange second_range(coords[1]);
    workload.add_program_and_compile(
        second_range,
        experimental::MakeProgramFromSpec(
            *mesh_device, MakeScratchpadSpec("manual_second_compile", second_kernel_name)),
        *mesh_device);
    ASSERT_EQ(workload.get_programs().size(), 2u);

    // -------------------------------------------------------
    // Set runtime args for both programs, dispatch, and verify both devices.
    // -------------------------------------------------------
    SetScratchpadArgs(workload.get_programs().at(first_range), first_kernel_name);
    SetScratchpadArgs(workload.get_programs().at(second_range), second_kernel_name);
    EnqueueAndVerifyScratchpadResult(mesh_device, workload, {coords[0], coords[1]});
}

TEST_F(MeshWorkloadFactoryHWTest, AddProgramAfterInitialProgramBeforeEnqueue) {
    auto mesh_device = devices_.at(0);
    const std::vector<distributed::MeshCoordinate> coords = GetLocalMeshCoordinates(mesh_device);
    if (coords.size() < 2) {
        GTEST_SKIP() << "Skipping: test requires at least two local mesh devices";
    }

    // -------------------------------------------------------
    // Create an initial factory workload on the first device.
    // -------------------------------------------------------
    constexpr auto first_kernel_name = "manual_first_plain_kernel";
    const distributed::MeshCoordinateRange first_range(coords[0]);
    std::unordered_map<distributed::MeshCoordinateRange, experimental::ProgramSpec> program_specs;
    program_specs.emplace(first_range, MakeScratchpadSpec("manual_first_plain", first_kernel_name));
    distributed::MeshWorkload workload = experimental::MakeMeshWorkloadFromSpecs(*mesh_device, program_specs);

    // -------------------------------------------------------
    // Add a second plain program before first enqueue.
    // -------------------------------------------------------
    constexpr auto second_kernel_name = "manual_second_plain_kernel";
    const distributed::MeshCoordinateRange second_range(coords[1]);
    workload.add_program(
        second_range,
        experimental::MakeProgramFromSpec(*mesh_device, MakeScratchpadSpec("manual_second_plain", second_kernel_name)));
    ASSERT_EQ(workload.get_programs().size(), 2u);

    // -------------------------------------------------------
    // Set runtime args for both programs, dispatch, and verify both devices.
    // -------------------------------------------------------
    SetScratchpadArgs(workload.get_programs().at(first_range), first_kernel_name);
    SetScratchpadArgs(workload.get_programs().at(second_range), second_kernel_name);
    EnqueueAndVerifyScratchpadResult(mesh_device, workload, {coords[0], coords[1]});
}

TEST_F(MeshWorkloadFactoryHWTest, MakeMeshWorkloadFromSpecHandlesRepeatedDfbSizeOverrides) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // -------------------------------------------------------
    // Build one DFB loopback ProgramSpec and create the workload.
    // -------------------------------------------------------
    distributed::MeshWorkload workload = experimental::MakeMeshWorkloadFromSpec(*mesh_device, MakeDfbLoopbackSpec());
    ASSERT_EQ(workload.get_programs().size(), 1u);

    InterleavedBufferConfig dram_config{
        .device = device, .size = kDfbTotalBytes, .page_size = kDfbTotalBytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    // -------------------------------------------------------
    // Repeatedly resize the DFB at runtime, dispatch, and verify loopback.
    // -------------------------------------------------------
    for (uint32_t dfb_num_entries : {2u, 4u, 6u}) {
        std::vector<uint32_t> input_data(kDfbTotalBytes / sizeof(uint32_t));
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = (dfb_num_entries << 16) + static_cast<uint32_t>(i);
        }
        std::vector<uint32_t> zero_output(input_data.size(), 0u);

        detail::WriteToBuffer(input_buffer, input_data);
        detail::WriteToBuffer(output_buffer, zero_output);
        SetDfbLoopbackArgs(workload, input_buffer, output_buffer, dfb_num_entries);

        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

        std::vector<uint32_t> output_data;
        detail::ReadFromBuffer(output_buffer, output_data);
        ASSERT_EQ(output_data.size(), input_data.size());
        EXPECT_EQ(output_data, input_data);
    }
}

TEST_F(MeshWorkloadFactoryHWTest, MakeMeshWorkloadFromSpecsUsesScratchpad) {
    auto mesh_device = devices_.at(0);

    // -------------------------------------------------------
    // Build a region-mapped scratchpad ProgramSpec and create the workload.
    // -------------------------------------------------------
    constexpr auto kernel_name = "factory_map_kernel";

    std::unordered_map<distributed::MeshCoordinateRange, experimental::ProgramSpec> program_specs;
    program_specs.emplace(
        distributed::MeshCoordinateRange(mesh_device->shape()), MakeScratchpadSpec("factory_map", kernel_name));
    distributed::MeshWorkload workload = experimental::MakeMeshWorkloadFromSpecs(*mesh_device, program_specs);

    ASSERT_EQ(workload.get_programs().size(), 1u);

    // -------------------------------------------------------
    // Repeatedly set runtime args, dispatch, and verify scratchpad access.
    // -------------------------------------------------------
    for (uint32_t launch = 0; launch < kNumLaunches; launch++) {
        SetScratchpadArgs(workload, kernel_name);
        VerifyScratchpadResult(mesh_device, workload);
    }
}

}  // namespace
}  // namespace tt::tt_metal
