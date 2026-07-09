// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kNumUserDMThreads = 6;
constexpr uint32_t kNumComputeNEOs = 4;
constexpr uint32_t kNumTRISCsPerNEO = 4;
constexpr uint32_t kL1CacheLineBytes = 64;

// L1 slots consumed by one workload: kNumUserDMThreads (DM) + kNumComputeNEOs*kNumTRISCsPerNEO (compute)
constexpr uint32_t kWorkloadOutputCount = kNumUserDMThreads + kNumComputeNEOs * kNumTRISCsPerNEO;

// Expected output of risc_math.cpp kernel with kNumComputeNEOs=4 (4 NEOs × 4 TRISCs = 16 writes).
const std::vector<uint32_t> kExpectedComputeValues = {4, 6, 5, 9, 8, 10, 9, 13, 12, 14, 13, 17, 16, 18, 17, 21};

// Builds a MeshWorkload with kNumUserDMThreads DM kernels and one 4-NEO compute kernel.
//
// DM kernel layout: each of the kNumUserDMThreads kernel specs uses num_threads=1, allowing
// each DM processor to receive its own address arg.  DM processor i writes
//   dm_base_value + i  →  dm_base_address + i * sizeof(uint32_t)
//
// Compute kernel layout: risc_math.cpp with num_threads=kNumComputeNEOs writes 16 uint32_t
// values starting at compute_address (outputs match kExpectedComputeValues).
//
// workload_id_str must be unique per call (used to derive kernel names).
distributed::MeshWorkload create_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const experimental::NodeCoord& node,
    uint32_t dm_base_address,
    uint32_t dm_base_value,
    uint32_t compute_address,
    const std::string& workload_id_str) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    std::vector<experimental::KernelSpec> kernel_specs;
    std::vector<experimental::KernelSpecName> wu_kernel_names;

    for (uint32_t i = 0; i < kNumUserDMThreads; i++) {
        experimental::KernelSpecName kernel_id{std::string("dm_") + workload_id_str + "_" + std::to_string(i)};
        kernel_specs.push_back(experimental::KernelSpec{
            .unique_id = kernel_id,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
            .hw_config = experimental::DataMovementGen2Config{},
        });
        wu_kernel_names.push_back(std::move(kernel_id));
    }

    const experimental::KernelSpecName COMPUTE_KERNEL{std::string("compute_") + workload_id_str};
    kernel_specs.push_back(experimental::KernelSpec{
        .unique_id = COMPUTE_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        .num_threads = kNumComputeNEOs,
        .runtime_arg_schema = {.runtime_arg_names = {"l1_address"}},
        .hw_config = experimental::ComputeGen2Config{},
    });
    wu_kernel_names.push_back(COMPUTE_KERNEL);

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = wu_kernel_names,
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = std::string("l1_write_") + workload_id_str,
        .kernels = kernel_specs,
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    for (uint32_t i = 0; i < kNumUserDMThreads; i++) {
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{std::string("dm_") + workload_id_str + "_" + std::to_string(i)},
            .runtime_arg_values = {{node, {{"address", dm_base_address + i * sizeof(uint32_t)}}}},
            .common_runtime_arg_values = {{"value", dm_base_value + i}},
        });
    }
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = COMPUTE_KERNEL,
        .runtime_arg_values = {{node, {{"l1_address", compute_address}}}},
    });
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    return workload;
}

}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, TestSingleWorkloadNonBlockingEnqueueFinish) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t base_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t dm_base_address = base_address;
    const uint32_t compute_address = base_address + kNumUserDMThreads * sizeof(uint32_t);
    const uint32_t dm_base_value = 0xdead0000;

    std::vector<uint32_t> zeros(kWorkloadOutputCount, 0);
    tt_metal::detail::WriteToDeviceL1(dev, node, base_address, zeros);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload =
        create_workload(mesh_device, node, dm_base_address, dm_base_value, compute_address, "k0");

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> dm_output(kNumUserDMThreads, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, dm_base_address, kNumUserDMThreads * sizeof(uint32_t), dm_output);
    for (uint32_t i = 0; i < kNumUserDMThreads; i++) {
        ASSERT_EQ(dm_output[i], dm_base_value + i);
    }

    std::vector<uint32_t> compute_output(kNumComputeNEOs * kNumTRISCsPerNEO, 0);
    tt_metal::detail::ReadFromDeviceL1(
        dev, node, compute_address, kNumComputeNEOs * kNumTRISCsPerNEO * sizeof(uint32_t), compute_output);
    ASSERT_EQ(compute_output, kExpectedComputeValues);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TestMultipleWorkloadsNonBlockingEnqueueFinish) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t base_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    constexpr uint32_t kNumWorkloads = 4;
    const std::array<uint32_t, kNumWorkloads> dm_base_values = {0x11110000, 0x22220000, 0x33330000, 0x44440000};

    // Put each workload's DM and compute outputs on separate 64-byte cache lines so the DM cache flush can't clobber
    // the compute kernel's uncached writes.
    auto dm_base_addr_for = [&](uint32_t w) { return base_address + w * 2 * kL1CacheLineBytes; };
    auto compute_addr_for = [&](uint32_t w) { return dm_base_addr_for(w) + kL1CacheLineBytes; };

    std::vector<uint32_t> zeros(kNumWorkloads * 2 * kL1CacheLineBytes / sizeof(uint32_t), 0);
    tt_metal::detail::WriteToDeviceL1(dev, node, base_address, zeros);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    std::vector<distributed::MeshWorkload> workloads;
    workloads.reserve(kNumWorkloads);
    for (uint32_t w = 0; w < kNumWorkloads; w++) {
        const std::string kernel_id = "k" + std::to_string(w + 1);
        workloads.push_back(
            create_workload(mesh_device, node, dm_base_addr_for(w), dm_base_values[w], compute_addr_for(w), kernel_id));
    }

    for (uint32_t w = 0; w < kNumWorkloads; w++) {
        distributed::EnqueueMeshWorkload(cq, workloads[w], false);
    }
    distributed::Finish(cq);

    for (uint32_t w = 0; w < kNumWorkloads; w++) {
        const uint32_t dm_base_addr = dm_base_addr_for(w);
        const uint32_t compute_addr = compute_addr_for(w);

        std::vector<uint32_t> dm_output(kNumUserDMThreads, 0);
        tt_metal::detail::ReadFromDeviceL1(dev, node, dm_base_addr, kNumUserDMThreads * sizeof(uint32_t), dm_output);
        for (uint32_t i = 0; i < kNumUserDMThreads; i++) {
            EXPECT_EQ(dm_output[i], dm_base_values[w] + i);
        }

        std::vector<uint32_t> compute_output(kNumComputeNEOs * kNumTRISCsPerNEO, 0);
        tt_metal::detail::ReadFromDeviceL1(
            dev, node, compute_addr, kNumComputeNEOs * kNumTRISCsPerNEO * sizeof(uint32_t), compute_output);
        EXPECT_EQ(compute_output, kExpectedComputeValues);
    }
}
