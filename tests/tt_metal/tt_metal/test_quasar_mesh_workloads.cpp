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
            .hw_config = experimental::DataMovementHardwareConfig{},
        });
        wu_kernel_names.push_back(std::move(kernel_id));
    }

    const experimental::KernelSpecName COMPUTE_KERNEL{std::string("compute_") + workload_id_str};
    kernel_specs.push_back(experimental::KernelSpec{
        .unique_id = COMPUTE_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        .num_threads = kNumComputeNEOs,
        .runtime_arg_schema = {.runtime_arg_names = {"l1_address"}},
        .hw_config = experimental::ComputeHardwareConfig{},
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
            .runtime_arg_values =
                experimental::MakeRuntimeArgsForSingleNode(node, {{"address", dm_base_address + i * sizeof(uint32_t)}}),
            .common_runtime_arg_values = {{"value", dm_base_value + i}},
        });
    }
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = COMPUTE_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"l1_address", compute_address}}),
    });
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    return workload;
}

void set_multi_node_run_args(
    Program& program,
    const std::vector<experimental::NodeCoord>& nodes,
    const std::vector<uint32_t>& dm_addresses,
    uint32_t dm_value,
    const std::vector<uint32_t>& compute_addresses) {
    TT_FATAL(
        nodes.size() == dm_addresses.size() && nodes.size() == compute_addresses.size(),
        "Per-node address vectors must match the node list size");

    const experimental::KernelSpecName dm_kernel{"multi_node_dm"};
    const experimental::KernelSpecName compute_kernel{"multi_node_compute"};

    experimental::ProgramRunArgs params;
    experimental::ProgramRunArgs::KernelRunArgs dm_args{
        .kernel = dm_kernel, .common_runtime_arg_values = {{"value", dm_value}}};
    experimental::ProgramRunArgs::KernelRunArgs compute_args{.kernel = compute_kernel};
    for (size_t i = 0; i < nodes.size(); ++i) {
        experimental::AddRuntimeArgsForNode(dm_args.runtime_arg_values, nodes[i], {{"address", dm_addresses[i]}});
        experimental::AddRuntimeArgsForNode(
            compute_args.runtime_arg_values, nodes[i], {{"l1_address", compute_addresses[i]}});
    }
    params.kernel_run_args = {std::move(dm_args), std::move(compute_args)};
    experimental::SetProgramRunArgs(program, params);
}

distributed::MeshWorkload create_multi_node_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const experimental::NodeRange& node_range,
    const std::vector<uint32_t>& dm_addresses,
    uint32_t dm_value,
    const std::vector<uint32_t>& compute_addresses) {
    const experimental::KernelSpecName dm_kernel{"multi_node_dm"};
    const experimental::KernelSpecName compute_kernel{"multi_node_compute"};
    const std::vector<experimental::NodeCoord> nodes =
        experimental::grid_to_nodes(node_range.start_coord, node_range.end_coord);

    experimental::ProgramSpec spec{
        .name = "multi_node_l1_write",
        .kernels =
            {experimental::KernelSpec{
                 .unique_id = dm_kernel,
                 .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
                 .num_threads = 1,
                 .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
                 .hw_config = experimental::DataMovementHardwareConfig{}},
             experimental::KernelSpec{
                 .unique_id = compute_kernel,
                 .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
                 .num_threads = kNumComputeNEOs,
                 .runtime_arg_schema = {.runtime_arg_names = {"l1_address"}},
                 .hw_config = experimental::ComputeHardwareConfig{}}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "multi_node", .kernels = {dm_kernel, compute_kernel}, .target_nodes = node_range}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    set_multi_node_run_args(program, nodes, dm_addresses, dm_value, compute_addresses);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
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

TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, TestInterleavedWorkloadsAcrossTwoCQs) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t base_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    constexpr uint32_t num_workloads = 8;

    // Distinct DM base value per workload so a cross-CQ clobber shows up as a mismatch.
    auto dm_base_value_for = [](uint32_t w) { return 0xa0000000u + (w << 16); };

    // Each workload's DM and compute outputs live on separate 64-byte cache lines so the DM cache
    // flush can't clobber the compute kernel's uncached writes.
    auto dm_base_addr_for = [&](uint32_t w) { return base_address + w * 2 * kL1CacheLineBytes; };
    auto compute_addr_for = [&](uint32_t w) { return dm_base_addr_for(w) + kL1CacheLineBytes; };

    std::vector<uint32_t> zeros(num_workloads * 2 * kL1CacheLineBytes / sizeof(uint32_t), 0);
    tt_metal::detail::WriteToDeviceL1(dev, node, base_address, zeros);

    distributed::MeshCommandQueue& cq0 = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& cq1 = mesh_device->mesh_command_queue(1);

    std::vector<distributed::MeshWorkload> workloads;
    workloads.reserve(num_workloads);
    for (uint32_t w = 0; w < num_workloads; w++) {
        const std::string kernel_id = "k" + std::to_string(w + 1);
        workloads.push_back(create_workload(
            mesh_device, node, dm_base_addr_for(w), dm_base_value_for(w), compute_addr_for(w), kernel_id));
    }

    // Interleave: w even -> CQ0, w odd -> CQ1, so consecutive enqueues alternate queues.
    for (uint32_t w = 0; w < num_workloads; w++) {
        distributed::EnqueueMeshWorkload(w % 2 == 0 ? cq0 : cq1, workloads[w], true);
    }

    for (uint32_t w = 0; w < num_workloads; w++) {
        const uint32_t dm_base_addr = dm_base_addr_for(w);
        const uint32_t compute_addr = compute_addr_for(w);

        std::vector<uint32_t> dm_output(kNumUserDMThreads, 0);
        tt_metal::detail::ReadFromDeviceL1(dev, node, dm_base_addr, kNumUserDMThreads * sizeof(uint32_t), dm_output);
        for (uint32_t i = 0; i < kNumUserDMThreads; i++) {
            EXPECT_EQ(dm_output[i], dm_base_value_for(w) + i);
        }

        std::vector<uint32_t> compute_output(kNumComputeNEOs * kNumTRISCsPerNEO, 0);
        tt_metal::detail::ReadFromDeviceL1(
            dev, node, compute_addr, kNumComputeNEOs * kNumTRISCsPerNEO * sizeof(uint32_t), compute_output);
        EXPECT_EQ(compute_output, kExpectedComputeValues);
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TestWorkloadAcrossMultipleWorkerNodes) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeRange node_range{{0, 0}, {worker_grid.x - 1, worker_grid.y - 1}};
    const std::vector<experimental::NodeCoord> nodes =
        experimental::grid_to_nodes(node_range.start_coord, node_range.end_coord);

    const uint32_t base_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    constexpr uint32_t kNumRounds = 2;
    const std::array<uint32_t, kNumRounds> dm_values = {0x12340000, 0xabcd0000};

    auto dm_address_for = [&](uint32_t round, uint32_t node_index) {
        return base_address + (round * num_nodes + node_index) * 2 * kL1CacheLineBytes;
    };
    auto compute_address_for = [&](uint32_t round, uint32_t node_index) {
        return dm_address_for(round, node_index) + kL1CacheLineBytes;
    };

    std::vector<uint32_t> zeros(kNumRounds * num_nodes * 2 * kL1CacheLineBytes / sizeof(uint32_t), 0);
    for (const auto& node : nodes) {
        tt_metal::detail::WriteToDeviceL1(dev, node, base_address, zeros);
    }

    auto addresses_for_round = [&](uint32_t round) {
        std::vector<uint32_t> dm_addresses(num_nodes);
        std::vector<uint32_t> compute_addresses(num_nodes);
        for (uint32_t node_index = 0; node_index < num_nodes; ++node_index) {
            dm_addresses[node_index] = dm_address_for(round, node_index);
            compute_addresses[node_index] = compute_address_for(round, node_index);
        }
        return std::pair{std::move(dm_addresses), std::move(compute_addresses)};
    };

    auto [dm_addresses_0, compute_addresses_0] = addresses_for_round(0);
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    distributed::MeshWorkload workload =
        create_multi_node_workload(mesh_device, node_range, dm_addresses_0, dm_values[0], compute_addresses_0);
    Program& program = workload.get_programs().at(device_range);
    auto& cq = mesh_device->mesh_command_queue();

    for (uint32_t round = 0; round < kNumRounds; ++round) {
        auto [dm_addresses, compute_addresses] = addresses_for_round(round);
        // Round 0 args were already applied in create_multi_node_workload; only later rounds need an update.
        // Reuse the same workload (instead of rebuilding each round) so this also exercises updating runtime
        // args on an existing program between enqueues.
        if (round != 0) {
            set_multi_node_run_args(program, nodes, dm_addresses, dm_values[round], compute_addresses);
        }
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

        for (uint32_t node_index = 0; node_index < num_nodes; ++node_index) {
            const experimental::NodeCoord& node = nodes[node_index];
            std::vector<uint32_t> dm_output(1, 0);
            tt_metal::detail::ReadFromDeviceL1(dev, node, dm_addresses[node_index], sizeof(uint32_t), dm_output);
            EXPECT_EQ(dm_output[0], dm_values[round]);

            std::vector<uint32_t> compute_output(kNumComputeNEOs * kNumTRISCsPerNEO, 0);
            tt_metal::detail::ReadFromDeviceL1(
                dev,
                node,
                compute_addresses[node_index],
                kNumComputeNEOs * kNumTRISCsPerNEO * sizeof(uint32_t),
                compute_output);
            EXPECT_EQ(compute_output, kExpectedComputeValues);
        }
    }
}
