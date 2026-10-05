// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

using namespace tt;
using namespace tt::tt_metal;

// TRISC0 computes c = a + b with RISC-V Vector instructions (enable_trisc0_rvv).
TEST_F(QuasarMeshDeviceSingleCardFixture, Trisc0RvvVectorAdd) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator.";
    }

    auto mesh_device = devices_[0];

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    const experimental::NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // a, b and c (32 x int32 each) laid out back to back in L1.
    constexpr uint32_t num_elements = 32;
    std::vector<uint32_t> a(num_elements), b(num_elements);
    for (uint32_t i = 0; i < num_elements; i++) {
        a[i] = static_cast<uint32_t>(static_cast<int32_t>(i) - 7);
        b[i] = 1000 + 3 * i;
    }
    std::vector<uint32_t> init_values;
    init_values.insert(init_values.end(), a.begin(), a.end()); // initialize a
    init_values.insert(init_values.end(), b.begin(), b.end()); // initialize b
    init_values.insert(init_values.end(), num_elements, 0); // initialize c

    const uint32_t l1_address = MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    slow_dispatch::WriteToL1(*mesh_device, node, l1_address, init_values);

    const experimental::KernelSpecName COMPUTE_KERNEL{"trisc0_rvv_vadd"};

    experimental::ComputeHardwareConfig hw_config{};
    hw_config.config_2xx = experimental::ComputeHardwareConfig::Compute2XXConfig{.enable_trisc0_rvv = true};

    experimental::KernelSpec compute_kernel_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/trisc0_rvv_vadd_quasar.cpp",
        .num_threads = 1,
        .runtime_arg_schema = { .runtime_arg_names = {"l1_address"}, },
        .hw_config = hw_config,
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {COMPUTE_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "trisc0_rvv_vadd",
        .kernels = {compute_kernel_spec},
        .work_units = {main_wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = COMPUTE_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"l1_address", l1_address}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_values(num_elements, 0);
    slow_dispatch::ReadFromL1(
        *mesh_device,
        node,
        l1_address + 2 * num_elements * sizeof(uint32_t),
        num_elements * sizeof(uint32_t),
        actual_values);

    std::vector<uint32_t> expected_values(num_elements);
    for (uint32_t i = 0; i < num_elements; i++) {
        expected_values[i] = a[i] + b[i];
    }

    ASSERT_EQ(actual_values, expected_values);
}
