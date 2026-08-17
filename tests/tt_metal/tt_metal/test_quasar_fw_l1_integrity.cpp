// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kLocalL1Size = 3200;

CoreRangeSet full_worker_grid(const CoreCoord& worker_grid) {
    return CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1}));
}

uint32_t worker_boot_word(IDevice* device, const CoreCoord& logical_core) {
    std::vector<uint32_t> output(1, 0);
    detail::ReadFromDeviceL1(device, logical_core, /*address=*/0, sizeof(uint32_t), output);
    return output[0];
}

distributed::MeshWorkload make_l1_write_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& worker_grid,
    uint32_t address,
    uint32_t value) {
    const experimental::KernelSpecName kernel_name{"boot_word_l1_writer"};
    experimental::ProgramSpec spec{
        .name = "boot_word_l1_write",
        .kernels = {experimental::KernelSpec{
            .unique_id = kernel_name,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
            .hw_config = experimental::DataMovementGen2Config{}}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "writer",
            .kernels = {kernel_name},
            .target_nodes = experimental::NodeRangeSet{full_worker_grid(worker_grid)}}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::ProgramRunArgs::KernelRunArgs kernel_args{
        .kernel = kernel_name, .common_runtime_arg_values = {{"value", value}}};
    for (const auto& node : experimental::node_range_to_nodes(full_worker_grid(worker_grid))) {
        experimental::AddRuntimeArgsForNode(kernel_args.runtime_arg_values, node, {{"address", address}});
    }
    experimental::SetProgramRunArgs(program, experimental::ProgramRunArgs{.kernel_run_args = {std::move(kernel_args)}});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    return workload;
}

}  // namespace

// Quasar has no stream registers, so a worker signals completion with a NOC atomic whose INCR_GET
// response is written back to the address held in the atomic command buffer. Only a kernel launch
// programs that address, so a worker that acks RUN_MSG_RESET_READ_PTR before it has ever launched a
// kernel would write the response over the boot code at L1[0].
//
// Reaching that state needs two device cycles: a worker only acks the reset go signal if a workload
// ran on a previous device in this process, so the warm-up cycle below is load bearing. The reopened
// device restarts worker firmware with an unprogrammed atomic command buffer.
TEST_F(QuasarMeshDeviceSingleCardFixture, WorkerBootWordSurvivesPreKernelGoSignalAck) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    const auto& hal = MetalContext::instance().hal();
    if (hal.has_stream_registers()) {
        GTEST_SKIP() << "Workers signal completion via stream registers, not an L1 atomic";
    }

    const uint32_t core_type_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const uint32_t fw_launch_value = hal.get_jit_build_config(core_type_index, 0, 0).fw_launch_addr_value;

    {
        auto mesh_device = devices_[0];
        const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
        const auto manager = mesh_device->create_sub_device_manager(
            std::array{SubDevice(std::array{full_worker_grid(worker_grid)})}, kLocalL1Size);
        mesh_device->load_sub_device_manager(manager);

        const uint32_t address =
            hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        auto workload = make_l1_write_workload(mesh_device, worker_grid, address, 0xa5a50000u);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
    }

    devices_.clear();
    id_to_device_.clear();
    create_devices();

    auto mesh_device = devices_[0];
    IDevice* device = mesh_device->get_devices()[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();

    // No kernel has run on these workers yet, so the boot word must still be the firmware launch
    // jump. If this fails the test is no longer exercising the pre-kernel state it intends to.
    for (uint32_t y = 0; y < worker_grid.y; ++y) {
        for (uint32_t x = 0; x < worker_grid.x; ++x) {
            ASSERT_EQ(worker_boot_word(device, CoreCoord{x, y}), fw_launch_value)
                << "boot word already invalid before any go signal on worker (" << x << "," << y << ")";
        }
    }

    // Loading a sub device manager sends RUN_MSG_RESET_READ_PTR to every worker; the dispatcher then
    // waits for all of them to ack, so draining the queue means the acking atomics have been issued.
    const auto manager = mesh_device->create_sub_device_manager(
        std::array{SubDevice(std::array{full_worker_grid(worker_grid)})}, kLocalL1Size);
    mesh_device->load_sub_device_manager(manager);
    distributed::Finish(mesh_device->mesh_command_queue());

    for (uint32_t y = 0; y < worker_grid.y; ++y) {
        for (uint32_t x = 0; x < worker_grid.x; ++x) {
            EXPECT_EQ(worker_boot_word(device, CoreCoord{x, y}), fw_launch_value)
                << "worker (" << x << "," << y << ") clobbered L1[0] while acking the go signal";
        }
    }
}
