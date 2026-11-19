// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cstdint>
#include <cstdlib>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;

void RunTest(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up program
    const CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();
    CoreRange core_range(
        {0, 0}, {std::min<std::size_t>(worker_grid_size.x, 6) - 1, std::min<std::size_t>(worker_grid_size.y, 6) - 1});

    auto l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    // Write runtime args
    const auto device_id = mesh_device->get_device_ids()[0];
    auto get_first_arg =
        [device_id](
            const std::shared_ptr<distributed::MeshDevice>& /*mesh_device*/, CoreCoord& core, uint32_t multiplier) {
            return (uint32_t)device_id + ((uint32_t)core.x * 10 * multiplier);
        };
    auto get_second_arg = [](const std::shared_ptr<distributed::MeshDevice>& /*mesh_device*/,
                             CoreCoord& core,
                             uint32_t multiplier) { return (uint32_t)core.y * 100 * multiplier; };

    const KernelSpecName first_kernel_name{"first_adder"};
    const KernelSpecName second_kernel_name{"second_adder"};
    auto make_adder_kernel =
        [](const KernelSpecName& kernel_name, uint32_t l1_address, DataMovementProcessor processor, NOC noc) {
            return KernelSpec{
                .unique_id = kernel_name,
                .source = "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints_2_0.cpp",
                .compile_time_args = {{"l1_address", l1_address}},
                .runtime_arg_schema = {.runtime_arg_names = {"a", "b"}},
                .hw_config =
                    DataMovementHardwareConfig{
                        .config_1xx =
                            DataMovementHardwareConfig::DataMovement1XXConfig{.processor = processor, .noc = noc}},
            };
        };
    const ProgramSpec spec{
        .name = "add_two_ints",
        .kernels =
            {make_adder_kernel(
                 first_kernel_name, l1_unreserved_base, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default),
             make_adder_kernel(
                 second_kernel_name, l1_unreserved_base + 4, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default)},
        .work_units = {WorkUnitSpec{
            .name = "work_unit", .kernels = {first_kernel_name, second_kernel_name}, .target_nodes = core_range}},
    };
    Program program = Program(*mesh_device, spec);

    ProgramRunArgs::KernelRunArgs first_kernel_run_args{.kernel = first_kernel_name};
    ProgramRunArgs::KernelRunArgs second_kernel_run_args{.kernel = second_kernel_name};
    for (CoreCoord core : core_range) {
        AddRuntimeArgsForNode(
            first_kernel_run_args.runtime_arg_values,
            core,
            {{"a", get_first_arg(mesh_device, core, 1)}, {"b", get_second_arg(mesh_device, core, 1)}});
        AddRuntimeArgsForNode(
            second_kernel_run_args.runtime_arg_values,
            core,
            {{"a", get_first_arg(mesh_device, core, 2)}, {"b", get_second_arg(mesh_device, core, 2)}});
    }
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {first_kernel_run_args, second_kernel_run_args};
    program.set_program_run_args(run_args);

    distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange({0, 0}, {0, 0}), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    // Check results
    for (CoreCoord core : core_range) {
        std::vector<uint32_t> brisc_result;
        slow_dispatch::ReadFromL1(*mesh_device, core, l1_unreserved_base, sizeof(uint32_t), brisc_result);
        std::vector<uint32_t> ncrisc_result;
        slow_dispatch::ReadFromL1(*mesh_device, core, l1_unreserved_base + 4, sizeof(uint32_t), ncrisc_result);
        uint32_t expected_result = get_first_arg(mesh_device, core, 1) + get_second_arg(mesh_device, core, 1);
        if (expected_result != brisc_result[0]) {
            log_warning(
                LogTest,
                "Device {}, Core {}, BRISC result was incorrect. Expected {} but got {}",
                device_id,
                core.str(),
                expected_result,
                brisc_result[0]);
        }
        EXPECT_TRUE(expected_result == brisc_result[0]);
        expected_result = get_first_arg(mesh_device, core, 2) + get_second_arg(mesh_device, core, 2);
        if (expected_result != ncrisc_result[0]) {
            log_warning(
                LogTest,
                "Device {}, Core {}, NCRISC result was incorrect. Expected {} but got {}",
                device_id,
                core.str(),
                expected_result,
                ncrisc_result[0]);
        }
        EXPECT_TRUE(expected_result == ncrisc_result[0]);
    }
}

TEST(DispatchStress, TensixRunManyTimes) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    // Skip fast dispatch until it's supported for remote device.
    if (!slow_dispatch) {
        GTEST_SKIP();
    }
    // Run 500 times to make sure that things work
    for (int idx = 0; idx < 400; idx++) {
        log_info(LogTest, "Running iteration #{}", idx);
        // Need to open/close the device each time in order to reproduce original issue.
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<ChipId> chip_ids;
        chip_ids.reserve(num_devices);
        for (unsigned int id = 0; id < num_devices; id++) {
            chip_ids.push_back(id);
        }
        vector<std::shared_ptr<distributed::MeshDevice>> devices_;
        auto reserved_devices_ = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(chip_ids);
        devices_.reserve(reserved_devices_.size());
        for (const auto& [id, device] : reserved_devices_) {
            devices_.push_back(device);
        }

        // Run the test on each device
        for (auto& device : devices_) {
            RunTest(device);
        }

        // Close all devices
        for (const auto& [id, device] : reserved_devices_) {
            device->close();
        }
    }
}

}  // namespace tt::tt_metal
