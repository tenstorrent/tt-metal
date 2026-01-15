// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <gtest/gtest.h>

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "debug_tools_fixture.hpp"

namespace tt::tt_metal {

// Test if RTA payload region initialized by HostMemDeviceCommand
// is filled with known garbage (with 0xBEEF####) for cores with unset RTAs
TEST_F(MeshWatcherFixture, WatcherKnownGarbageRTAs) {
    auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    // First run the program with the initial runtime args
    CoreRange first_core_range(CoreCoord(0, 0), CoreCoord(1, 1));
    CoreRange second_core_range(CoreCoord(3, 3), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{first_core_range, second_core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    const std::vector<uint32_t> rtas = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};

    // Configure CB to store read-back args
    uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t cb_size = tt::align(rtas.size() * sizeof(uint32_t), l1_alignment);
    CircularBufferConfig cb_config =
        CircularBufferConfig(cb_size, {{0, tt::DataFormat::Float32}}).set_page_size(0, cb_size);

    distributed::MeshWorkload workload;
    Program program;
    CreateCircularBuffer(program, core_range_set, cb_config);

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_runtime_args_prefill.cpp",
        core_range_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // SetRuntimeArgs only for first_core_range ; second_core_range has unset RTAs and should read known garbage back
    SetRuntimeArgs(program, kernel, first_core_range, rtas);

    workload.add_program(device_range, std::move(program));
    EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false));
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<uint32_t> read_result;
    // Verify each core in first_core_range (valid data)
    for (const auto& core : first_core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, cb_addr, rtas.size() * sizeof(uint32_t), read_result);
        EXPECT_EQ(read_result.size(), rtas.size());
        for (uint32_t i = 0; i < read_result.size(); i++) {
            EXPECT_EQ(read_result[i], rtas[i]);
        }
    }

    // Verify each core in second_core_range (expect known garbage: 0xBEEF####)
    for (const auto& core : second_core_range) {
        read_result.clear();
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, cb_addr, rtas.size() * sizeof(uint32_t), read_result);
        EXPECT_EQ(read_result.size(), rtas.size());
        for (const auto& result : read_result) {
            EXPECT_EQ(result & 0xFFFF0000, 0xBEEF0000);
        }
    }
}
}  // namespace tt::tt_metal
