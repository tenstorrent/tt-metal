// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "mesh_dispatch_fixture.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"

using namespace tt::tt_metal;

namespace {

constexpr uint32_t kCbPageSize = 2048;
constexpr uint32_t kCbPages = 8;
constexpr auto kPastRateLimit = std::chrono::milliseconds(150);

std::unique_ptr<distributed::MeshWorkload> make_cb_workload() {
    Program program = CreateProgram();
    const CoreCoord core(0, 0);
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(kCbPageSize * kCbPages, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, kCbPageSize));
    auto workload = std::make_unique<distributed::MeshWorkload>();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    workload->add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
    return workload;
}

}  // namespace

TEST_F(MeshDispatchFixture, TensixShmCbStatsReleasedWhenProgramsAreDestroyed) {
    auto& mesh_device = devices_.front();
    auto* device = dynamic_cast<Device*>(mesh_device->get_devices().front());
    ASSERT_NE(device, nullptr);
    auto* shm = device->get_shm_stats_provider();
    if (shm == nullptr || !shm->is_initialized()) {
        GTEST_SKIP() << "SHM tracking is disabled";
    }
    const uint64_t baseline = device->get_total_cb_allocated();
    if (baseline != 0) {
        GTEST_SKIP() << "Other programs still hold CBs on this device";
    }

    auto first = make_cb_workload();
    RunProgram(mesh_device, *first);
    std::this_thread::sleep_for(kPastRateLimit);
    auto second = make_cb_workload();
    RunProgram(mesh_device, *second);
    ASSERT_GT(shm->get_device_stats().cb_allocated, baseline);

    second.reset();
    first.reset();
    EXPECT_EQ(device->get_total_cb_allocated(), baseline);
    EXPECT_EQ(shm->get_device_stats().cb_allocated, baseline);
}
