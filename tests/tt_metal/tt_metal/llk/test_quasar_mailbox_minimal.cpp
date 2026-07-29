// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "host_api.hpp"
#include "impl/host_api/temp_quasar_api.hpp"
#include "llk_device_fixture.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal {

namespace {

constexpr CoreCoord MAILBOX_MIN_WORKER_CORE = {0, 0};
constexpr std::uint32_t MAILBOX_MIN_EXPECTED_VALUE = 0xfacefaceu;

}  // namespace

// Minimal reproducer for the QuasarCbL1ReadApi fault: a single mailbox_write (UNPACK ->
// MathThreadId) matched by a single mailbox_read (MATH <- UnpackThreadId). No CB/DFB is
// involved, isolating whether the mailbox mechanism itself faults independent of the
// dataflow-buffer address computation in QuasarCbL1ReadApi.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarMailboxMinimal) {
    auto mesh_device = devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto compute_kernel = experimental::quasar::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_mailbox_minimal_compute.cpp",
        MAILBOX_MIN_WORKER_CORE,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
        });

    // Kernel (MATH thread) writes the mailbox-read value here; host reads it back after the run.
    const std::uint32_t result_l1_addr = static_cast<std::uint32_t>(device->l1_size_per_core()) - sizeof(std::uint32_t);
    std::vector<std::uint32_t> result_init(1, 0);
    detail::WriteToDeviceL1(device, MAILBOX_MIN_WORKER_CORE, result_l1_addr, result_init);

    SetRuntimeArgs(program_, compute_kernel, MAILBOX_MIN_WORKER_CORE, {result_l1_addr});

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::vector<std::uint32_t> host_buffer;
    detail::ReadFromDeviceL1(device, MAILBOX_MIN_WORKER_CORE, result_l1_addr, sizeof(std::uint32_t), host_buffer);

    ASSERT_EQ(host_buffer.size(), 1u);
    EXPECT_EQ(host_buffer[0], MAILBOX_MIN_EXPECTED_VALUE);
}

}  // namespace tt::tt_metal
