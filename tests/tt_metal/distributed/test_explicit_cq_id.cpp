// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Metal has no implicit command queue selection for these APIs: the queue that performs the write is passed as a
// MeshCommandQueue& (command queue 0 when the overload without a queue is used). These tests check that the given
// queue is honored and that a queue belonging to a different MeshDevice is rejected.

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/forge_backdoor/global_semaphore.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/inspector.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {
namespace {

using ExplicitCqTest = GenericMultiCQMeshDeviceFixture;

uint32_t read_semaphore(const MeshDevice& mesh_device, const CoreCoord& logical_core, DeviceAddr address) {
    const auto device_id = mesh_device.get_device_ids()[0];
    return MetalContext::instance().get_cluster().read_core(
        device_id, mesh_device.worker_core_from_logical_core(logical_core), address, sizeof(uint32_t))[0];
}

TEST_F(ExplicitCqTest, GlobalSemaphoreCreateAndResetOnSecondQueue) {
    auto mesh_device = get_mesh_device();
    ASSERT_EQ(mesh_device->num_hw_cqs(), 2);
    auto& cq1 = mesh_device->mesh_command_queue(1);
    const CoreCoord core(0, 0);
    const CoreRangeSet cores(CoreRange(core, core));

    GlobalSemaphore semaphore(*mesh_device, cq1, cores, /*initial_value=*/3);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, semaphore.address()), 3u);

    semaphore.reset_semaphore_value(7, cq1);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, semaphore.address()), 7u);

    // The overload without a queue uses cq 0.
    semaphore.reset_semaphore_value(9);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, semaphore.address()), 9u);
}

TEST_F(ExplicitCqTest, AddressBackedGlobalSemaphoreInitOnGivenQueue) {
    auto mesh_device = get_mesh_device();
    const CoreCoord core(0, 0);
    const CoreRangeSet cores(CoreRange(core, core));

    // Borrow the address of a regular semaphore as the caller-owned backing memory.
    GlobalSemaphore backing(*mesh_device, cores, /*initial_value=*/0);
    const DeviceAddr address = backing.address();

    auto at_address = experimental::CreateGlobalSemaphore(
        *mesh_device, mesh_device->mesh_command_queue(1), cores, /*initial_value=*/5, BufferType::L1, address);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, address), 5u);

    auto at_address_default_cq =
        experimental::CreateGlobalSemaphore(*mesh_device, cores, /*initial_value=*/6, BufferType::L1, address);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, address), 6u);
}

TEST_F(ExplicitCqTest, GlobalCircularBufferConfigWriteOnSecondQueue) {
    auto mesh_device = get_mesh_device();
    const std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange(CoreCoord(1, 1), CoreCoord(1, 1)))}};

    {
        experimental::GlobalCircularBuffer gcb(*mesh_device, mesh_device->mesh_command_queue(1), mapping, 3200);
        Synchronize(mesh_device.get(), std::nullopt);
    }
    {
        // The overload without a queue uses cq 0.
        experimental::GlobalCircularBuffer gcb(*mesh_device, mapping, 3200);
        Synchronize(mesh_device.get(), std::nullopt);
    }
}

TEST_F(ExplicitCqTest, InspectorTraceIdOfGivenQueue) {
    auto mesh_device = get_mesh_device();

    EXPECT_FALSE(experimental::inspector::GetCurrentMeshTraceId(mesh_device->mesh_command_queue(0)).has_value());
    EXPECT_FALSE(experimental::inspector::GetCurrentMeshTraceId(mesh_device->mesh_command_queue(1)).has_value());
}

TEST_F(ExplicitCqTest, QueueOfAnotherMeshDeviceIsRejected) {
    auto mesh_device = get_mesh_device();
    if (mesh_device->num_devices() < 2) {
        GTEST_SKIP() << "Needs a mesh with at least two devices to build a submesh that differs from the parent";
    }
    auto submeshes = mesh_device->create_submeshes(MeshShape(1, 1));
    ASSERT_GE(submeshes.size(), 1u);
    auto& submesh = *submeshes[0];
    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    // The parent's queue does not belong to the submesh.
    auto& parent_cq = mesh_device->mesh_command_queue(0);
    EXPECT_ANY_THROW(GlobalSemaphore(submesh, parent_cq, cores, /*initial_value=*/1));

    GlobalSemaphore semaphore(submesh, cores, /*initial_value=*/1);
    EXPECT_ANY_THROW(semaphore.reset_semaphore_value(2, parent_cq));

    const std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange(CoreCoord(1, 1), CoreCoord(1, 1)))}};
    EXPECT_ANY_THROW(experimental::GlobalCircularBuffer(submesh, parent_cq, mapping, 3200));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
