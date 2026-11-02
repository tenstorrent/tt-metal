// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Metal has no implicit (thread-local) command queue selection. APIs that issue a write on behalf of the caller take
// the queue id explicitly (cq 0 by default). These tests check that the explicit id is honored: a valid second queue
// works, and an out-of-range id is rejected (which can only happen if the id reaches the queue lookup).

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
#include <tt-metalium/mesh_device.hpp>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {
namespace {

using ExplicitCqIdTest = GenericMultiCQMeshDeviceFixture;

constexpr uint8_t kOutOfRangeCqId = 2;  // The fixture opens exactly 2 command queues.

uint32_t read_semaphore(const MeshDevice& mesh_device, const CoreCoord& logical_core, DeviceAddr address) {
    const auto device_id = mesh_device.get_device_ids()[0];
    return MetalContext::instance().get_cluster().read_core(
        device_id, mesh_device.worker_core_from_logical_core(logical_core), address, sizeof(uint32_t))[0];
}

TEST_F(ExplicitCqIdTest, GlobalSemaphoreCreateAndResetOnSecondQueue) {
    auto mesh_device = get_mesh_device();
    ASSERT_EQ(mesh_device->num_hw_cqs(), 2);
    const CoreCoord core(0, 0);
    const CoreRangeSet cores(CoreRange(core, core));

    GlobalSemaphore semaphore(*mesh_device, cores, /*initial_value=*/3, BufferType::L1, /*cq_id=*/1);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, semaphore.address()), 3u);

    semaphore.reset_semaphore_value(7, /*cq_id=*/1);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, semaphore.address()), 7u);

    // Omitted queue id is cq 0.
    semaphore.reset_semaphore_value(9);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, semaphore.address()), 9u);
}

TEST_F(ExplicitCqIdTest, GlobalSemaphoreRejectsOutOfRangeQueue) {
    auto mesh_device = get_mesh_device();
    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));

    EXPECT_ANY_THROW(GlobalSemaphore(*mesh_device, cores, /*initial_value=*/1, BufferType::L1, kOutOfRangeCqId));

    GlobalSemaphore semaphore(*mesh_device, cores, /*initial_value=*/1);
    EXPECT_ANY_THROW(semaphore.reset_semaphore_value(2, kOutOfRangeCqId));
}

TEST_F(ExplicitCqIdTest, AddressBackedGlobalSemaphoreInitOnRequestedQueue) {
    auto mesh_device = get_mesh_device();
    const CoreCoord core(0, 0);
    const CoreRangeSet cores(CoreRange(core, core));

    // Borrow the address of a regular semaphore as the caller-owned backing memory.
    GlobalSemaphore backing(*mesh_device, cores, /*initial_value=*/0);
    const DeviceAddr address = backing.address();

    auto at_address = experimental::CreateGlobalSemaphore(
        *mesh_device, cores, /*initial_value=*/5, BufferType::L1, address, /*cq_id=*/1);
    Synchronize(mesh_device.get(), std::nullopt);
    EXPECT_EQ(read_semaphore(*mesh_device, core, address), 5u);

    EXPECT_ANY_THROW(experimental::CreateGlobalSemaphore(
        *mesh_device, cores, /*initial_value=*/6, BufferType::L1, address, kOutOfRangeCqId));
}

TEST_F(ExplicitCqIdTest, GlobalCircularBufferConfigWriteOnSecondQueue) {
    auto mesh_device = get_mesh_device();
    const std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
        {CoreCoord(0, 0), CoreRangeSet(CoreRange(CoreCoord(1, 1), CoreCoord(1, 1)))}};

    {
        experimental::GlobalCircularBuffer gcb(*mesh_device, mapping, 3200, BufferType::L1, /*cq_id=*/1);
        Synchronize(mesh_device.get(), std::nullopt);
    }
    EXPECT_ANY_THROW(experimental::GlobalCircularBuffer(*mesh_device, mapping, 3200, BufferType::L1, kOutOfRangeCqId));
}

TEST_F(ExplicitCqIdTest, InspectorTraceIdLooksUpRequestedQueue) {
    auto mesh_device = get_mesh_device();

    EXPECT_FALSE(experimental::inspector::GetCurrentMeshTraceId(mesh_device.get(), /*cq_id=*/0).has_value());
    EXPECT_FALSE(experimental::inspector::GetCurrentMeshTraceId(mesh_device.get(), /*cq_id=*/1).has_value());
    EXPECT_ANY_THROW(experimental::inspector::GetCurrentMeshTraceId(mesh_device.get(), kOutOfRangeCqId));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
