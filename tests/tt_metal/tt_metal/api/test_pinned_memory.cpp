// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Coverage for the device-access mode of experimental::PinnedMemory: which mapping permissions Create() will
// hand out, how PinnedMemoryCache reconciles a request against an already-cached mapping with different
// permissions, and what the transfer paths do when a caller supplies a mapping the device may only read.
//
// Device-based tests use GenericMeshDeviceFixture and require hardware access.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <tt_stl/span.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "impl/context/metal_context.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "tt_metal/impl/dispatch/vector_aligned.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using PinnedMemoryTestSuite = GenericMeshDeviceFixture;

// A device-read-only mapping cannot be a D2H destination: the device would NOC-write into an IOMMU mapping that
// faults, which shows up as a hung read rather than a test failure. issue_read_buffer_dispatch_command_sequence
// guards against that by excluding read-only mappings from the direct pinned-write path. Nothing else in the tree
// reaches that guard -- every in-tree read pins ReadWrite -- so this attaches a read-only pin to a read transfer
// explicitly and checks the data still arrives via the fallback.
TEST_F(PinnedMemoryTestSuite, EnqueueReadShardsWithReadOnlyPinnedMemoryUsesUnpinnedPath) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (!pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    std::vector<uint32_t> src(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer = distributed::ShardDataTransfer{coord}
                              .host_data(static_cast<void*>(const_cast<uint32_t*>(src.data())))
                              .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    auto dst = std::make_shared<vector_aligned<uint32_t>>(bytes_per_device / sizeof(uint32_t), 0);
    uint32_t* dst_ptr_aligned = reinterpret_cast<uint32_t*>(dst->data());
    HostBuffer host_buffer(ttsl::Span<uint32_t>(dst_ptr_aligned, bytes_per_device / sizeof(uint32_t)), MemoryPin(dst));

    // The host memory is writable; only the device's view of it is restricted, which is what makes this the case
    // the guard has to catch rather than one the driver would reject up front.
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    std::shared_ptr<experimental::PinnedMemory> pinned_read_only = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true,
        experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(pinned_read_only);
    ASSERT_EQ(pinned_read_only->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadOnly);

    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(static_cast<void*>(dst_ptr_aligned))
                             .region(BufferRegion(0, bytes_per_device));
    experimental::ShardDataTransferSetPinnedMemory(read_transfer, pinned_read_only);
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);

    std::vector<uint32_t> dst_aligned(dst_ptr_aligned, dst_ptr_aligned + (bytes_per_device / sizeof(uint32_t)));

    EXPECT_EQ(dst_aligned, src);
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheReadOnlyRequestReusesReadWriteMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    ASSERT_TRUE(read_write);
    ASSERT_EQ(read_write->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    EXPECT_EQ(read_only.get(), read_write.get());
    EXPECT_EQ(read_only->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheCreatesBestSupportedReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));

    auto pinned = experimental::PinnedMemoryCache::instance().try_pin(
        *mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(pinned);
    const auto expected_access = pinning_params.supports_read_only ? experimental::PinnedMemoryDeviceAccess::ReadOnly
                                                                   : experimental::PinnedMemoryDeviceAccess::ReadWrite;
    EXPECT_EQ(pinned->get_device_access(), expected_access);
}

// The mirror image of the read-only tests above: they need the feature present, this one needs it absent. That
// makes it the only one of the group that runs on the current fleet, which is still below the KMD 2.9.0 floor,
// and the only coverage of PinnedMemory::Create's public contract. PinnedMemoryCache::try_pin cannot cover it:
// try_pin widens an unsupported read-only request to ReadWrite, while Create -- reachable directly by callers --
// must reject it up front rather than let a raw UMD error escape a tt-metal API.
TEST_F(PinnedMemoryTestSuite, PinnedMemoryCreateRejectsReadOnlyWhenUnsupported) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (pinning_params.supports_read_only) {
        GTEST_SKIP() << "Requires a system without device-read-only pinning support";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));

    try {
        experimental::PinnedMemory::Create(
            *mesh_device_, range, host_buffer, /*map_to_noc=*/true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
        FAIL() << "Create() must reject a ReadOnly request on a system without support for it";
    } catch (const std::exception& e) {
        // Checking the text, not just that something threw: the contract is that the caller is told which
        // parameter to consult, so a bare throw from somewhere else in Create() would not satisfy it.
        EXPECT_NE(std::string(e.what()).find("supports_read_only"), std::string::npos) << e.what();
    }
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheReadWriteRequestReplacesUnreferencedReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    std::weak_ptr<experimental::PinnedMemory> old_mapping = read_only;
    read_only.reset();

    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    ASSERT_TRUE(read_write);
    EXPECT_TRUE(old_mapping.expired());
    EXPECT_EQ(read_write->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheReadWriteRequestRejectsHeldReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    EXPECT_EQ(read_write, nullptr);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
