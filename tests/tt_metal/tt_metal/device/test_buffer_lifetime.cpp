// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #57381: a Buffer whose allocation ended must not touch a dead allocator or free a range a newer buffer holds.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_allocator.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-umd/types/arch.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/emulation/emule_live_ranges.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {
namespace {

class BufferLifetimeMockTest : public ::testing::TestWithParam<tt::ARCH> {
protected:
    void SetUp() override {
        env_ = std::make_unique<MetalEnv>(
            MetalEnvDescriptor{.mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(GetParam(), chips_)});
        device_ = distributed::MeshDevice::open_mesh_device(
            *env_, distributed::MeshDeviceConfig(distributed::MeshShape(1, chips_)));
    }

    void TearDown() override {
        device_.reset();
        env_.reset();
    }

    uint32_t chips_ = 1;  // a 1 x chips_ mesh
    std::unique_ptr<MetalEnv> env_;
    std::shared_ptr<distributed::MeshDevice> device_;
};

void expect_inert(const std::shared_ptr<Buffer>& buffer) {
    EXPECT_FALSE(buffer->is_allocated());
    // An explicit release followed by destruction must both be harmless.
    buffer->impl().deallocate(*buffer);
    buffer->impl().deallocate(*buffer);
}

// Plain Buffers: a MeshBuffer drops its backing buffer along with its MeshDevice, which would hide the bug.
TEST_P(BufferLifetimeMockTest, CPU_BufferOutlivesDeviceAndContext) {
    auto dram = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    auto l1 = BufferImpl::create(device_.get(), 4096, 4096, BufferType::L1);
    auto empty = BufferImpl::create(device_.get(), 0, 32, BufferType::DRAM);
    auto non_owning = BufferImpl::create(device_.get(), dram->address(), 4096, 4096, BufferType::DRAM);
    for (const auto& buffer : {dram, l1, empty, non_owning}) {
        EXPECT_TRUE(buffer->is_allocated());
    }

    device_->close();
    device_.reset();
    env_.reset();

    for (const auto& buffer : {dram, l1, empty, non_owning}) {
        expect_inert(buffer);
    }
}

TEST_P(BufferLifetimeMockTest, CPU_BufferDestructionAfterDeviceClose) {
    auto buffer = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    device_->close();
    expect_inert(buffer);
}

// clear() keeps the allocator usable; a buffer from before it must not free its same-address replacement.
TEST_P(BufferLifetimeMockTest, CPU_BufferDestructionAfterAllocatorClear) {
    const auto& allocator = device_->allocator_impl();
    auto buffer = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    const auto address = buffer->address();
    allocator->clear();
    EXPECT_EQ(allocator->get_num_allocated_buffers(), 0u);
    EXPECT_FALSE(buffer->is_allocated());

    auto replacement = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    EXPECT_EQ(replacement->address(), address);
    EXPECT_TRUE(replacement->is_allocated());
    const auto allocated_bytes = allocator->get_statistics(BufferType::DRAM).total_allocated_bytes;

    buffer.reset();
    EXPECT_TRUE(replacement->is_allocated());
    EXPECT_EQ(allocator->get_num_allocated_buffers(), 1u);
    EXPECT_EQ(allocator->get_statistics(BufferType::DRAM).total_allocated_bytes, allocated_bytes);

    replacement.reset();
    EXPECT_EQ(allocator->get_num_allocated_buffers(), 0u);
    EXPECT_EQ(allocator->get_statistics(BufferType::DRAM).total_allocated_bytes, 0u);
}

DeviceAddr allocated_dram_bytes(const AllocatorImpl& allocator) {
    return allocator.get_statistics(BufferType::DRAM).total_allocated_bytes;
}

// After ttnn.deallocate_buffers(), surviving tensor storage must not free its same-address replacement.
TEST_P(BufferLifetimeMockTest, CPU_MeshBufferDestructionAfterDeallocateBuffers) {
    const auto& allocator = device_->allocator_impl();
    const distributed::DeviceLocalBufferConfig local{.page_size = 4096, .buffer_type = BufferType::DRAM};
    const distributed::ReplicatedBufferConfig global{.size = 4096};
    auto stale = distributed::MeshBuffer::create(global, local, device_.get());
    const auto address = stale->address();
    device_->allocator()->deallocate_buffers();
    EXPECT_EQ(allocator->get_num_allocated_buffers(), 0u);
    EXPECT_FALSE(stale->get_backing_buffer()->is_allocated());

    auto replacement = distributed::MeshBuffer::create(global, local, device_.get());
    EXPECT_EQ(replacement->address(), address);
    const auto allocated_bytes = allocated_dram_bytes(*allocator);
    EXPECT_GT(allocated_bytes, 0u);

    stale.reset();
    EXPECT_EQ(allocated_dram_bytes(*allocator), allocated_bytes);
    EXPECT_EQ(allocator->get_num_allocated_buffers(), 1u);

    replacement.reset();
    EXPECT_EQ(allocated_dram_bytes(*allocator), 0u);
}

// The same with a Buffer, next to one that does not own its memory: deallocate_buffers() does not end that one.
TEST_P(BufferLifetimeMockTest, CPU_BufferDestructionAfterDeallocateBuffers) {
    const auto& allocator = device_->allocator_impl();
    auto stale = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    const auto address = stale->address();
    auto non_owning = BufferImpl::create(device_.get(), address, 4096, 4096, BufferType::DRAM);
    allocator->deallocate_buffers();
    EXPECT_FALSE(stale->is_allocated());
    EXPECT_TRUE(non_owning->is_allocated());
    EXPECT_EQ(allocator->get_num_allocated_buffers(), 0u);

    auto replacement = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    EXPECT_EQ(replacement->address(), address);
    const auto allocated_bytes = allocated_dram_bytes(*allocator);

    stale.reset();
    non_owning.reset();
    EXPECT_TRUE(replacement->is_allocated());
    EXPECT_EQ(allocated_dram_bytes(*allocator), allocated_bytes);

    replacement.reset();
    EXPECT_EQ(allocated_dram_bytes(*allocator), 0u);
}

// override_state() ends owning buffers' allocations; a non-owning one (like unit-mesh aggregation's) stays allocated.
TEST_P(BufferLifetimeMockTest, CPU_BufferDestructionAfterOverrideState) {
    const auto& allocator = device_->allocator_impl();
    const auto empty = experimental::extract_mock_allocator_state(*device_);
    auto stale = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    const auto address = stale->address();
    auto non_owning = BufferImpl::create(device_.get(), address, 4096, 4096, BufferType::DRAM);
    experimental::override_mock_allocator_state(*device_, empty);
    EXPECT_FALSE(stale->is_allocated());
    EXPECT_TRUE(non_owning->is_allocated());

    auto replacement = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    EXPECT_EQ(replacement->address(), address);
    const auto allocated_bytes = allocated_dram_bytes(*allocator);

    stale.reset();
    EXPECT_TRUE(replacement->is_allocated());
    EXPECT_EQ(allocated_dram_bytes(*allocator), allocated_bytes);

    replacement.reset();
    EXPECT_EQ(allocated_dram_bytes(*allocator), 0u);
}

// A deallocation that checked the buffer before clear() and reaches the allocator after a reallocation frees nothing.
TEST_P(BufferLifetimeMockTest, CPU_ReleasedAllocationIsNotFreedAgain) {
    const auto& allocator = device_->allocator_impl();
    auto stale = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    const auto address = stale->address();
    allocator->clear();
    auto replacement = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    ASSERT_EQ(replacement->address(), address);
    const auto allocated_bytes = allocated_dram_bytes(*allocator);

    allocator->deallocate_buffer(stale.get());
    EXPECT_TRUE(replacement->is_allocated());
    EXPECT_EQ(allocated_dram_bytes(*allocator), allocated_bytes);

    replacement.reset();
    EXPECT_EQ(allocated_dram_bytes(*allocator), 0u);
}

// Runs AllocatorImpl::clear() inside Buffer allocation, after construction and before AllocatorImpl::allocate_buffer.
class ClearBeforeAllocateHook : public IGraphHooks {
public:
    explicit ClearBeforeAllocateHook(AllocatorImpl* allocator) : allocator_(allocator) {}
    bool hook_allocate(const Buffer* /*buffer*/) override {
        allocator_->clear();
        return false;
    }
    bool hook_deallocate(Buffer* /*buffer*/) override { return false; }
    bool hook_program(Program* /*program*/) override { return false; }
    bool hook_write_to_device(const Buffer* /*buffer*/) override { return false; }
    bool hook_read_from_device(Buffer* /*buffer*/) override { return false; }
    bool hook_read_from_device(const distributed::MeshBuffer* /*mesh_buffer*/) override { return false; }
    bool hook_write_to_device(const distributed::MeshBuffer* /*mesh_buffer*/) override { return false; }

private:
    AllocatorImpl* allocator_;
};

struct ScopedGraphHook {
    explicit ScopedGraphHook(const std::shared_ptr<IGraphHooks>& hook) {
        EXPECT_TRUE(GraphTracker::instance().add_hook(hook));
    }
    ~ScopedGraphHook() { GraphTracker::instance().clear_hook(); }
    ScopedGraphHook(const ScopedGraphHook&) = delete;
    ScopedGraphHook& operator=(const ScopedGraphHook&) = delete;
};

// A clear() between a buffer's construction and its allocation does not end that allocation.
TEST_P(BufferLifetimeMockTest, CPU_ClearBetweenConstructionAndAllocation) {
    const auto& allocator = device_->allocator_impl();
    std::shared_ptr<Buffer> buffer;
    {
        const ScopedGraphHook hook(std::make_shared<ClearBeforeAllocateHook>(allocator.get()));
        buffer = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    }
    EXPECT_TRUE(buffer->is_allocated());
    EXPECT_GT(allocated_dram_bytes(*allocator), 0u);

    buffer.reset();
    EXPECT_EQ(allocated_dram_bytes(*allocator), 0u);
}

class DeallocationCounter : public IGraphProcessor {
public:
    bool is_capture_processor() const override { return false; }
    void track_deallocate(Buffer* /*buffer*/) override { ++deallocations; }
    int deallocations = 0;
};

// Destroying a buffer whose allocation clear() released still reports the deallocation to the graph tracker. With
// the allocator gone it is not reported: GraphProcessor::track_deallocate reads the allocator and the device.
TEST_P(BufferLifetimeMockTest, CPU_DeallocationAfterClearIsTracked) {
    auto buffer = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    device_->allocator_impl()->clear();
    auto counter = std::make_shared<DeallocationCounter>();
    GraphTracker::instance().push_processor(counter);
    buffer.reset();
    GraphTracker::instance().pop_processor();
    EXPECT_EQ(counter->deallocations, 1);
}

// Graph capture's NO_DISPATCH mode claims every allocation this way, so the buffer never reaches the allocator.
class HookEveryAllocation : public IGraphHooks {
public:
    bool hook_allocate(const Buffer* /*buffer*/) override { return true; }
    bool hook_deallocate(Buffer* /*buffer*/) override { return true; }
    bool hook_program(Program* /*program*/) override { return false; }
    bool hook_write_to_device(const Buffer* /*buffer*/) override { return false; }
    bool hook_read_from_device(Buffer* /*buffer*/) override { return false; }
    bool hook_read_from_device(const distributed::MeshBuffer* /*mesh_buffer*/) override { return false; }
    bool hook_write_to_device(const distributed::MeshBuffer* /*mesh_buffer*/) override { return false; }
};

// A bulk release ends only allocator allocations; hooked and empty buffers hold none and stay allocated.
TEST_P(BufferLifetimeMockTest, CPU_BulkReleaseKeepsBuffersWithoutAllocations) {
    std::shared_ptr<Buffer> hooked;
    {
        const ScopedGraphHook hook(std::make_shared<HookEveryAllocation>());
        hooked = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    }
    auto empty = BufferImpl::create(device_.get(), 0, 32, BufferType::DRAM);
    ASSERT_TRUE(hooked->is_allocated());
    ASSERT_TRUE(empty->is_allocated());

    device_->allocator()->deallocate_buffers();
    EXPECT_TRUE(hooked->is_allocated());
    EXPECT_TRUE(empty->is_allocated());
}

// Whether the emulator's padding registry holds [logical_end, physical_end) on the device.
bool has_padding(int device_id, uint32_t logical_end, uint32_t physical_end) {
    const auto regions = emule::LiveL1PaddingRanges::snapshot(device_id);
    const uint64_t packed = (static_cast<uint64_t>(logical_end) << 32) | physical_end;
    return std::find(regions.begin(), regions.end(), packed) != regions.end();
}

// A buffer whose allocation a bulk release ended drops the padding region it registered, not a replacement's.
TEST_P(BufferLifetimeMockTest, CPU_ReleasedBufferDropsItsPaddingRegion) {
    auto stale = BufferImpl::create(device_.get(), 4096, 4096, BufferType::L1);
    const int device_id = stale->device()->id();
    const auto start = static_cast<uint32_t>(stale->address());
    emule::LiveL1PaddingRanges::set(device_id, start, start + 2048, start + 4096, stale->unique_id());
    device_->allocator()->deallocate_buffers();
    stale.reset();
    EXPECT_FALSE(has_padding(device_id, start + 2048, start + 4096));

    auto older = BufferImpl::create(device_.get(), 4096, 4096, BufferType::L1);
    device_->allocator()->deallocate_buffers();
    auto replacement = BufferImpl::create(device_.get(), 4096, 4096, BufferType::L1);
    ASSERT_EQ(replacement->address(), start);
    emule::LiveL1PaddingRanges::set(device_id, start, start + 1024, start + 4096, replacement->unique_id());
    older.reset();
    EXPECT_TRUE(has_padding(device_id, start + 1024, start + 4096));
    emule::LiveL1PaddingRanges::clear(device_id, start);
}

// The same for a buffer that outlives its allocator.
TEST_P(BufferLifetimeMockTest, CPU_BufferOutlivingAllocatorDropsItsPaddingRegion) {
    auto buffer = BufferImpl::create(device_.get(), 4096, 4096, BufferType::L1);
    const int device_id = buffer->device()->id();
    const auto start = static_cast<uint32_t>(buffer->address());
    emule::LiveL1PaddingRanges::set(device_id, start, start + 2048, start + 4096, buffer->unique_id());
    device_->close();
    buffer.reset();
    const bool kept = has_padding(device_id, start + 2048, start + 4096);
    emule::LiveL1PaddingRanges::clear(device_id, start);
    EXPECT_FALSE(kept);
}

INSTANTIATE_TEST_SUITE_P(
    Architectures,
    BufferLifetimeMockTest,
    ::testing::Values(tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE),
    [](const ::testing::TestParamInfo<tt::ARCH>& info) {
        return info.param == tt::ARCH::BLACKHOLE ? "Blackhole" : "Wormhole";
    });

// Mock devices with the HYBRID allocator, which mirrors each mesh L1 allocation into the device's lockstep allocator.
class BufferLifetimeHybridMockTest : public BufferLifetimeMockTest {
protected:
    void SetUp() override {
        if (const char* value = std::getenv(kHybrid)) {
            previous_ = value;
        }
        setenv(kHybrid, "1", /*overwrite=*/1);
        BufferLifetimeMockTest::SetUp();
    }

    void TearDown() override {
        BufferLifetimeMockTest::TearDown();
        if (previous_) {
            setenv(kHybrid, previous_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(kHybrid);
        }
    }

    // The mesh's one device, whose allocator holds the mirrors.
    IDevice* chip() const { return device_->impl().get_device(distributed::MeshCoordinate(0, 0)); }
    AllocatorImpl& physical() const { return *chip()->allocator_impl(); }

    // A bulk release, then a replacement of another size that overlaps the stale MeshBuffer's mirror.
    void check_resized_replacement(DeviceAddr stale_size, DeviceAddr replacement_size, bool bottom_up);

    static constexpr const char* kHybrid = "TT_METAL_ALLOCATOR_MODE_HYBRID";
    std::optional<std::string> previous_;
};

DeviceAddr mirrored_l1_bytes(const AllocatorImpl& allocator) {
    const auto ranges = allocator.get_l1_allocated_ranges(BankManager::AllocatorDependencies::AllocatorID{0});
    return std::accumulate(ranges.begin(), ranges.end(), DeviceAddr{0}, [](DeviceAddr sum, const auto& r) {
        return sum + r.second - r.first;
    });
}

// One page per buffer, so each device mirrors exactly `size` bytes.
std::shared_ptr<distributed::MeshBuffer> l1_mesh_buffer(
    distributed::MeshDevice* device, DeviceAddr size, bool bottom_up = false) {
    const distributed::DeviceLocalBufferConfig local{
        .page_size = size, .buffer_type = BufferType::L1, .bottom_up = bottom_up};
    return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = size}, local, device);
}

// Destroying a MeshBuffer released by a bulk release must keep the mirror of its same-address replacement.
TEST_P(BufferLifetimeHybridMockTest, CPU_StaleMeshBufferKeepsReplacementMirror) {
    const DeviceAddr before = mirrored_l1_bytes(physical());
    auto stale = l1_mesh_buffer(device_.get(), 4096);
    ASSERT_EQ(mirrored_l1_bytes(physical()), before + 4096);

    device_->allocator()->deallocate_buffers();
    auto replacement = l1_mesh_buffer(device_.get(), 4096);
    ASSERT_EQ(replacement->address(), stale->address());
    stale.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 4096);

    replacement.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before);
}

// Destroying the replacement first frees the range, so a larger buffer can be mirrored over it.
TEST_P(BufferLifetimeHybridMockTest, CPU_ReplacementDestroyedBeforeStaleMeshBuffer) {
    const DeviceAddr before = mirrored_l1_bytes(physical());
    auto stale = l1_mesh_buffer(device_.get(), 4096);
    device_->allocator()->deallocate_buffers();
    auto replacement = l1_mesh_buffer(device_.get(), 4096);
    ASSERT_EQ(replacement->address(), stale->address());
    replacement.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before);

    std::shared_ptr<distributed::MeshBuffer> larger;
    ASSERT_NO_THROW(larger = l1_mesh_buffer(device_.get(), 8192));
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 8192);
    stale.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 8192);
    larger.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before);
}

void BufferLifetimeHybridMockTest::check_resized_replacement(
    DeviceAddr stale_size, DeviceAddr replacement_size, bool bottom_up) {
    const DeviceAddr before = mirrored_l1_bytes(physical());
    auto stale = l1_mesh_buffer(device_.get(), stale_size, bottom_up);
    device_->allocator()->deallocate_buffers();
    std::shared_ptr<distributed::MeshBuffer> replacement;
    ASSERT_NO_THROW(replacement = l1_mesh_buffer(device_.get(), replacement_size, bottom_up));
    ASSERT_LT(replacement->address(), stale->address() + stale_size);
    ASSERT_GT(replacement->address() + replacement_size, stale->address());
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + replacement_size);

    stale.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + replacement_size);
    replacement.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before);
}

TEST_P(BufferLifetimeHybridMockTest, CPU_LargerBottomUpReplacementOfStaleMeshBuffer) {
    check_resized_replacement(4096, 8192, /*bottom_up=*/true);
}

TEST_P(BufferLifetimeHybridMockTest, CPU_SmallerReplacementOfStaleMeshBuffer) {
    check_resized_replacement(4096, 2048, /*bottom_up=*/false);
}

TEST_P(BufferLifetimeHybridMockTest, CPU_LargerReplacementOfStaleMeshBuffer) {
    check_resized_replacement(4096, 8192, /*bottom_up=*/false);
}

// Bytes the device's per-core L1 allocators hold, over all banks.
DeviceAddr per_core_l1_bytes(const AllocatorImpl& allocator) {
    DeviceAddr bytes = 0;
    for (uint32_t bank = 0; bank < allocator.get_num_banks(BufferType::L1); ++bank) {
        for (const auto& [begin, end] :
             allocator.get_l1_allocated_ranges(BankManager::AllocatorDependencies::AllocatorID{bank + 1})) {
            bytes += end - begin;
        }
    }
    return bytes;
}

// A per-core buffer whose allocation a bulk release ended leaves its replacement's per-core ranges allocated.
TEST_P(BufferLifetimeHybridMockTest, CPU_StalePerCoreBufferKeepsReplacement) {
    const ShardSpecBuffer shard(
        CoreRangeSet(CoreRange({0, 0}, {1, 0})), {32, 32}, ShardOrientation::ROW_MAJOR, {32, 32}, {2, 1});
    auto args = BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(args, true);
    const DeviceAddr before = per_core_l1_bytes(physical());
    auto stale = BufferImpl::create(chip(), 2048, 1024, BufferType::L1, args);
    const DeviceAddr allocated = per_core_l1_bytes(physical());
    ASSERT_GT(allocated, before);

    physical().deallocate_buffers();
    auto replacement = BufferImpl::create(chip(), 2048, 1024, BufferType::L1, args);
    ASSERT_EQ(per_core_l1_bytes(physical()), allocated);
    stale.reset();
    EXPECT_EQ(per_core_l1_bytes(physical()), allocated);
    replacement.reset();
    EXPECT_EQ(per_core_l1_bytes(physical()), before);
}

// A mesh buffer mirrored where a device-level buffer already holds the range borrows it, and leaves it held.
TEST_P(BufferLifetimeHybridMockTest, CPU_MirrorBorrowsRangeHeldOnDevice) {
    const DeviceAddr before = mirrored_l1_bytes(physical());
    auto held = BufferImpl::create(chip(), 4096, 4096, BufferType::L1);
    ASSERT_EQ(mirrored_l1_bytes(physical()), before + 4096);
    auto mesh_buffer = l1_mesh_buffer(device_.get(), 4096);
    ASSERT_EQ(mesh_buffer->address(), held->address());
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 4096);

    mesh_buffer.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 4096);
    EXPECT_TRUE(held->is_allocated());
    held.reset();
    EXPECT_EQ(mirrored_l1_bytes(physical()), before);
}

// Live holders share one mirror; a later mark there drops only a released holder and keeps the range.
TEST_P(BufferLifetimeHybridMockTest, CPU_MirrorKeepsLiveHoldersPastReleasedOne) {
    const DeviceAddr before = mirrored_l1_bytes(physical());
    const DeviceAddr address = l1_mesh_buffer(device_.get(), 4096)->address();
    ASSERT_EQ(mirrored_l1_bytes(physical()), before);
    auto released = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    auto on_chip = BufferImpl::create(chip(), 4096, 4096, BufferType::DRAM);
    physical().mirror_lockstep_allocation(address, 4096, *released);
    physical().mirror_lockstep_allocation(address, 4096, *on_chip);
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 4096);

    device_->allocator()->deallocate_buffers();
    auto replacement = BufferImpl::create(device_.get(), 4096, 4096, BufferType::DRAM);
    physical().mirror_lockstep_allocation(address, 4096, *replacement);
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 4096);
    physical().unmirror_lockstep_allocation(address, *released);
    physical().unmirror_lockstep_allocation(address, *on_chip);
    EXPECT_EQ(mirrored_l1_bytes(physical()), before + 4096);
    physical().unmirror_lockstep_allocation(address, *replacement);
    EXPECT_EQ(mirrored_l1_bytes(physical()), before);
}

INSTANTIATE_TEST_SUITE_P(
    Architectures,
    BufferLifetimeHybridMockTest,
    ::testing::Values(tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE),
    [](const ::testing::TestParamInfo<tt::ARCH>& info) {
        return info.param == tt::ARCH::BLACKHOLE ? "Blackhole" : "Wormhole";
    });

// The same on a two-chip mesh: each device keeps the mirror of a resized replacement, at the replacement's size.
class BufferLifetimeHybridTwoChipMockTest : public BufferLifetimeHybridMockTest {
protected:
    BufferLifetimeHybridTwoChipMockTest() { chips_ = 2; }
};

TEST_P(BufferLifetimeHybridTwoChipMockTest, CPU_StaleMeshBufferOnEveryDevice) {
    auto mirrored = [this](uint32_t column) {
        return mirrored_l1_bytes(*device_->impl().get_device(distributed::MeshCoordinate(0, column))->allocator_impl());
    };
    const DeviceAddr before0 = mirrored(0);
    const DeviceAddr before1 = mirrored(1);
    auto stale = l1_mesh_buffer(device_.get(), 4096);
    device_->allocator()->deallocate_buffers();
    std::shared_ptr<distributed::MeshBuffer> replacement;
    ASSERT_NO_THROW(replacement = l1_mesh_buffer(device_.get(), 8192));
    stale.reset();
    EXPECT_EQ(mirrored(0), before0 + 8192);
    EXPECT_EQ(mirrored(1), before1 + 8192);
    replacement.reset();
    EXPECT_EQ(mirrored(0), before0);
    EXPECT_EQ(mirrored(1), before1);
}

INSTANTIATE_TEST_SUITE_P(
    Architectures,
    BufferLifetimeHybridTwoChipMockTest,
    ::testing::Values(tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE),
    [](const ::testing::TestParamInfo<tt::ARCH>& info) {
        return info.param == tt::ARCH::BLACKHOLE ? "Blackhole" : "Wormhole";
    });

// Loading a sub-device manager needs a command queue, so these cases need hardware and fast dispatch.
class BufferLifetimeDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
            GTEST_SKIP() << "Sub-device managers require fast dispatch";
        }
        if (get_physical_architecture() == tt::ARCH::Invalid) {
            GTEST_SKIP() << "No TT hardware detected";
        }
        if (MetalContext::instance().get_cluster().is_mock_or_emulated()) {
            GTEST_SKIP() << "Requires real hardware";
        }
    }
};

// Closing the device destroys the loaded manager's allocators while a buffer from one of them is still alive.
TEST_F(BufferLifetimeDeviceTest, TensixBufferOutlivesDeviceWithSubDeviceManager) {
    auto device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1, 1)));
    const CoreRangeSet cores(CoreRange({0, 0}));
    const SubDevice sub_device(std::array{cores});
    device->load_sub_device_manager(device->create_sub_device_manager({sub_device}, 4096));
    const ShardSpecBuffer shard(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    auto sub_device_buffer = BufferImpl::create(
        device.get(),
        32,
        32,
        BufferType::L1,
        BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED),
        false,
        SubDeviceId{0});
    auto dram = BufferImpl::create(device.get(), 4096, 4096, BufferType::DRAM);
    EXPECT_TRUE(sub_device_buffer->is_allocated());
    EXPECT_TRUE(dram->is_allocated());

    device->close();
    device.reset();

    expect_inert(sub_device_buffer);
    expect_inert(dram);
}

// A bulk release lets the device leave a manager whose buffer is still alive; destroying the buffer must not throw.
TEST_F(BufferLifetimeDeviceTest, TensixReleasedSubDeviceBufferAfterManagerSwitch) {
    auto device = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1, 1)));
    const CoreRangeSet cores(CoreRange({0, 0}));
    const SubDevice sub_device(std::array{cores});
    const auto manager = device->create_sub_device_manager({sub_device}, 4096);
    device->load_sub_device_manager(manager);
    const ShardSpecBuffer shard(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    auto buffer = BufferImpl::create(
        device.get(),
        32,
        32,
        BufferType::L1,
        BufferShardingArgs(shard, TensorMemoryLayout::HEIGHT_SHARDED),
        false,
        SubDeviceId{0});

    device->allocator(SubDeviceId{0})->deallocate_buffers();
    ASSERT_NO_THROW(device->clear_loaded_sub_device_manager());
    EXPECT_FALSE(buffer->is_allocated());
    EXPECT_NO_THROW(buffer->impl().deallocate(*buffer));
    buffer.reset();
    device->remove_sub_device_manager(manager);
}

}  // namespace
}  // namespace tt::tt_metal
