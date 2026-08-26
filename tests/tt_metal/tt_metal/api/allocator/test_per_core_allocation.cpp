// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Integration tests for per-core L1 allocation via experimental::per_core_allocation.
// These tests require a real device (slow dispatch).

#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tests/tt_metal/tt_metal/common/device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace per_core = experimental::per_core_allocation;
namespace range_lockstep = experimental::range_lockstep_allocation;

class PerCoreAllocationTest : public MeshDeviceSingleCardBufferFixture {
protected:
    void SetUp() override {
        // Enable HYBRID allocator mode before device creation.
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);

        if (!this->validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        std::vector<ChipId> ids;
        for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
            ids.push_back(id);
        }
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        id_to_device_ = distributed::MeshDevice::create_unit_meshes(
            ids, l1_small_size_, trace_region_size_, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
        devices_.clear();
        for (const auto& [device_id, device] : id_to_device_) {
            devices_.push_back(device);
        }
        init_max_cbs();
    }

    void TearDown() override {
        MeshDeviceSingleCardBufferFixture::TearDown();
        unsetenv("TT_METAL_ALLOCATOR_MODE_HYBRID");
    }
};

// Use 1024-byte page size to be safely above all alignment requirements
// (FreeListOpt internally uses DRAM alignment which may be larger than L1 alignment)
static constexpr DeviceAddr PAGE_SIZE = 1024;

TEST_F(PerCoreAllocationTest, BasicPerCoreAllocation) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u) << "Need at least 2 compute cores";

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    DeviceAddr total_size = PAGE_SIZE * num_cores;

    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(shard_args, true);

    auto buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);

    ASSERT_TRUE(per_core::is_per_core_allocation(*buf));

    // Verify each core has an address within L1 range
    auto cores = corerange_to_cores(CoreRangeSet(core_range), std::nullopt, true);
    for (uint32_t i = 0; i < num_cores; i++) {
        auto addr = per_core::get_per_core_address(*buf, cores[i]);
        EXPECT_GT(addr, 0u) << "Core " << i << " address should be non-zero (above L1 base)";
        EXPECT_LT(addr, device->l1_size_per_core()) << "Core " << i << " address exceeds L1 size";
    }
}

TEST_F(PerCoreAllocationTest, PerCoreAndLockstepCoexist) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u);

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    DeviceAddr total_size = 2 * PAGE_SIZE * num_cores;

    // Create per-core buffer
    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(shard_args, true);
    auto per_core_buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);

    // Create lockstep buffer on same cores
    auto lockstep_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    auto lockstep_buf = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, lockstep_args);

    // Both should be allocated successfully
    EXPECT_TRUE(per_core_buf->is_allocated());
    EXPECT_TRUE(lockstep_buf->is_allocated());

    // Lockstep address should not overlap any per-core address
    auto lockstep_addr = lockstep_buf->address();
    auto cores = corerange_to_cores(CoreRangeSet(core_range), std::nullopt, true);
    for (uint32_t i = 0; i < num_cores; i++) {
        auto pc_addr = per_core::get_per_core_address(*per_core_buf, cores[i]);
        EXPECT_NE(lockstep_addr, pc_addr) << "Lockstep address overlaps per-core address at core " << i;
    }
}

// A lockstep buffer takes one address. By default the allocator keeps that address clear of
// per-core allocations on EVERY bank, because an op may reach the buffer on a core outside its
// own shard grid (a multicast writes to its whole rectangle). Opting into range lockstep narrows
// that to the cores the buffer occupies.
//
// `expect_range_lockstep` selects which behaviour is asserted, so the same scenario covers both:
// hog most of one core's L1 with a per-core allocation, then ask for the same size on a DIFFERENT
// core. Scoped to its own cores that fits; scanning chip-wide it does not.
//
// Goes through MeshBuffer: hybrid_device_allocators_ is only populated for mesh allocations, and
// the ranges are only gathered when it is non-empty. A buffer states its cores either through a
// shard spec or a distribution spec, and both must scope the scan, so each gets its own test --
// a failed allocation perturbs the allocator, so they cannot share one.
void run_lockstep_beside_per_core_hog(
    const std::shared_ptr<distributed::MeshDevice>& md,
    const std::function<BufferShardingArgs(const CoreCoord&)>& make_lockstep_args,
    bool expect_range_lockstep) {
    // Under LOCKSTEP the ranges are never gathered, so this would pass without proving anything.
    // The mode is latched at the first MetalContext construction, so skip loudly.
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        GTEST_SKIP() << "HYBRID allocator mode is not active in this process (it is latched at the first "
                        "MetalContext construction); run this binary with TT_METAL_ALLOCATOR_MODE_HYBRID=1";
    }
    auto* device = md->get_devices()[0];
    ASSERT_GE(device->compute_with_storage_grid_size().x, 2u) << "Need two disjoint compute cores";

    const CoreCoord hogged_core(0, 0);
    const CoreCoord free_core(1, 0);

    // Over half of L1, so a chip-wide scan cannot fit the second allocation.
    const auto stats = device->allocator()->get_statistics(BufferType::L1);
    const DeviceAddr alloc_size = (stats.largest_free_block_bytes * 6 / 10) / PAGE_SIZE * PAGE_SIZE;
    ASSERT_GT(alloc_size, stats.largest_free_block_bytes / 2)
        << "Sized under half of L1; both allocations would fit even under a chip-wide scan";

    // Occupy most of hogged_core's L1 with a per-core allocation.
    auto hog_args = BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(hogged_core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    per_core::set_per_core_allocation(hog_args, true);
    const distributed::DeviceLocalBufferConfig hog_local_config{
        .page_size = alloc_size,
        .buffer_type = BufferType::L1,
        .sharding_args = hog_args,
        .bottom_up = false,
    };
    auto hog_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = alloc_size}, hog_local_config, md.get());
    ASSERT_TRUE(per_core::is_per_core_allocation(*hog_buffer));

    // free_core holds nothing, so the same size must still fit there.
    const distributed::DeviceLocalBufferConfig lockstep_local_config{
        .page_size = alloc_size,
        .buffer_type = BufferType::L1,
        .sharding_args = make_lockstep_args(free_core),
        .bottom_up = false,
    };
    auto allocate_lockstep = [&] {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = alloc_size}, lockstep_local_config, md.get());
    };

    if (!expect_range_lockstep) {
        // Default lockstep still avoids per-core allocations everywhere, so this cannot be placed.
        EXPECT_ANY_THROW(allocate_lockstep())
            << "Default lockstep placed on " << free_core.str() << " despite a per-core allocation on "
            << hogged_core.str() << "; the chip-wide scan is what makes a multicast past its own cores safe";
        return;
    }

    std::shared_ptr<distributed::MeshBuffer> lockstep_buffer;
    ASSERT_NO_THROW(lockstep_buffer = allocate_lockstep())
        << "Range lockstep allocation on " << free_core.str() << " was blocked by a per-core allocation on "
        << hogged_core.str();
    ASSERT_NE(lockstep_buffer, nullptr);
    EXPECT_TRUE(lockstep_buffer->is_allocated());
}

namespace {
BufferShardingArgs shard_spec_args(const CoreCoord& core) {
    return BufferShardingArgs(
        ShardSpecBuffer(CoreRangeSet(core), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
}

// shard_spec_ is unset on this path, and cores_with_data() is what Buffer::num_cores() reports.
BufferShardingArgs distribution_spec_args(const CoreCoord& core) {
    return BufferShardingArgs(BufferDistributionSpec(Shape({1, 1}), Shape({1, 1}), std::vector<CoreCoord>{core}));
}
}  // namespace

TEST_F(PerCoreAllocationTest, RangeLockstepIgnoresPerCoreRangesOnOtherCores) {
    run_lockstep_beside_per_core_hog(
        this->devices_[0],
        [](const CoreCoord& core) {
            auto args = shard_spec_args(core);
            range_lockstep::set_range_lockstep_allocation(args, true);
            return args;
        },
        /*expect_range_lockstep=*/true);
}

TEST_F(PerCoreAllocationTest, RangeLockstepIgnoresPerCoreRangesOnOtherCoresWithDistributionSpec) {
    run_lockstep_beside_per_core_hog(
        this->devices_[0],
        [](const CoreCoord& core) {
            auto args = distribution_spec_args(core);
            range_lockstep::set_range_lockstep_allocation(args, true);
            return args;
        },
        /*expect_range_lockstep=*/true);
}

// Without the opt-in the chip-wide scan stays, which is what keeps a multicast past a buffer's
// own cores from landing on a per-core allocation. Same scenario, opposite expectation.
TEST_F(PerCoreAllocationTest, DefaultLockstepStillAvoidsPerCoreRangesEverywhere) {
    run_lockstep_beside_per_core_hog(this->devices_[0], shard_spec_args, /*expect_range_lockstep=*/false);
}

TEST_F(PerCoreAllocationTest, RangeLockstepRejectsPerCoreAllocation) {
    auto args = shard_spec_args(CoreCoord(0, 0));
    per_core::set_per_core_allocation(args, true);
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(args, true))
        << "A buffer cannot both take one address across its cores and an independent address on each";
}

TEST_F(PerCoreAllocationTest, RangeLockstepRejectsInterleaved) {
    BufferShardingArgs interleaved;
    EXPECT_ANY_THROW(range_lockstep::set_range_lockstep_allocation(interleaved, true))
        << "An interleaved buffer spans every bank, so there is no narrower core set to scope to";
}

TEST_F(PerCoreAllocationTest, DeallocationFreesPerCoreSpace) {
    auto* device = this->devices_[0]->get_devices()[0];
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores = static_cast<uint32_t>(std::min<size_t>(4, compute_grid.x));
    ASSERT_GE(num_cores, 2u);

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(num_cores - 1, 0));
    std::array<uint32_t, 2> shard_shape = {32, 32};
    std::array<uint32_t, 2> page_shape = {32, 32};
    std::array<uint32_t, 2> tensor2d_shape = {num_cores, 1};

    ShardSpecBuffer shard_spec(
        CoreRangeSet(core_range), shard_shape, ShardOrientation::ROW_MAJOR, page_shape, tensor2d_shape);

    DeviceAddr total_size = 2 * PAGE_SIZE * num_cores;

    auto shard_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    experimental::per_core_allocation::set_per_core_allocation(shard_args, true);

    // Create and destroy a buffer
    {
        auto buf1 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
        EXPECT_TRUE(per_core::is_per_core_allocation(*buf1));
        // buf1 destroyed here, freeing per-core allocations
    }

    // Create another buffer on same cores — should succeed (space was freed)
    auto buf2 = Buffer::create(device, total_size, PAGE_SIZE, BufferType::L1, shard_args);
    EXPECT_TRUE(per_core::is_per_core_allocation(*buf2));
    EXPECT_TRUE(buf2->is_allocated());
}

// ================== Per-core socket data-buffer allocation (Phase B) ==================
// Exercises SocketMemoryConfig{..., per_core_allocation=true}: the receiver FIFO should be allocated with the
// per-core allocator (only on the receiver connection core) instead of lockstep across every worker core, and
// the handshake metadata should carry that per-core address.

namespace {

// Builds a single-connection, single-device (md == md) per-core socket on the fixture's first unit mesh.
std::pair<distributed::MeshSocket, distributed::MeshSocket> make_per_core_socket(
    const std::shared_ptr<distributed::MeshDevice>& md,
    const CoreCoord& sender_core,
    const CoreCoord& receiver_core,
    uint32_t fifo_size) {
    distributed::SocketConnection connection(
        distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), sender_core),
        distributed::MeshCoreCoord(distributed::MeshCoordinate(0, 0), receiver_core));
    distributed::SocketMemoryConfig mem_config(
        BufferType::L1,
        fifo_size,
        /*sender_sub_device=*/std::nullopt,
        /*receiver_sub_device=*/std::nullopt,
        /*per_core_allocation=*/true);
    distributed::SocketConfig socket_config({connection}, mem_config);
    return distributed::MeshSocket::create_socket_pair(md, md, socket_config);
}

}  // namespace

TEST_F(PerCoreAllocationTest, PerCoreSocketDataBufferPlacement) {
    auto md = this->devices_[0];
    auto* device = md->get_devices()[0];

    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(0, 1);
    const uint32_t fifo_size = 2048;

    auto [send_socket, recv_socket] = make_per_core_socket(md, sender_core, receiver_core, fifo_size);

    auto data_buffer = recv_socket.get_data_buffer();
    ASSERT_TRUE(per_core::is_per_core_allocation(*data_buffer));

    // The FIFO occupies L1 only on the receiver core, at a valid per-core address.
    auto pc_addr = per_core::get_per_core_address(*data_buffer, distributed::MeshCoordinate(0, 0), receiver_core);
    EXPECT_GT(pc_addr, 0u) << "Receiver per-core address should be above the L1 base";
    EXPECT_LT(pc_addr, device->l1_size_per_core()) << "Receiver per-core address exceeds L1 size";

    // A per-core buffer has no single lockstep address.
    EXPECT_EQ(data_buffer->address(), 0u);
}

TEST_F(PerCoreAllocationTest, PerCoreSocketCoexistsWithLockstep) {
    auto md = this->devices_[0];
    auto* device = md->get_devices()[0];

    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(0, 1);
    const uint32_t fifo_size = 2048;

    auto [send_socket, recv_socket] = make_per_core_socket(md, sender_core, receiver_core, fifo_size);
    auto pc_addr = per_core::get_per_core_address(
        *recv_socket.get_data_buffer(), distributed::MeshCoordinate(0, 0), receiver_core);

    // A subsequent lockstep L1 buffer spanning the receiver core must not alias the per-core socket FIFO.
    CoreRange grid(CoreCoord(0, 0), CoreCoord(1, 0));
    ShardSpecBuffer shard_spec(CoreRangeSet(grid), {32, 32}, ShardOrientation::ROW_MAJOR, {32, 32}, {2, 1});
    auto lockstep_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    auto lockstep_buf = Buffer::create(device, 2 * PAGE_SIZE, PAGE_SIZE, BufferType::L1, lockstep_args);
    EXPECT_TRUE(lockstep_buf->is_allocated());
    EXPECT_NE(lockstep_buf->address(), pc_addr) << "Lockstep buffer aliases the per-core socket FIFO";
}

TEST_F(PerCoreAllocationTest, PerCoreSocketConfigMetadataUsesPerCoreAddress) {
    auto md = this->devices_[0];

    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(0, 1);
    const uint32_t fifo_size = 2048;

    auto [send_socket, recv_socket] = make_per_core_socket(md, sender_core, receiver_core, fifo_size);
    auto pc_addr = per_core::get_per_core_address(
        *recv_socket.get_data_buffer(), distributed::MeshCoordinate(0, 0), receiver_core);

    // The receiver's on-device config must point read_ptr/fifo_addr at the receiver core's per-core address.
    std::vector<receiver_socket_md> recv_config_readback;
    distributed::ReadShard(
        md->mesh_command_queue(),
        recv_config_readback,
        recv_socket.get_config_buffer(),
        distributed::MeshCoordinate(0, 0));
    ASSERT_EQ(recv_config_readback.size(), 1u);
    EXPECT_EQ(recv_config_readback[0].fifo_addr, pc_addr);
    EXPECT_EQ(recv_config_readback[0].read_ptr, pc_addr);
    EXPECT_EQ(recv_config_readback[0].fifo_total_size, fifo_size);

    // The sender's downstream_fifo_addr must match the same per-core address.
    std::vector<uint8_t> sender_config_bytes;
    distributed::ReadShard(
        md->mesh_command_queue(),
        sender_config_bytes,
        send_socket.get_config_buffer(),
        distributed::MeshCoordinate(0, 0));
    ASSERT_GE(sender_config_bytes.size(), sizeof(sender_socket_md));
    sender_socket_md sender_md{};
    std::memcpy(&sender_md, sender_config_bytes.data(), sizeof(sender_socket_md));
    EXPECT_EQ(sender_md.downstream_fifo_addr, pc_addr);
    EXPECT_EQ(sender_md.downstream_fifo_total_size, fifo_size);
}

// ================== Per-core H2D socket FIFO (Phase C) ==================
// In HYBRID mode the receiver FIFO is allocated per-core rather than reserving fifo_size on every
// worker core. A per-core buffer has no lockstep address(), so reading address() would silently
// point the host at the L1 base; data_buffer_ is private, so these assert via the public surfaces.

namespace {

// The FIFO base must stay aligned to this, and fifo_size must be a multiple of it (ctor TT_FATAL).
uint32_t h2d_host_alignment() { return MetalContext::instance().hal().get_alignment(HalMemType::HOST); }

// Why per-core H2D sockets can't run here, or nullopt when they can.
std::optional<std::string> per_core_h2d_skip_reason(const std::shared_ptr<distributed::MeshDevice>& md) {
    // The allocator mode is latched at the first MetalContext construction, so skip loudly rather
    // than assert per-core properties against a silently-LOCKSTEP device.
    if (!MetalContext::instance().rtoptions().get_allocator_mode_hybrid()) {
        return "HYBRID allocator mode is not active in this process (it is latched at the first "
               "MetalContext construction); run this binary with TT_METAL_ALLOCATOR_MODE_HYBRID=1";
    }
    if (!experimental::GetMemoryPinningParameters(*md).can_map_to_noc) {
        return "Mapping host memory to NOC is not supported on this system";
    }
    return std::nullopt;
}

// Reads the receiver_socket_md the socket wrote to the receiver core's L1 config buffer.
receiver_socket_md read_h2d_receiver_config(
    const std::shared_ptr<distributed::MeshDevice>& md,
    const distributed::H2DSocket& socket,
    const distributed::MeshCoreCoord& recv_core) {
    receiver_socket_md config{};
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto virtual_core = md->worker_core_from_logical_core(recv_core.core_coord);
    cluster.read_core(
        &config,
        sizeof(config),
        tt_cxy_pair(md->get_device(recv_core.device_coord)->id(), virtual_core),
        socket.get_config_buffer_address());
    return config;
}

}  // namespace

// The FIFO base handed to the host must be a real per-core L1 address. Under the lockstep-address
// bug this is align(0) == 0, so the assertion below is the direct regression guard.
TEST_F(PerCoreAllocationTest, H2DSocketPerCoreFifoBaseIsAPerCoreAddress) {
    auto md = this->devices_[0];
    if (auto reason = per_core_h2d_skip_reason(md)) {
        GTEST_SKIP() << *reason;
    }
    auto* device = md->get_devices()[0];

    const uint32_t fifo_size = 4 * h2d_host_alignment();
    const distributed::MeshCoreCoord recv_core(distributed::MeshCoordinate(0, 0), CoreCoord(0, 1));
    distributed::H2DSocket socket(md, recv_core, BufferType::L1, fifo_size, distributed::H2DMode::HOST_PUSH);

    const auto desc = socket.populate_descriptor();
    EXPECT_GT(desc.aligned_data_buf_start, 0u)
        << "FIFO base is 0 — a per-core buffer has no lockstep address(), so the host would push to the L1 base";
    EXPECT_LT(desc.aligned_data_buf_start, device->l1_size_per_core()) << "FIFO base exceeds L1 size";
    EXPECT_EQ(desc.aligned_data_buf_start % h2d_host_alignment(), 0u) << "FIFO base must stay PCIe-aligned";
    EXPECT_EQ(desc.fifo_size, fifo_size);
}

// The on-device config the receiver kernel reads must point at the same per-core address, or the
// kernel drains from one place while the host writes to another.
TEST_F(PerCoreAllocationTest, H2DSocketPerCoreConfigMetadataUsesPerCoreAddress) {
    auto md = this->devices_[0];
    if (auto reason = per_core_h2d_skip_reason(md)) {
        GTEST_SKIP() << *reason;
    }

    const uint32_t fifo_size = 4 * h2d_host_alignment();
    const distributed::MeshCoreCoord recv_core(distributed::MeshCoordinate(0, 0), CoreCoord(0, 1));
    distributed::H2DSocket socket(md, recv_core, BufferType::L1, fifo_size, distributed::H2DMode::HOST_PUSH);

    const auto desc = socket.populate_descriptor();
    const auto config = read_h2d_receiver_config(md, socket, recv_core);
    EXPECT_EQ(config.fifo_addr, desc.aligned_data_buf_start);
    EXPECT_EQ(config.read_ptr, desc.aligned_data_buf_start);
    EXPECT_EQ(config.fifo_total_size, fifo_size);
}

// A lockstep L1 buffer spanning the receiver core must not overlap the per-core FIFO. MeshBuffer
// mirrors every lockstep L1 allocation into each device's per-core allocator precisely so this
// holds; compare RANGES, since a per-core FIFO sitting just below a lockstep base still corrupts.
TEST_F(PerCoreAllocationTest, H2DSocketPerCoreFifoDoesNotAliasLockstep) {
    auto md = this->devices_[0];
    if (auto reason = per_core_h2d_skip_reason(md)) {
        GTEST_SKIP() << *reason;
    }
    auto* device = md->get_devices()[0];

    const uint32_t fifo_size = 4 * h2d_host_alignment();
    const distributed::MeshCoreCoord recv_core(distributed::MeshCoordinate(0, 0), CoreCoord(0, 1));
    distributed::H2DSocket socket(md, recv_core, BufferType::L1, fifo_size, distributed::H2DMode::HOST_PUSH);
    const DeviceAddr fifo_base = socket.populate_descriptor().aligned_data_buf_start;

    // Lockstep buffer over a grid that includes the receiver core.
    CoreRange grid(CoreCoord(0, 0), CoreCoord(0, 1));
    ShardSpecBuffer shard_spec(CoreRangeSet(grid), {32, 32}, ShardOrientation::ROW_MAJOR, {32, 32}, {2, 1});
    auto lockstep_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
    auto lockstep_buf = Buffer::create(device, 2 * PAGE_SIZE, PAGE_SIZE, BufferType::L1, lockstep_args);
    ASSERT_TRUE(lockstep_buf->is_allocated());

    const DeviceAddr lockstep_base = lockstep_buf->address();
    const DeviceAddr lockstep_end = lockstep_base + lockstep_buf->aligned_size_per_bank();
    const DeviceAddr fifo_end = fifo_base + fifo_size;
    EXPECT_TRUE(fifo_end <= lockstep_base || lockstep_end <= fifo_base)
        << "per-core FIFO [" << fifo_base << ", " << fifo_end << ") overlaps lockstep buffer [" << lockstep_base << ", "
        << lockstep_end << ")";
}

// ---------- End-to-end ----------
// The receiver kernel drains from receiver_socket.read_ptr, exactly the value init_data_buffer
// computed, so a wrong FIFO base surfaces here as mismatched payload bytes (or a hang).
TEST_F(PerCoreAllocationTest, H2DSocketPerCoreEndToEndTransfer) {
    auto md = this->devices_[0];
    if (auto reason = per_core_h2d_skip_reason(md)) {
        GTEST_SKIP() << *reason;
    }

    const uint32_t page_size = h2d_host_alignment();
    const uint32_t fifo_size = 4 * page_size;   // 4-page FIFO
    const uint32_t data_size = 12 * page_size;  // 12 pages => forces multiple wraps
    const uint32_t num_iterations = 4;
    const distributed::MeshCoreCoord recv_core(distributed::MeshCoordinate(0, 0), CoreCoord(0, 1));

    distributed::H2DSocket socket(md, recv_core, BufferType::L1, fifo_size, distributed::H2DMode::HOST_PUSH);
    socket.set_page_size(page_size);

    // Lockstep landing buffer the receiver kernel writes each drained page into.
    const distributed::ReplicatedBufferConfig landing_config{.size = data_size};
    auto landing_shard =
        ShardSpecBuffer(CoreRangeSet(recv_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const distributed::DeviceLocalBufferConfig landing_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(landing_shard, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto landing_buffer = distributed::MeshBuffer::create(landing_config, landing_local_config, md.get());

    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(socket.get_config_buffer_address()),
                static_cast<uint32_t>(landing_buffer->address()),
                page_size,
                data_size,
                num_iterations,
            }});

    auto workload = distributed::MeshWorkload();
    workload.add_program(distributed::MeshCoordinateRange(recv_core.device_coord), std::move(recv_program));
    distributed::EnqueueMeshWorkload(md->mesh_command_queue(), workload, false);

    const uint32_t page_size_words = page_size / sizeof(uint32_t);
    const uint32_t num_pages = data_size / page_size;
    std::vector<uint32_t> src(data_size / sizeof(uint32_t));

    const auto& cluster = MetalContext::instance().get_cluster();
    const auto recv_core_virtual = md->worker_core_from_logical_core(recv_core.core_coord);
    for (uint32_t i = 0; i < num_iterations; i++) {
        // Distinct payload per iteration, so a stale-read would not accidentally match.
        std::iota(src.begin(), src.end(), i);
        for (uint32_t p = 0; p < num_pages; p++) {
            socket.write(src.data() + (p * page_size_words), 1);
        }
        socket.barrier();

        std::vector<uint32_t> readback(data_size / sizeof(uint32_t));
        cluster.read_core(
            readback.data(),
            data_size,
            tt_cxy_pair(md->get_device(recv_core.device_coord)->id(), recv_core_virtual),
            landing_buffer->address());
        EXPECT_EQ(src, readback) << "payload mismatch on iteration " << i;
    }
}

}  // namespace tt::tt_metal
