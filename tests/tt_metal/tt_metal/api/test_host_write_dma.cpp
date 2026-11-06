// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>

#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

// How the case writes its data: an L1 write with the default Strict ordering, an L1 write that asks for Relaxed
// ordering, or a DRAM channel write, which is Relaxed.
enum class HostWriteKind { L1Strict, L1Relaxed, Dram };

struct HostWriteCase {
    std::string name;
    HostWriteKind kind;
    uint32_t size_bytes;
    uint32_t address_offset = 0;
};

namespace {

// Cluster::write_core sends Relaxed writes of 64 KiB or more through PCIe DMA. Strict writes stay on MMIO on Blackhole
// and use DMA from 32 B on Wormhole, and anything whose size or address is not a multiple of 4 stays on MMIO. These
// cases land on both sides of each of those rules.
const std::vector<HostWriteCase> kHostWriteCases = {
    {"L1Strict1KiB", HostWriteKind::L1Strict, 1024},
    {"L1Strict256KiB", HostWriteKind::L1Strict, 256 * 1024},
    {"L1StrictUnalignedSize", HostWriteKind::L1Strict, 2050},
    {"L1RelaxedMmio32KiB", HostWriteKind::L1Relaxed, 32 * 1024},
    {"L1RelaxedDma64KiB", HostWriteKind::L1Relaxed, 64 * 1024},
    {"L1RelaxedMmio64KiBAtUnalignedAddress", HostWriteKind::L1Relaxed, 64 * 1024, 2},
    {"DramMmio32KiB", HostWriteKind::Dram, 32 * 1024},
    {"DramDma64KiB", HostWriteKind::Dram, 64 * 1024},
    {"DramDma4MiB", HostWriteKind::Dram, 4 * 1024 * 1024},
    {"DramMmio64KiBAtUnalignedAddress", HostWriteKind::Dram, 64 * 1024, 2},
};

}  // namespace

class HostWriteDmaTest : public UnitMeshFixture, public ::testing::WithParamInterface<HostWriteCase> {};

// A host write reads back intact whichever transport it is routed through.
TEST_P(HostWriteDmaTest, WriteReadsBack) {
    auto& mesh_device = *this->devices_[0];
    const HostWriteCase& test_case = GetParam();
    const HalMemType mem_type = test_case.kind == HostWriteKind::Dram ? HalMemType::DRAM : HalMemType::L1;
    const uint32_t address = mesh_device.allocator()->get_base_allocator_addr(mem_type) + test_case.address_offset;

    std::vector<uint8_t> pattern(test_case.size_bytes);
    for (uint32_t i = 0; i < test_case.size_bytes; ++i) {
        pattern[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    std::vector<uint8_t> readback(test_case.size_bytes, 0);

    switch (test_case.kind) {
        case HostWriteKind::L1Strict:
            slow_dispatch::WriteToL1(mesh_device, CoreCoord{0, 0}, address, std::span<const uint8_t>(pattern));
            break;
        case HostWriteKind::L1Relaxed: {
            const ChipId chip_id = mesh_device.get_device_ids()[0];
            const CoreCoord worker_core = mesh_device.worker_core_from_logical_core(CoreCoord{0, 0});
            MetalContext::instance().get_cluster().write_core(
                pattern.data(),
                pattern.size(),
                tt_cxy_pair(chip_id, worker_core),
                address,
                tt::umd::IoOrdering::Relaxed);
            break;
        }
        case HostWriteKind::Dram:
            slow_dispatch::WriteToDRAMChannel(mesh_device, 0, address, std::span<const uint8_t>(pattern));
            break;
    }

    if (mem_type == HalMemType::L1) {
        slow_dispatch::ReadFromL1(mesh_device, CoreCoord{0, 0}, address, std::span<uint8_t>(readback));
    } else {
        slow_dispatch::ReadFromDRAMChannel(mesh_device, 0, address, std::span<uint8_t>(readback));
    }

    EXPECT_EQ(readback, pattern);
}

INSTANTIATE_TEST_SUITE_P(
    HostWrites,
    HostWriteDmaTest,
    ::testing::ValuesIn(kHostWriteCases),
    [](const ::testing::TestParamInfo<HostWriteCase>& info) { return info.param.name; });

namespace {

// Two cores, 32 bf16 tiles each. Height sharded, each core's shard is one contiguous 64 KiB write, which takes DMA.
// Interleaved, every 2 KiB tile is its own write, which takes Relaxed MMIO.
constexpr uint32_t kTileBytes = tt::constants::TILE_HW * 2;
constexpr uint32_t kTilesPerCore = 32;
constexpr uint32_t kNumCores = 2;
constexpr uint32_t kBufferBytes = kTileBytes * kTilesPerCore * kNumCores;
constexpr int kOrderingIterations = 1000;

std::shared_ptr<distributed::MeshBuffer> make_l1_buffer(distributed::MeshDevice& mesh_device, bool sharded) {
    distributed::DeviceLocalBufferConfig local_config{.page_size = kTileBytes, .buffer_type = BufferType::L1};
    if (sharded) {
        const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(0, kNumCores - 1)));
        local_config.sharding_args = BufferShardingArgs(
            ShardSpecBuffer(
                cores,
                {kTilesPerCore * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
                ShardOrientation::ROW_MAJOR,
                {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
                {kTilesPerCore * kNumCores, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED);
    }
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kBufferBytes}, local_config, &mesh_device);
}

}  // namespace

class L1BufferWriteOrderingTest : public UnitMeshFixture, public ::testing::WithParamInterface<bool> {};

// L1 buffer data is written Relaxed, so these rely on WriteToBuffer's barrier to keep back-to-back writes to the
// same bytes in issue order.
TEST_P(L1BufferWriteOrderingTest, BackToBackWritesKeepTheSecond) {
    auto& mesh_device = *this->devices_[0];
    const auto buffer = make_l1_buffer(mesh_device, GetParam());
    const std::vector<uint32_t> first(kBufferBytes / sizeof(uint32_t), 0x11111111);
    const std::vector<uint32_t> second(kBufferBytes / sizeof(uint32_t), 0x22222222);

    for (int i = 0; i < kOrderingIterations; ++i) {
        slow_dispatch::WriteToBuffer(*buffer, first);
        slow_dispatch::WriteToBuffer(*buffer, second);

        std::vector<uint32_t> readback;
        slow_dispatch::ReadFromBuffer(*buffer, readback);
        ASSERT_EQ(readback, std::vector<uint32_t>(kBufferBytes / sizeof(uint32_t), 0x22222222)) << "iteration " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    L1Layouts,
    L1BufferWriteOrderingTest,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? std::string("Sharded") : std::string("Interleaved");
    });

// A small Strict write right after a Relaxed buffer write lands last, even where it overlaps the DMA'd shard.
TEST_F(UnitMeshFixture, L1BufferWriteThenSmallStrictWriteKeepsTheSmallWrite) {
    auto& mesh_device = *this->devices_[0];
    const auto buffer = make_l1_buffer(mesh_device, /*sharded=*/true);
    const uint32_t shard_bytes = kTileBytes * kTilesPerCore;
    const uint32_t small_bytes = 1024;
    const uint32_t small_address = buffer->address() + shard_bytes - small_bytes;
    const std::vector<uint32_t> buffer_data(kBufferBytes / sizeof(uint32_t), 0x33333333);
    std::vector<uint32_t> small_data(small_bytes / sizeof(uint32_t), 0x44444444);

    std::vector<uint32_t> expected(shard_bytes / sizeof(uint32_t), 0x33333333);
    std::fill(expected.end() - small_bytes / sizeof(uint32_t), expected.end(), 0x44444444);

    for (int i = 0; i < kOrderingIterations; ++i) {
        slow_dispatch::WriteToBuffer(*buffer, buffer_data);
        slow_dispatch::WriteToL1(mesh_device, CoreCoord{0, 0}, small_address, small_data);

        std::vector<uint32_t> readback;
        slow_dispatch::ReadFromL1(mesh_device, CoreCoord{0, 0}, buffer->address(), shard_bytes, readback);
        ASSERT_EQ(readback, expected) << "iteration " << i;
    }
}

}  // namespace tt::tt_metal
