// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/unit_mesh_group.hpp>

namespace tt::tt_metal::distributed {
namespace {

// Writes a pattern to DRAM on the member and reads it back.
void expect_dram_roundtrip(MeshDevice& mesh, uint32_t seed) {
    constexpr uint32_t num_words = 1024;
    const DeviceLocalBufferConfig local_config{
        .page_size = sizeof(uint32_t) * num_words, .buffer_type = BufferType::DRAM};
    const ReplicatedBufferConfig global_config{.size = sizeof(uint32_t) * num_words};
    auto buffer = MeshBuffer::create(global_config, local_config, &mesh);

    std::vector<uint32_t> src(num_words);
    std::iota(src.begin(), src.end(), seed);
    EnqueueWriteMeshBuffer(mesh.mesh_command_queue(), buffer, src, /*blocking=*/true);

    std::vector<uint32_t> dst;
    EnqueueReadMeshBuffer(mesh.mesh_command_queue(), dst, buffer, /*blocking=*/true);
    EXPECT_EQ(dst, src);
}

TEST(UnitMeshGroupTest, OpenSingleChipAndUseMember) {
    const std::vector<ChipId> chip_ids{0};
    auto group = UnitMeshGroup::open(chip_ids);

    EXPECT_EQ(group.size(), 1u);
    EXPECT_TRUE(group.contains(0));
    EXPECT_FALSE(group.contains(1000));
    EXPECT_EQ(group.chip_ids(), chip_ids);

    MeshDevice& mesh = group.at(0);
    EXPECT_EQ(mesh.num_devices(), 1u);
    EXPECT_EQ(mesh.get_devices().front()->id(), 0);
    expect_dram_roundtrip(mesh, /*seed=*/7);
}

TEST(UnitMeshGroupTest, EmptyChipListThrows) { EXPECT_THROW(UnitMeshGroup::open({}), std::runtime_error); }

TEST(UnitMeshGroupTest, UnknownChipThrows) {
    auto group = UnitMeshGroup::open(std::vector<ChipId>{0});
    EXPECT_THROW(group.at(1000), std::runtime_error);
}

TEST(UnitMeshGroupTest, CloseIsIdempotentAndLeavesGroupEmpty) {
    auto group = UnitMeshGroup::open(std::vector<ChipId>{0});
    group.close();
    EXPECT_TRUE(group.empty());
    EXPECT_FALSE(group.contains(0));
    EXPECT_NO_THROW(group.close());
}

// The chips can be opened again once the group is gone, both after close() and after plain destruction.
TEST(UnitMeshGroupTest, ChipsAreReusableAfterCloseAndDestruction) {
    {
        auto group = UnitMeshGroup::open(std::vector<ChipId>{0});
        expect_dram_roundtrip(group.at(0), /*seed=*/1);
        group.close();
    }
    {
        auto group = UnitMeshGroup::open(std::vector<ChipId>{0});
        expect_dram_roundtrip(group.at(0), /*seed=*/2);
    }
    auto group = UnitMeshGroup::open(std::vector<ChipId>{0});
    expect_dram_roundtrip(group.at(0), /*seed=*/3);
}

TEST(UnitMeshGroupTest, MoveTransfersOwnership) {
    auto group = UnitMeshGroup::open(std::vector<ChipId>{0});
    const MeshDevice* member = &group.at(0);

    UnitMeshGroup moved = std::move(group);
    EXPECT_TRUE(group.empty());  // NOLINT(bugprone-use-after-move): a moved-from group is specified to be empty
    ASSERT_EQ(moved.size(), 1u);
    EXPECT_EQ(&moved.at(0), member);
    expect_dram_roundtrip(moved.at(0), /*seed=*/4);

    // Move-assigning a group over another one closes what the target held.
    UnitMeshGroup target;
    target = std::move(moved);
    EXPECT_EQ(target.size(), 1u);
    EXPECT_TRUE(moved.empty());  // NOLINT(bugprone-use-after-move)
}

TEST(UnitMeshGroupTest, OpenSeveralChipsTogether) {
    if (GetNumAvailableDevices() < 2) {
        GTEST_SKIP() << "Needs at least 2 devices";
    }
    auto group = UnitMeshGroup::open(std::vector<ChipId>{1, 0});
    EXPECT_EQ(group.chip_ids(), (std::vector<ChipId>{0, 1}));

    uint32_t seed = 100;
    for (auto [chip_id, mesh] : group) {
        EXPECT_EQ(mesh.get_devices().front()->id(), chip_id);
        expect_dram_roundtrip(mesh, seed++);
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed
