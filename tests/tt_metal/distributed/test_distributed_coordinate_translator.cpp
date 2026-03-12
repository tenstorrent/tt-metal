// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/mesh_coord.hpp>

#include "tt_metal/distributed/distributed_coordinate_translator.hpp"

namespace tt::tt_metal {
namespace {

using distributed::MeshCoordinate;
using distributed::MeshShape;

TEST(DistributedCoordinateTranslatorTest, FullLocalMesh) {
    DistributedCoordinateTranslator translator(MeshShape(2, 4), MeshShape(2, 4), MeshCoordinate(0, 0));

    for (uint32_t row = 0; row < 2; row++) {
        for (uint32_t col = 0; col < 4; col++) {
            EXPECT_TRUE(translator.is_local(MeshCoordinate(row, col)));
        }
    }
}

// Dual-host T3K split: host-0 owns left 2x2, host-1 owns right 2x2.
// SDMeshCommandQueue::write_shard_to_device was missing an is_local guard and would
// crash on get_device_buffer() for remote coords.
TEST(DistributedCoordinateTranslatorTest, PartialLocalMeshLeftHalf) {
    DistributedCoordinateTranslator translator(MeshShape(2, 4), MeshShape(2, 2), MeshCoordinate(0, 0));

    EXPECT_TRUE(translator.is_local(MeshCoordinate(0, 0)));
    EXPECT_TRUE(translator.is_local(MeshCoordinate(0, 1)));
    EXPECT_TRUE(translator.is_local(MeshCoordinate(1, 0)));
    EXPECT_TRUE(translator.is_local(MeshCoordinate(1, 1)));

    EXPECT_FALSE(translator.is_local(MeshCoordinate(0, 2)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(0, 3)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(1, 2)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(1, 3)));
}

TEST(DistributedCoordinateTranslatorTest, PartialLocalMeshRightHalf) {
    DistributedCoordinateTranslator translator(MeshShape(2, 4), MeshShape(2, 2), MeshCoordinate(0, 2));

    EXPECT_FALSE(translator.is_local(MeshCoordinate(0, 0)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(0, 1)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(1, 0)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(1, 1)));

    EXPECT_TRUE(translator.is_local(MeshCoordinate(0, 2)));
    EXPECT_TRUE(translator.is_local(MeshCoordinate(0, 3)));
    EXPECT_TRUE(translator.is_local(MeshCoordinate(1, 2)));
    EXPECT_TRUE(translator.is_local(MeshCoordinate(1, 3)));
}

TEST(DistributedCoordinateTranslatorTest, LinearMeshOffset) {
    DistributedCoordinateTranslator translator(MeshShape(1, 8), MeshShape(1, 4), MeshCoordinate(0, 4));

    for (uint32_t col = 0; col < 4; col++) {
        EXPECT_FALSE(translator.is_local(MeshCoordinate(0, col)));
    }
    for (uint32_t col = 4; col < 8; col++) {
        EXPECT_TRUE(translator.is_local(MeshCoordinate(0, col)));
    }
}

TEST(DistributedCoordinateTranslatorTest, OutOfBoundsCoordThrows) {
    DistributedCoordinateTranslator translator(MeshShape(2, 4), MeshShape(2, 2), MeshCoordinate(0, 0));

    EXPECT_ANY_THROW(translator.is_local(MeshCoordinate(2, 0)));
    EXPECT_ANY_THROW(translator.is_local(MeshCoordinate(0, 4)));
}

TEST(DistributedCoordinateTranslatorTest, InvalidConfigThrows) {
    EXPECT_ANY_THROW(DistributedCoordinateTranslator(MeshShape(2, 4), MeshShape(2, 3), MeshCoordinate(0, 2)));
    EXPECT_ANY_THROW(DistributedCoordinateTranslator(MeshShape(2, 4), MeshShape(3, 2), MeshCoordinate(0, 0)));
}

}  // namespace
}  // namespace tt::tt_metal
