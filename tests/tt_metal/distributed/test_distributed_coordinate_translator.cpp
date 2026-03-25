// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/distributed/distributed_coordinate_translator.hpp"

namespace tt::tt_metal {

using distributed::MeshCoordinate;
using distributed::MeshShape;

// All coords local, single-host mesh where everything is owned locally.
TEST(DistributedCoordinateTranslatorTest, FullLocalMesh) {
    DistributedCoordinateTranslator translator(MeshShape(2, 4), MeshShape(2, 4), MeshCoordinate(0, 0));

    for (uint32_t r = 0; r < 2; r++) {
        for (uint32_t c = 0; c < 4; c++) {
            EXPECT_TRUE(translator.is_local(MeshCoordinate(r, c)));
        }
    }
}

// Dual-host T3K split: host-0 owns left 2x2, host-1 owns right 2x2.
// SDMeshCommandQueue::write_shard_to_device was missing an is_local guard and
// would crash on get_device_buffer() for the remote coords.
TEST(DistributedCoordinateTranslatorTest, PartialLocalMesh) {
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

// Linear 1x4 mesh split across two hosts.
TEST(DistributedCoordinateTranslatorTest, LinearMesh) {
    DistributedCoordinateTranslator translator(MeshShape(1, 4), MeshShape(1, 2), MeshCoordinate(0, 0));

    EXPECT_TRUE(translator.is_local(MeshCoordinate(0, 0)));
    EXPECT_TRUE(translator.is_local(MeshCoordinate(0, 1)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(0, 2)));
    EXPECT_FALSE(translator.is_local(MeshCoordinate(0, 3)));
}

// Out-of-bounds coord must throw, not silently return false.
TEST(DistributedCoordinateTranslatorTest, OutOfBoundsThrows) {
    DistributedCoordinateTranslator translator(MeshShape(2, 4), MeshShape(2, 2), MeshCoordinate(0, 0));
    EXPECT_THROW(translator.is_local(MeshCoordinate(5, 5)), std::exception);
}

}  // namespace tt::tt_metal
