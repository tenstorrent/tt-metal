// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <tt-metalium/distributed_mesh_shape.hpp>
#include <tt-metalium/maybe_remote.hpp>

namespace tt::tt_metal {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::SizeIs;

TEST(DistributedMeshShapeTest, InvalidLocalShapeExceedsGlobal) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(3, 2);  // Exceeds global in first dimension
    distributed::MeshCoordinate local_offset(0, 0);

    EXPECT_ANY_THROW(DistributedMeshShape(global_shape, local_shape, local_offset));
}

TEST(DistributedMeshShapeTest, InvalidLocalOffsetExceedsGlobal) {
    distributed::MeshShape global_shape(4, 4);
    distributed::MeshShape local_shape(2, 2);
    distributed::MeshCoordinate local_offset(3, 3);  // Offset + shape exceeds global

    EXPECT_ANY_THROW(DistributedMeshShape(global_shape, local_shape, local_offset));
}

TEST(DistributedMeshShapeTest, DimensionMismatch) {
    distributed::MeshShape global_shape(2, 3);
    distributed::MeshShape local_shape(2, 2, 1);  // 3D vs 2D
    distributed::MeshCoordinate local_offset(0, 0);

    EXPECT_ANY_THROW(DistributedMeshShape(global_shape, local_shape, local_offset));
}

TEST(DistributedMeshShapeTest, FullyLocal) {
    distributed::MeshShape shape(2, 3);
    DistributedMeshShape dist_shape(shape);

    EXPECT_EQ(dist_shape.shape(), shape);
    EXPECT_TRUE(dist_shape.fully_local());

    for (const auto& coord : distributed::MeshCoordinateRange(shape)) {
        EXPECT_TRUE(dist_shape.is_local(coord));
    }
}

TEST(DistributedMeshShapeTest, FullyLocalWithLocalShapeAndOffset) {
    distributed::MeshShape global_shape(3, 4);
    distributed::MeshShape local_shape(3, 4);
    distributed::MeshCoordinate local_offset(0, 0);

    DistributedMeshShape dist_shape(global_shape, local_shape, local_offset);

    EXPECT_EQ(dist_shape.shape(), global_shape);
    EXPECT_TRUE(dist_shape.fully_local());

    // Verify all coordinates are local
    for (const auto& coord : distributed::MeshCoordinateRange(global_shape)) {
        EXPECT_TRUE(dist_shape.is_local(coord));
    }
}

TEST(DistributedMeshShapeTest, FullyLocalWithDistributedMeshContainer) {
    distributed::MeshShape shape(2, 3);
    distributed::DistributedMeshContainer<int> container(shape);

    // Make all devices local
    for (const auto& coord : distributed::MeshCoordinateRange(shape)) {
        container.at(coord) = distributed::MaybeRemote<int>::local(42);
    }

    DistributedMeshShape dist_shape(container);

    EXPECT_EQ(dist_shape.shape(), shape);
    EXPECT_TRUE(dist_shape.fully_local());

    // Verify all coordinates are local
    for (const auto& coord : distributed::MeshCoordinateRange(shape)) {
        EXPECT_TRUE(dist_shape.is_local(coord));
    }
}

TEST(DistributedMeshShapeTest, UnitMesh) {
    distributed::MeshShape shape(1);
    DistributedMeshShape dist_shape(shape);

    EXPECT_EQ(dist_shape.shape(), shape);
    EXPECT_TRUE(dist_shape.fully_local());
    EXPECT_TRUE(dist_shape.is_local(distributed::MeshCoordinate(0)));
}

TEST(DistributedMeshShapeTest, WithLocalShapeAndOffset) {
    distributed::MeshShape global_shape(4, 6);
    distributed::MeshShape local_shape(2, 3);
    distributed::MeshCoordinate local_offset(1, 2);

    DistributedMeshShape dist_shape(global_shape, local_shape, local_offset);

    EXPECT_EQ(dist_shape.shape(), global_shape);
    EXPECT_FALSE(dist_shape.fully_local());

    // Check that only the local region is marked as local
    for (const auto& coord : distributed::MeshCoordinateRange(global_shape)) {
        bool expected_local = (coord[0] >= 1 && coord[0] < 3) && (coord[1] >= 2 && coord[1] < 5);
        EXPECT_EQ(dist_shape.is_local(coord), expected_local)
            << "Coordinate " << coord << " expected to be " << (expected_local ? "local" : "remote");
    }
}

TEST(DistributedMeshShapeTest, FromDistributedMeshContainer) {
    distributed::MeshShape shape(2, 3);
    distributed::DistributedMeshContainer<int> container(shape);

    // Set up a pattern where some devices are local and some are remote
    container.at(distributed::MeshCoordinate(0, 0)) = distributed::MaybeRemote<int>::local(10);
    container.at(distributed::MeshCoordinate(0, 1)) = distributed::MaybeRemote<int>::remote();
    container.at(distributed::MeshCoordinate(0, 2)) = distributed::MaybeRemote<int>::local(20);
    container.at(distributed::MeshCoordinate(1, 0)) = distributed::MaybeRemote<int>::remote();
    container.at(distributed::MeshCoordinate(1, 1)) = distributed::MaybeRemote<int>::local(30);
    container.at(distributed::MeshCoordinate(1, 2)) = distributed::MaybeRemote<int>::remote();

    DistributedMeshShape dist_shape(container);

    EXPECT_EQ(dist_shape.shape(), shape);
    EXPECT_FALSE(dist_shape.fully_local());

    // Check local/remote pattern matches
    EXPECT_TRUE(dist_shape.is_local(distributed::MeshCoordinate(0, 0)));
    EXPECT_FALSE(dist_shape.is_local(distributed::MeshCoordinate(0, 1)));
    EXPECT_TRUE(dist_shape.is_local(distributed::MeshCoordinate(0, 2)));
    EXPECT_FALSE(dist_shape.is_local(distributed::MeshCoordinate(1, 0)));
    EXPECT_TRUE(dist_shape.is_local(distributed::MeshCoordinate(1, 1)));
    EXPECT_FALSE(dist_shape.is_local(distributed::MeshCoordinate(1, 2)));
}

TEST(DistributedMeshShapeTest, OutOfBounds) {
    distributed::MeshShape shape(2, 3);
    DistributedMeshShape dist_shape(shape);

    // Valid coordinates should not throw
    EXPECT_NO_THROW(dist_shape.is_local(distributed::MeshCoordinate(0, 0)));
    EXPECT_NO_THROW(dist_shape.is_local(distributed::MeshCoordinate(1, 2)));

    // Out of bounds coordinates should throw
    EXPECT_ANY_THROW(dist_shape.is_local(distributed::MeshCoordinate(2, 0)));
    EXPECT_ANY_THROW(dist_shape.is_local(distributed::MeshCoordinate(0, 3)));
    EXPECT_ANY_THROW(dist_shape.is_local(distributed::MeshCoordinate(2, 3)));
}

}  // namespace
}  // namespace tt::tt_metal
