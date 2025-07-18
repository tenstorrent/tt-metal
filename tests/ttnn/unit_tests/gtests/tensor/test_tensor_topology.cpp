// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn_test_fixtures.hpp"
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>

namespace ttnn::distributed::test {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

using TensorTopologyTest = GenericMeshDeviceFixture;
using TensorTopologyT3000Test = T3000MeshDeviceFixture;

TEST_F(TensorTopologyTest, SingleDevice) {
    const auto tensor_spec =
        TensorSpec(ttnn::Shape{1, 1, 1, 3}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_tensor = Tensor::from_vector(std::vector<float>{42.F, 13.F, -99.F}, tensor_spec);

    const auto mesh_shape_override = MeshShape(1);
    auto mapper = create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}},
            .mesh_shape_override = mesh_shape_override,
        });
    Tensor replicated_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    // Tensor topology for tensor replicated across 2D (with override) should be same as mesh shape override
    const auto tensor_topology = replicated_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.mesh_shape, mesh_shape_override);

    auto coord = MeshCoordinate(0);

    // Check that get_device_coord returns the correct device coordinate
    EXPECT_EQ(tensor_topology.get_device_coord(coord), MeshCoordinate(0, 0));

    // Check that get_neighbor returns the correct neighbor coordinate (ie. same coordinate)
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 1, 0), coord);
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 2, 0), coord);
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -1, 0), coord);
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -2, 0), coord);

    // Check that get_neighbor throws for invalid dimension
    EXPECT_THAT(
        std::function<void()>([&tensor_topology, &coord]() { tensor_topology.get_neighbor(coord, 0, 1); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Index out of bounds: 1 not in [-1, 1)")));

    // Check that get_next_neighbor and get_prev_neighbor return the correct neighbor coordinate
    EXPECT_EQ(tensor_topology.get_next_neighbor(coord, 0), coord);
    EXPECT_EQ(tensor_topology.get_prev_neighbor(coord, 0), coord);
}

TEST_F(TensorTopologyTest, Replicate2D) {
    const auto tensor_spec =
        TensorSpec(ttnn::Shape{1, 1, 1, 3}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_tensor = Tensor::from_vector(std::vector<float>{42.F, 13.F, -99.F}, tensor_spec);

    const auto mesh_shape_override = MeshShape(2, 3);
    auto mapper = create_mesh_mapper(
        *mesh_device_,
        MeshMapperConfig{
            .placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Replicate{}},
            .mesh_shape_override = mesh_shape_override,
        });
    Tensor replicated_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    // Tensor topology for tensor replicated across 2D (with override) should be same as mesh shape override
    const auto tensor_topology = replicated_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.mesh_shape, mesh_shape_override);

    auto check_neighbors_2d = [&tensor_topology](const MeshCoordinate& coord) {
        EXPECT_EQ(tensor_topology.get_next_neighbor(coord, 0), tensor_topology.get_neighbor(coord, 1, 0));
        EXPECT_EQ(tensor_topology.get_prev_neighbor(coord, 0), tensor_topology.get_neighbor(coord, -1, 0));
        EXPECT_EQ(tensor_topology.get_next_neighbor(coord, 1), tensor_topology.get_neighbor(coord, 1, 1));
        EXPECT_EQ(tensor_topology.get_prev_neighbor(coord, 1), tensor_topology.get_neighbor(coord, -1, 1));
    };

    auto coord = MeshCoordinate(0, 0);
    EXPECT_EQ(tensor_topology.get_device_coord(coord), MeshCoordinate(0, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 1, 0), MeshCoordinate(1, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 2, 0), MeshCoordinate(0, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -1, 0), MeshCoordinate(1, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -2, 0), MeshCoordinate(0, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 1, 1), MeshCoordinate(0, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 2, 1), MeshCoordinate(0, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -1, 1), MeshCoordinate(0, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -2, 1), MeshCoordinate(0, 1));

    check_neighbors_2d(coord);

    coord = MeshCoordinate(1, 1);
    EXPECT_EQ(tensor_topology.get_device_coord(coord), MeshCoordinate(1, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 1, 0), MeshCoordinate(0, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 2, 0), MeshCoordinate(1, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -1, 0), MeshCoordinate(0, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -2, 0), MeshCoordinate(1, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 1, 1), MeshCoordinate(1, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 2, 1), MeshCoordinate(1, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -1, 1), MeshCoordinate(1, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -2, 1), MeshCoordinate(1, 2));

    check_neighbors_2d(coord);

    coord = MeshCoordinate(1, 2);
    EXPECT_EQ(tensor_topology.get_device_coord(coord), MeshCoordinate(1, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 1, 0), MeshCoordinate(0, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 2, 0), MeshCoordinate(1, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -1, 0), MeshCoordinate(0, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -2, 0), MeshCoordinate(1, 2));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 1, 1), MeshCoordinate(1, 0));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, 2, 1), MeshCoordinate(1, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -1, 1), MeshCoordinate(1, 1));
    EXPECT_EQ(tensor_topology.get_neighbor(coord, -2, 1), MeshCoordinate(1, 0));

    check_neighbors_2d(coord);
}

}  // namespace ttnn::distributed::test
