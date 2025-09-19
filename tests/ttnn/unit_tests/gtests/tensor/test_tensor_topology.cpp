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
using tt::tt_metal::distributed::MeshMapperConfig;

using TensorTopologyTest = GenericMeshDeviceFixture;
using TensorTopology2x4Test = MeshDevice2x4Fixture;

TEST_F(TensorTopologyTest, SingleDevice) {
    const auto tensor_spec =
        TensorSpec(ttnn::Shape{1, 1, 1, 3}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_tensor = Tensor::from_vector(std::vector<float>{42.F, 13.F, -99.F}, tensor_spec);

    const auto mesh_mapper_config = MeshMapperConfig{
        .placements = {MeshMapperConfig::Replicate{}},
        .mesh_shape_override = MeshShape(1),
    };
    auto mapper = create_mesh_mapper(*mesh_device_, mesh_mapper_config);
    Tensor replicated_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    // Tensor topology for tensor replicated across 2D (with override) should be same as mesh mapper config
    const auto& tensor_topology = replicated_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), mesh_mapper_config.mesh_shape_override);
    EXPECT_EQ(tensor_topology.placements(), mesh_mapper_config.placements);

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

TEST_F(TensorTopology2x4Test, Replicate2D) {
    const auto tensor_spec =
        TensorSpec(ttnn::Shape{1, 1, 1, 3}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_tensor = Tensor::from_vector(std::vector<float>{42.F, 13.F, -99.F}, tensor_spec);

    const auto mesh_mapper_config = MeshMapperConfig{
        .placements = {MeshMapperConfig::Replicate{}, MeshMapperConfig::Replicate{}},
        .mesh_shape_override = MeshShape(2, 3),
    };
    auto mapper = create_mesh_mapper(*mesh_device_, mesh_mapper_config);
    Tensor replicated_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);

    // Tensor topology for tensor replicated across 2D (with override) should be same as mesh mapper config
    const auto& tensor_topology = replicated_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), mesh_mapper_config.mesh_shape_override);
    EXPECT_EQ(tensor_topology.placements(), mesh_mapper_config.placements);

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

TEST_F(TensorTopology2x4Test, Shard1DRowMajor) {
    const int num_devices = mesh_device_->num_devices();
    // Test only works on 8 devices in 2x4 mesh
    ASSERT_EQ(num_devices, 8);
    ASSERT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    const auto tensor_spec = TensorSpec(
        ttnn::Shape{1, num_devices, 3, 1}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_tensor = Tensor::from_vector(test_data, tensor_spec);

    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    // Tensor topology for tensor sharded across 1 dimension should be 1D shape with number of actual shards (ie.
    // num_devices)
    const auto& tensor_topology = sharded_tensor.tensor_topology();
    EXPECT_EQ(tensor_topology.distribution_shape(), MeshShape(num_devices));
    EXPECT_EQ(
        tensor_topology.placements(), (tt::stl::SmallVector<MeshMapperConfig::Placement>{MeshMapperConfig::Shard{1}}));

    const auto& mesh_coords = tensor_topology.mesh_coords();
    EXPECT_EQ(mesh_coords.size(), num_devices);
    EXPECT_EQ(mesh_coords[0], MeshCoordinate(0, 0));
    EXPECT_EQ(mesh_coords[1], MeshCoordinate(0, 1));
    EXPECT_EQ(mesh_coords[2], MeshCoordinate(0, 2));
    EXPECT_EQ(mesh_coords[3], MeshCoordinate(0, 3));
    EXPECT_EQ(mesh_coords[4], MeshCoordinate(1, 0));
    EXPECT_EQ(mesh_coords[5], MeshCoordinate(1, 1));
    EXPECT_EQ(mesh_coords[6], MeshCoordinate(1, 2));
    EXPECT_EQ(mesh_coords[7], MeshCoordinate(1, 3));
}

TEST_F(TensorTopology2x4Test, GetTensorCoord) {
    const int num_devices = mesh_device_->num_devices();
    // Test only works on 8 devices in 2x4 mesh
    ASSERT_EQ(num_devices, 8);
    ASSERT_EQ(mesh_device_->shape(), MeshShape(2, 4));

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    const auto tensor_spec = TensorSpec(
        ttnn::Shape{1, num_devices, 3, 1}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    Tensor input_tensor = Tensor::from_vector(test_data, tensor_spec);

    // The sharding creates a 1D distribution with 8 devices in row-major order
    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    const auto& tensor_topology = sharded_tensor.tensor_topology();
    const auto& mesh_coords = tensor_topology.mesh_coords();

    // Test that get_tensor_coord returns the correct tensor coordinate for each device coordinate
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(0, 0)), MeshCoordinate(0));
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(0, 1)), MeshCoordinate(1));
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(0, 2)), MeshCoordinate(2));
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(0, 3)), MeshCoordinate(3));
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(1, 0)), MeshCoordinate(4));
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(1, 1)), MeshCoordinate(5));
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(1, 2)), MeshCoordinate(6));
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(1, 3)), MeshCoordinate(7));

    // Test that get_tensor_coord is the inverse of get_device_coord
    for (const auto& device_coord : mesh_coords) {
        auto tensor_coord = tensor_topology.get_tensor_coord(device_coord);
        ASSERT_TRUE(tensor_coord.has_value());
        EXPECT_EQ(tensor_topology.get_device_coord(tensor_coord.value()), device_coord);
    }

    // Test that get_tensor_coord returns nullopt for invalid device coordinates
    // ie. These device coordinates don't exist in the mesh
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(2, 0)), std::nullopt);
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(0, 4)), std::nullopt);
    EXPECT_EQ(tensor_topology.get_tensor_coord(MeshCoordinate(3, 3)), std::nullopt);
}

}  // namespace ttnn::distributed::test
