// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <type_traits>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/operation_concepts.hpp"
#include "ttnn/operations/examples/example/device/example_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::ttnn::device_operation::mesh_device_operation_utils::all_tensors_have_uniform_storage;
using ::ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates;
using ::ttnn::device_operation::mesh_device_operation_utils::filter_tensor_shards;

// Returns a dummy device tensor with `num_device_shards` populated.
Tensor make_tensor_with_num_shards(int num_device_shards, MeshDevice* mesh_device, int shard_dim = 0) {
    TT_FATAL(num_device_shards > 0 && num_device_shards <= mesh_device->num_devices(), "Invalid number of shards");

    const auto global_shape = ttnn::Shape{num_device_shards, 1, 32, 32};
    auto buffer = std::make_shared<std::vector<float>>(global_shape.volume());
    return distributed::create_distributed_tensor(
        tt::stl::make_span(*buffer),
        global_shape,
        tt::tt_metal::MemoryPin{buffer},
        tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::TILE, MemoryConfig{}),
        *distributed::shard_tensor_to_mesh_mapper(*mesh_device, shard_dim),
        *mesh_device);
}

// Returns a dummy device tensor distributed according to the `mapper_config`.
Tensor make_tensor_with_mapper_config(
    int num_device_shards, MeshDevice* mesh_device, const distributed::MeshMapperConfig& mapper_config) {
    auto mapper = distributed::create_mesh_mapper(*mesh_device, mapper_config);
    const auto global_shape = ttnn::Shape{num_device_shards, 1, 32, 32};
    auto buffer = std::make_shared<std::vector<float>>(global_shape.volume());
    return distributed::create_distributed_tensor(
        tt::stl::make_span(*buffer),
        global_shape,
        tt::tt_metal::MemoryPin{buffer},
        tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::TILE, MemoryConfig{}),
        *mapper,
        *mesh_device);
}

struct SharedVariables {};
struct OperationAttributes {};

// New-infra style program factory that uses the "create" method (non-heterogeneous dispatch)
struct NewInfraProgramFactory {
    using shared_variables_t = SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_program_t create(
        const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*tensor_return_value*/) {
        return cached_program_t(tt::tt_metal::Program(), SharedVariables{});
    }

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

// New-infra style program factory that uses the "create_at" method (heterogeneous dispatch)
struct NewInfraWorkloadFactory {
    using shared_variables_t = SharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_mesh_workload_t create_mesh_workload(
        const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*tensor_return_value*/) {
        return cached_mesh_workload_t(
            tt::tt_metal::distributed::MeshWorkload(),
            std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t>());
    }

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<NewInfraWorkloadFactory>);
static_assert(ttnn::device_operation::ProgramFactoryConcept<NewInfraProgramFactory>);

TEST(LaunchOperationTest, MeshDeviceOperationAdapterGetName) {
    using ::ttnn::operations::examples::ExampleDeviceOperation;
    EXPECT_EQ(
        device_operation::MeshDeviceOperationAdapter<ExampleDeviceOperation>::get_type_name(
            ExampleDeviceOperation::operation_attributes_t{.attribute = true, .some_other_attribute = 42}),
        "ExampleDeviceOperation");
}

using LaunchOperation2x4Test = tt::tt_metal::MeshDevice2x4Fixture;

TEST_F(LaunchOperation2x4Test, UniformTensor) {
    const TensorSpec tensor_spec = TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = tt::tt_metal::create_device_tensor(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(all_tensors_have_uniform_storage(full_tensor));

    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));
}

TEST_F(LaunchOperation2x4Test, UnevenTensor) {
    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());

    EXPECT_THAT(uneven_tensor.device_storage().coords, SizeIs(2));

    EXPECT_FALSE(all_tensors_have_uniform_storage(uneven_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(uneven_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));
}

TEST_F(LaunchOperation2x4Test, FilterTensorShards) {
    const TensorSpec tensor_spec = TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = tt::tt_metal::create_device_tensor(tensor_spec, mesh_device_.get());

    EXPECT_TRUE(all_tensors_have_uniform_storage(full_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    // Filter the first 2 shards and the last 3 shards.
    filter_tensor_shards(
        {ttnn::MeshCoordinate{0, 0},
         ttnn::MeshCoordinate{0, 1},
         ttnn::MeshCoordinate{1, 1},
         ttnn::MeshCoordinate{1, 2},
         ttnn::MeshCoordinate{1, 3}},
        full_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(full_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    // Filter the first and the last shards.
    filter_tensor_shards(
        {ttnn::MeshCoordinate{0, 0},  //
         ttnn::MeshCoordinate{1, 3}},
        full_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(full_tensor));
    EXPECT_THAT(
        extract_tensor_coordinates(full_tensor),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{1, 3}));

    // Filter the rest.
    filter_tensor_shards(/*tensor_coordinates=*/{}, full_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(full_tensor));
    EXPECT_THAT(extract_tensor_coordinates(full_tensor), IsEmpty());
}

TEST_F(LaunchOperation2x4Test, LaunchOpFilterTensorShards) {
    auto full_tensor = make_tensor_with_num_shards(8, mesh_device_.get());
    auto sum = ttnn::add(full_tensor, full_tensor);

    EXPECT_TRUE(all_tensors_have_uniform_storage(sum));
    EXPECT_THAT(
        extract_tensor_coordinates(sum),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1},
            ttnn::MeshCoordinate{0, 2},
            ttnn::MeshCoordinate{0, 3},
            ttnn::MeshCoordinate{1, 0},
            ttnn::MeshCoordinate{1, 1},
            ttnn::MeshCoordinate{1, 2},
            ttnn::MeshCoordinate{1, 3}));

    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());
    auto sum_uneven = ttnn::add(uneven_tensor, uneven_tensor);

    EXPECT_FALSE(all_tensors_have_uniform_storage(sum_uneven));
    EXPECT_THAT(
        extract_tensor_coordinates(sum_uneven),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));
}

TEST_F(LaunchOperation2x4Test, CachingHeterogeneousDispatch) {
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 0);

    auto full_tensor = make_tensor_with_num_shards(8, mesh_device_.get());
    auto sum = ttnn::add(full_tensor, full_tensor);

    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1);

    auto sum2 = ttnn::add(full_tensor, full_tensor);
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 1);

    auto uneven_tensor = make_tensor_with_num_shards(2, mesh_device_.get());
    auto sum_uneven = ttnn::add(uneven_tensor, uneven_tensor);

    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 2);

    auto sum3 = ttnn::add(uneven_tensor, uneven_tensor);
    EXPECT_EQ(mesh_device_->get_program_cache().num_entries(), 2);
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopology) {
    auto input_tensor_1 = make_tensor_with_num_shards(8, mesh_device_.get());
    auto input_tensor_2 = make_tensor_with_num_shards(8, mesh_device_.get());

    auto sum = ttnn::add(input_tensor_1, input_tensor_2);

    EXPECT_EQ(sum.tensor_topology().distribution_shape(), MeshShape(8));
    EXPECT_EQ(
        sum.tensor_topology().placements(),
        (tt::stl::SmallVector<distributed::MeshMapperConfig::Placement>{distributed::MeshMapperConfig::Shard{0}}));
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopologyAugmentedDistribution) {
    auto config_1 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Shard{0}, distributed::MeshMapperConfig::Replicate{}},
        .mesh_shape_override = MeshShape(2, 2),
    };
    auto input_tensor_1 = make_tensor_with_mapper_config(4, mesh_device_.get(), config_1);
    auto config_2 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}},
        .mesh_shape_override = MeshShape(1, 4),
    };
    auto input_tensor_2 = make_tensor_with_mapper_config(8, mesh_device_.get(), config_2);
    auto config_3 = distributed::MeshMapperConfig{
        .placements = {distributed::MeshMapperConfig::Shard{0}},
        .mesh_shape_override = MeshShape(8),
    };
    auto input_tensor_3 = make_tensor_with_mapper_config(16, mesh_device_.get(), config_3);

    auto sum_1 = ttnn::add(input_tensor_1, input_tensor_2);
    auto sum_2 = ttnn::add(input_tensor_2, input_tensor_1);
    auto sum_3 = ttnn::add(input_tensor_3, input_tensor_2);

    EXPECT_EQ(sum_1.tensor_topology().distribution_shape(), MeshShape(2, 4));
    EXPECT_EQ(
        sum_1.tensor_topology().placements(),
        (tt::stl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Shard{0}, distributed::MeshMapperConfig::Replicate{}}));
    EXPECT_EQ(sum_2.tensor_topology().distribution_shape(), MeshShape(2, 4));
    EXPECT_EQ(
        sum_2.tensor_topology().placements(),
        (tt::stl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}}));
    EXPECT_EQ(sum_3.tensor_topology().distribution_shape(), MeshShape(1, 4));
    EXPECT_EQ(
        sum_3.tensor_topology().placements(),
        (tt::stl::SmallVector<distributed::MeshMapperConfig::Placement>{
            distributed::MeshMapperConfig::Replicate{}, distributed::MeshMapperConfig::Shard{0}}));
}

TEST_F(LaunchOperation2x4Test, OutputTensorTopologyMultipleShardDims) {
    auto input_tensor_1 = make_tensor_with_num_shards(8, mesh_device_.get());
    auto input_tensor_2 = make_tensor_with_num_shards(8, mesh_device_.get(), /*shard_dim=*/1);

    auto sum = ttnn::add(input_tensor_1, input_tensor_2);

    EXPECT_EQ(sum.tensor_topology().distribution_shape(), MeshShape(8));
    EXPECT_EQ(
        sum.tensor_topology().placements(),
        (tt::stl::SmallVector<distributed::MeshMapperConfig::Placement>{distributed::MeshMapperConfig::Shard{0}}));
}

}  // namespace
}  // namespace ttnn
