// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/old_infra_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

struct SharedVariables {};
struct OperationAttributes {};

// New-infra style program factory that uses the "create" method (non-heterogeneous dispatch)
struct NewInfraProgramFactoryWithCreate {
    using shared_variables_t = SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        return cached_program_t(tt::tt_metal::Program(), SharedVariables{});
    }

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

// New-infra style program factory that uses the "create_at" method (heterogeneous dispatch)
struct NewInfraProgramFactoryWithCreateAt {
    using shared_variables_t = SharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_program_t create_at(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coord,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        return cached_program_t(tt::tt_metal::Program(), SharedVariables{});
    }

    static void override_runtime_arguments_at(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coord,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}
};

// Old infrastructure device operation that uses the "create_program" method
struct OldInfraDeviceOpWithCreate {
    void validate(const std::vector<Tensor>& input_tensors) const {}
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const { return {}; }
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const { return {}; }

    auto create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
        return tt::tt_metal::operation::ProgramWithCallbacks();
    }
};

// Old infrastructure device operation that uses the "create_program_at" method
struct OldInfraDeviceOpWithCreateAt {
    void validate(const std::vector<Tensor>& input_tensors) const {}
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const { return {}; }
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const { return {}; }

    auto create_program_at(
        const ttnn::MeshCoordinate& mesh_coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const {
        return tt::tt_metal::operation::ProgramWithCallbacks();
    }
};

TEST(UsesHeterogeneousDispatchTest, OldInfra) {
    auto old_infra_attributes = tt::tt_metal::operation::DeviceOperation(OldInfraDeviceOpWithCreate{});
    auto old_infra_attributes_with_at = tt::tt_metal::operation::DeviceOperation(OldInfraDeviceOpWithCreateAt{});

    auto old_infra_program_factory = tt::tt_metal::operation::OldInfraDeviceOperation<Tensors>::program_factory_t();

    EXPECT_FALSE(std::visit(
        [&]<typename ConcreteProgramFactory>(const ConcreteProgramFactory&) {
            return ttnn::mesh_device_operation_utils::uses_heterogenous_dispatch<ConcreteProgramFactory>(
                old_infra_attributes);
        },
        old_infra_program_factory));

    EXPECT_TRUE(std::visit(
        [&]<typename ConcreteProgramFactory>(const ConcreteProgramFactory&) {
            return ttnn::mesh_device_operation_utils::uses_heterogenous_dispatch<ConcreteProgramFactory>(
                old_infra_attributes_with_at);
        },
        old_infra_program_factory));
}

TEST(UsesHeterogeneousDispatchTest, NewInfra) {
    EXPECT_FALSE(ttnn::mesh_device_operation_utils::uses_heterogenous_dispatch<NewInfraProgramFactoryWithCreate>(
        OperationAttributes{}));

    EXPECT_TRUE(ttnn::mesh_device_operation_utils::uses_heterogenous_dispatch<NewInfraProgramFactoryWithCreateAt>(
        OperationAttributes{}));
}

using TensorCoordinatesT3000Test = tt::tt_metal::T3000MeshDeviceFixture;

TEST_F(TensorCoordinatesT3000Test, UniformTensor) {
    const TensorSpec tensor_spec = TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = tt::tt_metal::allocate_tensor_on_mesh(tensor_spec, mesh_device_.get());
    std::vector<Tensor> tensor_args = {full_tensor};

    EXPECT_TRUE(ttnn::mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args));

    EXPECT_THAT(
        ttnn::mesh_device_operation_utils::extract_tensor_coordinates(tensor_args),
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

TEST_F(TensorCoordinatesT3000Test, IncompleteShards) {
    const TensorSpec tensor_spec = TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    // Make a tensor spanning 2 devices using `aggregate_as_tensor` with 2 host (owned) tensors.
    auto host_tensor = Tensor::from_vector(std::vector<float>(tensor_spec.logical_shape().volume()), tensor_spec);
    auto device_tensor = distributed::aggregate_as_tensor({host_tensor, host_tensor}, tt::tt_metal::AllGatherTensor{})
                             .to_device(mesh_device_.get());
    std::vector<Tensor> tensor_args = {device_tensor};

    EXPECT_THAT(device_tensor.device_storage().specs, SizeIs(2));

    EXPECT_FALSE(ttnn::mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args));
    EXPECT_THAT(
        ttnn::mesh_device_operation_utils::extract_tensor_coordinates(tensor_args),
        ElementsAre(
            ttnn::MeshCoordinate{0, 0},  //
            ttnn::MeshCoordinate{0, 1}));
}

TEST_F(TensorCoordinatesT3000Test, MismatchedTensors) {
    const TensorSpec tensor_spec = TensorSpec(
        ttnn::Shape{1, 1, 32, 32}, tt::tt_metal::TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto full_tensor = tt::tt_metal::allocate_tensor_on_mesh(tensor_spec, mesh_device_.get());
    // Make a tensor spanning 2 devices using `aggregate_as_tensor` with 2 host (owned) tensors.
    auto host_tensor = Tensor::from_vector(std::vector<float>(tensor_spec.logical_shape().volume()), tensor_spec);
    auto device_tensor = distributed::aggregate_as_tensor({host_tensor, host_tensor}, tt::tt_metal::AllGatherTensor{})
                             .to_device(mesh_device_.get());
    std::vector<Tensor> tensor_args = {full_tensor, device_tensor};

    EXPECT_ANY_THROW(ttnn::mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args));
    EXPECT_ANY_THROW(ttnn::mesh_device_operation_utils::extract_tensor_coordinates(tensor_args));
}

}  // namespace
}  // namespace ttnn
