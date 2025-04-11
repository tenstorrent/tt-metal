// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <type_traits>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include "ttnn/old_infra_device_operation.hpp"
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

// Returns device tensor with `num_device_shards` populated.
Tensor make_tensor_with_num_shards(const TensorSpec& tensor_spec, int num_device_shards, MeshDevice* mesh_device) {
    TT_FATAL(num_device_shards > 0 && num_device_shards <= mesh_device->num_devices(), "Invalid number of shards");

    auto host_tensor = Tensor::from_vector(std::vector<float>(tensor_spec.logical_shape().volume()), tensor_spec);
    std::vector<Tensor> host_tensors;
    for (int i = 0; i < num_device_shards; ++i) {
        host_tensors.push_back(host_tensor);
    }

    return distributed::aggregate_as_tensor(host_tensors, tt::tt_metal::AllGatherTensor{}).to_device(mesh_device);
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
struct NewInfraWorkloadFactory {
    using shared_variables_t = SharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using operation_attributes_t = OperationAttributes;
    using tensor_args_t = Tensor;
    using tensor_return_value_t = Tensor;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
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

// Old infrastructure device operation that uses the "create_program" method
struct OldInfraDeviceOpWithCreateProgram {
    void validate(const std::vector<Tensor>& input_tensors) const {}
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const { return {}; }
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const { return {}; }

    auto create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
        return tt::tt_metal::operation::ProgramWithCallbacks();
    }
};

// Old infrastructure device operation that uses the "create_program_at" method
struct OldInfraDeviceOpWithCreateMeshWorkload {
    void validate(const std::vector<Tensor>& input_tensors) const {}
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const { return {}; }
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const { return {}; }

    auto create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const {
        return tt::tt_metal::operation::MeshWorkloadWithCallbacks();
    }
};

TEST(LaunchOperationTest, OldInfraSelectsMeshWorkloadFactory) {
    auto old_infra_attributes_create_program =
        tt::tt_metal::operation::DeviceOperation(OldInfraDeviceOpWithCreateProgram{});
    auto old_infra_attributes_create_mesh_workload =
        tt::tt_metal::operation::DeviceOperation(OldInfraDeviceOpWithCreateMeshWorkload{});

    using OldInfraDeviceOperation = tt::tt_metal::operation::OldInfraDeviceOperation<Tensors>;

    EXPECT_TRUE(std::holds_alternative<OldInfraDeviceOperation::ProgramFactory>(
        OldInfraDeviceOperation::select_program_factory(old_infra_attributes_create_program, {})));

    EXPECT_TRUE(std::holds_alternative<OldInfraDeviceOperation::MeshWorkloadFactory>(
        OldInfraDeviceOperation::select_program_factory(old_infra_attributes_create_mesh_workload, {})));
}

TEST(LaunchOperationTest, MeshDeviceOperationAdapterGetName) {
    auto old_infra_attrs = tt::tt_metal::operation::DeviceOperation(OldInfraDeviceOpWithCreateProgram{});

    EXPECT_EQ(
        device_operation::MeshDeviceOperationAdapter<
            tt::tt_metal::operation::OldInfraDeviceOperation<Tensors>>::get_type_name(old_infra_attrs),
        "OldInfraDeviceOpWithCreateProgram");

    using ::ttnn::operations::examples::ExampleDeviceOperation;
    EXPECT_EQ(
        device_operation::MeshDeviceOperationAdapter<ExampleDeviceOperation>::get_type_name(
            ExampleDeviceOperation::operation_attributes_t{.attribute = true, .some_other_attribute = 42}),
        "ExampleDeviceOperation");
}

}  // namespace
}  // namespace ttnn
