// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim::test {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::PageConfig;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshShape;

class EmbeddingBackwardValidationFixture : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    EmbeddingBackwardValidationFixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 2}}) {}
};

Tensor make_index_tensor(tt::tt_metal::distributed::MeshDevice* device) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 1, 32}), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})),
        device);
}

Tensor make_gradient_tensor(tt::tt_metal::distributed::MeshDevice* device) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 32, 32}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{})),
        device);
}

Tensor make_output_tensor(tt::tt_metal::distributed::MeshDevice* device) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 64, 32}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{})),
        device);
}

using Operation = EmbeddingBackwardDeviceOperation;
using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;

const Operation::operation_attributes_t kAttributes{
    .output_mem_config = MemoryConfig{}, .output_dtype = DataType::BFLOAT16, .num_embeddings = 64};

void expect_miss_and_hit_reject(const Operation::tensor_args_t& args, const char* message) {
    EXPECT_THAT(
        [&] { Adapter::validate_on_program_cache_miss(kAttributes, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr(message)));
    EXPECT_THAT(
        [&] { Adapter::validate_on_program_cache_hit(kAttributes, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr(message)));
}

}  // namespace

TEST_F(EmbeddingBackwardValidationFixture, RejectsGradientOnAnotherMeshBeforeColdOrCachedDispatch) {
    auto local_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 0});
    auto foreign_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 1});

    const Operation::tensor_args_t local_args{
        .index_tensor = make_index_tensor(local_mesh.get()),
        .grad_tensor = make_gradient_tensor(local_mesh.get()),
        .preallocated_output = std::nullopt,
    };
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_miss(kAttributes, local_args));
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_hit(kAttributes, local_args));

    auto foreign_args = local_args;
    foreign_args.grad_tensor = make_gradient_tensor(foreign_mesh.get());
    expect_miss_and_hit_reject(foreign_args, "index and gradient tensors to be on the same MeshDevice");
}

TEST_F(EmbeddingBackwardValidationFixture, RejectsPreallocatedOutputOnAnotherMeshBeforeColdOrCachedDispatch) {
    auto local_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 0});
    auto foreign_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 1});

    const Operation::tensor_args_t local_args{
        .index_tensor = make_index_tensor(local_mesh.get()),
        .grad_tensor = make_gradient_tensor(local_mesh.get()),
        .preallocated_output = make_output_tensor(local_mesh.get()),
    };
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_miss(kAttributes, local_args));
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_hit(kAttributes, local_args));

    auto foreign_args = local_args;
    foreign_args.preallocated_output = make_output_tensor(foreign_mesh.get());
    expect_miss_and_hit_reject(foreign_args, "preallocated output to be on the same MeshDevice");
}

}  // namespace ttnn::prim::test
