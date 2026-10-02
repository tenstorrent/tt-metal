// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <exception>
#include <optional>

#include <tt-metalium/mesh_coord.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim::test {
namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::MeshDevice1x2Fixture;
using tt::tt_metal::PageConfig;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshShape;

Tensor make_indices(ttnn::MeshDevice* device) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 1, 32}), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})),
        device);
}

Tensor make_weights(ttnn::MeshDevice* device) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 32, 32}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})),
        device);
}

EmbeddingParams make_params() {
    return EmbeddingParams{
        .output_mem_config = MemoryConfig{},
        .tilized = false,
        .embeddings_type = EmbeddingsType::GENERIC,
        .pad_token = std::nullopt,
    };
}

using EmbeddingPrimValidationFixture = MeshDevice1x2Fixture;

TEST_F(EmbeddingPrimValidationFixture, RequiresInputAndWeightsOnSameMeshDevice) {
    auto input_mesh = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto weight_mesh = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 1));

    const auto indices = make_indices(input_mesh.get());
    const auto local_weights = make_weights(input_mesh.get());
    const auto remote_weights = make_weights(weight_mesh.get());
    const auto params = make_params();

    EXPECT_NO_THROW(EmbeddingsDeviceOperation::validate_on_program_cache_miss(
        params,
        EmbeddingInputs{
            .input_tensor_arg = indices,
            .weight_arg = local_weights,
            .optional_output_tensor = std::nullopt,
        }));

    const auto remote_args = EmbeddingInputs{
        .input_tensor_arg = indices,
        .weight_arg = remote_weights,
        .optional_output_tensor = std::nullopt,
    };
    EXPECT_THROW(EmbeddingsDeviceOperation::validate_on_program_cache_miss(params, remote_args), std::exception);
    EXPECT_THROW(
        ttnn::device_operation::MeshDeviceOperationAdapter<EmbeddingsDeviceOperation>::validate_on_program_cache_hit(
            params, remote_args),
        std::exception);
}

}  // namespace
}  // namespace ttnn::prim::test
