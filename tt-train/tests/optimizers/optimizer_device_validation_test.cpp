// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>

#include "metal/optimizers/adamw/device/adamw_device_operation.hpp"
#include "metal/optimizers/sgd/device/sgd_device_operation.hpp"
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/moreh/moreh_adamw/device/moreh_adamw_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

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
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshShape;

class OptimizerDeviceValidationFixture : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    OptimizerDeviceValidationFixture() : MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 2}}) {
    }
};

ttnn::Tensor make_tensor(MeshDevice* device) {
    return ttnn::create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, 32, 32}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

template <typename Operation>
void expect_miss_and_hit_accept(
    const typename Operation::operation_attributes_t& attributes, const typename Operation::tensor_args_t& args) {
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_miss(attributes, args));
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_hit(attributes, args));
}

template <typename Operation>
void expect_miss_and_hit_reject(
    const typename Operation::operation_attributes_t& attributes,
    const typename Operation::tensor_args_t& args,
    const char* message) {
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;
    EXPECT_THAT(
        [&] { Adapter::validate_on_program_cache_miss(attributes, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr(message)));
    EXPECT_THAT(
        [&] { Adapter::validate_on_program_cache_hit(attributes, args); },
        ThrowsMessage<std::runtime_error>(HasSubstr(message)));
}

}  // namespace

TEST_F(OptimizerDeviceValidationFixture, FusedAdamWRejectsEveryForeignCompanion) {
    using Operation = ttml::metal::optimizers::adamw::device::AdamWDeviceOperation;
    auto local_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 0});
    auto foreign_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 1});

    const auto param = make_tensor(local_mesh.get());
    const auto grad = make_tensor(local_mesh.get());
    const auto exp_avg = make_tensor(local_mesh.get());
    const auto exp_avg_sq = make_tensor(local_mesh.get());
    const auto max_exp_avg_sq = make_tensor(local_mesh.get());
    const auto foreign = make_tensor(foreign_mesh.get());
    const Operation::operation_attributes_t attributes{.amsgrad = true};

    expect_miss_and_hit_accept<Operation>(
        attributes,
        {.param = param, .grad = grad, .exp_avg = exp_avg, .exp_avg_sq = exp_avg_sq, .max_exp_avg_sq = max_exp_avg_sq});

    constexpr auto message = "same MeshDevice as the parameter";
    expect_miss_and_hit_reject<Operation>(
        attributes,
        {.param = param,
         .grad = foreign,
         .exp_avg = exp_avg,
         .exp_avg_sq = exp_avg_sq,
         .max_exp_avg_sq = max_exp_avg_sq},
        message);
    expect_miss_and_hit_reject<Operation>(
        attributes,
        {.param = param, .grad = grad, .exp_avg = foreign, .exp_avg_sq = exp_avg_sq, .max_exp_avg_sq = max_exp_avg_sq},
        message);
    expect_miss_and_hit_reject<Operation>(
        attributes,
        {.param = param, .grad = grad, .exp_avg = exp_avg, .exp_avg_sq = foreign, .max_exp_avg_sq = max_exp_avg_sq},
        message);
    expect_miss_and_hit_reject<Operation>(
        attributes,
        {.param = param, .grad = grad, .exp_avg = exp_avg, .exp_avg_sq = exp_avg_sq, .max_exp_avg_sq = foreign},
        message);
}

TEST_F(OptimizerDeviceValidationFixture, FusedSGDRejectsEveryForeignCompanion) {
    using Operation = ttml::metal::optimizers::sgd::device::SGDDeviceOperation;
    auto local_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 0});
    auto foreign_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 1});

    const auto param = make_tensor(local_mesh.get());
    const auto grad = make_tensor(local_mesh.get());
    const auto momentum = make_tensor(local_mesh.get());
    const auto foreign = make_tensor(foreign_mesh.get());
    const Operation::operation_attributes_t attributes{.momentum = 0.9F};

    expect_miss_and_hit_accept<Operation>(attributes, {.param = param, .grad = grad, .momentum_buffer = momentum});

    constexpr auto message = "same MeshDevice as the parameter";
    expect_miss_and_hit_reject<Operation>(
        attributes, {.param = param, .grad = foreign, .momentum_buffer = momentum}, message);
    expect_miss_and_hit_reject<Operation>(
        attributes, {.param = param, .grad = grad, .momentum_buffer = foreign}, message);
}

TEST_F(OptimizerDeviceValidationFixture, MorehAdamWRejectsEveryForeignCompanionAndPreallocatedOutput) {
    using Operation = ttnn::operations::moreh::moreh_adamw::MorehAdamWDeviceOperation;
    auto local_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 0});
    auto foreign_mesh = mesh_device_->create_submesh(MeshShape{1, 1}, MeshCoordinate{0, 1});

    const auto param = make_tensor(local_mesh.get());
    const auto grad = make_tensor(local_mesh.get());
    const auto exp_avg = make_tensor(local_mesh.get());
    const auto exp_avg_sq = make_tensor(local_mesh.get());
    const std::optional<ttnn::Tensor> max_exp_avg_sq = make_tensor(local_mesh.get());
    const std::optional<ttnn::Tensor> param_out = make_tensor(local_mesh.get());
    const std::optional<ttnn::Tensor> exp_avg_out = make_tensor(local_mesh.get());
    const std::optional<ttnn::Tensor> exp_avg_sq_out = make_tensor(local_mesh.get());
    const std::optional<ttnn::Tensor> max_exp_avg_sq_out = make_tensor(local_mesh.get());
    const std::optional<ttnn::Tensor> foreign = make_tensor(foreign_mesh.get());
    const Operation::operation_attributes_t attributes{
        .amsgrad = true, .memory_config = ttnn::DRAM_MEMORY_CONFIG, .compute_kernel_config = {}};

    auto expect_reject = [&](const ttnn::Tensor& actual_grad,
                             const ttnn::Tensor& actual_exp_avg,
                             const ttnn::Tensor& actual_exp_avg_sq,
                             const std::optional<ttnn::Tensor>& actual_max_exp_avg_sq,
                             const std::optional<ttnn::Tensor>& actual_param_out,
                             const std::optional<ttnn::Tensor>& actual_exp_avg_out,
                             const std::optional<ttnn::Tensor>& actual_exp_avg_sq_out,
                             const std::optional<ttnn::Tensor>& actual_max_exp_avg_sq_out) {
        const Operation::tensor_args_t args{
            .param_in = param,
            .grad = actual_grad,
            .exp_avg_in = actual_exp_avg,
            .exp_avg_sq_in = actual_exp_avg_sq,
            .max_exp_avg_sq_in = actual_max_exp_avg_sq,
            .param_out = actual_param_out,
            .exp_avg_out = actual_exp_avg_out,
            .exp_avg_sq_out = actual_exp_avg_sq_out,
            .max_exp_avg_sq_out = actual_max_exp_avg_sq_out,
        };
        expect_miss_and_hit_reject<Operation>(attributes, args, "same MeshDevice as param_in");
    };

    const Operation::tensor_args_t local_args{
        .param_in = param,
        .grad = grad,
        .exp_avg_in = exp_avg,
        .exp_avg_sq_in = exp_avg_sq,
        .max_exp_avg_sq_in = max_exp_avg_sq,
        .param_out = param_out,
        .exp_avg_out = exp_avg_out,
        .exp_avg_sq_out = exp_avg_sq_out,
        .max_exp_avg_sq_out = max_exp_avg_sq_out,
    };
    expect_miss_and_hit_accept<Operation>(attributes, local_args);

    expect_reject(
        foreign.value(),
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        param_out,
        exp_avg_out,
        exp_avg_sq_out,
        max_exp_avg_sq_out);
    expect_reject(
        grad, foreign.value(), exp_avg_sq, max_exp_avg_sq, param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out);
    expect_reject(
        grad, exp_avg, foreign.value(), max_exp_avg_sq, param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out);
    expect_reject(grad, exp_avg, exp_avg_sq, foreign, param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out);
    expect_reject(grad, exp_avg, exp_avg_sq, max_exp_avg_sq, foreign, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out);
    expect_reject(grad, exp_avg, exp_avg_sq, max_exp_avg_sq, param_out, foreign, exp_avg_sq_out, max_exp_avg_sq_out);
    expect_reject(grad, exp_avg, exp_avg_sq, max_exp_avg_sq, param_out, exp_avg_out, foreign, max_exp_avg_sq_out);
    expect_reject(grad, exp_avg, exp_avg_sq, max_exp_avg_sq, param_out, exp_avg_out, exp_avg_sq_out, foreign);
}
