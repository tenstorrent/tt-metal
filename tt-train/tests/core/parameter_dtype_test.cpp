// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/parameter_dtype.hpp"

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "test_utils/optimizer_state.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

using namespace ttml;

class ParameterDtypeTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        autograd::ctx().close_device();
    }
};

namespace {

const std::array<std::size_t, 4> kShape = {1, 1, 32, 32};

ttnn::Tensor random_values(ttnn::DataType dtype) {
    autograd::ctx().set_seed(123U);
    const xt::xarray<float> values =
        test_utils::make_uniform_xarray<float>(kShape, -1.0F, 1.0F, autograd::ctx().get_generator()());
    auto* device = &autograd::ctx().get_device();
    return dtype == ttnn::DataType::FLOAT32 ? core::from_xtensor<float, ttnn::DataType::FLOAT32>(values, device)
                                            : core::from_xtensor(values, device);
}

const ttnn::Tensor& native(const autograd::TensorPtr& tensor) {
    return tensor->get_value(autograd::PreferredPrecision::NATIVE);
}

}  // namespace

// The two names are applied, any other is rejected, and unset or null means "keep the dtype the model builds".
TEST_F(ParameterDtypeTest, ReadsParamDtypeFromTheTrainingConfig) {
    const auto apply = [](const char* training_config) {
        return core::apply_precision_config(YAML::Load(training_config), serialization::NamedParameters{});
    };
    EXPECT_EQ(apply("precision: {param_dtype: float32}"), ttnn::DataType::FLOAT32);
    EXPECT_EQ(apply("precision: {param_dtype: bfloat16}"), ttnn::DataType::BFLOAT16);
    EXPECT_ANY_THROW(apply("precision: {param_dtype: float16}"));
    EXPECT_EQ(apply("seed: 5489"), std::nullopt);
    EXPECT_EQ(apply("precision: {param_dtype: }"), std::nullopt);
    EXPECT_EQ(apply("precision: {param_dtype: ~}"), std::nullopt);
}

TEST_F(ParameterDtypeTest, AppliesThePrecisionConfig) {
    auto parameter = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    const serialization::NamedParameters parameters{{"parameter", parameter}};

    EXPECT_EQ(core::apply_precision_config(YAML::Load("seed: 5489"), parameters), std::nullopt);
    EXPECT_EQ(native(parameter).dtype(), ttnn::DataType::BFLOAT16);

    EXPECT_EQ(
        core::apply_precision_config(YAML::Load("precision: {param_dtype: float32}"), parameters),
        ttnn::DataType::FLOAT32);
    EXPECT_EQ(native(parameter).dtype(), ttnn::DataType::FLOAT32);
}

// bf16 to fp32 is exact, so a trainable parameter keeps its values. Frozen parameters and non-float tensors stay.
TEST_F(ParameterDtypeTest, StoresTrainableParametersAsFp32) {
    auto trainable = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    auto frozen = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ false);
    auto ids = autograd::create_tensor(
        core::full(ttnn::Shape({1, 1, 32, 32}), 3.0F, &autograd::ctx().get_device(), ttnn::DataType::UINT32),
        /* requires_grad */ true);
    const auto values_before = core::to_xtensor(native(trainable));

    core::store_trainable_parameters_as(
        serialization::NamedParameters{{"trainable", trainable}, {"frozen", frozen}, {"ids", ids}},
        ttnn::DataType::FLOAT32);

    ASSERT_EQ(native(trainable).dtype(), ttnn::DataType::FLOAT32);
    EXPECT_TRUE(core::to_xtensor(native(trainable)) == values_before);
    EXPECT_EQ(native(frozen).dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(native(ids).dtype(), ttnn::DataType::UINT32);
}

TEST_F(ParameterDtypeTest, StoresFp32ParametersAsBf16) {
    auto parameter = autograd::create_tensor(random_values(ttnn::DataType::FLOAT32), /* requires_grad */ true);
    const auto expected = core::to_xtensor(ttnn::typecast(native(parameter), ttnn::DataType::BFLOAT16));

    core::store_trainable_parameters_as(
        serialization::NamedParameters{{"parameter", parameter}}, ttnn::DataType::BFLOAT16);

    ASSERT_EQ(native(parameter).dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_TRUE(core::to_xtensor(native(parameter)) == expected);
}

// The optimizer created afterwards keeps its state in the parameters' new dtype: fp32 moments for fp32 masters.
TEST_F(ParameterDtypeTest, FusedAdamWStateFollowsTheStoredDtype) {
    auto parameter = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    serialization::NamedParameters parameters{{"parameter", parameter}};
    core::store_trainable_parameters_as(parameters, ttnn::DataType::FLOAT32);

    optimizers::AdamWConfig config;
    config.amsgrad = true;
    optimizers::AdamW optimizer(parameters, config);
    const auto state = test_utils::state_tensors(optimizer);
    ASSERT_FALSE(state.empty());
    for (const auto& [name, tensor] : state) {
        EXPECT_EQ(native(tensor).dtype(), ttnn::DataType::FLOAT32) << name;
    }
}

TEST_F(ParameterDtypeTest, RejectsDtypesOtherThanBf16AndFp32) {
    auto parameter = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    EXPECT_ANY_THROW(core::store_trainable_parameters_as(
        serialization::NamedParameters{{"parameter", parameter}}, ttnn::DataType::UINT32));
    EXPECT_EQ(native(parameter).dtype(), ttnn::DataType::BFLOAT16);
}
