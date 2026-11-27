// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/parameter_dtype.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "test_utils/optimizer_state.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"
#include "tt-metalium/data_formats/bfloat16.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

using namespace ttml;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

class ParameterDtypeTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        autograd::ctx().close_device();
    }

protected:
    // Tests store gradient sync defaults; the next test starts from none.
    void TearDown() override {
        autograd::ctx().set_gradient_reduce_dtype(std::nullopt);
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

// The runners' config parsers check the block before the model is built: the same errors as apply_precision_config,
// and no reduce dtype stored. It gets no parameters, so there is nothing it could convert.
TEST_F(ParameterDtypeTest, ValidatesThePrecisionConfigWithoutApplyingIt) {
    autograd::ctx().set_gradient_reduce_dtype(ttnn::DataType::FLOAT32);
    for (const char* training_config :
         {"seed: 5489",
          "precision: ~",
          "precision: {}",
          "precision: {master_dtype: ~}",
          "precision: {master_dtype: float32, reduce_dtype: float32}",
          "precision: {master_dtype: bfloat16, reduce_dtype: bfloat16}"}) {
        EXPECT_NO_THROW(core::validate_precision_config(YAML::Load(training_config))) << training_config;
        // Applied, every block here but the fp32 one would store bf16 or nothing instead.
        EXPECT_EQ(autograd::ctx().get_gradient_reduce_dtype(), ttnn::DataType::FLOAT32) << training_config;
    }

    const auto validate = [](const char* training_config) {
        core::validate_precision_config(YAML::Load(training_config));
    };
    EXPECT_THAT(
        [&] { validate("precision: {params_dtype: float32}"); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("Unknown key precision.params_dtype (supported: master_dtype, reduce_dtype)")));
    EXPECT_THAT(
        [&] { validate("precision: {master_dtype: float16}"); },
        ThrowsMessage<std::runtime_error>(HasSubstr(R"(precision.master_dtype must be "bfloat16" or "float32")")));
    // The second key is checked too, not only the first.
    EXPECT_THAT(
        [&] { validate("precision: {master_dtype: float32, reduce_dtype: float16}"); },
        ThrowsMessage<std::runtime_error>(HasSubstr(R"(precision.reduce_dtype must be "bfloat16" or "float32")")));
    EXPECT_THAT(
        [&] { validate("precision: float32"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("training_config.precision must be a map")));
}

// The two names are applied, any other is rejected, and unset or null means "keep the dtype the model builds". The
// block holds known keys only, so a typo stops the run instead of training in the model's dtype.
TEST_F(ParameterDtypeTest, ReadsMasterDtypeFromTheTrainingConfig) {
    const auto apply = [](const char* training_config) {
        return core::apply_precision_config(YAML::Load(training_config), serialization::NamedParameters{});
    };
    EXPECT_EQ(apply("precision: {master_dtype: float32}"), ttnn::DataType::FLOAT32);
    EXPECT_EQ(apply("precision: {master_dtype: bfloat16}"), ttnn::DataType::BFLOAT16);
    EXPECT_THAT(
        [&] { apply("precision: {master_dtype: float16}"); },
        ThrowsMessage<std::runtime_error>(HasSubstr(R"(precision.master_dtype must be "bfloat16" or "float32")")));
    EXPECT_EQ(apply("seed: 5489"), std::nullopt);
    EXPECT_EQ(apply("precision: ~"), std::nullopt);
    EXPECT_EQ(apply("precision: {master_dtype: }"), std::nullopt);
    EXPECT_EQ(apply("precision: {master_dtype: ~}"), std::nullopt);
    // reduce_dtype is a known key of the block too.
    EXPECT_EQ(apply("precision: {master_dtype: float32, reduce_dtype: float32}"), ttnn::DataType::FLOAT32);
    EXPECT_THAT(
        [&] { apply("precision: {params_dtype: float32}"); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("Unknown key precision.params_dtype (supported: master_dtype, reduce_dtype)")));
    EXPECT_THAT(
        [&] { apply("precision: float32"); },
        ThrowsMessage<std::runtime_error>(HasSubstr("training_config.precision must be a map")));
}

TEST_F(ParameterDtypeTest, AppliesThePrecisionConfig) {
    auto parameter = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    const serialization::NamedParameters parameters{{"parameter", parameter}};

    EXPECT_EQ(core::apply_precision_config(YAML::Load("seed: 5489"), parameters), std::nullopt);
    EXPECT_EQ(native(parameter).dtype(), ttnn::DataType::BFLOAT16);

    EXPECT_EQ(
        core::apply_precision_config(YAML::Load("precision: {master_dtype: float32}"), parameters),
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

// apply_precision_config stores the context's gradient sync default from every config it applies, an unset one too,
// and closing the device clears it.
TEST_F(ParameterDtypeTest, AppliesReduceDtypeAsTheGradientSyncDefault) {
    const auto apply = [](const char* training_config) {
        return core::apply_precision_config(YAML::Load(training_config), serialization::NamedParameters{});
    };
    (void)apply("precision: {reduce_dtype: float32}");
    EXPECT_EQ(autograd::ctx().get_gradient_reduce_dtype(), ttnn::DataType::FLOAT32);

    // A config without the key, without a precision block, or with the key null, stores none: nothing carries over
    // from the config before, nor from a default set by hand.
    (void)apply("precision: {master_dtype: bfloat16}");
    EXPECT_EQ(autograd::ctx().get_gradient_reduce_dtype(), std::nullopt);
    autograd::ctx().set_gradient_reduce_dtype(ttnn::DataType::FLOAT32);
    (void)apply("seed: 5489");
    EXPECT_EQ(autograd::ctx().get_gradient_reduce_dtype(), std::nullopt);
    autograd::ctx().set_gradient_reduce_dtype(ttnn::DataType::FLOAT32);
    (void)apply("precision: {reduce_dtype: ~}");
    EXPECT_EQ(autograd::ctx().get_gradient_reduce_dtype(), std::nullopt);

    // A wrong name changes nothing.
    (void)apply("precision: {reduce_dtype: bfloat16}");
    EXPECT_THAT(
        [&] { (void)apply("precision: {reduce_dtype: float32, master_dtype: float16}"); },
        ThrowsMessage<std::runtime_error>(HasSubstr(R"(precision.master_dtype must be "bfloat16" or "float32")")));
    EXPECT_THAT(
        [&] { (void)apply("precision: {reduce_dtype: float16}"); },
        ThrowsMessage<std::runtime_error>(HasSubstr(R"(precision.reduce_dtype must be "bfloat16" or "float32")")));
    EXPECT_EQ(autograd::ctx().get_gradient_reduce_dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_THAT(
        [] { autograd::ctx().set_gradient_reduce_dtype(ttnn::DataType::UINT32); },
        ThrowsMessage<std::invalid_argument>(HasSubstr("reduce_dtype must be BFLOAT16, FLOAT32 or unset")));

    autograd::ctx().close_device();
    EXPECT_EQ(autograd::ctx().get_gradient_reduce_dtype(), std::nullopt);
    autograd::ctx().open_device();
}

// Stochastic rounding does nothing for fp32 parameters, so fused AdamW asked for both warns once, at construction.
TEST_F(ParameterDtypeTest, AdamWWarnsOnceWhenStochasticRoundingMeetsFp32Parameters) {
    const auto construct = [](const serialization::NamedParameters& parameters, bool stochastic_rounding) {
        ::testing::internal::CaptureStdout();
        ::testing::internal::CaptureStderr();
        {
            optimizers::AdamW optimizer(
                parameters, optimizers::AdamWConfig{.stochastic_rounding = stochastic_rounding});
        }
        const auto output = ::testing::internal::GetCapturedStdout() + ::testing::internal::GetCapturedStderr();
        size_t warnings = 0;
        for (size_t at = output.find("stochastic_rounding has no effect"); at != std::string::npos;
             at = output.find("stochastic_rounding has no effect", at + 1)) {
            ++warnings;
        }
        return warnings;
    };
    const serialization::NamedParameters fp32_parameters{
        {"first", autograd::create_tensor(random_values(ttnn::DataType::FLOAT32), /* requires_grad */ true)},
        {"second", autograd::create_tensor(random_values(ttnn::DataType::FLOAT32), /* requires_grad */ true)}};
    const serialization::NamedParameters bf16_parameters{
        {"only", autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true)}};

    EXPECT_EQ(construct(fp32_parameters, /* stochastic_rounding */ true), 1U);
    EXPECT_EQ(construct(fp32_parameters, /* stochastic_rounding */ false), 0U);
    EXPECT_EQ(construct(bf16_parameters, /* stochastic_rounding */ true), 0U);
}

// The bf16-only gradient consumers call this right before they read the gradients. An fp32 gradient is replaced by its
// bf16 cast, rounded to nearest even as the host rounds it; a bf16 gradient, a parameter without one and a second call
// are left alone.
TEST_F(ParameterDtypeTest, CastsFp32GradientsToBf16Once) {
    auto fp32_grad = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    auto bf16_grad = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    auto no_grad = autograd::create_tensor(random_values(ttnn::DataType::BFLOAT16), /* requires_grad */ true);
    const auto fp32_values = random_values(ttnn::DataType::FLOAT32);
    fp32_grad->set_grad(fp32_values);
    bf16_grad->set_grad(random_values(ttnn::DataType::BFLOAT16));
    const auto bf16_address = bf16_grad->get_grad().buffer()->address();
    const serialization::NamedParameters parameters{
        {"fp32_grad", fp32_grad}, {"bf16_grad", bf16_grad}, {"no_grad", no_grad}};

    core::cast_fp32_gradients_to_bf16(parameters);

    ASSERT_EQ(fp32_grad->get_grad().dtype(), ttnn::DataType::BFLOAT16);
    const xt::xarray<float> expected = xt::cast<float>(xt::cast<bfloat16>(core::to_xtensor(fp32_values)));
    EXPECT_EQ(test_utils::count_bit_mismatches(expected, core::to_xtensor(fp32_grad->get_grad())), 0U);
    // The values are random fp32 ones, so the cast changed them: the check above isn't a copy of the input.
    EXPECT_GT(test_utils::count_bit_mismatches(expected, core::to_xtensor(fp32_values)), 0U);
    EXPECT_EQ(bf16_grad->get_grad().buffer()->address(), bf16_address);
    EXPECT_FALSE(no_grad->is_grad_initialized());

    const auto cast_address = fp32_grad->get_grad().buffer()->address();
    core::cast_fp32_gradients_to_bf16(parameters);
    EXPECT_EQ(fp32_grad->get_grad().buffer()->address(), cast_address);
    EXPECT_EQ(bf16_grad->get_grad().buffer()->address(), bf16_address);
}
