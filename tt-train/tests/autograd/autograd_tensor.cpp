// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/sgd.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"

using namespace ttml;

class AutogradTensorTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

struct Parameter {
    autograd::TensorPtr theta;
    ttnn::Tensor grad;
};

// A parameter stored as `dtype` (BFLOAT16 or FLOAT32) with fixed random values, and a bf16 gradient for one step.
Parameter make_parameter(ttnn::DataType dtype) {
    const std::array<std::size_t, 4> shape = {1, 1, 32, 32};
    autograd::ctx().set_seed(123U);
    auto& gen = autograd::ctx().get_generator();
    const xt::xarray<float> w0 = test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, gen());
    const xt::xarray<float> g0 = test_utils::make_uniform_xarray<float>(shape, 0.25F, 1.0F, gen());

    auto* device = &autograd::ctx().get_device();
    auto value = dtype == ttnn::DataType::FLOAT32 ? core::from_xtensor<float, ttnn::DataType::FLOAT32>(w0, device)
                                                  : core::from_xtensor(w0, device);
    return {autograd::create_tensor(value, /* requires_grad */ true), core::from_xtensor(g0, device)};
}

// Fused optimizers update a bf16 parameter in place. A FULL view read before the step must show the
// updated values afterwards.
template <typename Optimizer, typename Config>
void expect_full_view_tracks_fused_step(const Config& config) {
    auto [theta, grad] = make_parameter(ttnn::DataType::BFLOAT16);
    ASSERT_EQ(theta->get_value(autograd::PreferredPrecision::NATIVE).dtype(), ttnn::DataType::BFLOAT16);

    // Reading FULL before the step leaves a cached fp32 copy behind.
    (void)theta->get_value(autograd::PreferredPrecision::FULL);
    const auto native_before = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::NATIVE));

    theta->set_grad(grad);
    Optimizer optimizer(serialization::NamedParameters{{"theta", theta}}, config);
    optimizer.step();

    const auto native_after = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::NATIVE));
    ASSERT_FALSE(native_after == native_before) << "the step did not change the stored bf16 value";

    // bf16 -> fp32 is lossless, so the FULL view must equal the stored value exactly.
    const auto full_after = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::FULL));
    EXPECT_TRUE(full_after == native_after) << "the FULL view is stale after an in-place step";
}

}  // namespace

TEST_F(AutogradTensorTest, AutogradTensorFLOAT32) {
    auto tensor = autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32));
    const auto& half_precision_tensor = tensor->get_value();
    const auto& full_precision_tensor = tensor->get_value(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutogradTensorBFLOAT16) {
    auto tensor = autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16));
    const auto& half_precision_tensor = tensor->get_value();
    const auto& full_precision_tensor = tensor->get_value(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutocastTensorFromFLOAT32) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);

    EXPECT_TRUE(autocast_tensor.has_full());
    EXPECT_FALSE(autocast_tensor.has_half());

    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);
    EXPECT_FALSE(autocast_tensor.has_half());

    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_TRUE(autocast_tensor.has_half());
}

TEST_F(AutogradTensorTest, AutocastTensorFromBFLOAT16) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);

    EXPECT_TRUE(autocast_tensor.has_half());
    EXPECT_FALSE(autocast_tensor.has_full());

    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_FALSE(autocast_tensor.has_full());

    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);
    EXPECT_TRUE(autocast_tensor.has_full());
}

TEST_F(AutogradTensorTest, AutocastTensorSetTensorInvalidatesCache) {
    auto fp32_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(fp32_tensor);

    EXPECT_TRUE(autocast_tensor.has_full());
    EXPECT_FALSE(autocast_tensor.has_half());

    [[maybe_unused]] const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_TRUE(autocast_tensor.has_full());
    EXPECT_TRUE(autocast_tensor.has_half());

    auto bf16_tensor =
        ttml::core::zeros(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    autocast_tensor.set_tensor(bf16_tensor);

    EXPECT_TRUE(autocast_tensor.has_half());
    EXPECT_FALSE(autocast_tensor.has_full());

    [[maybe_unused]] const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_TRUE(autocast_tensor.has_half());
    EXPECT_TRUE(autocast_tensor.has_full());
}

namespace {

const ttnn::Shape kShape({1, 1, 32, 32});

ttnn::Tensor filled(float value, ttnn::DataType dtype) {
    return core::full(kShape, value, &autograd::ctx().get_device(), dtype);
}

// Writes value into the native tensor in place, the way an in-place kernel does.
void write_in_place(autograd::AutocastTensor& tensor, float value) {
    auto view = tensor.get_value_for_update();
    ttnn::copy(filled(value, view.tensor().dtype()), view.tensor());
}

bool all_equal(const ttnn::Tensor& tensor, float value) {
    const auto values = core::to_xtensor(tensor);
    return xt::all(xt::equal(values, value));
}

void expect_derived_view_tracks_writes(ttnn::DataType native_dtype, autograd::PreferredPrecision derived) {
    auto tensor = autograd::AutocastTensor(filled(1.0F, native_dtype));
    const auto& before = tensor.get_tensor(derived);
    ASSERT_TRUE(all_equal(before, 1.0F));
    const auto address = before.buffer()->address();

    write_in_place(tensor, 2.0F);

    const auto& after = tensor.get_tensor(derived);
    EXPECT_TRUE(all_equal(after, 2.0F)) << "the derived view is stale after an in-place write";
    EXPECT_EQ(after.buffer()->address(), address) << "the refresh allocated a new buffer";
}

}  // namespace

TEST_F(AutogradTensorTest, AutocastTensorFullViewTracksWritesToBf16Native) {
    expect_derived_view_tracks_writes(ttnn::DataType::BFLOAT16, autograd::PreferredPrecision::FULL);
}

TEST_F(AutogradTensorTest, AutocastTensorHalfViewTracksWritesToFp32Native) {
    expect_derived_view_tracks_writes(ttnn::DataType::FLOAT32, autograd::PreferredPrecision::HALF);
}

TEST_F(AutogradTensorTest, AutocastTensorDerivedViewIsNotRecastWhenCurrent) {
    auto tensor = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16));
    const auto& full = tensor.get_tensor(autograd::PreferredPrecision::FULL);
    // Overwrite the derived copy behind the tensor's back: a read with no write in between must not recast it.
    ttnn::copy(filled(7.0F, ttnn::DataType::FLOAT32), full);
    EXPECT_TRUE(all_equal(tensor.get_tensor(autograd::PreferredPrecision::FULL), 7.0F))
        << "the derived copy was recast although the native tensor was not written";

    write_in_place(tensor, 2.0F);
    EXPECT_TRUE(all_equal(tensor.get_tensor(autograd::PreferredPrecision::FULL), 2.0F));
}

TEST_F(AutogradTensorTest, AutocastTensorUpdateRequiresNativePrecision) {
    auto bf16 = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16));
    EXPECT_ANY_THROW({
        auto view = bf16.get_value_for_update(autograd::PreferredPrecision::FULL);
        (void)view;
    });
    {
        auto view = bf16.get_value_for_update(autograd::PreferredPrecision::HALF);
    }
    {
        auto view = bf16.get_value_for_update(autograd::PreferredPrecision::NATIVE);
    }

    auto fp32 = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::FLOAT32));
    EXPECT_ANY_THROW({
        auto view = fp32.get_value_for_update(autograd::PreferredPrecision::HALF);
        (void)view;
    });
    {
        auto view = fp32.get_value_for_update(autograd::PreferredPrecision::FULL);
    }
}

TEST_F(AutogradTensorTest, AutocastTensorRejectsAccessWhileBeingWritten) {
    auto tensor = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16));
    {
        auto view = tensor.get_value_for_update();
        EXPECT_ANY_THROW((void)tensor.get_tensor(autograd::PreferredPrecision::FULL));
        EXPECT_NO_THROW((void)tensor.get_tensor(autograd::PreferredPrecision::NATIVE));
        EXPECT_ANY_THROW({
            auto second = tensor.get_value_for_update();
            (void)second;
        });
        EXPECT_ANY_THROW(tensor.set_tensor(filled(3.0F, ttnn::DataType::BFLOAT16)));
    }
    EXPECT_NO_THROW((void)tensor.get_tensor(autograd::PreferredPrecision::FULL));
}

TEST_F(AutogradTensorTest, AutocastTensorCopiesShareStorageAndVersion) {
    auto original = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16));
    auto copy = original;
    // The copy creates the derived buffer after the copy was made.
    (void)copy.get_tensor(autograd::PreferredPrecision::FULL);

    write_in_place(original, 2.0F);
    EXPECT_TRUE(all_equal(copy.get_tensor(autograd::PreferredPrecision::FULL), 2.0F))
        << "a copy missed a write made through another copy";
    EXPECT_TRUE(all_equal(original.get_tensor(autograd::PreferredPrecision::FULL), 2.0F));

    copy.set_tensor(filled(5.0F, ttnn::DataType::BFLOAT16));
    EXPECT_TRUE(all_equal(copy.get_tensor(autograd::PreferredPrecision::NATIVE), 5.0F));
    EXPECT_TRUE(all_equal(original.get_tensor(autograd::PreferredPrecision::NATIVE), 2.0F))
        << "set_tensor on a copy changed the original";

    write_in_place(original, 3.0F);
    EXPECT_TRUE(all_equal(original.get_tensor(autograd::PreferredPrecision::FULL), 3.0F));
    EXPECT_TRUE(all_equal(copy.get_tensor(autograd::PreferredPrecision::FULL), 5.0F));
}

TEST_F(AutogradTensorTest, AutocastTensorSetTensorKeepsHeldReferencesValid) {
    auto tensor = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16));
    const auto& held = tensor.get_tensor(autograd::PreferredPrecision::NATIVE);
    tensor.set_tensor(filled(4.0F, ttnn::DataType::BFLOAT16));
    EXPECT_TRUE(all_equal(held, 4.0F)) << "a reference returned by get_tensor() no longer follows the tensor";
}

TEST_F(AutogradTensorTest, FullViewTracksFusedAdamWStep) {
    optimizers::AdamWConfig config;
    config.lr = 1e-2F;
    expect_full_view_tracks_fused_step<optimizers::AdamW>(config);
}

// MorehAdamW writes the parameter through ttnn::moreh_adamw output tensors, outside tt-train's fused wrappers.
TEST_F(AutogradTensorTest, FullViewTracksMorehAdamWStep) {
    optimizers::AdamWCompositeConfig config;
    config.lr = 1e-2F;
    expect_full_view_tracks_fused_step<optimizers::MorehAdamW>(config);
}

TEST_F(AutogradTensorTest, FullViewTracksFusedSGDStep) {
    optimizers::SGDConfig config;
    config.lr = 1e-1F;
    expect_full_view_tracks_fused_step<optimizers::SGD>(config);
}

TEST_F(AutogradTensorTest, NativeValueTracksFusedAdamWStepOnFp32Parameter) {
    auto [theta, grad] = make_parameter(ttnn::DataType::FLOAT32);
    ASSERT_EQ(theta->get_value(autograd::PreferredPrecision::NATIVE).dtype(), ttnn::DataType::FLOAT32);

    // A forward pass reads the bf16 compute copy before any step.
    (void)theta->get_value(autograd::PreferredPrecision::HALF);
    const auto native_before = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::NATIVE));

    theta->set_grad(grad);
    optimizers::AdamWConfig config;
    config.lr = 1e-2F;
    optimizers::AdamW optimizer(serialization::NamedParameters{{"theta", theta}}, config);
    optimizer.step();

    const auto& native = theta->get_value(autograd::PreferredPrecision::NATIVE);
    ASSERT_FALSE(core::to_xtensor(native) == native_before)
        << "the stored fp32 value did not change after an in-place step";

    // The bf16 compute copy must be exactly the cast of the stored value.
    const auto expected_half = core::to_xtensor(ttnn::typecast(native, ttnn::DataType::BFLOAT16));
    const auto half_after = core::to_xtensor(theta->get_value(autograd::PreferredPrecision::HALF));
    EXPECT_TRUE(half_after == expected_half) << "the bf16 copy is stale after an in-place step";
}
