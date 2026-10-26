// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/adamw_composite.hpp"
#include "optimizers/adamw_full_precision.hpp"
#include "optimizers/sgd.hpp"
#include "test_utils/optimizer_state.hpp"
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

using test_utils::NamedTensors;
using test_utils::state_tensors;

// Reads each tensor's `derived` view, so a cached copy exists, then runs `step`. Afterwards each stored value must
// have moved, and its `derived` view must be exactly the cast of it.
template <typename Step>
void expect_views_track(const NamedTensors& tensors, autograd::PreferredPrecision derived, Step step) {
    const bool half = derived == autograd::PreferredPrecision::HALF;
    const auto dtype = half ? ttnn::DataType::BFLOAT16 : ttnn::DataType::FLOAT32;
    std::vector<xt::xarray<float>> native_before;
    for (const auto& [name, tensor] : tensors) {
        (void)tensor->get_value(derived);
        native_before.push_back(core::to_xtensor(tensor->get_value(autograd::PreferredPrecision::NATIVE)));
    }
    step();
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& [name, tensor] = tensors[i];
        const auto& native = tensor->get_value(autograd::PreferredPrecision::NATIVE);
        ASSERT_FALSE(core::to_xtensor(native) == native_before[i])
            << name << ": the step did not change the stored value";
        const auto expected = core::to_xtensor(ttnn::typecast(native, dtype));
        EXPECT_TRUE(core::to_xtensor(tensor->get_value(derived)) == expected)
            << name << ": the " << (half ? "HALF" : "FULL") << " view is stale after an in-place step";
    }
}

// Fused optimizers update a bf16 parameter and their state in place. A FULL view read before a step must show the
// updated values afterwards.
template <typename Optimizer, typename Config>
void expect_full_view_tracks_fused_step(const Config& config) {
    auto [theta, grad] = make_parameter(ttnn::DataType::BFLOAT16);
    ASSERT_EQ(theta->get_value(autograd::PreferredPrecision::NATIVE).dtype(), ttnn::DataType::BFLOAT16);
    theta->set_grad(grad);
    Optimizer optimizer(serialization::NamedParameters{{"theta", theta}}, config);
    expect_views_track({{"theta", theta}}, autograd::PreferredPrecision::FULL, [&] { optimizer.step(); });

    // The state is written in place too. The first step created all of it (SGD allocates its momentum buffer then),
    // so a second step checks it.
    expect_views_track(state_tensors(optimizer), autograd::PreferredPrecision::FULL, [&] { optimizer.step(); });
}

// AdamWFullPrecision updates its fp32 master weights and moments in place. Their bf16 views must follow. It creates
// all of its state at construction, so one step checks every tensor.
void expect_half_view_tracks_full_precision_step(const optimizers::AdamWFullPrecisionConfig& config) {
    auto [theta, grad] = make_parameter(ttnn::DataType::BFLOAT16);
    theta->set_grad(grad);
    optimizers::AdamWFullPrecision optimizer(serialization::NamedParameters{{"theta", theta}}, config);
    expect_views_track(state_tensors(optimizer), autograd::PreferredPrecision::HALF, [&] { optimizer.step(); });
}

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

    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);

    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);
}

TEST_F(AutogradTensorTest, AutocastTensorFromBFLOAT16) {
    auto tt_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);

    const auto& half = autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);
    EXPECT_EQ(half.dtype(), ttnn::DataType::BFLOAT16);

    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AutocastTensorSetTensorInvalidatesCache) {
    auto fp32_tensor =
        ttml::core::ones(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(fp32_tensor);
    // Caches a bf16 copy of the ones.
    (void)autocast_tensor.get_tensor(autograd::PreferredPrecision::HALF);

    auto bf16_tensor =
        ttml::core::zeros(ttnn::Shape({1, 1, 1, 32}), &autograd::ctx().get_device(), ttnn::DataType::BFLOAT16);
    autocast_tensor.set_tensor(bf16_tensor);

    const auto& full = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);
    EXPECT_EQ(full.dtype(), ttnn::DataType::FLOAT32);
    EXPECT_TRUE(all_equal(full, 0.0F)) << "the copy cached before set_tensor() was returned";
}

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
    EXPECT_ANY_THROW((void)bf16.get_value_for_update(autograd::PreferredPrecision::FULL));
    EXPECT_NO_THROW((void)bf16.get_value_for_update(autograd::PreferredPrecision::HALF));
    EXPECT_NO_THROW((void)bf16.get_value_for_update(autograd::PreferredPrecision::NATIVE));

    auto fp32 = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::FLOAT32));
    EXPECT_ANY_THROW((void)fp32.get_value_for_update(autograd::PreferredPrecision::HALF));
    EXPECT_NO_THROW((void)fp32.get_value_for_update(autograd::PreferredPrecision::FULL));
}

TEST_F(AutogradTensorTest, AutocastTensorRejectsAccessWhileBeingWritten) {
    auto tensor = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16));
    {
        auto view = tensor.get_value_for_update();
        EXPECT_ANY_THROW((void)tensor.get_tensor(autograd::PreferredPrecision::FULL));
        EXPECT_NO_THROW((void)tensor.get_tensor(autograd::PreferredPrecision::NATIVE));
        EXPECT_ANY_THROW((void)tensor.get_value_for_update());
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

TEST_F(AutogradTensorTest, AutocastTensorSetTensorFromItsOwnDerivedView) {
    auto tensor = autograd::AutocastTensor(filled(1.5F, ttnn::DataType::BFLOAT16));
    tensor.set_tensor(tensor.get_tensor(autograd::PreferredPrecision::FULL));
    const auto& native = tensor.get_tensor(autograd::PreferredPrecision::NATIVE);
    EXPECT_EQ(native.dtype(), ttnn::DataType::FLOAT32);
    EXPECT_TRUE(all_equal(native, 1.5F));
}

TEST_F(AutogradTensorTest, AutocastTensorMovedFromStaysUsable) {
    auto original = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16));
    auto moved = std::move(original);
    original.set_tensor(filled(3.0F, ttnn::DataType::BFLOAT16));  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(all_equal(original.get_tensor(autograd::PreferredPrecision::NATIVE), 3.0F));
    EXPECT_TRUE(all_equal(moved.get_tensor(autograd::PreferredPrecision::NATIVE), 1.0F));
}

TEST_F(AutogradTensorTest, AutocastTensorRefreshesHostTensor) {
    auto tensor = autograd::AutocastTensor(filled(1.0F, ttnn::DataType::BFLOAT16).cpu());
    (void)tensor.get_tensor(autograd::PreferredPrecision::FULL);
    // A write that leaves the values as they are: the next read takes the refresh path.
    (void)tensor.get_value_for_update();
    EXPECT_TRUE(all_equal(tensor.get_tensor(autograd::PreferredPrecision::FULL), 1.0F));
}

// Loading values into a tensor keeps the dtype it is stored in, in both directions.
TEST_F(AutogradTensorTest, AssignCastsToTheStoredDtype) {
    for (const auto& [stored, loaded] :
         {std::pair{ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32},
          std::pair{ttnn::DataType::FLOAT32, ttnn::DataType::BFLOAT16}}) {
        auto tensor = autograd::create_tensor(filled(0.0F, stored));
        tensor->assign(filled(1.5F, loaded));
        const auto& native = tensor->get_value(autograd::PreferredPrecision::NATIVE);
        EXPECT_EQ(native.dtype(), stored);
        EXPECT_TRUE(all_equal(native, 1.5F));
    }
}

TEST_F(AutogradTensorTest, AssignToEmptyTensorTakesTheValueAsIs) {
    auto tensor = autograd::create_tensor();
    tensor->assign(filled(1.5F, ttnn::DataType::FLOAT32));
    EXPECT_EQ(tensor->get_value(autograd::PreferredPrecision::NATIVE).dtype(), ttnn::DataType::FLOAT32);
}

TEST_F(AutogradTensorTest, AssignEmptyValueEmptiesTheTensor) {
    auto tensor = autograd::create_tensor(filled(1.5F, ttnn::DataType::BFLOAT16));
    tensor->assign(ttnn::Tensor());
    EXPECT_FALSE(core::is_tensor_initialized(tensor->get_value(autograd::PreferredPrecision::NATIVE)));
}

// Only bf16 and fp32 are cast; a value of any other dtype must match the stored one.
TEST_F(AutogradTensorTest, AssignRejectsNonFloatDtypeMismatch) {
    for (const auto float_dtype : {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}) {
        auto stored_float = autograd::create_tensor(filled(1.0F, float_dtype));
        EXPECT_ANY_THROW(stored_float->assign(filled(2.0F, ttnn::DataType::UINT32)));
        EXPECT_EQ(stored_float->get_value(autograd::PreferredPrecision::NATIVE).dtype(), float_dtype);

        auto stored_uint = autograd::create_tensor(filled(1.0F, ttnn::DataType::UINT32));
        EXPECT_ANY_THROW(stored_uint->assign(filled(2.0F, float_dtype)));
    }
    auto stored_uint = autograd::create_tensor(filled(1.0F, ttnn::DataType::UINT32));
    EXPECT_NO_THROW(stored_uint->assign(filled(2.0F, ttnn::DataType::UINT32)));
    EXPECT_EQ(stored_uint->get_value(autograd::PreferredPrecision::NATIVE).dtype(), ttnn::DataType::UINT32);
}

TEST_F(AutogradTensorTest, FullViewTracksFusedAdamWStep) {
    optimizers::AdamWConfig config;
    config.lr = 1e-2F;
    expect_full_view_tracks_fused_step<optimizers::AdamW>(config);
}

// Each write path of an optimizer needs its own test: the version bump is checked at run time, not by the compiler.
TEST_F(AutogradTensorTest, FullViewTracksFusedAdamWStepWithAmsgrad) {
    optimizers::AdamWConfig config;
    config.lr = 1e-2F;
    config.amsgrad = true;
    expect_full_view_tracks_fused_step<optimizers::AdamW>(config);
}

// MorehAdamW writes the parameter through ttnn::moreh_adamw output tensors.
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

TEST_F(AutogradTensorTest, FullViewTracksFusedSGDStepWithMomentum) {
    optimizers::SGDConfig config;
    config.lr = 1e-1F;
    config.momentum = 0.9F;
    expect_full_view_tracks_fused_step<optimizers::SGD>(config);
}

TEST_F(AutogradTensorTest, NativeValueTracksFusedAdamWStepOnFp32Parameter) {
    auto [theta, grad] = make_parameter(ttnn::DataType::FLOAT32);
    ASSERT_EQ(theta->get_value(autograd::PreferredPrecision::NATIVE).dtype(), ttnn::DataType::FLOAT32);
    theta->set_grad(grad);
    optimizers::AdamWConfig config;
    config.lr = 1e-2F;
    optimizers::AdamW optimizer(serialization::NamedParameters{{"theta", theta}}, config);
    // A forward pass reads the bf16 compute copy before the step.
    expect_views_track({{"theta", theta}}, autograd::PreferredPrecision::HALF, [&] { optimizer.step(); });
}

TEST_F(AutogradTensorTest, HalfViewTracksAdamWFullPrecisionStep) {
    optimizers::AdamWFullPrecisionConfig config;
    config.lr = 1e-2F;
    expect_half_view_tracks_full_precision_step(config);
}

TEST_F(AutogradTensorTest, HalfViewTracksAdamWFullPrecisionStepWithAmsgrad) {
    optimizers::AdamWFullPrecisionConfig config;
    config.lr = 1e-2F;
    config.amsgrad = true;
    expect_half_view_tracks_full_precision_step(config);
}

// Stochastic rounding applies to bf16 parameters only; an fp32 parameter is updated without it.
TEST_F(AutogradTensorTest, FusedAdamWWithStochasticRoundingUpdatesFp32Parameter) {
    auto [theta, grad] = make_parameter(ttnn::DataType::FLOAT32);
    theta->set_grad(grad);
    optimizers::AdamWConfig config;
    config.lr = 1e-2F;
    config.stochastic_rounding = true;
    optimizers::AdamW optimizer(serialization::NamedParameters{{"theta", theta}}, config);
    expect_views_track({{"theta", theta}}, autograd::PreferredPrecision::HALF, [&] { optimizer.step(); });
}
