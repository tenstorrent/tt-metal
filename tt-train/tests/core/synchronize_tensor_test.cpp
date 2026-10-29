// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <core/xtensor_utils.hpp>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <tt-metalium/bfloat16.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "core/parameter_dtype.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/mesh_utils.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

using namespace ttml;

namespace {

constexpr uint32_t kDdpAxis = 1U;  // the 1x2 mesh's axis with two devices

// Round to the nearest bf16 (ties to even), which is what from_xtensor does on upload.
float round_to_bf16(float value) {
    return static_cast<float>(bfloat16(value));
}

// Two devices' bf16 gradients for one [1, 1, height, width] tensor, stacked on dim 0. Each element scatters around a
// per-element mean with a spread as large as the mean, so the two values often cancel partly, which is where a bf16
// sum loses bits.
xt::xarray<float> make_gradients(uint32_t height, uint32_t width, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> exponent(-3.0F, 0.0F);
    std::bernoulli_distribution sign;
    std::normal_distribution<float> noise;
    xt::xarray<float> values = xt::zeros<float>(std::array<std::size_t, 4>{2U, 1U, height, width});
    for (uint32_t r = 0; r < height; ++r) {
        for (uint32_t c = 0; c < width; ++c) {
            const float mean = (sign(rng) ? 1.0F : -1.0F) * std::pow(10.0F, exponent(rng));
            for (uint32_t device = 0; device < 2U; ++device) {
                values(device, 0, r, c) = round_to_bf16(mean + std::abs(mean) * noise(rng));
            }
        }
    }
    return values;
}

// bf16 of the exact mean of the two devices' values: what a perfect reduction returns.
xt::xarray<float> exact_mean(const xt::xarray<float>& values) {
    xt::xarray<float> mean = xt::zeros<float>(std::array<std::size_t, 4>{1U, 1U, values.shape(2), values.shape(3)});
    for (size_t i = 0; i < mean.size(); ++i) {
        const double sum = static_cast<double>(values.flat(i)) + static_cast<double>(values.flat(mean.size() + i));
        mean.flat(i) = round_to_bf16(static_cast<float>(sum / 2.0));
    }
    return mean;
}

double share_equal(const xt::xarray<float>& result, const xt::xarray<float>& expected) {
    const auto equal = xt::sum(xt::cast<std::int64_t>(xt::equal(result, expected)))();
    return static_cast<double>(equal) / static_cast<double>(expected.size());
}

// The device operations `call` runs, in order, recorded with ttnn's graph capture.
std::vector<std::string> device_operations(const std::function<ttnn::Tensor()>& call) {
    ttnn::graph::ScopedGraphCapture capture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
    {
        [[maybe_unused]] const auto result = call();
    }
    std::vector<std::string> operations;
    for (const auto& name : ttnn::graph::extract_calltrace(capture.end_graph_capture())) {
        if (name.ends_with("DeviceOperation")) {
            operations.push_back(name);
        }
    }
    return operations;
}

const std::vector<std::string> kReduceScatterRoute = {
    "TypecastDeviceOperation",
    "ReduceScatterMinimalAsyncDeviceOperation",
    "BinaryNgDeviceOperation",
    "AllGatherAsyncDeviceOperation"};

// Whether a sum's device operations divide by the device count before an all-gather, as the reduce-scatter route
// does so that its all-gather carries the narrower dtype. ttnn's all_reduce divides after its sum is complete, whether
// it runs a reduce-scatter and an all-gather or, for a shape that doesn't split, a gather and a local reduce.
bool divides_before_gathering(const std::vector<std::string>& operations) {
    const auto mean = std::ranges::find(operations, "BinaryNgDeviceOperation");
    const auto gather = std::ranges::find(operations, "AllGatherAsyncDeviceOperation");
    return mean != operations.end() && gather != operations.end() && mean < gather;
}

bool casts(const std::vector<std::string>& operations) {
    return std::ranges::find(operations, "TypecastDeviceOperation") != operations.end();
}

class SynchronizeTensorTest : public test_utils::Mesh1x2Fixture {
protected:
    SynchronizeTensorTest() : Mesh1x2Fixture(/*seed=*/42) {
    }

    // One [1, 1, height, width] gradient per device: device i holds values[i].
    ttnn::Tensor upload(const xt::xarray<float>& values) {
        auto* device = &autograd::ctx().get_device();
        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0);
        return core::from_xtensor<float, ttnn::DataType::BFLOAT16>(values, device, ttnn::Layout::TILE, mapper.get());
    }

    static std::vector<xt::xarray<float>> per_device(const ttnn::Tensor& tensor) {
        return core::to_xtensor<float>(tensor, core::IdentityComposer{});
    }

    // An fp32 all-reduce written out with public ops, to check the helper's routes against.
    static ttnn::Tensor fp32_all_reduce_mean(const ttnn::Tensor& tensor) {
        const auto sum = ttnn_fixed::distributed::all_reduce(ttnn::typecast(tensor, ttnn::DataType::FLOAT32), kDdpAxis);
        return ttnn::typecast(ttnn::multiply(sum, 0.5F), ttnn::DataType::BFLOAT16);
    }

    static void expect_same_on_every_device(const ttnn::Tensor& actual, const ttnn::Tensor& expected) {
        const auto actual_values = per_device(actual);
        const auto expected_values = per_device(expected);
        ASSERT_EQ(actual_values.size(), expected_values.size());
        for (size_t device = 0; device < actual_values.size(); ++device) {
            EXPECT_TRUE(actual_values[device] == expected_values[device]) << "device " << device;
        }
    }
};

}  // namespace

TEST_F(SynchronizeTensorTest, UnsetAndBf16ReduceDtypeSumAsBefore) {
    const auto gradients = upload(make_gradients(64, 128, 1));
    const auto before = ttnn::multiply(ttnn_fixed::distributed::all_reduce(gradients, kDdpAxis), 0.5F);

    expect_same_on_every_device(core::distributed::synchronize_tensor(gradients, {kDdpAxis}), before);
    expect_same_on_every_device(
        core::distributed::synchronize_tensor(gradients, {kDdpAxis}, ttnn::DataType::BFLOAT16), before);
}

// With two devices and this spread, the fp32 sum gets about 99.5% of the elements exact and the bf16 sum about 86%.
TEST_F(SynchronizeTensorTest, Fp32SumIsCloserToTheExactMean) {
    const auto values = make_gradients(64, 128, 2);
    const auto expected = exact_mean(values);
    const auto gradients = upload(values);

    const auto fp32 = core::distributed::synchronize_tensor(gradients, {kDdpAxis}, ttnn::DataType::FLOAT32);
    const auto bf16 = core::distributed::synchronize_tensor(gradients, {kDdpAxis});

    EXPECT_EQ(fp32.dtype(), ttnn::DataType::BFLOAT16);
    const auto fp32_values = per_device(fp32);
    EXPECT_TRUE(fp32_values[0] == fp32_values[1]);
    const double fp32_share = share_equal(fp32_values[0], expected);
    const double bf16_share = share_equal(per_device(bf16)[0], expected);
    EXPECT_GE(fp32_share, 0.99);
    EXPECT_GT(fp32_share, bf16_share);
}

// 512 x 1024 bf16 values per device is the size from which the reduce-scatter route is used.
TEST_F(SynchronizeTensorTest, LargeFp32SumTakesTheReduceScatterRouteWithTheSameResult) {
    const auto values = make_gradients(512, 1024, 3);
    const auto gradients = upload(values);

    const auto fp32 = core::distributed::synchronize_tensor(gradients, {kDdpAxis}, ttnn::DataType::FLOAT32);

    expect_same_on_every_device(fp32, fp32_all_reduce_mean(gradients));
    EXPECT_GE(share_equal(per_device(fp32)[0], exact_mean(values)), 0.99);
}

// 544 and 992 are whole tiles but not two whole tiles per device, so neither dim can be split for two devices.
TEST_F(SynchronizeTensorTest, LargeFp32SumWithoutWholeTileSlicesUsesTheAllReduce) {
    const auto gradients = upload(make_gradients(544, 992, 4));

    const auto fp32 = core::distributed::synchronize_tensor(gradients, {kDdpAxis}, ttnn::DataType::FLOAT32);

    expect_same_on_every_device(fp32, fp32_all_reduce_mean(gradients));
}

TEST_F(SynchronizeTensorTest, AxisWithOneDeviceLeavesTheTensorUnchanged) {
    const auto gradients = upload(make_gradients(32, 64, 5));

    const auto result = core::distributed::synchronize_tensor(gradients, {0U}, ttnn::DataType::FLOAT32);

    EXPECT_EQ(result.dtype(), ttnn::DataType::BFLOAT16);
    expect_same_on_every_device(result, gradients);
}

TEST_F(SynchronizeTensorTest, RejectsReduceDtypesOtherThanBf16AndFp32) {
    const auto gradients = upload(make_gradients(32, 64, 6));
    EXPECT_ANY_THROW((void)core::distributed::synchronize_tensor(gradients, {kDdpAxis}, ttnn::DataType::UINT32));
}

// The C++ runners call this overload, which takes the DDP axis from the parallelism context.
TEST_F(SynchronizeTensorTest, SynchronizeGradientsOverTheDdpAxisUsesTheReduceDtype) {
    autograd::ctx().initialize_parallelism_context({.enable_ddp = true});
    auto* device = &autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    // The value is replicated, as a data-parallel parameter's is; only its gradient differs per device.
    auto parameter = autograd::create_tensor(
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(
            xt::zeros<float>(std::array<std::size_t, 4>{1U, 1U, 64U, 128U}),
            device,
            ttnn::Layout::TILE,
            replicate.get()),
        /* requires_grad */ true);
    const auto gradients = upload(make_gradients(64, 128, 8));
    const auto fp32 = core::distributed::synchronize_tensor(gradients, {kDdpAxis}, ttnn::DataType::FLOAT32);
    parameter->set_grad(gradients);

    core::distributed::synchronize_gradients({{"parameter", parameter}}, ttnn::DataType::FLOAT32);
    expect_same_on_every_device(parameter->get_grad(), fp32);
}

// The runners set the default from precision.reduce_dtype; a dtype the call passes wins over it, both ways.
TEST_F(SynchronizeTensorTest, SynchronizeGradientsTakesTheConfiguredReduceDtypeUnlessTheCallPassesOne) {
    const auto gradients = upload(make_gradients(64, 128, 9));
    const auto fp32 = core::distributed::synchronize_tensor(gradients, {kDdpAxis}, ttnn::DataType::FLOAT32);
    const auto bf16 = core::distributed::synchronize_tensor(gradients, {kDdpAxis});
    ASSERT_FALSE(per_device(fp32)[0] == per_device(bf16)[0]);
    const auto synchronize = [&](std::optional<ttnn::DataType> reduce_dtype) {
        auto parameter = autograd::create_tensor(gradients, /* requires_grad */ true);
        parameter->set_grad(gradients);
        core::distributed::synchronize_gradients({{"parameter", parameter}}, {kDdpAxis}, reduce_dtype);
        return parameter->get_grad();
    };

    expect_same_on_every_device(synchronize(std::nullopt), bf16);
    (void)core::apply_precision_config(YAML::Load("precision: {reduce_dtype: float32}"), {});
    expect_same_on_every_device(synchronize(std::nullopt), fp32);
    expect_same_on_every_device(synchronize(ttnn::DataType::BFLOAT16), bf16);

    autograd::ctx().set_gradient_reduce_dtype(ttnn::DataType::BFLOAT16);
    expect_same_on_every_device(synchronize(std::nullopt), bf16);
    expect_same_on_every_device(synchronize(ttnn::DataType::FLOAT32), fp32);
}

// Which route each sum takes, from the device operations it runs: only an fp32 sum of a large bf16 tensor that splits
// into whole-tile slices takes the reduce-scatter route; an fp32 tensor needs no cast and gains nothing from it.
TEST_F(SynchronizeTensorTest, OnlyALargeNarrowerTensorTakesTheReduceScatterRoute) {
    const auto large = upload(make_gradients(512, 1024, 3));
    const auto small = upload(make_gradients(64, 128, 1));
    const auto no_whole_tile_slices = upload(make_gradients(544, 992, 4));
    const auto large_fp32 = ttnn::typecast(large, ttnn::DataType::FLOAT32);
    const auto operations = [](const ttnn::Tensor& tensor, std::optional<ttnn::DataType> reduce_dtype) {
        return device_operations(
            [&] { return core::distributed::synchronize_tensor(tensor, {kDdpAxis}, reduce_dtype); });
    };

    EXPECT_EQ(operations(large, ttnn::DataType::FLOAT32), kReduceScatterRoute);
    for (const auto& tensor : {small, no_whole_tile_slices}) {
        const auto all_reduce = operations(tensor, ttnn::DataType::FLOAT32);
        EXPECT_TRUE(casts(all_reduce));
        EXPECT_FALSE(divides_before_gathering(all_reduce));
    }
    for (const auto& [tensor, reduce_dtype] : std::vector<std::pair<ttnn::Tensor, std::optional<ttnn::DataType>>>{
             {large_fp32, ttnn::DataType::FLOAT32}, {large, std::nullopt}, {large, ttnn::DataType::BFLOAT16}}) {
        const auto all_reduce = operations(tensor, reduce_dtype);
        EXPECT_FALSE(casts(all_reduce));
        EXPECT_FALSE(divides_before_gathering(all_reduce));
    }
}

// The parallelism context describes the mesh it was made for, so closing the device drops it and the next mesh
// initializes its own.
TEST_F(SynchronizeTensorTest, CloseDeviceDropsTheParallelismContext) {
    autograd::ctx().initialize_parallelism_context({.enable_ddp = true});
    autograd::ctx().close_device();
    EXPECT_FALSE(autograd::ctx().is_parallelism_context_initialized());

    SetUp();  // reopens the mesh as the fixture does; TearDown closes it again
    EXPECT_NO_THROW(autograd::ctx().initialize_parallelism_context({.enable_ddp = true}));
}
