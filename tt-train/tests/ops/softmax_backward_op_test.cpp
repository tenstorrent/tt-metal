// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <core/ttnn_all_includes.hpp>
#include <string_view>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

namespace {

struct SoftmaxBackwardCase {
    const char* name;
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
    int32_t dim;
    float atol;
    float rtol;
    float grad_min;
    float grad_max;
};

struct DTypeParam {
    const char* name;
    ttnn::DataType dtype;
    bool is_supported;
};

constexpr uint32_t kSuiteSeed = 42U;

uint32_t make_case_seed(const SoftmaxBackwardCase& test_case, uint32_t salt) {
    // Small deterministic hash for stable test data generation regardless of execution order.
    uint32_t hash = 2166136261U ^ (kSuiteSeed + salt);
    auto mix = [&hash](uint32_t value) {
        hash ^= value + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        hash *= 16777619U;
    };
    for (unsigned char ch : std::string_view(test_case.name)) {
        hash ^= static_cast<uint32_t>(ch);
        hash *= 16777619U;
    }
    mix(test_case.n);
    mix(test_case.c);
    mix(test_case.h);
    mix(test_case.w);
    mix(static_cast<uint32_t>(test_case.dim));
    return hash;
}

xt::xarray<float> xt_softmax(const xt::xarray<float>& input, uint32_t dim = 3U) {
    xt::xarray<float> max_value = xt::amax(input, dim, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_value;
    xt::xarray<float> exp_shifted_input = xt::exp(shifted_input);
    xt::xarray<float> exp_sum = xt::sum(exp_shifted_input, dim, xt::keep_dims);
    return exp_shifted_input / exp_sum;
}

xt::xarray<float> reference_softmax_backward(const xt::xarray<float>& y, const xt::xarray<float>& grad, uint32_t dim) {
    xt::xarray<float> dot = xt::sum(y * grad, {dim}, xt::keep_dims);
    return y * (grad - dot);
}

ttnn::Tensor to_device_tensor(
    const xt::xarray<float>& host_tensor, ttnn::distributed::MeshDevice* device, ttnn::DataType dtype) {
    switch (dtype) {
        case ttnn::DataType::BFLOAT16:
            return ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(host_tensor, device);
        case ttnn::DataType::FLOAT32:
            return ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(host_tensor, device);
        default: TT_THROW("Unsupported dtype in softmax backward test");
    }
}

void run_softmax_backward_case(
    const SoftmaxBackwardCase& test_case, const DTypeParam& dtype_param, ttnn::distributed::MeshDevice* device) {
    using namespace ttml;

    xt::xarray<float> logits_tensor = xt::empty<float>({test_case.n, test_case.c, test_case.h, test_case.w});
    xt::xarray<float> grad_tensor = xt::empty<float>({test_case.n, test_case.c, test_case.h, test_case.w});

    const uint32_t logits_seed = make_case_seed(test_case, 0xA5A5A5A5U);
    const uint32_t grad_seed = make_case_seed(test_case, 0x5A5A5A5AU);
    ttml::core::parallel_generate(
        std::span{logits_tensor.data(), logits_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        logits_seed);
    const float grad_min = test_case.grad_min;
    const float grad_max = test_case.grad_max;
    ttml::core::parallel_generate(
        std::span{grad_tensor.data(), grad_tensor.size()},
        [grad_min, grad_max]() { return std::uniform_real_distribution<float>(grad_min, grad_max); },
        grad_seed);

    const int32_t rank = 4;
    const int32_t normalized_dim = test_case.dim >= 0 ? test_case.dim : rank + test_case.dim;
    const uint32_t dim_u32 = static_cast<uint32_t>(normalized_dim);

    auto y_tensor = xt_softmax(logits_tensor, dim_u32);
    auto y_tt = to_device_tensor(y_tensor, device, dtype_param.dtype);
    auto grad_tt = to_device_tensor(grad_tensor, device, dtype_param.dtype);

    if (!dtype_param.is_supported) {
        EXPECT_ANY_THROW((void)ttml::metal::softmax_backward(y_tt, grad_tt, test_case.dim))
            << "case=" << test_case.name << ", dtype=" << dtype_param.name;
        return;
    }

    auto result_tt = ttml::metal::softmax_backward(y_tt, grad_tt, test_case.dim);
    auto result_xtensor = core::to_xtensor(result_tt);

    auto expected = reference_softmax_backward(core::to_xtensor(y_tt), core::to_xtensor(grad_tt), dim_u32);
    const auto max_abs_diff = xt::amax(xt::abs(result_xtensor - expected))();
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, test_case.rtol, test_case.atol))
        << "case=" << test_case.name << ", max_abs_diff=" << max_abs_diff << ", atol=" << test_case.atol
        << ", rtol=" << test_case.rtol;
}

}  // namespace

class SoftmaxBackwardOpTest : public ::testing::Test {
protected:
    static ttnn::distributed::MeshDevice* s_device;

    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(kSuiteSeed);
        s_device = &ttml::autograd::ctx().get_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
        s_device = nullptr;
    }
};

ttnn::distributed::MeshDevice* SoftmaxBackwardOpTest::s_device = nullptr;

class SoftmaxBackwardOpTypedTest : public SoftmaxBackwardOpTest, public ::testing::WithParamInterface<DTypeParam> {};

#define SOFTMAX_BW_SKIP_IF_UNSUPPORTED(description)                                                       \
    do {                                                                                                  \
        if (!GetParam().is_supported) {                                                                   \
            GTEST_SKIP() << "Skipping " << (description) << " for unsupported dtype " << GetParam().name; \
            return;                                                                                       \
        }                                                                                                 \
    } while (0)

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_LastDim_1Tile) {
    SOFTMAX_BW_SKIP_IF_UNSUPPORTED("1-tile last-dim");
    constexpr SoftmaxBackwardCase test_case{
        .name = "1tile_last_dim",
        .n = 1,
        .c = 1,
        .h = 32,
        .w = 32,
        .dim = 3,
        .atol = 2.5e-2F,
        .rtol = 2.5e-2F,
        .grad_min = -10.0F,
        .grad_max = 10.0F,
    };
    run_softmax_backward_case(test_case, GetParam(), s_device);
}

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_LongRows) {
    SOFTMAX_BW_SKIP_IF_UNSUPPORTED("long-row shape");
    constexpr std::array<SoftmaxBackwardCase, 2> cases = {{
        {"long_rows_streaming_300_tiles", 1, 5, 32, 300 * 32 + 3, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"long_rows_streaming_639_tiles", 1, 1, 32, 639 * 32, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_ManyShortRows) {
    SOFTMAX_BW_SKIP_IF_UNSUPPORTED("many-short-rows shape");
    constexpr SoftmaxBackwardCase test_case{
        .name = "many_short_rows_non_streaming_5_tiles",
        .n = 1,
        .c = 30,
        .h = 6400,
        .w = 5 * 32,
        .dim = -1,
        .atol = 6e-2F,
        .rtol = 6e-2F,
        .grad_min = -10.0F,
        .grad_max = 10.0F,
    };
    run_softmax_backward_case(test_case, GetParam(), s_device);
}

// Type A - first face must be padded
// Type B - first face is full, second face is empty
// Type C - first face is full, second face must be padded

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_Padding_NonStreaming) {
    SOFTMAX_BW_SKIP_IF_UNSUPPORTED("padded non-streaming shape");
    constexpr std::array<SoftmaxBackwardCase, 3> cases = {{
        {"padded_non_streaming_type_a", 1, 1, 128, 14 * 32 + 2, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"padded_non_streaming_type_b", 2, 1, 256, 14 * 32 + 16, 3, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"padded_non_streaming_type_c", 2, 1, 128, 14 * 32 + 18, 3, 6e-2F, 6e-2F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_Padding_Streaming) {
    SOFTMAX_BW_SKIP_IF_UNSUPPORTED("padded streaming shape");
    constexpr std::array<SoftmaxBackwardCase, 3> cases = {{
        {"padded_streaming_type_a", 1, 5, 32, 300 * 32 + 3, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"padded_streaming_type_b", 7, 1, 64, 300 * 32 + 16, 3, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"padded_streaming_type_c", 3, 1, 32, 300 * 32 + 19, 3, 6e-2F, 6e-2F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_WidthBoundaryStreamingSwitch) {
    SOFTMAX_BW_SKIP_IF_UNSUPPORTED("boundary shape");
    constexpr std::array<SoftmaxBackwardCase, 5> cases = {{
        {"boundary_63_tiles", 1, 2, 32, 63 * 32, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"boundary_64_tiles", 1, 3, 32, 64 * 32, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"boundary_65_tiles", 1, 4, 32, 65 * 32, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"boundary_127_tiles", 1, 1, 32, 127 * 32, -1, 6e-2F, 6e-2F, -10.0F, 10.0F},
        {"boundary_128_tiles", 3, 1, 32, 128 * 32, -1, 7e-2F, 7e-2F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

// 16384 rows by 64 tiles each
// Test takes around 105 seconds on BH
TEST_P(SoftmaxBackwardOpTypedTest, NIGHTLY_SoftmaxBackward_llama8b) {
    SOFTMAX_BW_SKIP_IF_UNSUPPORTED("llama shape");
    constexpr std::array<SoftmaxBackwardCase, 1> cases = {{
        {"llama8b_b1", 8, 32, 2048, 2048, 3, 1e-2F, 1e-2F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

constexpr std::array<DTypeParam, 2> kDTypeParams = {{
    {"bf16", ttnn::DataType::BFLOAT16, true},
    {"fp32", ttnn::DataType::FLOAT32, false},
}};

INSTANTIATE_TEST_SUITE_P(
    SoftmaxBackward,
    SoftmaxBackwardOpTypedTest,
    ::testing::ValuesIn(kDTypeParams),
    [](const ::testing::TestParamInfo<SoftmaxBackwardOpTypedTest::ParamType>& info) { return info.param.name; });
