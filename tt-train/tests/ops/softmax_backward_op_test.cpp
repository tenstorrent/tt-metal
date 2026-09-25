// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tt-metalium/core_coord.hpp>
#include <umd/device/cluster.hpp>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/softmax_backward/device/softmax_backward_device_operation.hpp"
#include "test_utils/random_data.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace {

class ProgramCacheGuard {
public:
    explicit ProgramCacheGuard(ttnn::distributed::MeshDevice* device) : device_(device) {
        device_->set_program_cache_misses_allowed(true);
        device_->disable_and_clear_program_cache();
        device_->enable_program_cache();
    }

    ~ProgramCacheGuard() {
        device_->set_program_cache_misses_allowed(true);
        device_->disable_and_clear_program_cache();
    }

private:
    ttnn::distributed::MeshDevice* device_;
};

template <typename Adapter, typename Attributes, typename TensorArgs>
void expect_validation_failure_on_miss_and_hit(
    const Attributes& attributes, const TensorArgs& tensor_args, std::string_view diagnostic) {
    EXPECT_THAT(
        ([&] { Adapter::validate_on_program_cache_miss(attributes, tensor_args); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
    EXPECT_THAT(
        ([&] { Adapter::validate_on_program_cache_hit(attributes, tensor_args); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
}

using SoftmaxBackwardOp = ttml::metal::ops::softmax_backward::device::SoftmaxBackwardDeviceOperation;
using SoftmaxBackwardAdapter = ttnn::device_operation::MeshDeviceOperationAdapter<SoftmaxBackwardOp>;

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

// Converts float xtensor to bfloat16 so from_xtensor<bfloat16, BFLOAT16> is used; that path does
// tilization on host (to_layout TILE then to_device), avoiding device tilize.
static xt::xarray<bfloat16> to_bf16_xtensor(const xt::xarray<float>& src) {
    xt::xarray<bfloat16> out = xt::empty<bfloat16>(src.shape());
    for (size_t i = 0; i < src.size(); ++i) {
        out.data()[i] = bfloat16(src.data()[i]);
    }
    return out;
}

ttnn::Tensor to_device_tensor(
    const xt::xarray<float>& host_tensor, ttnn::distributed::MeshDevice* device, ttnn::DataType dtype) {
    switch (dtype) {
        case ttnn::DataType::BFLOAT16: {
            auto bf16_host = to_bf16_xtensor(host_tensor);
            return ttml::core::from_xtensor<bfloat16, ttnn::DataType::BFLOAT16>(bf16_host, device);
        }
        case ttnn::DataType::FLOAT32:
            return ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(host_tensor, device);
        default: TT_THROW("Unsupported dtype in softmax backward test");
    }
}

void run_softmax_backward_case(
    const SoftmaxBackwardCase& test_case,
    const DTypeParam& dtype_param,
    ttnn::distributed::MeshDevice* device,
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids = std::nullopt) {
    using namespace ttml;

    const uint32_t logits_seed = make_case_seed(test_case, 0xA5A5A5A5U);
    const uint32_t grad_seed = make_case_seed(test_case, 0x5A5A5A5AU);
    xt::xarray<float> logits_tensor = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{test_case.n, test_case.c, test_case.h, test_case.w}, -10.0F, 10.0F, logits_seed);
    const float grad_min = test_case.grad_min;
    const float grad_max = test_case.grad_max;
    xt::xarray<float> grad_tensor = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{test_case.n, test_case.c, test_case.h, test_case.w}, grad_min, grad_max, grad_seed);

    const int32_t rank = 4;
    const int32_t normalized_dim = test_case.dim >= 0 ? test_case.dim : rank + test_case.dim;
    const uint32_t dim_u32 = static_cast<uint32_t>(normalized_dim);

    auto y_tensor = xt_softmax(logits_tensor, dim_u32);
    auto y_tt = to_device_tensor(y_tensor, device, dtype_param.dtype);
    auto grad_tt = to_device_tensor(grad_tensor, device, dtype_param.dtype);

    ttnn::Tensor result_tt = sub_core_grids.has_value()
                                 ? ttml::metal::softmax_backward(y_tt, grad_tt, test_case.dim, *sub_core_grids)
                                 : ttml::metal::softmax_backward(y_tt, grad_tt, test_case.dim);
    auto result_xtensor = core::to_xtensor(result_tt);

    auto expected = reference_softmax_backward(core::to_xtensor(y_tt), core::to_xtensor(grad_tt), dim_u32);
    const auto max_abs_diff = xt::amax(xt::abs(result_xtensor - expected))();
    EXPECT_TRUE(xt::allclose(result_xtensor, expected, test_case.rtol, test_case.atol))
        << "case=" << test_case.name << ", max_abs_diff=" << max_abs_diff << ", atol=" << test_case.atol
        << ", rtol=" << test_case.rtol;
}

ttnn::Tensor make_overpadded_tensor(
    const xt::xarray<float>& host_tensor, ttnn::distributed::MeshDevice* device, const ttnn::Shape& padded_shape) {
    auto bf16_host = to_bf16_xtensor(host_tensor);
    auto row_major =
        ttml::core::from_xtensor<bfloat16, ttnn::DataType::BFLOAT16>(bf16_host, device, ttnn::Layout::ROW_MAJOR);
    return ttnn::tilize_with_val_padding(row_major, padded_shape, 0.0F);
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

TEST_F(SoftmaxBackwardOpTest, PreservesOverpaddedSpecAndRejectsPhysicalMismatch) {
    constexpr SoftmaxBackwardCase test_case{
        .name = "overpadded_h",
        .n = 1,
        .c = 2,
        .h = 59,
        .w = 64,
        .dim = 3,
        .atol = 2e-2F,
        .rtol = 2e-2F,
        .grad_min = -2.0F,
        .grad_max = 2.0F,
    };
    const auto shape = std::array<std::size_t, 4>{test_case.n, test_case.c, test_case.h, test_case.w};
    const auto logits = ttml::test_utils::make_uniform_xarray<float>(shape, -3.0F, 3.0F, 72U);
    const auto grad = ttml::test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, 73U);
    const auto y = xt_softmax(logits);
    const auto padded_shape = ttnn::Shape({1, 2, 96, 64});
    auto y_overpadded = make_overpadded_tensor(y, s_device, padded_shape);
    auto grad_overpadded = make_overpadded_tensor(grad, s_device, padded_shape);

    const SoftmaxBackwardOp::operation_attributes_t attributes{.dim = 3, .sub_core_grids = std::nullopt};
    const SoftmaxBackwardOp::tensor_args_t tensor_args{
        .softmax_output = y_overpadded, .upstream_grad = grad_overpadded};
    ASSERT_EQ(SoftmaxBackwardOp::compute_output_specs(attributes, tensor_args), y_overpadded.tensor_spec());

    auto result = ttml::metal::softmax_backward(y_overpadded, grad_overpadded, 3);
    EXPECT_EQ(result.tensor_spec(), y_overpadded.tensor_spec());
    const auto expected =
        reference_softmax_backward(ttml::core::to_xtensor(y_overpadded), ttml::core::to_xtensor(grad_overpadded), 3);
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(result), expected, test_case.rtol, test_case.atol));

    auto default_grad = to_device_tensor(grad, s_device, ttnn::DataType::BFLOAT16);
    expect_validation_failure_on_miss_and_hit<SoftmaxBackwardAdapter>(
        attributes,
        SoftmaxBackwardOp::tensor_args_t{.softmax_output = y_overpadded, .upstream_grad = default_grad},
        "Softmax output and upstream gradient tensors must have the same padded shape");

    const auto custom_tile_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, 32, 64}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, tt::tt_metal::Tile({16, 32})),
            ttnn::DRAM_MEMORY_CONFIG));
    auto custom_tile_y = ttnn::create_device_tensor(custom_tile_spec, s_device);
    auto custom_tile_grad = ttnn::create_device_tensor(custom_tile_spec, s_device);
    expect_validation_failure_on_miss_and_hit<SoftmaxBackwardAdapter>(
        attributes,
        SoftmaxBackwardOp::tensor_args_t{.softmax_output = custom_tile_y, .upstream_grad = custom_tile_grad},
        "Softmax backward requires the canonical 32x32 tile");

    const auto transposed_tile_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, 32, 64}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, tt::tt_metal::Tile({32, 32}, /*transpose_tile=*/true)),
            ttnn::DRAM_MEMORY_CONFIG));
    auto transposed_tile_y = ttnn::create_device_tensor(transposed_tile_spec, s_device);
    auto transposed_tile_grad = ttnn::create_device_tensor(transposed_tile_spec, s_device);
    expect_validation_failure_on_miss_and_hit<SoftmaxBackwardAdapter>(
        attributes,
        SoftmaxBackwardOp::tensor_args_t{.softmax_output = transposed_tile_y, .upstream_grad = transposed_tile_grad},
        "Softmax backward requires the canonical 32x32 tile");
}

TEST_F(SoftmaxBackwardOpTest, ReusesProgramForFreshAddressesWithSameSpec) {
    const auto shape = std::array<std::size_t, 4>{1U, 1U, 32U, 64U};
    const auto logits_first = ttml::test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, 74U);
    const auto grad_first_host = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, 75U);
    const auto logits_second = ttml::test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, 76U);
    const auto grad_second_host = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, 77U);
    const auto y_first_host = xt_softmax(logits_first);
    const auto y_second_host = xt_softmax(logits_second);

    auto y_first = to_device_tensor(y_first_host, s_device, ttnn::DataType::BFLOAT16);
    auto grad_first = to_device_tensor(grad_first_host, s_device, ttnn::DataType::BFLOAT16);
    auto y_second = to_device_tensor(y_second_host, s_device, ttnn::DataType::BFLOAT16);
    auto grad_second = to_device_tensor(grad_second_host, s_device, ttnn::DataType::BFLOAT16);

    ASSERT_EQ(y_first.tensor_spec(), y_second.tensor_spec());
    ASSERT_EQ(grad_first.tensor_spec(), grad_second.tensor_spec());
    ASSERT_NE(y_first.buffer()->address(), y_second.buffer()->address());
    ASSERT_NE(grad_first.buffer()->address(), grad_second.buffer()->address());

    ProgramCacheGuard cache_guard(s_device);
    ASSERT_EQ(s_device->num_program_cache_entries(), 0U);

    const auto entries_before_first = s_device->num_program_cache_entries();
    auto first = ttml::metal::softmax_backward(y_first, grad_first, 3);
    const auto entries_after_first = s_device->num_program_cache_entries();
    ASSERT_GT(entries_after_first, entries_before_first);
    const auto expected_first =
        reference_softmax_backward(ttml::core::to_xtensor(y_first), ttml::core::to_xtensor(grad_first), 3);
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(first), expected_first, 2e-2F, 2e-2F));

    const auto entries_before_hit = s_device->num_program_cache_entries();
    s_device->set_program_cache_misses_allowed(false);
    auto second = ttml::metal::softmax_backward(y_second, grad_second, 3);
    s_device->set_program_cache_misses_allowed(true);
    EXPECT_EQ(s_device->num_program_cache_entries(), entries_before_hit);
    EXPECT_NE(first.buffer()->address(), second.buffer()->address());
    const auto expected_second =
        reference_softmax_backward(ttml::core::to_xtensor(y_second), ttml::core::to_xtensor(grad_second), 3);
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(second), expected_second, 2e-2F, 2e-2F));
}

class SoftmaxBackwardOpTypedTest : public SoftmaxBackwardOpTest, public ::testing::WithParamInterface<DTypeParam> {};

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_LastDim_1Tile) {
    constexpr SoftmaxBackwardCase test_case{
        .name = "1tile_last_dim",
        .n = 1,
        .c = 1,
        .h = 32,
        .w = 32,
        .dim = 3,
        .atol = 2e-2F,
        .rtol = 2e-2F,
        .grad_min = -10.0F,
        .grad_max = 10.0F,
    };
    run_softmax_backward_case(test_case, GetParam(), s_device);
}

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_SubCoreGrid_Rectangular) {
    // TODO: Accuracy issue with P150. Tracking: https://github.com/tenstorrent/tt-metal/issues/39312
    auto board = tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0);
    if (board == tt::BoardType::P150) {
        GTEST_SKIP() << "Skipping on P150 boards";
    }
    // Rectangular sub-grid: 2x2 cores starting at (0,0). Requires device with at least 2x2 compute grid.
    const tt::tt_metal::CoreRange sub_range(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(1, 1));
    const tt::tt_metal::CoreRangeSet sub_core_grids(sub_range);
    constexpr SoftmaxBackwardCase test_case{
        .name = "sub_grid_2x2",
        .n = 23,
        .c = 1,
        .h = 32,
        .w = 64,
        .dim = 3,
        .atol = 1e-2F,
        .rtol = 1e-2F,
        .grad_min = -10.0F,
        .grad_max = 10.0F,
    };
    run_softmax_backward_case(test_case, GetParam(), s_device, sub_core_grids);
}

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_SubCoreGrid_NonRectangular) {
    // Non-rectangular (L-shaped) sub-grid
    std::vector<tt::tt_metal::CoreRange> ranges = {
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(2, 0)),  // row y=0
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(2, 1), tt::tt_metal::CoreCoord(2, 1)),  // (0,1) only
    };
    const tt::tt_metal::CoreRangeSet sub_core_grids(std::move(ranges));
    constexpr SoftmaxBackwardCase test_case{
        .name = "sub_grid_L_shape",
        .n = 13,
        .c = 1,
        .h = 32,
        .w = 512,
        .dim = 3,
        .atol = 5e-3F,
        .rtol = 5e-3F,
        .grad_min = -10.0F,
        .grad_max = 10.0F,
    };
    run_softmax_backward_case(test_case, GetParam(), s_device, sub_core_grids);
}

TEST_P(SoftmaxBackwardOpTypedTest, NIGHTLY_SoftmaxBackward_LongRows) {
    constexpr std::array<SoftmaxBackwardCase, 2> cases = {{
        {"long_rows_streaming_300_tiles", 1, 5, 32, 300 * 32, 3, 1e-3F, 1e-3F, -10.0F, 10.0F},
        {"long_rows_streaming_639_tiles", 1, 1, 32, 639 * 32, -1, 1e-3F, 1e-3F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

TEST_P(SoftmaxBackwardOpTypedTest, NIGHTLY_SoftmaxBackward_ManyShortRows) {
    constexpr SoftmaxBackwardCase test_case{
        .name = "many_short_rows_non_streaming",
        .n = 1,
        .c = 30,
        .h = 6400,
        .w = 5 * 32,
        .dim = -1,
        .atol = 2.5e-2F,
        .rtol = 2.5e-2F,
        .grad_min = -10.0F,
        .grad_max = 10.0F,
    };
    run_softmax_backward_case(test_case, GetParam(), s_device);
}

// Type A - first face must be padded
// Type B - first face is full, second face is empty
// Type C - first face is full, second face must be padded

TEST_P(SoftmaxBackwardOpTypedTest, NIGHTLY_SoftmaxBackward_Padding_NonStreaming) {
    constexpr std::array<SoftmaxBackwardCase, 3> cases = {{
        {"padded_non_streaming_type_a", 1, 1, 128, 14 * 32 + 2, -1, 5e-3F, 1e-3F, -10.0F, 10.0F},
        {"padded_non_streaming_type_b", 2, 1, 256, 14 * 32 + 16, 3, 5e-3F, 1e-3F, -10.0F, 10.0F},
        {"padded_non_streaming_type_c", 2, 1, 128, 14 * 32 + 18, 3, 5e-3F, 1e-3F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

TEST_P(SoftmaxBackwardOpTypedTest, NIGHTLY_SoftmaxBackward_Padding_Streaming) {
    constexpr std::array<SoftmaxBackwardCase, 3> cases = {{
        {"padded_streaming_type_a", 1, 5, 32, 300 * 32 + 3, -1, 1e-3F, 1e-3F, -10.0F, 10.0F},
        {"padded_streaming_type_b", 7, 1, 64, 300 * 32 + 16, 3, 1e-3F, 1e-3F, -10.0F, 10.0F},
        {"padded_streaming_type_c", 3, 1, 32, 300 * 32 + 19, 3, 1e-3F, 1e-3F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

TEST_P(SoftmaxBackwardOpTypedTest, SoftmaxBackward_ManyRowsPaddedWidth) {
    constexpr std::array<SoftmaxBackwardCase, 2> cases = {{
        // N*C*(W_padded - W) exceeds W here: a row count derived from the logical shape
        // instead of the padded shape overcounts tile-rows and writes past the output buffer.
        // bf16 on device measures max_abs_diff ~0.011 on both old and fixed kernels for the
        // padded-W shapes (grad ~O(1) row sums); 2e-2 keeps margin without hiding corruption.
        {"many_rows_padded_w", 64, 1, 32, 197, 3, 2e-2F, 1e-3F, -10.0F, 10.0F},
        // H not tile-aligned: the padded-H tile-rows are processed too; per-lane compute
        // must keep all logical rows exact.
        {"unaligned_h_padded_w", 1, 2, 59, 197, 3, 2e-2F, 1e-3F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

TEST_P(SoftmaxBackwardOpTypedTest, NIGHTLY_SoftmaxBackward_WidthBoundaryStreamingSwitch) {
    constexpr std::array<SoftmaxBackwardCase, 5> cases = {{
        {"boundary_63_tiles", 1, 2, 32, 63 * 32, -1, 1e-3F, 1e-3F, -10.0F, 10.0F},
        {"boundary_64_tiles", 1, 3, 32, 64 * 32, -1, 1e-3F, 1e-3F, -10.0F, 10.0F},
        {"boundary_65_tiles", 1, 4, 32, 65 * 32, -1, 1e-3F, 1e-3F, -10.0F, 10.0F},
        {"boundary_127_tiles", 1, 1, 32, 127 * 32, -1, 1e-3F, 1e-3F, -10.0F, 10.0F},
        {"boundary_128_tiles", 3, 1, 32, 128 * 32, -1, 1e-3F, 1e-3F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

// 2048 rows by 64 tiles each
TEST_P(SoftmaxBackwardOpTypedTest, NIGHTLY_SoftmaxBackward_llama8b) {
    constexpr std::array<SoftmaxBackwardCase, 1> cases = {{
        {"llama8b_b1", 1, 32, 2048, 2048, 3, 3e-3F, 3e-3F, -10.0F, 10.0F},
    }};
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        run_softmax_backward_case(test_case, GetParam(), s_device);
    }
}

constexpr std::array<DTypeParam, 2> kDTypeParams = {{
    {"bf16", ttnn::DataType::BFLOAT16},
    {"fp32", ttnn::DataType::FLOAT32},
}};

INSTANTIATE_TEST_SUITE_P(
    SoftmaxBackward,
    SoftmaxBackwardOpTypedTest,
    ::testing::ValuesIn(kDTypeParams),
    [](const ::testing::TestParamInfo<SoftmaxBackwardOpTypedTest::ParamType>& info) { return info.param.name; });
