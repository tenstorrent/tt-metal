// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Smoke tests for the ttnn reduction op family: one test per program factory,
// for every op under ttnn/cpp/ttnn/operations/reduction/, plus a few tests for
// the front-end code paths that pick between factories. Each test uses small
// deterministic inputs with exact (or op-tolerance) expected outputs, so a
// kernel that runs but produces garbage fails.
//
// Each test states in a comment which program factory or code path it covers.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/reduction/accumulation/cumprod/cumprod.hpp"
#include "ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp"
#include "ttnn/operations/reduction/accumulation/ema/ema.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/manual_seed/manual_seed.hpp"
#include "ttnn/operations/reduction/moe/moe.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/reduction/sampling/sampling.hpp"
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"
#include "ttnn/operations/reduction/topk/topk.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "smoke_test_utils.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::reduction::test {

class ReductionSmoke : public TTNNFixtureWithSuiteDevice<ReductionSmoke> {};

namespace detail {

using ttnn::test_utils::make_device_tensor;

}  // namespace detail

// ---------------------------------------------------------------------------
// Generic reduce: sum and max/min on the tiled W / H / HW factories (SUM and
// MAX / MIN-negate pool math). Integrated from the original test_reduction.cpp
// parametrized suites: tall unaligned + aligned shapes per dim, positive and
// negative data for the MIN-negate / pad-sentinel paths, and both the
// single-tile HW single-core factory (30x30) and the multi-tile two-step
// W-then-H path (64x64).
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, SumReduceW) {
    auto& device = *device_;
    for (const auto [h, w] : std::array<std::array<uint32_t, 2>, 2>{{{3100, 63}, {3200, 64}}}) {
        const auto input = ttnn::ones(ttnn::Shape{h, w}, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output = ttnn::sum(input, -1, true);
        ASSERT_EQ(output.logical_shape(), (ttnn::Shape{h, 1})) << "h=" << h << " w=" << w;
        const auto result = output.to_vector<bfloat16>();
        for (uint32_t r = 0; r < h; r++) {
            ASSERT_EQ(static_cast<float>(result[r]), static_cast<float>(w)) << "h=" << h << " w=" << w << " row " << r;
        }
    }
}

TEST_F(ReductionSmoke, SumReduceH) {
    auto& device = *device_;
    for (const auto [h, w] : std::array<std::array<uint32_t, 2>, 2>{{{63, 3100}, {64, 3200}}}) {
        const auto input = ttnn::ones(ttnn::Shape{h, w}, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output = ttnn::sum(input, -2, true);
        ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, w})) << "h=" << h << " w=" << w;
        const auto result = output.to_vector<bfloat16>();
        for (uint32_t c = 0; c < w; c++) {
            ASSERT_EQ(static_cast<float>(result[c]), static_cast<float>(h)) << "h=" << h << " w=" << w << " col " << c;
        }
    }
}

TEST_F(ReductionSmoke, SumReduceBothDims) {
    auto& device = *device_;
    // 30x30 = one padded tile -> HW single-core factory; 64x64 = two-step W-then-H.
    for (const auto [h, w] : std::array<std::array<uint32_t, 2>, 2>{{{30, 30}, {64, 64}}}) {
        const auto input = ttnn::ones(ttnn::Shape{h, w}, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output = ttnn::sum(input, ttsl::SmallVector<int>{0, 1}, false);
        const auto result = output.to_vector<bfloat16>();
        ASSERT_EQ(result.size(), 1u) << "h=" << h << " w=" << w;
        ASSERT_EQ(static_cast<float>(result[0]), static_cast<float>(h * w)) << "h=" << h << " w=" << w;
    }
}

namespace detail {

// offset+index ramps: max/min land on exactly-representable bf16 endpoints.
struct MinMaxCase {
    int h;
    int w;
    int offset;
};

}  // namespace detail

TEST_F(ReductionSmoke, MinMaxReduceW) {
    auto& device = *device_;
    for (const auto& c :
         std::array<detail::MinMaxCase, 4>{{{3100, 63, 4}, {3200, 64, 4}, {3100, 63, -128}, {3200, 64, -128}}}) {
        std::vector<float> data(static_cast<size_t>(c.h) * c.w);
        for (int i = 0; i < c.h; i++) {
            for (int j = 0; j < c.w; j++) {
                data[static_cast<size_t>(i) * c.w + j] = static_cast<float>(c.offset + j);
            }
        }
        const auto input = detail::make_device_tensor(
            device,
            ttnn::Shape{1, 1, static_cast<uint32_t>(c.h), static_cast<uint32_t>(c.w)},
            data,
            DataType::BFLOAT16,
            Layout::TILE);
        const auto out_max = ttnn::max(input, -1, true).to_vector<bfloat16>();
        const auto out_min = ttnn::min(input, -1, true).to_vector<bfloat16>();
        ASSERT_EQ(out_max.size(), static_cast<size_t>(c.h));
        for (int i = 0; i < c.h; i++) {
            ASSERT_EQ(static_cast<float>(out_max[i]), static_cast<float>(c.offset + c.w - 1))
                << "offset=" << c.offset << " row " << i;
            ASSERT_EQ(static_cast<float>(out_min[i]), static_cast<float>(c.offset)) << "offset=" << c.offset;
        }
    }
}

TEST_F(ReductionSmoke, MinMaxReduceH) {
    auto& device = *device_;
    for (const auto& c :
         std::array<detail::MinMaxCase, 4>{{{63, 3100, 4}, {64, 3200, 4}, {63, 3100, -128}, {64, 3200, -128}}}) {
        std::vector<float> data(static_cast<size_t>(c.h) * c.w);
        for (int i = 0; i < c.h; i++) {
            for (int j = 0; j < c.w; j++) {
                data[static_cast<size_t>(i) * c.w + j] = static_cast<float>(c.offset + i);
            }
        }
        const auto input = detail::make_device_tensor(
            device,
            ttnn::Shape{1, 1, static_cast<uint32_t>(c.h), static_cast<uint32_t>(c.w)},
            data,
            DataType::BFLOAT16,
            Layout::TILE);
        const auto out_max = ttnn::max(input, -2, true).to_vector<bfloat16>();
        const auto out_min = ttnn::min(input, -2, true).to_vector<bfloat16>();
        ASSERT_EQ(out_max.size(), static_cast<size_t>(c.w));
        for (int j = 0; j < c.w; j++) {
            ASSERT_EQ(static_cast<float>(out_max[j]), static_cast<float>(c.offset + c.h - 1))
                << "offset=" << c.offset << " col " << j;
            ASSERT_EQ(static_cast<float>(out_min[j]), static_cast<float>(c.offset)) << "offset=" << c.offset;
        }
    }
}

TEST_F(ReductionSmoke, MinMaxReduceBothDims) {
    auto& device = *device_;
    for (const auto& c : std::array<detail::MinMaxCase, 2>{{{64, 64, 1}, {30, 30, -1004}}}) {
        std::vector<float> data(static_cast<size_t>(c.h) * c.w);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<float>(c.offset) + static_cast<float>(i);
        }
        const auto input = detail::make_device_tensor(
            device,
            ttnn::Shape{1, 1, static_cast<uint32_t>(c.h), static_cast<uint32_t>(c.w)},
            data,
            DataType::BFLOAT16,
            Layout::TILE);
        const auto out_max = ttnn::max(input, ttsl::SmallVector<int>{-2, -1}, true).to_vector<bfloat16>();
        const auto out_min = ttnn::min(input, ttsl::SmallVector<int>{-2, -1}, true).to_vector<bfloat16>();
        ASSERT_EQ(out_max.size(), 1u);
        ASSERT_EQ(out_min.size(), 1u);
        ASSERT_EQ(static_cast<float>(out_max[0]), static_cast<float>(c.offset + c.h * c.w - 1))
            << "offset=" << c.offset;
        ASSERT_EQ(static_cast<float>(out_min[0]), static_cast<float>(c.offset)) << "offset=" << c.offset;
    }
}

// ---------------------------------------------------------------------------
// Generic reduce: mean (AVG pool math) on the tiled W / H / multi-axis paths
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, MeanReduceW) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(r + 1);  // constant per row -> exact mean
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::mean(input, -1, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{h, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(static_cast<float>(result[r]), static_cast<float>(r + 1)) << "row " << r;
    }
}

TEST_F(ReductionSmoke, MeanReduceH) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(c + 1);  // constant per column
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::mean(input, -2, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, w}));
    const auto result = output.to_vector<bfloat16>();
    for (int c = 0; c < w; c++) {
        EXPECT_EQ(static_cast<float>(result[c]), static_cast<float>(c + 1)) << "col " << c;
    }
}

TEST_F(ReductionSmoke, MeanReduceBothDims) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    const std::vector<float> data(h * w, 3.0f);
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::mean(input, ttsl::SmallVector<int>{0, 1}, false);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(static_cast<float>(result[0]), 3.0f);
}

// ---------------------------------------------------------------------------
// Generic reduce: fp32 accurate SFPU path (FLOAT32 + !fast_and_approximate)
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, Fp32AccurateMeanW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = 0.25f * static_cast<float>(c);
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::FLOAT32, Layout::TILE);
    const auto output = ttnn::mean(input, -1, true);
    const auto result = output.to_vector<float>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(result[r], 7.875f) << "row " << r;  // sum(0.25*j, j<64)/64
    }
}

TEST_F(ReductionSmoke, Fp32AccurateMaxW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = 0.25f * static_cast<float>(c);
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::FLOAT32, Layout::TILE);
    const auto output = ttnn::max(input, -1, true);
    const auto result = output.to_vector<float>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(result[r], 15.75f) << "row " << r;
    }
}

// ---------------------------------------------------------------------------
// Generic reduce: INT32 SFPU paths
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, Int32SumW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<int32_t> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = c + 1;
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::INT32, Layout::TILE);
    const auto output = ttnn::sum(input, -1, true);
    const auto result = output.to_vector<int32_t>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(result[r], (w * (w + 1)) / 2) << "row " << r;
    }
}

TEST_F(ReductionSmoke, Int32MinMaxW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<int32_t> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = c - 500;  // negatives exercise the INT32 pad sentinels
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::INT32, Layout::TILE);
    const auto out_max = ttnn::max(input, -1, true);
    const auto out_min = ttnn::min(input, -1, true);
    const auto max_result = out_max.to_vector<int32_t>();
    const auto min_result = out_min.to_vector<int32_t>();
    ASSERT_EQ(max_result.size(), static_cast<size_t>(h));
    ASSERT_EQ(min_result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(max_result[r], w - 1 - 500) << "row " << r;
        EXPECT_EQ(min_result[r], -500) << "row " << r;
    }
}

TEST_F(ReductionSmoke, Int32SumBothDims) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    const std::vector<int32_t> data(h * w, 1);
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::INT32, Layout::TILE);
    const auto output = ttnn::sum(input, ttsl::SmallVector<int>{0, 1}, false);
    const auto result = output.to_vector<int32_t>();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], h * w);
}

// ---------------------------------------------------------------------------
// Generic reduce: dense row-major paths (stay ROW_MAJOR by default)
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, RowMajorSumW) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>((r % 4) + 1);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::sum(input, -1, true);
    EXPECT_EQ(output.layout(), Layout::ROW_MAJOR);  // dense-RM path contract
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, h, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(static_cast<float>(result[r]), static_cast<float>(w * ((r % 4) + 1))) << "row " << r;
    }
}

TEST_F(ReductionSmoke, RowMajorSumHSplit) {
    auto& device = *device_;
    constexpr int h = 3136, w = 32;  // tall H engages the H-axis slicing heuristic
    const std::vector<float> data(static_cast<size_t>(h) * w, 1.0f);
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::sum(input, -2, true);
    EXPECT_EQ(output.layout(), Layout::ROW_MAJOR);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, 1, w}));
    const auto result = output.to_vector<bfloat16>();
    for (int c = 0; c < w; c++) {
        EXPECT_EQ(static_cast<float>(result[c]), static_cast<float>(h)) << "col " << c;
    }
}

TEST_F(ReductionSmoke, RowMajorMeanW) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(r % 8);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::mean(input, -1, true);
    EXPECT_EQ(output.layout(), Layout::ROW_MAJOR);  // dense-RM path contract (AVG)
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(static_cast<float>(result[r]), static_cast<float>(r % 8)) << "row " << r;
    }
}

// ---------------------------------------------------------------------------
// Generic reduce: fast_reduce_nc and the multi-axis loop
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, FastReduceNCSum) {
    auto& device = *device_;
    // Sum over a non-H/W dim with BFLOAT16 and scalar==1.0 routes to fast_reduce_nc.
    const auto input = ttnn::ones(ttnn::Shape{2, 3, 32, 32}, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto output = ttnn::sum(input, 1, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{2, 1, 32, 32}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_EQ(static_cast<float>(result[i]), 3.0f) << "element " << i;
    }
}

TEST_F(ReductionSmoke, MultiAxisSumChain) {
    auto& device = *device_;
    // A dim list mixing a batch dim and W takes the per-axis transpose loop
    // (bf16 multi-axis sum keeps fp32 intermediates between stages).
    const auto input = ttnn::ones(ttnn::Shape{2, 1, 32, 64}, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto output = ttnn::sum(input, ttsl::SmallVector<int>{0, 3}, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, 32, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_EQ(static_cast<float>(result[i]), 128.0f) << "element " << i;  // 2 * 64
    }
}

// ---------------------------------------------------------------------------
// std/var: Welford kernel variants
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, WelfordVarW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = (c % 2 == 0) ? -1.0f : 1.0f;  // mean 0, population var 1
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::var(input, -1, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_NEAR(static_cast<float>(result[r]), 1.0f, 0.03f) << "row " << r;
    }
}

TEST_F(ReductionSmoke, WelfordStdH) {
    auto& device = *device_;
    constexpr int h = 64, w = 32;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = (r % 2 == 0) ? -1.0f : 1.0f;
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::std(input, -2, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(w));
    for (int c = 0; c < w; c++) {
        EXPECT_NEAR(static_cast<float>(result[c]), 1.0f, 0.03f) << "col " << c;
    }
}

TEST_F(ReductionSmoke, WelfordVarDim0) {
    auto& device = *device_;
    // A single non-H/W dim takes the permute -> H-reduce -> inverse-permute path.
    constexpr int n = 4, h = 32, w = 32;
    std::vector<float> data(n * h * w);
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < h * w; i++) {
            data[p * h * w + i] = static_cast<float>(p);  // {0,1,2,3}: population var 1.25
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{n, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::var(input, 0, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, h, w}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_NEAR(static_cast<float>(result[i]), 1.25f, 0.03f) << "element " << i;
    }
}

TEST_F(ReductionSmoke, WelfordStdMultiDim) {
    auto& device = *device_;
    // 2+ reduce dims take the unified HW welford path.
    constexpr int n = 2, h = 64, w = 64;
    std::vector<float> data(n * h * w);
    for (int i = 0; i < n * h * w; i++) {
        data[i] = (i % 2 == 0) ? -1.0f : 1.0f;
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{n, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::std(
        input, ttsl::SmallVector<int>{1, 2}, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{n, 1, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(static_cast<float>(result[i]), 1.0f, 0.03f) << "element " << i;
    }
}

TEST_F(ReductionSmoke, Fp32WelfordVarW) {
    auto& device = *device_;
    // FLOAT32 welford requires fp32_dest_acc_en, which the op defaults to.
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = 0.5f * static_cast<float>(c);
        }
    }
    const auto input = detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::FLOAT32, Layout::TILE);
    const auto output = ttnn::var(input, -1, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    const auto result = output.to_vector<float>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    // var(0.5*j, j<64) = 0.25 * (64^2 - 1) / 12 = 85.3125
    for (int r = 0; r < h; r++) {
        EXPECT_NEAR(result[r], 85.3125f, 0.1f) << "row " << r;
    }
}

// ---------------------------------------------------------------------------
// argmax: multicore RM, single-core TILE W/H, NC, global
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, ArgmaxMultiCoreRowMajorLastDim) {
    auto& device = *device_;
    constexpr int h = 32, w = 512;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(c % 3);
        }
        data[r * w + (7 * r + 3) % w] = 100.0f;
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::argmax(input, -1, false);
    const auto result = output.to_vector<uint32_t>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(result[r], static_cast<uint32_t>((7 * r + 3) % w)) << "row " << r;
    }
}

TEST_F(ReductionSmoke, ArgmaxSingleCoreTileLastDim) {
    auto& device = *device_;
    constexpr int h = 64, w = 128;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(c % 5);
        }
        data[r * w + (5 * r + 1) % w] = 100.0f;
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::argmax(input, -1, false);
    const auto result = output.to_vector<uint32_t>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(result[r], static_cast<uint32_t>((5 * r + 1) % w)) << "row " << r;
    }
}

TEST_F(ReductionSmoke, ArgmaxSingleCoreTileSecondLastDim) {
    auto& device = *device_;
    constexpr int h = 64, w = 128;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(r % 5);
        }
    }
    for (int c = 0; c < w; c++) {
        data[((3 * c + 2) % h) * w + c] = 100.0f;
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::argmax(input, -2, false);
    const auto result = output.to_vector<uint32_t>();
    ASSERT_EQ(result.size(), static_cast<size_t>(w));
    for (int c = 0; c < w; c++) {
        EXPECT_EQ(result[c], static_cast<uint32_t>((3 * c + 2) % h)) << "col " << c;
    }
}

TEST_F(ReductionSmoke, ArgmaxNCDim) {
    auto& device = *device_;
    constexpr int n = 2, ch = 4, h = 32, w = 32;
    std::vector<float> data(static_cast<size_t>(n) * ch * h * w);
    for (int b = 0; b < n; b++) {
        for (int c = 0; c < ch; c++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    const size_t idx = ((static_cast<size_t>(b) * ch + c) * h + i) * w + j;
                    data[idx] = (c == (b + i + j) % ch) ? 50.0f : 0.5f * static_cast<float>(c);
                }
            }
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{n, ch, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::argmax(input, 1, false);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{n, h, w}));
    const auto result = output.to_vector<uint32_t>();
    for (int b = 0; b < n; b++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                const size_t idx = (static_cast<size_t>(b) * h + i) * w + j;
                ASSERT_EQ(result[idx], static_cast<uint32_t>((b + i + j) % ch))
                    << "b=" << b << " i=" << i << " j=" << j;
            }
        }
    }
}

TEST_F(ReductionSmoke, ArgmaxGlobalRowMajor) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int i = 0; i < h * w; i++) {
        data[i] = static_cast<float>(i % 5);
    }
    constexpr int peak = 777;
    data[peak] = 100.0f;
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::argmax(input, std::nullopt, false);
    const auto result = output.to_vector<uint32_t>();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], static_cast<uint32_t>(peak));
}

// ---------------------------------------------------------------------------
// topk: single-core, multi-core with >32 rows, smallest, non-tile-multiple k,
// index-dtype selection
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, TopkSingleCoreLastDim) {
    auto& device = *device_;
    constexpr int h = 32, w = 64, k = 32;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(w - c);  // strictly descending
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto outputs = ttnn::topk(input, k, -1, /*largest=*/true, /*sorted=*/true);
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[1].dtype(), DataType::UINT16);  // W < 65536 and non-fp32 input
    const auto values = outputs[0].to_vector<bfloat16>();
    const auto indices = outputs[1].to_vector<uint16_t>();
    ASSERT_EQ(values.size(), static_cast<size_t>(h * k));
    for (int r = 0; r < h; r++) {
        for (int m = 0; m < k; m++) {
            ASSERT_EQ(static_cast<float>(values[r * k + m]), static_cast<float>(w - m)) << "r=" << r << " m=" << m;
            ASSERT_EQ(indices[r * k + m], static_cast<uint16_t>(m)) << "r=" << r << " m=" << m;
        }
    }
}

TEST_F(ReductionSmoke, TopkSingleCoreSmallest) {
    auto& device = *device_;
    constexpr int h = 32, w = 64, k = 32;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(w - c);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto outputs = ttnn::topk(input, k, -1, /*largest=*/false, /*sorted=*/true);
    const auto values = outputs[0].to_vector<bfloat16>();
    const auto indices = outputs[1].to_vector<uint16_t>();
    for (int r = 0; r < h; r++) {
        for (int m = 0; m < k; m++) {
            ASSERT_EQ(static_cast<float>(values[r * k + m]), static_cast<float>(m + 1)) << "r=" << r << " m=" << m;
            ASSERT_EQ(indices[r * k + m], static_cast<uint16_t>(w - 1 - m)) << "r=" << r << " m=" << m;
        }
    }
}

TEST_F(ReductionSmoke, TopkNonTileMultipleK) {
    auto& device = *device_;
    // k=50 is padded to a tile multiple internally and sliced back to 50.
    constexpr int h = 32, w = 128, k = 50;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = 512.0f - 4.0f * static_cast<float>(c);  // exact in bf16
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto outputs = ttnn::topk(input, k, -1, /*largest=*/true, /*sorted=*/true);
    ASSERT_EQ(outputs[0].logical_shape(), (ttnn::Shape{1, 1, h, k}));
    const auto values = outputs[0].to_vector<bfloat16>();
    const auto indices = outputs[1].to_vector<uint16_t>();
    for (int r = 0; r < h; r++) {
        for (int m = 0; m < k; m++) {
            ASSERT_EQ(static_cast<float>(values[r * k + m]), 512.0f - 4.0f * m) << "r=" << r << " m=" << m;
            ASSERT_EQ(indices[r * k + m], static_cast<uint16_t>(m)) << "r=" << r << " m=" << m;
        }
    }
}

TEST_F(ReductionSmoke, TopkFp32IndicesUint32) {
    auto& device = *device_;
    constexpr int h = 32, w = 64, k = 32;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(w - c);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::FLOAT32, Layout::TILE);
    const auto outputs = ttnn::topk(input, k, -1, /*largest=*/true, /*sorted=*/true);
    EXPECT_EQ(outputs[1].dtype(), DataType::UINT32);  // fp32 input forces 32-bit indices
    const auto values = outputs[0].to_vector<float>();
    const auto indices = outputs[1].to_vector<uint32_t>();
    for (int r = 0; r < h; r++) {
        for (int m = 0; m < k; m++) {
            ASSERT_EQ(values[r * k + m], static_cast<float>(w - m)) << "r=" << r << " m=" << m;
            ASSERT_EQ(indices[r * k + m], static_cast<uint32_t>(m)) << "r=" << r << " m=" << m;
        }
    }
}

TEST_F(ReductionSmoke, TopkMultiCoreRows64) {
    auto& device = *device_;
    // W=8192 (power of two, >= multi_core_min_width) with 64 flattened rows:
    // the multi-core factory cell where issue #53453 lived. On SKUs where the
    // L1 cost model rejects multi-core, this falls back to single-core and
    // still checks correctness.
    constexpr int n = 1, ch = 2, h = 32, w = 8192, k = 50;
    constexpr int rows = ch * h;
    std::vector<float> data(static_cast<size_t>(rows) * w, 0.0f);
    auto peak_col = [](int row, int m) { return (131 * m + 977 * row) % w; };
    for (int row = 0; row < rows; row++) {
        for (int m = 0; m < k; m++) {
            data[static_cast<size_t>(row) * w + peak_col(row, m)] = 512.0f - 4.0f * m;
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{n, ch, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto outputs = ttnn::topk(input, k, -1, /*largest=*/true, /*sorted=*/true);
    ASSERT_EQ(outputs[0].logical_shape(), (ttnn::Shape{n, ch, h, k}));
    EXPECT_EQ(outputs[1].dtype(), DataType::UINT16);
    const auto values = outputs[0].to_vector<bfloat16>();
    const auto indices = outputs[1].to_vector<uint16_t>();
    for (int row = 0; row < rows; row++) {
        for (int m = 0; m < k; m++) {
            ASSERT_EQ(static_cast<float>(values[row * k + m]), 512.0f - 4.0f * m) << "row=" << row << " m=" << m;
            ASSERT_EQ(indices[row * k + m], static_cast<uint16_t>(peak_col(row, m))) << "row=" << row << " m=" << m;
        }
    }
}

// Host-side lock on the split-selection contract of find_topk_core_config.
// The function is a pure function of its arguments (host-only math, it never
// queries a device), which is why this test runs without silicon attached.
//
// Contract locked here:
// - Among all valid power-of-two splits, the function selects the one that
//   minimizes the makespan score kLocalCostFactor * Wt_local +
//   kFinalCostFactor * Wt_final (constants defined next to the sweep in
//   topk_utils.cpp, fitted to silicon measurements). Warning to future
//   editors: a greedy first-valid / max-cores pick is NOT equivalent -- it
//   maximizes the serial final-stage gather and measures slower on silicon,
//   so don't simplify the sweep back to that.
// - num_cores is asserted as width / split (the only value consistent with a
//   returned config), so a re-fit of the cost constants edits one number per row.
// - The core-rectangle orientation (selected_x/selected_y): the arrangement
//   search takes the WIDEST (smallest-y) factorization that fits -- the
//   arrangement the cost constants were fitted against on silicon.
// - The Ht-gate domain (W in [multi_core_low_ht_min_width,
//   multi_core_min_width) = [1024, 8192)) yields valid configs, including
//   W=1024 right at the eligibility floor, where the start-split clamp in
//   the sweep must bind.
// - Non-tile-multiple K (50): Wt_final rounds K UP to the tile boundary
//   (ceil(k/32) tiles per core, matching what the writer kernels physically
//   gather), so k=50 must select the same config as k=64.
//
// The core range and tile sizes mirror what the multi-core factory passes on
// the p150a unit the constants were fitted on (bf16 values, uint16 indices,
// BH L1; e.g. a 13x10 compute grid on that box -- p150a worker grids vary
// per unit with harvesting, 12x10 on others, which is why production code
// queries the grid instead of assuming one). If kLocalCostFactor /
// kFinalCostFactor or the sweep order changes, these assertions catch it
// even though every numerical topk test stays green.
TEST(TopkCoreConfigModel, SelectsFittedMakespanMinimum) {
    constexpr uint32_t l1_size = 1536 * 1024;                                       // BH L1 per core
    constexpr uint32_t value_tile_size = tt::tile_size(tt::DataFormat::Float16_b);  // 2048
    constexpr uint32_t index_tile_size = tt::tile_size(tt::DataFormat::UInt16);     // 2048
    constexpr uint32_t min_dim = ttnn::prim::constants::min_dim_per_core;           // 64
    const tt::tt_metal::CoreRange core_range({0, 0}, {12, 9});  // grid of the fitting box (p150a grids vary per unit)

    struct Case {
        uint32_t width;
        uint32_t k;
        uint16_t expected_split;
        uint16_t expected_x;
        uint16_t expected_y;
    };
    // Expected values: the minimum of 7 * Wt_local + 2 * Wt_final over all
    // valid power-of-two splits, computable by hand -- recompute each row
    // when re-fitting the cost constants.
    const std::array<Case, 7> cases{{
        {8192, 64, 512, 8, 2},
        {8192, 50, 512, 8, 2},  // == the k=64 row: Wt_final rounds k up to the tile boundary
        {8192, 32, 256, 8, 4},
        {32768, 64, 1024, 8, 4},
        // Ht-gate domain (low-tile-row shapes below multi_core_min_width):
        {1024, 32, 128, 8, 1},  // eligibility floor: 32 tiles < lp2(96 cores), the start-split clamp must bind
        {2048, 32, 128, 8, 2},
        {4096, 32, 256, 8, 2},
    }};
    for (const auto& c : cases) {
        const auto config = ttnn::prim::find_topk_core_config(
            c.width, min_dim, c.width / 2, c.k, core_range, l1_size, value_tile_size, index_tile_size);
        ASSERT_TRUE(config.has_value()) << "W=" << c.width << " k=" << c.k;
        EXPECT_EQ(config->split_size, c.expected_split) << "W=" << c.width << " k=" << c.k;
        EXPECT_EQ(config->num_cores, c.width / c.expected_split) << "W=" << c.width << " k=" << c.k;
        EXPECT_EQ(config->selected_x, c.expected_x) << "W=" << c.width << " k=" << c.k;
        EXPECT_EQ(config->selected_y, c.expected_y) << "W=" << c.width << " k=" << c.k;
    }

    // Grids that cannot host the multi-core layout (local rectangle + final-core
    // row) must return nullopt -- never crash -- so the caller falls back to
    // single-core; single-row and two-row grids are the tight cases.
    for (const auto& small_range : {tt::tt_metal::CoreRange({0, 0}, {7, 0}), tt::tt_metal::CoreRange({0, 0}, {7, 1})}) {
        const auto config = ttnn::prim::find_topk_core_config(
            2048, min_dim, 1024, 32, small_range, l1_size, value_tile_size, index_tile_size);
        EXPECT_FALSE(config.has_value()) << "range=" << small_range.str();
    }
}

// ---------------------------------------------------------------------------
// prod: prod_all (dim=None), prod_nc (dim 0), dim=-1 via permute + prod_nc
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, ProdAllDims) {
    auto& device = *device_;
    constexpr int h = 32, w = 32;
    std::vector<float> data(h * w, 1.0f);
    data[5] = 2.0f;
    data[100] = 2.0f;
    data[999] = 2.0f;
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::prod(input);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(static_cast<float>(result[0]), 8.0f);
}

TEST_F(ReductionSmoke, ProdNCDim0) {
    auto& device = *device_;
    constexpr int n = 3, h = 32, w = 32;
    std::vector<float> data(static_cast<size_t>(n) * h * w);
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < h * w; i++) {
            data[p * h * w + i] = static_cast<float>(p + 1);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{n, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::prod(input, 0, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, h, w}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_EQ(static_cast<float>(result[i]), 6.0f) << "element " << i;  // 1*2*3
    }
}

TEST_F(ReductionSmoke, ProdLastDim) {
    auto& device = *device_;
    // dim in {-1, -2} is routed through a permute to dim 0 and prod_nc.
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w, 1.0f);
    for (int r = 0; r < h; r++) {
        data[r * w + (r % w)] = 2.0f;
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::prod(input, -1, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, h, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(static_cast<float>(result[r]), 2.0f) << "row " << r;
    }
}

// ---------------------------------------------------------------------------
// cumsum / cumprod: shared accumulation factory
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, CumsumLastDim) {
    auto& device = *device_;
    // rank - dim < 4 takes the reshape + permute-to-dim-0 path.
    constexpr int h = 32, w = 64;
    const auto input = ttnn::ones(ttnn::Shape{1, 1, h, w}, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto output = ttnn::cumsum(input, -1);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h * w));
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            ASSERT_EQ(static_cast<float>(result[r * w + c]), static_cast<float>(c + 1)) << "r=" << r << " c=" << c;
        }
    }
}

TEST_F(ReductionSmoke, CumsumDim0NoPermute) {
    auto& device = *device_;
    // rank 4, dim 0: accumulation runs directly on the outermost axis (no permute).
    constexpr int n = 4, h = 32, w = 32;
    const auto input = ttnn::ones(ttnn::Shape{n, 1, h, w}, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto output = ttnn::cumsum(input, 0);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(n * h * w));
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < h * w; i++) {
            ASSERT_EQ(static_cast<float>(result[p * h * w + i]), static_cast<float>(p + 1)) << "plane " << p;
        }
    }
}

TEST_F(ReductionSmoke, CumsumInt32) {
    auto& device = *device_;
    constexpr int n = 4, h = 32, w = 32;
    const std::vector<int32_t> data(static_cast<size_t>(n) * h * w, 2);
    const auto input = detail::make_device_tensor(device, ttnn::Shape{n, 1, h, w}, data, DataType::INT32, Layout::TILE);
    const auto output = ttnn::cumsum(input, 0);
    const auto result = output.to_vector<int32_t>();
    ASSERT_EQ(result.size(), static_cast<size_t>(n * h * w));
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < h * w; i++) {
            ASSERT_EQ(result[p * h * w + i], 2 * (p + 1)) << "plane " << p;
        }
    }
}

TEST_F(ReductionSmoke, CumprodLastDimReverse) {
    auto& device = *device_;
    constexpr int h = 32, w = 32;
    const std::vector<float> data(h * w, 2.0f);
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::cumprod(input, -1, std::nullopt, /*reverse_order=*/true);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h * w));
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            const float expected = std::ldexp(1.0f, w - c);  // 2^(w-c), exact in bf16
            ASSERT_EQ(static_cast<float>(result[r * w + c]), expected) << "r=" << r << " c=" << c;
        }
    }
}

TEST_F(ReductionSmoke, CumprodDim1) {
    auto& device = *device_;
    constexpr int n = 1, ch = 4, h = 32, w = 32;
    const std::vector<float> data(static_cast<size_t>(ch) * h * w, 2.0f);
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{n, ch, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::cumprod(input, 1);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(ch * h * w));
    for (int c = 0; c < ch; c++) {
        for (int i = 0; i < h * w; i++) {
            ASSERT_EQ(static_cast<float>(result[c * h * w + i]), std::ldexp(1.0f, c + 1)) << "channel " << c;
        }
    }
}

// ---------------------------------------------------------------------------
// ema
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, EmaConstantInput) {
    auto& device = *device_;
    constexpr int b = 2, c = 32, t = 64;
    constexpr float alpha = 0.5f;
    const std::vector<float> data(static_cast<size_t>(b) * c * t, 1.0f);
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, b, c, t}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::ema(input, alpha);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(b) * c * t);
    // out_t = alpha * out_{t-1} + (1 - alpha) * in_t with out_{-1} = 0
    // => for constant input 1: out_t = 1 - alpha^(t+1)
    for (int row = 0; row < b * c; row++) {
        for (int j = 0; j < t; j++) {
            const float expected = 1.0f - std::pow(alpha, static_cast<float>(j + 1));
            ASSERT_NEAR(static_cast<float>(result[row * t + j]), expected, 0.02f) << "row=" << row << " t=" << j;
        }
    }
}

// ---------------------------------------------------------------------------
// moe (Mixtral decode shape)
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, MoeMixtralShape) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;  // 32 users, expert dim padded to 64
    constexpr int experts = 8;     // E: real experts
    constexpr int top_e = 2;       // e: experts kept per token
    constexpr uint16_t k = 32;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Expert 0 dominates every row: softmax weight for expert 0 within the
    // top-2 is sigma(8) = e^8 / (e^8 + 1) ~= 0.99966.
    std::vector<float> input_data(h * w, 0.0f);
    for (int r = 0; r < h; r++) {
        input_data[r * w + 0] = 8.0f;
    }
    std::vector<float> expert_mask(w, 0.0f);
    for (int j = experts; j < w; j++) {
        expert_mask[j] = neg_inf;
    }
    std::vector<float> topk_mask(k, 0.0f);
    for (int j = top_e; j < k; j++) {
        topk_mask[j] = neg_inf;
    }

    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, h, w}, input_data, DataType::BFLOAT16, Layout::TILE);
    const auto expert_mask_tensor =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, 1, w}, expert_mask, DataType::BFLOAT16, Layout::TILE);
    const auto topk_mask_tensor =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, 1, k}, topk_mask, DataType::BFLOAT16, Layout::TILE);

    const auto output = ttnn::moe(input, expert_mask_tensor, topk_mask_tensor, k);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, h, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (int r = 0; r < h; r++) {
        EXPECT_NEAR(static_cast<float>(result[r]), 0.99966f, 0.01f) << "user " << r;
    }
}

// ---------------------------------------------------------------------------
// sampling (greedy: per-user k=1, p=0 makes the pick deterministic)
// ---------------------------------------------------------------------------

TEST_F(ReductionSmoke, SamplingGreedyTopK1) {
    auto& device = *device_;
    constexpr int users = 32, w = 64;  // Wt=2: smallest width clear of the Wt=1 hang (#52348)

    std::vector<float> values(static_cast<size_t>(users) * w);
    std::vector<int32_t> indices(static_cast<size_t>(users) * w);
    for (int r = 0; r < users; r++) {
        for (int j = 0; j < w; j++) {
            values[r * w + j] = static_cast<float>(w - j);  // descending, top-1 at j=0
            indices[r * w + j] = r * 100 + j;
        }
    }
    const auto values_tensor =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, users, w}, values, DataType::BFLOAT16, Layout::TILE);
    const auto indices_tensor =
        detail::make_device_tensor(device, ttnn::Shape{1, 1, users, w}, indices, DataType::INT32, Layout::ROW_MAJOR);

    const std::vector<int32_t> k_data(users, 1);
    const std::vector<float> p_data(users, 0.0f);
    const std::vector<float> temp_data(users, 1.0f);
    const ttnn::Shape user_shape(std::array<uint32_t, 1>{users});
    const auto k_tensor = detail::make_device_tensor(device, user_shape, k_data, DataType::INT32, Layout::ROW_MAJOR);
    const auto p_tensor = detail::make_device_tensor(device, user_shape, p_data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto temp_tensor =
        detail::make_device_tensor(device, user_shape, temp_data, DataType::BFLOAT16, Layout::ROW_MAJOR);

    const auto output = ttnn::sampling(values_tensor, indices_tensor, k_tensor, p_tensor, temp_tensor, 42u);
    ASSERT_TRUE(output.dtype() == DataType::UINT32 || output.dtype() == DataType::INT32);
    std::vector<int64_t> picks(users);
    if (output.dtype() == DataType::UINT32) {
        const auto result = output.to_vector<uint32_t>();
        ASSERT_EQ(result.size(), static_cast<size_t>(users));
        std::copy(result.begin(), result.end(), picks.begin());
    } else {
        const auto result = output.to_vector<int32_t>();
        ASSERT_EQ(result.size(), static_cast<size_t>(users));
        std::copy(result.begin(), result.end(), picks.begin());
    }
    for (int r = 0; r < users; r++) {
        EXPECT_EQ(picks[r], static_cast<int64_t>(r) * 100) << "user " << r;
    }
}

// ---------------------------------------------------------------------------
// manual_seed: all four program-factory variants. Verification mirrors
// test_manual_seed.py: seed -> sample without a per-call seed -> perturb the
// device RNG -> re-seed identically -> sample again; the picks must reproduce.
// A seed kernel that silently does nothing leaves the perturbed state in
// place and fails the comparison. (The returned tensor is synthetic - the op
// cannot return void - so its dtype alone proves nothing.)
// ---------------------------------------------------------------------------

namespace detail {

// Stochastic sampling with NO per-call seed: reads the device RNG state that
// ttnn::manual_seed set. Gentle value slope + k=10, p=0.9 keeps several
// plausible picks per user, so the picks depend on the RNG state.
inline std::vector<int64_t> sample_picks(tt::tt_metal::distributed::MeshDevice& device) {
    constexpr int users = 32, w = 64;
    std::vector<float> values(static_cast<size_t>(users) * w);
    std::vector<int32_t> indices(static_cast<size_t>(users) * w);
    for (int r = 0; r < users; r++) {
        for (int j = 0; j < w; j++) {
            values[static_cast<size_t>(r) * w + j] = 2.0f - 0.1f * static_cast<float>(j);
            indices[static_cast<size_t>(r) * w + j] = r * 1000 + j;
        }
    }
    const auto values_tensor =
        make_device_tensor(device, ttnn::Shape{1, 1, users, w}, values, DataType::BFLOAT16, Layout::TILE);
    const auto indices_tensor =
        make_device_tensor(device, ttnn::Shape{1, 1, users, w}, indices, DataType::INT32, Layout::ROW_MAJOR);
    const ttnn::Shape user_shape(std::array<uint32_t, 1>{users});
    const auto k =
        make_device_tensor(device, user_shape, std::vector<int32_t>(users, 10), DataType::INT32, Layout::ROW_MAJOR);
    const auto p =
        make_device_tensor(device, user_shape, std::vector<float>(users, 0.9f), DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto temp =
        make_device_tensor(device, user_shape, std::vector<float>(users, 1.0f), DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto out = ttnn::sampling(values_tensor, indices_tensor, k, p, temp, std::nullopt);
    std::vector<int64_t> picks(users);
    if (out.dtype() == DataType::UINT32) {
        const auto result = out.to_vector<uint32_t>();
        std::copy(result.begin(), result.end(), picks.begin());
    } else {
        const auto result = out.to_vector<int32_t>();
        std::copy(result.begin(), result.end(), picks.begin());
    }
    return picks;
}

}  // namespace detail

TEST_F(ReductionSmoke, ManualSeedAllCores) {
    auto& device = *device_;
    const auto out = ttnn::manual_seed(42u, std::ref(device));
    EXPECT_EQ(out.dtype(), DataType::UINT32);
    const auto picks1 = detail::sample_picks(device);
    ttnn::manual_seed(999u, std::ref(device));  // perturb the RNG state
    (void)detail::sample_picks(device);
    ttnn::manual_seed(42u, std::ref(device));
    const auto picks2 = detail::sample_picks(device);
    ASSERT_EQ(picks1, picks2);
}

TEST_F(ReductionSmoke, ManualSeedSingleUser) {
    auto& device = *device_;
    ttnn::manual_seed(7u, std::ref(device));  // known base state on all cores
    const auto out = ttnn::manual_seed(42u, std::ref(device), 3u);
    EXPECT_EQ(out.dtype(), DataType::UINT32);
    const auto picks1 = detail::sample_picks(device);
    ttnn::manual_seed(999u, std::ref(device));  // perturb all users
    (void)detail::sample_picks(device);
    ttnn::manual_seed(42u, std::ref(device), 3u);  // restore only user 3
    const auto picks2 = detail::sample_picks(device);
    ASSERT_EQ(picks1[3], picks2[3]);  // the re-seeded user's pick reproduces
}

TEST_F(ReductionSmoke, ManualSeedUserTensor) {
    auto& device = *device_;
    const std::vector<uint32_t> users = {0, 1, 2, 3};
    const auto user_tensor = detail::make_device_tensor(
        device, ttnn::Shape(std::array<uint32_t, 1>{4}), users, DataType::UINT32, Layout::ROW_MAJOR);
    const auto out = ttnn::manual_seed(43u, std::ref(device), user_tensor);
    EXPECT_EQ(out.dtype(), DataType::UINT32);
    const auto picks1 = detail::sample_picks(device);
    ttnn::manual_seed(999u, std::ref(device));  // perturb all users
    (void)detail::sample_picks(device);
    ttnn::manual_seed(43u, std::ref(device), user_tensor);  // restore users 0-3
    const auto picks2 = detail::sample_picks(device);
    for (int r = 0; r < 4; r++) {
        ASSERT_EQ(picks1[r], picks2[r]) << "user " << r;
    }
}

TEST_F(ReductionSmoke, ManualSeedPerUserSeeds) {
    auto& device = *device_;
    const std::vector<uint32_t> seeds = {7, 8, 9, 10};
    const std::vector<uint32_t> users = {0, 1, 2, 3};
    const ttnn::Shape shape(std::array<uint32_t, 1>{4});
    const auto seed_tensor = detail::make_device_tensor(device, shape, seeds, DataType::UINT32, Layout::ROW_MAJOR);
    const auto user_tensor = detail::make_device_tensor(device, shape, users, DataType::UINT32, Layout::ROW_MAJOR);
    const auto out = ttnn::manual_seed(seed_tensor, std::nullopt, user_tensor);
    EXPECT_EQ(out.dtype(), DataType::UINT32);
    const auto picks1 = detail::sample_picks(device);
    ttnn::manual_seed(999u, std::ref(device));  // perturb all users
    (void)detail::sample_picks(device);
    ttnn::manual_seed(seed_tensor, std::nullopt, user_tensor);  // restore users 0-3
    const auto picks2 = detail::sample_picks(device);
    for (int r = 0; r < 4; r++) {
        ASSERT_EQ(picks1[r], picks2[r]) << "user " << r;
    }
}

}  // namespace ttnn::operations::reduction::test
