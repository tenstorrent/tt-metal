// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Branch-keyed smoke tests for the ttnn reduction op family.
//
// One test per program-factory / dispatch branch, for every op under
// ttnn/cpp/ttnn/operations/reduction/. Each test uses small deterministic
// inputs with exact (or op-tolerance) expected outputs, so a branch that runs
// but produces garbage fails. Complements test_reduction.cpp, which covers the
// tiled W/H/HW sum/max/min factories; this file covers the branches that had
// no pre-merge coverage: mean/std/var/prod/argmax/topk/cumsum/cumprod/ema/moe/
// sampling/manual_seed, plus the int32, fp32-SFPU, row-major-dense,
// fast_reduce_nc and multi-axis paths of the generic reduce.
//
// Branch → test map (front-end dispatch in generic_reductions.cpp unless noted):
//   ReduceMultiCoreW/H (AVG)             MeanReduceW / MeanReduceH
//   multi-axis loop (mean)               MeanReduceBothDims
//   fp32 accurate SFPU path              Fp32AccurateMeanW / Fp32AccurateMaxW
//   INT32 SFPU W / HW two-step           Int32SumW, Int32MinMaxW / Int32SumBothDims
//   dense row-major W / H(split) / AVG   RowMajorSumW / RowMajorSumHSplit / RowMajorMeanW
//   fast_reduce_nc                       FastReduceNCSum
//   bf16 multi-axis fp32 chain           MultiAxisSumChain
//   welford W / H / non-HW permute / HW  WelfordVarW / WelfordStdH / WelfordVarDim0 / WelfordStdMultiDim
//   welford fp32 (fp32_dest_acc default) Fp32WelfordVarW
//   argmax multicore RM / TILE W / TILE H / NC / global
//   topk single-core / multi-core rows>32 / smallest / non-tile k / fp32→uint32 indices
//   prod_all / prod_nc dim0 / dim=-1 via permute+prod_nc
//   accumulation: cumsum permuted / dim0 direct / int32; cumprod reverse / dim1
//   ema single factory; moe single factory; sampling single factory
//   manual_seed: all four program-factory variants
//
// Not covered here (covered by python suites): sharded I/O variants of the
// generic reduce, block-float dtypes, zero-volume / rank-0 host paths.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
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
#include "ttnn/operations/reduction/topk/topk.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::reduction::test {

class ReductionBranches : public TTNNFixtureWithSuiteDevice<ReductionBranches> {};

namespace detail {

inline MemoryConfig dram_interleaved() {
    return MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
}

template <typename T>
Tensor make_device_tensor(
    tt::tt_metal::distributed::MeshDevice& device,
    const ttnn::Shape& shape,
    const std::vector<T>& data,
    DataType dtype,
    Layout layout) {
    const MemoryConfig mem_cfg = dram_interleaved();
    const TensorLayout tensor_layout(dtype, PageConfig(layout), mem_cfg);
    const tt::tt_metal::TensorSpec tensor_spec(shape, tensor_layout);
    return Tensor::from_vector(data, tensor_spec).to_device(&device, mem_cfg, ttnn::QueueId(0));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Generic reduce: mean (AVG pool math) on the tiled W / H / multi-axis paths
// ---------------------------------------------------------------------------

TEST_F(ReductionBranches, MeanReduceW) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(r + 1);  // constant per row -> exact mean
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::mean(input, -1, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{h, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(static_cast<float>(result[r]), static_cast<float>(r + 1)) << "row " << r;
    }
}

TEST_F(ReductionBranches, MeanReduceH) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(c + 1);  // constant per column
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::mean(input, -2, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, w}));
    const auto result = output.to_vector<bfloat16>();
    for (int c = 0; c < w; c++) {
        EXPECT_EQ(static_cast<float>(result[c]), static_cast<float>(c + 1)) << "col " << c;
    }
}

TEST_F(ReductionBranches, MeanReduceBothDims) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    const std::vector<float> data(h * w, 3.0f);
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::mean(input, ttsl::SmallVector<int>{0, 1}, false);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(static_cast<float>(result[0]), 3.0f);
}

// ---------------------------------------------------------------------------
// Generic reduce: fp32 accurate SFPU path (FLOAT32 + !fast_and_approximate)
// ---------------------------------------------------------------------------

TEST_F(ReductionBranches, Fp32AccurateMeanW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = 0.25f * static_cast<float>(c);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::FLOAT32, Layout::TILE);
    const auto output = ttnn::mean(input, -1, true);
    const auto result = output.to_vector<float>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(result[r], 7.875f) << "row " << r;  // sum(0.25*j, j<64)/64
    }
}

TEST_F(ReductionBranches, Fp32AccurateMaxW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = 0.25f * static_cast<float>(c);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::FLOAT32, Layout::TILE);
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

TEST_F(ReductionBranches, Int32SumW) {
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

TEST_F(ReductionBranches, Int32MinMaxW) {
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

TEST_F(ReductionBranches, Int32SumBothDims) {
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

TEST_F(ReductionBranches, RowMajorSumW) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>((r % 4) + 1);
        }
    }
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::sum(input, -1, true);
    EXPECT_EQ(output.layout(), Layout::ROW_MAJOR);  // dense-RM path contract
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, h, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(static_cast<float>(result[r]), static_cast<float>(w * ((r % 4) + 1))) << "row " << r;
    }
}

TEST_F(ReductionBranches, RowMajorSumHSplit) {
    auto& device = *device_;
    constexpr int h = 3136, w = 32;  // tall H engages the H-axis slicing heuristic
    const std::vector<float> data(static_cast<size_t>(h) * w, 1.0f);
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::sum(input, -2, true);
    EXPECT_EQ(output.layout(), Layout::ROW_MAJOR);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, 1, w}));
    const auto result = output.to_vector<bfloat16>();
    for (int c = 0; c < w; c++) {
        EXPECT_EQ(static_cast<float>(result[c]), static_cast<float>(h)) << "col " << c;
    }
}

TEST_F(ReductionBranches, RowMajorMeanW) {
    auto& device = *device_;
    constexpr int h = 64, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(r % 8);
        }
    }
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::mean(input, -1, true);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(static_cast<float>(result[r]), static_cast<float>(r % 8)) << "row " << r;
    }
}

// ---------------------------------------------------------------------------
// Generic reduce: fast_reduce_nc and the multi-axis loop
// ---------------------------------------------------------------------------

TEST_F(ReductionBranches, FastReduceNCSum) {
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

TEST_F(ReductionBranches, MultiAxisSumChain) {
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

TEST_F(ReductionBranches, WelfordVarW) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = (c % 2 == 0) ? -1.0f : 1.0f;  // mean 0, population var 1
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::var(
        input, -1, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_NEAR(static_cast<float>(result[r]), 1.0f, 0.03f) << "row " << r;
    }
}

TEST_F(ReductionBranches, WelfordStdH) {
    auto& device = *device_;
    constexpr int h = 64, w = 32;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = (r % 2 == 0) ? -1.0f : 1.0f;
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::std(
        input, -2, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    const auto result = output.to_vector<bfloat16>();
    ASSERT_EQ(result.size(), static_cast<size_t>(w));
    for (int c = 0; c < w; c++) {
        EXPECT_NEAR(static_cast<float>(result[c]), 1.0f, 0.03f) << "col " << c;
    }
}

TEST_F(ReductionBranches, WelfordVarDim0) {
    auto& device = *device_;
    // A single non-H/W dim takes the permute -> H-reduce -> inverse-permute branch.
    constexpr int n = 4, h = 32, w = 32;
    std::vector<float> data(n * h * w);
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < h * w; i++) {
            data[p * h * w + i] = static_cast<float>(p);  // {0,1,2,3}: population var 1.25
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{n, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::var(
        input, 0, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, h, w}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_NEAR(static_cast<float>(result[i]), 1.25f, 0.03f) << "element " << i;
    }
}

TEST_F(ReductionBranches, WelfordStdMultiDim) {
    auto& device = *device_;
    // 2+ reduce dims take the unified HW welford path.
    constexpr int n = 2, h = 64, w = 64;
    std::vector<float> data(n * h * w);
    for (int i = 0; i < n * h * w; i++) {
        data[i] = (i % 2 == 0) ? -1.0f : 1.0f;
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{n, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::std(
        input, ttsl::SmallVector<int>{1, 2}, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{n, 1, 1}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_NEAR(static_cast<float>(result[i]), 1.0f, 0.03f) << "element " << i;
    }
}

TEST_F(ReductionBranches, Fp32WelfordVarW) {
    auto& device = *device_;
    // FLOAT32 welford requires fp32_dest_acc_en, which the op defaults to.
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = 0.5f * static_cast<float>(c);
        }
    }
    const auto input =
        detail::make_device_tensor(device, ttnn::Shape{h, w}, data, DataType::FLOAT32, Layout::TILE);
    const auto output = ttnn::var(
        input, -1, true, std::nullopt, std::nullopt, /*scalar=*/1.0f, /*correction=*/false);
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

TEST_F(ReductionBranches, ArgmaxMultiCoreRowMajorLastDim) {
    auto& device = *device_;
    constexpr int h = 32, w = 512;
    std::vector<float> data(h * w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            data[r * w + c] = static_cast<float>(c % 3);
        }
        data[r * w + (7 * r + 3) % w] = 100.0f;
    }
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::argmax(input, -1, false);
    const auto result = output.to_vector<uint32_t>();
    ASSERT_EQ(result.size(), static_cast<size_t>(h));
    for (int r = 0; r < h; r++) {
        EXPECT_EQ(result[r], static_cast<uint32_t>((7 * r + 3) % w)) << "row " << r;
    }
}

TEST_F(ReductionBranches, ArgmaxSingleCoreTileLastDim) {
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

TEST_F(ReductionBranches, ArgmaxSingleCoreTileSecondLastDim) {
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

TEST_F(ReductionBranches, ArgmaxNCDim) {
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
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{n, ch, h, w}, data, DataType::BFLOAT16, Layout::TILE);
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

TEST_F(ReductionBranches, ArgmaxGlobalRowMajor) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;
    std::vector<float> data(h * w);
    for (int i = 0; i < h * w; i++) {
        data[i] = static_cast<float>(i % 5);
    }
    constexpr int peak = 777;
    data[peak] = 100.0f;
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, h, w}, data, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto output = ttnn::argmax(input, std::nullopt, false);
    const auto result = output.to_vector<uint32_t>();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], static_cast<uint32_t>(peak));
}

// ---------------------------------------------------------------------------
// topk: single-core, multi-core with >32 rows, smallest, non-tile-multiple k,
// index-dtype selection
// ---------------------------------------------------------------------------

TEST_F(ReductionBranches, TopkSingleCoreLastDim) {
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

TEST_F(ReductionBranches, TopkSingleCoreSmallest) {
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

TEST_F(ReductionBranches, TopkNonTileMultipleK) {
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

TEST_F(ReductionBranches, TopkFp32IndicesUint32) {
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

TEST_F(ReductionBranches, TopkMultiCoreRows64) {
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
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{n, ch, h, w}, data, DataType::BFLOAT16, Layout::TILE);
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

// ---------------------------------------------------------------------------
// prod: prod_all (dim=None), prod_nc (dim 0), dim=-1 via permute + prod_nc
// ---------------------------------------------------------------------------

TEST_F(ReductionBranches, ProdAllDims) {
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

TEST_F(ReductionBranches, ProdNCDim0) {
    auto& device = *device_;
    constexpr int n = 3, h = 32, w = 32;
    std::vector<float> data(static_cast<size_t>(n) * h * w);
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < h * w; i++) {
            data[p * h * w + i] = static_cast<float>(p + 1);
        }
    }
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{n, 1, h, w}, data, DataType::BFLOAT16, Layout::TILE);
    const auto output = ttnn::prod(input, 0, true);
    ASSERT_EQ(output.logical_shape(), (ttnn::Shape{1, 1, h, w}));
    const auto result = output.to_vector<bfloat16>();
    for (size_t i = 0; i < result.size(); i++) {
        ASSERT_EQ(static_cast<float>(result[i]), 6.0f) << "element " << i;  // 1*2*3
    }
}

TEST_F(ReductionBranches, ProdLastDim) {
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

TEST_F(ReductionBranches, CumsumLastDim) {
    auto& device = *device_;
    // rank - dim < 4 takes the reshape + permute-to-dim-0 branch.
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

TEST_F(ReductionBranches, CumsumDim0NoPermute) {
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

TEST_F(ReductionBranches, CumsumInt32) {
    auto& device = *device_;
    constexpr int n = 4, h = 32, w = 32;
    const std::vector<int32_t> data(static_cast<size_t>(n) * h * w, 2);
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{n, 1, h, w}, data, DataType::INT32, Layout::TILE);
    const auto output = ttnn::cumsum(input, 0);
    const auto result = output.to_vector<int32_t>();
    ASSERT_EQ(result.size(), static_cast<size_t>(n * h * w));
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < h * w; i++) {
            ASSERT_EQ(result[p * h * w + i], 2 * (p + 1)) << "plane " << p;
        }
    }
}

TEST_F(ReductionBranches, CumprodLastDimReverse) {
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

TEST_F(ReductionBranches, CumprodDim1) {
    auto& device = *device_;
    constexpr int n = 1, ch = 4, h = 32, w = 32;
    const std::vector<float> data(static_cast<size_t>(ch) * h * w, 2.0f);
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{n, ch, h, w}, data, DataType::BFLOAT16, Layout::TILE);
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

TEST_F(ReductionBranches, EmaConstantInput) {
    auto& device = *device_;
    constexpr int b = 2, c = 32, t = 64;
    constexpr float alpha = 0.5f;
    const std::vector<float> data(static_cast<size_t>(b) * c * t, 1.0f);
    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{1, b, c, t}, data, DataType::BFLOAT16, Layout::TILE);
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

TEST_F(ReductionBranches, MoeMixtralShape) {
    auto& device = *device_;
    constexpr int h = 32, w = 64;    // 32 users, expert dim padded to 64
    constexpr int experts = 8;       // E: real experts
    constexpr int top_e = 2;         // e: experts kept per token
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

    const auto input = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, h, w}, input_data, DataType::BFLOAT16, Layout::TILE);
    const auto expert_mask_tensor = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, 1, w}, expert_mask, DataType::BFLOAT16, Layout::TILE);
    const auto topk_mask_tensor = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, 1, k}, topk_mask, DataType::BFLOAT16, Layout::TILE);

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

TEST_F(ReductionBranches, SamplingGreedyTopK1) {
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
    const auto values_tensor = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, users, w}, values, DataType::BFLOAT16, Layout::TILE);
    const auto indices_tensor = detail::make_device_tensor(
        device, ttnn::Shape{1, 1, users, w}, indices, DataType::INT32, Layout::ROW_MAJOR);

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
// manual_seed: all four program-factory variants
// ---------------------------------------------------------------------------

TEST_F(ReductionBranches, ManualSeedAllCores) {
    auto& device = *device_;
    const auto out = ttnn::manual_seed(42u, std::ref(device));
    EXPECT_EQ(out.dtype(), DataType::UINT32);
}

TEST_F(ReductionBranches, ManualSeedSingleUser) {
    auto& device = *device_;
    const auto out = ttnn::manual_seed(42u, std::ref(device), 3u);
    EXPECT_EQ(out.dtype(), DataType::UINT32);
}

TEST_F(ReductionBranches, ManualSeedUserTensor) {
    auto& device = *device_;
    const std::vector<uint32_t> users = {0, 1, 2, 3};
    const auto user_tensor = detail::make_device_tensor(
        device, ttnn::Shape(std::array<uint32_t, 1>{4}), users, DataType::UINT32, Layout::ROW_MAJOR);
    const auto out = ttnn::manual_seed(43u, std::ref(device), user_tensor);
    EXPECT_EQ(out.dtype(), DataType::UINT32);
}

TEST_F(ReductionBranches, ManualSeedPerUserSeeds) {
    auto& device = *device_;
    const std::vector<uint32_t> seeds = {7, 8, 9, 10};
    const std::vector<uint32_t> users = {0, 1, 2, 3};
    const ttnn::Shape shape(std::array<uint32_t, 1>{4});
    const auto seed_tensor = detail::make_device_tensor(device, shape, seeds, DataType::UINT32, Layout::ROW_MAJOR);
    const auto user_tensor = detail::make_device_tensor(device, shape, users, DataType::UINT32, Layout::ROW_MAJOR);
    const auto out = ttnn::manual_seed(seed_tensor, std::nullopt, user_tensor);
    EXPECT_EQ(out.dtype(), DataType::UINT32);
}

}  // namespace ttnn::operations::reduction::test
