// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark & correctness test: Original Matmul (ProgramFactoryConcept)
// vs MatmulNew (ProgramDescriptorFactoryConcept).
//
// 1. Verifies both ops produce identical results on cache miss and cache hit.
// 2. Runs each operation N times and compares wall-clock dispatch time.

#include <chrono>
#include <iostream>

#include "gtest/gtest.h"
#include "ttnn_test_fixtures.hpp"

#include <tt-metalium/distributed.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul_new/device/matmul_new_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

// Helper: build fully-resolved attributes including program_config.
inline ttnn::prim::MatmulParams build_attrs(const Tensor& a, const Tensor& b) {
    ttnn::prim::MatmulParams params{};
    auto attrs = ttnn::prim::create_matmul_attributes(a, b, params, {std::nullopt});
    attrs.program_config =
        ttnn::operations::matmul::get_program_config(a, b, attrs.transpose_a, attrs.transpose_b, 0, attrs);
    return attrs;
}

inline ttnn::prim::MatmulInputs build_tensor_args(const Tensor& a, const Tensor& b) {
    return {{a, b}, {std::nullopt}, {std::nullopt}};
}

// Directly call launch<> so both paths are as symmetric as possible.
inline std::vector<Tensor> call_old(
    const ttnn::prim::MatmulParams& attrs, const ttnn::prim::MatmulInputs& tensor_args) {
    return ttnn::device_operation::launch<ttnn::prim::MatmulDeviceOperation>(attrs, tensor_args);
}
inline std::vector<Tensor> call_new(
    const ttnn::prim::MatmulParams& attrs, const ttnn::prim::MatmulInputs& tensor_args) {
    return ttnn::device_operation::launch<ttnn::prim::MatmulNewDeviceOperation>(attrs, tensor_args);
}

class MatmulDescriptorBenchmark : public ttnn::TTNNFixtureWithDevice {};

TEST_F(MatmulDescriptorBenchmark, CorrectnessNonCached) {
    ttnn::Shape a_shape{1, 1, 128, 64};
    ttnn::Shape b_shape{1, 1, 64, 128};
    Tensor a = ttnn::random::random(a_shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);
    Tensor b = ttnn::random::random(b_shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);
    auto attrs = build_attrs(a, b);
    auto targs = build_tensor_args(a, b);

    Tensor out_old = call_old(attrs, targs).at(0).cpu();
    Tensor out_new = call_new(attrs, targs).at(0).cpu();

    bool match = ttnn::allclose<bfloat16>(out_old, out_new, 0.1f, 0.1f);
    ASSERT_TRUE(match) << "Cache-miss outputs differ between Matmul and MatmulNew";
}

TEST_F(MatmulDescriptorBenchmark, CorrectnessCached) {
    ttnn::Shape a_shape{1, 1, 128, 64};
    ttnn::Shape b_shape{1, 1, 64, 128};
    Tensor a = ttnn::random::random(a_shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);
    Tensor b = ttnn::random::random(b_shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);
    auto attrs = build_attrs(a, b);
    auto targs = build_tensor_args(a, b);

    {
        auto _ = call_old(attrs, targs);
    }
    {
        auto _ = call_new(attrs, targs);
    }

    Tensor out_old = call_old(attrs, targs).at(0).cpu();
    Tensor out_new = call_new(attrs, targs).at(0).cpu();

    bool match = ttnn::allclose<bfloat16>(out_old, out_new, 0.1f, 0.1f);
    ASSERT_TRUE(match) << "Cache-hit outputs differ between Matmul and MatmulNew";
}

// Performance: run new FIRST, then old, to avoid instruction cache bias.
TEST_F(MatmulDescriptorBenchmark, DispatchPerformance) {
    constexpr uint32_t N = 1000000;

    ttnn::Shape a_shape{1, 1, 128, 64};
    ttnn::Shape b_shape{1, 1, 64, 128};
    Tensor a = ttnn::random::random(a_shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);
    Tensor b = ttnn::random::random(b_shape, DataType::BFLOAT16, Layout::TILE).to_device(this->device_);
    auto attrs = build_attrs(a, b);
    auto targs = build_tensor_args(a, b);

    // Warm up both.
    {
        auto x = call_old(attrs, targs);
        auto y = call_new(attrs, targs);
    }

    // ---- Benchmark MatmulNew FIRST ----
    auto t2 = Clock::now();
    for (uint32_t i = 0; i < N; ++i) {
        auto out = call_new(attrs, targs);
        (void)out;
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});
    auto t3 = Clock::now();
    Duration new_time = t3 - t2;

    // ---- Benchmark original Matmul ----
    auto t0 = Clock::now();
    for (uint32_t i = 0; i < N; ++i) {
        auto out = call_old(attrs, targs);
        (void)out;
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});
    auto t1 = Clock::now();
    Duration old_time = t1 - t0;

    double old_per_op = old_time.count() / N;
    double new_per_op = new_time.count() / N;
    double overhead_pct = ((new_per_op - old_per_op) / old_per_op) * 100.0;

    std::cout << "\n=== Matmul ProgramDescriptor Benchmark ===\n";
    std::cout << "  Iterations:         " << N << "\n";
    std::cout << "  Original Matmul:    " << old_time.count() << " ms total, " << old_per_op << " ms/op\n";
    std::cout << "  MatmulNew (desc):   " << new_time.count() << " ms total, " << new_per_op << " ms/op\n";
    std::cout << "  Overhead:           " << overhead_pct << " %\n";
    std::cout << "==========================================\n\n";

    EXPECT_LT(new_per_op, old_per_op * 1.03) << "MatmulNew is more than 3% slower than original Matmul.";
}

}  // namespace
