// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <iomanip>
#include <iostream>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

class KSplitGramMatmulTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

ttnn::Tensor make_test_tensor(uint32_t M_tiles, uint32_t K_dim = 0) {
    auto* device = &ttml::autograd::ctx().get_device();
    uint32_t M = M_tiles * 32;
    uint32_t K = (K_dim > 0) ? K_dim : M;
    std::vector<float> data(M * K);
    std::generate(data.begin(), data.end(), []() {
        static std::mt19937 gen(42);
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        return dist(gen);
    });
    auto shape = ttnn::Shape({1, 1, M, K});
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

double tile_pcc(const std::vector<float>& a, const std::vector<float>& b) {
    double sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    for (size_t i = 0; i < a.size(); i++) {
        sum_ab += (double)a[i] * b[i];
        sum_a2 += (double)a[i] * a[i];
        sum_b2 += (double)b[i] * b[i];
    }
    return sum_ab / (std::sqrt(sum_a2) * std::sqrt(sum_b2));
}

std::vector<float> compute_gram_tile(
    const std::vector<float>& in_vec, uint32_t K, uint32_t m_tile, uint32_t n_tile, int k_stride = 1, int k_start = 0) {
    std::vector<float> result(32 * 32, 0.0f);
    uint32_t K_tiles = K / 32;
    for (uint32_t k_tile = k_start; k_tile < K_tiles; k_tile += k_stride) {
        for (uint32_t i = 0; i < 32; i++) {
            for (uint32_t j = 0; j < 32; j++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < 32; l++) {
                    sum += in_vec[(m_tile * 32 + i) * K + k_tile * 32 + l] *
                           in_vec[(n_tile * 32 + j) * K + k_tile * 32 + l];
                }
                result[i * 32 + j] += sum;
            }
        }
    }
    return result;
}

std::vector<float> extract_output_tile(
    const std::vector<float>& out_vec, uint32_t padded_out, uint32_t tile_r, uint32_t tile_c) {
    std::vector<float> tile(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) tile[i * 32 + j] = out_vec[(tile_r * 32 + i) * padded_out + tile_c * 32 + j];
    return tile;
}

}  // namespace

TEST_F(KSplitGramMatmulTest, Verification4096x4096) {
    auto input = make_test_tensor(128, 4096);
    auto output = ttml::metal::gram_matmul(input);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t padded_out = output.logical_shape()[-1];

    // Upper off-diagonal: full accumulated result (own odd-K + partner's even-K)
    auto ref = compute_gram_tile(in_vec, K, 2, 15);
    auto dev = extract_output_tile(out_vec, padded_out, 2, 15);
    double p = tile_pcc(ref, dev);
    std::cout << "G[2,15] (upper) PCC=" << p << "\n";
    EXPECT_GT(p, 0.99);

    // Diagonal: full accumulated result (from helper)
    ref = compute_gram_tile(in_vec, K, 0, 0);
    dev = extract_output_tile(out_vec, padded_out, 0, 0);
    p = tile_pcc(ref, dev);
    std::cout << "G[0,0] (diag) PCC=" << p << "\n";
    EXPECT_GT(p, 0.99);
}

TEST_F(KSplitGramMatmulTest, Verification4096x11008) {
    auto input = make_test_tensor(128, 11008);
    auto output = ttml::metal::gram_matmul(input);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t padded_out = output.logical_shape()[-1];

    auto ref = compute_gram_tile(in_vec, K, 2, 15);
    auto dev = extract_output_tile(out_vec, padded_out, 2, 15);
    double p = tile_pcc(ref, dev);
    std::cout << "G[2,15] PCC=" << p << "\n";
    EXPECT_GT(p, 0.99);
}

TEST_F(KSplitGramMatmulTest, VerificationMirror) {
    auto input = make_test_tensor(20);
    auto output = ttml::metal::gram_matmul(input, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t padded_out = output.logical_shape()[-1];

    // Upper: full result
    auto ref_upper = compute_gram_tile(in_vec, K, 2, 4);
    auto dev_upper = extract_output_tile(out_vec, padded_out, 2, 4);
    double p = tile_pcc(ref_upper, dev_upper);
    std::cout << "Upper G[2,4] PCC=" << p << "\n";
    EXPECT_GT(p, 0.99);

    // Mirror: transposed
    auto dev_mirror = extract_output_tile(out_vec, padded_out, 4, 2);
    std::vector<float> ref_mirror(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) ref_mirror[i * 32 + j] = ref_upper[j * 32 + i];
    double p_mirror = tile_pcc(ref_mirror, dev_mirror);
    std::cout << "Mirror G[4,2] PCC=" << p_mirror << "\n";
    EXPECT_GT(p_mirror, 0.99);
}

TEST_F(KSplitGramMatmulTest, PreallocatedOutput) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_test_tensor(64, 2048);
    uint32_t M = input.logical_shape()[-2];

    // Preallocate output tensor
    auto output_spec = ttnn::TensorSpec(
        ttnn::Shape({1, 1, M, M}),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    auto preallocated = create_device_tensor(output_spec, device);

    auto output =
        ttml::metal::gram_matmul(input, ttml::metal::OutputMode::UpperTriangle, MathFidelity::HiFi4, preallocated);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    // Verify it wrote to the preallocated tensor (same buffer address)
    EXPECT_EQ(output.buffer()->address(), preallocated.buffer()->address());

    // Verify correctness
    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t padded_out = output.logical_shape()[-1];

    auto ref = compute_gram_tile(in_vec, K, 2, 15);
    auto dev = extract_output_tile(out_vec, padded_out, 2, 15);
    double p = tile_pcc(ref, dev);
    std::cout << "Preallocated G[2,15] PCC=" << p << "\n";
    EXPECT_GT(p, 0.99);
}

TEST_F(KSplitGramMatmulTest, SmokeAllShapes) {
    struct Shape {
        uint32_t M_tiles, K_dim;
    };
    Shape shapes[] = {{10, 0}, {64, 2048}, {64, 5632}, {128, 4096}, {128, 11008}, {256, 8192}};
    for (auto& s : shapes) {
        auto input = make_test_tensor(s.M_tiles, s.K_dim);
        auto output = ttml::metal::gram_matmul(input);
        tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);
    }
    SUCCEED();
}

TEST_F(KSplitGramMatmulTest, StressTest8192x8192) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_test_tensor(256, 8192);
    constexpr int N = 5;
    for (int i = 0; i < N; i++) {
        auto out = ttml::metal::gram_matmul(input);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        out.deallocate();
        std::cout << "  dispatch " << (i + 1) << "/" << N << " OK\n" << std::flush;
    }
    SUCCEED();
}

// NOTE: 8192x8192 (subs>1) deadlocks on repeated dispatch.
// 8192x8192 uses subs=3 (per-msb reduction path).
TEST_F(KSplitGramMatmulTest, Benchmark) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto device_grid = device->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(device_grid.x, device_grid.y);

    struct Shape {
        uint32_t M_tiles, K_dim;
        const char* label;
    };
    Shape shapes[] = {
        {64, 2048, "2048x2048"},
        {64, 5632, "2048x5632"},
        {128, 4096, "4096x4096"},
        {128, 11008, "4096x11008"},
        {256, 8192, "8192x8192"},
    };

    constexpr int warmup = 2;
    constexpr int iters = 5;

    auto bench = [&](auto fn, const char* name) {
        for (int i = 0; i < warmup; i++) {
            fn();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            fn();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    };

    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();

    for (auto& s : shapes) {
        auto input = make_test_tensor(s.M_tiles, s.K_dim);
        auto input_t = ttnn::transpose(input, -2, -1);

        uint64_t M = s.M_tiles * 32;
        uint64_t K = s.K_dim;
        auto tflops = [&](double us) { return 2.0 * M * M * K / 1e12 / (us / 1e6); };

        double t_gram = bench(
            [&]() {
                auto out = ttml::metal::gram_matmul(input);
                out.deallocate();
            },
            "gram_matmul");

        double t_ttnn = bench(
            [&]() {
                auto out = ttnn::matmul(
                    input,
                    input_t,
                    false,
                    false,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    compute_kernel_config,
                    core_grid,
                    std::nullopt);
                out.deallocate();
            },
            "ttnn::matmul");

        std::cout << "\n  " << s.label << ":\n" << std::flush;
        std::cout << "    gram_matmul:    " << std::fixed << std::setprecision(1) << t_gram << " us  ("
                  << std::setprecision(2) << tflops(t_gram) << " TF)\n";
        std::cout << "    ttnn::matmul:   " << std::setprecision(1) << t_ttnn << " us  (" << std::setprecision(2)
                  << tflops(t_ttnn) << " TF)\n";
        std::cout << "    speedup:        " << std::setprecision(2) << t_ttnn / t_gram << "x\n" << std::flush;

        input.deallocate();
        input_t.deallocate();
    }
    SUCCEED();
}
