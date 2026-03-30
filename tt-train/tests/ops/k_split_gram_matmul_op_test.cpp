// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <iostream>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "metal/operations.hpp"

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

ttnn::Tensor make_random_tensor(uint32_t M, uint32_t N, uint32_t seed = 42) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> data(M * N);
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);
    auto shape = ttnn::Shape({1, 1, M, N});
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

// Compute reference gram matmul tile G[m_tile, n_tile] on CPU
std::vector<float> compute_gram_tile(const std::vector<float>& in_vec, uint32_t K, uint32_t m_tile, uint32_t n_tile) {
    std::vector<float> result(32 * 32, 0.0f);
    uint32_t K_tiles = K / 32;
    for (uint32_t k_tile = 0; k_tile < K_tiles; k_tile++) {
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

// Extract a 32x32 tile from a flattened row-major matrix
std::vector<float> extract_output_tile(
    const std::vector<float>& out_vec, uint32_t out_width, uint32_t tile_r, uint32_t tile_c) {
    std::vector<float> tile(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) tile[i * 32 + j] = out_vec[(tile_r * 32 + i) * out_width + tile_c * 32 + j];
    return tile;
}

// Max relative error between two tile vectors: max(|ref-dev| / max(|ref|, 1))
float max_rel_error(const std::vector<float>& ref, const std::vector<float>& dev) {
    float max_abs_ref = 1.0f;
    for (size_t i = 0; i < ref.size(); i++) {
        max_abs_ref = std::max(max_abs_ref, std::abs(ref[i]));
    }
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); i++) {
        max_err = std::max(max_err, std::abs(ref[i] - dev[i]) / max_abs_ref);
    }
    return max_err;
}

// Check a tile against reference with relative tolerance
void check_tile(
    const std::vector<float>& in_vec,
    const std::vector<float>& out_vec,
    uint32_t K,
    uint32_t out_width,
    uint32_t tile_r,
    uint32_t tile_c,
    float rtol,
    const char* label) {
    auto ref = compute_gram_tile(in_vec, K, tile_r, tile_c);
    auto dev = extract_output_tile(out_vec, out_width, tile_r, tile_c);
    float err = max_rel_error(ref, dev);
    std::cout << label << " max_rel_error=" << err << "\n";
    EXPECT_LT(err, rtol) << label << " exceeded tolerance";
}

}  // namespace

TEST_F(KSplitGramMatmulTest, Verification4096x4096) {
    auto input = make_random_tensor(4096, 4096);
    auto output = ttml::metal::gram_matmul(input);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    constexpr float rtol = 0.01f;
    check_tile(in_vec, out_vec, K, W, 2, 15, rtol, "G[2,15] (upper)");
    check_tile(in_vec, out_vec, K, W, 0, 0, rtol, "G[0,0] (diag)");
}

TEST_F(KSplitGramMatmulTest, Verification4096x11008) {
    auto input = make_random_tensor(4096, 11008);
    auto output = ttml::metal::gram_matmul(input);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    constexpr float rtol = 0.01f;
    check_tile(in_vec, out_vec, K, W, 2, 15, rtol, "G[2,15]");
}

TEST_F(KSplitGramMatmulTest, VerificationMirror) {
    auto input = make_random_tensor(640, 640);
    auto output = ttml::metal::gram_matmul(input, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    constexpr float rtol = 0.01f;
    check_tile(in_vec, out_vec, K, W, 2, 4, rtol, "Upper G[2,4]");

    // Mirror: G[4,2] should equal G[2,4]^T
    auto ref_upper = compute_gram_tile(in_vec, K, 2, 4);
    auto dev_mirror = extract_output_tile(out_vec, W, 4, 2);
    std::vector<float> ref_mirror(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) ref_mirror[i * 32 + j] = ref_upper[j * 32 + i];
    float err = max_rel_error(ref_mirror, dev_mirror);
    std::cout << "Mirror G[4,2] max_rel_error=" << err << "\n";
    EXPECT_LT(err, rtol) << "Mirror exceeded tolerance";
}

TEST_F(KSplitGramMatmulTest, PreallocatedOutput) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_random_tensor(2048, 2048);
    uint32_t M = input.logical_shape()[-2];

    auto output_spec = ttnn::TensorSpec(
        ttnn::Shape({1, 1, M, M}),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    auto preallocated = create_device_tensor(output_spec, device);

    auto output =
        ttml::metal::gram_matmul(input, ttml::metal::OutputMode::UpperTriangle, MathFidelity::HiFi4, preallocated);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    EXPECT_EQ(output.buffer()->address(), preallocated.buffer()->address());

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    constexpr float rtol = 0.01f;
    check_tile(in_vec, out_vec, K, W, 2, 15, rtol, "Preallocated G[2,15]");
}

TEST_F(KSplitGramMatmulTest, SmokeAllShapes) {
    struct Shape {
        uint32_t M, K;
    };
    Shape shapes[] = {{320, 320}, {2048, 2048}, {2048, 5632}, {4096, 4096}, {4096, 11008}, {8192, 8192}};
    for (auto& s : shapes) {
        auto input = make_random_tensor(s.M, s.K);
        auto output = ttml::metal::gram_matmul(input);
        tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);
    }
    SUCCEED();
}

TEST_F(KSplitGramMatmulTest, StressTest8192x8192) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_random_tensor(8192, 8192);
    constexpr int N = 5;
    for (int i = 0; i < N; i++) {
        auto out = ttml::metal::gram_matmul(input);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        out.deallocate();
        std::cout << "  dispatch " << (i + 1) << "/" << N << " OK\n" << std::flush;
    }
    SUCCEED();
}
