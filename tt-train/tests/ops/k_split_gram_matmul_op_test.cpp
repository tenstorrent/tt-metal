// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"

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

// Diagonal tiles G[i,i] = sum(X[i,k]^2) scale linearly with K
// absolute error is inherently high on diagonal for this operation in BF16
constexpr float kRtol = 1e-2f;
constexpr float kAtol = 15.0f;

ttnn::Tensor make_random_tensor(uint32_t M, uint32_t N, uint32_t seed = 42) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto data = ttml::test_utils::make_uniform_vector<float>(M * N, -1.0f, 1.0f, seed);
    return ttml::core::from_vector(data, ttnn::Shape({1, 1, M, N}), device);
}

// Compute reference gram matmul tile G[m_tile, n_tile] on CPU
std::vector<float> compute_gram_tile(const std::vector<float>& in_vec, uint32_t K, uint32_t m_tile, uint32_t n_tile) {
    size_t M = in_vec.size() / K;
    xt::xarray<float> x = xt::adapt(in_vec, std::array<size_t, 2>{M, K});
    auto a = xt::view(x, xt::range(m_tile * 32, m_tile * 32 + 32), xt::all());
    auto b = xt::view(x, xt::range(n_tile * 32, n_tile * 32 + 32), xt::all());
    xt::xarray<float> c = xt::linalg::dot(a, xt::transpose(b));
    return std::vector<float>(c.begin(), c.end());
}

std::vector<float> extract_output_tile(
    const std::vector<float>& out_vec, uint32_t out_width, uint32_t tile_r, uint32_t tile_c) {
    std::vector<float> tile(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) tile[i * 32 + j] = out_vec[(tile_r * 32 + i) * out_width + tile_c * 32 + j];
    return tile;
}

void check_tile(
    const std::vector<float>& in_vec,
    const std::vector<float>& out_vec,
    uint32_t K,
    uint32_t out_width,
    uint32_t tile_r,
    uint32_t tile_c,
    float rtol,
    float atol,
    const char* label) {
    auto ref = compute_gram_tile(in_vec, K, tile_r, tile_c);
    auto dev = extract_output_tile(out_vec, out_width, tile_r, tile_c);
    auto ref_xt = xt::adapt(ref, {32u, 32u});
    auto dev_xt = xt::adapt(dev, {32u, 32u});
    EXPECT_TRUE(xt::allclose(ref_xt, dev_xt, rtol, atol)) << label << " exceeded tolerance";
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

    check_tile(in_vec, out_vec, K, W, 2, 15, kRtol, kAtol, "G[2,15] (upper)");
    check_tile(in_vec, out_vec, K, W, 0, 0, kRtol, kAtol, "G[0,0] (diag)");
}

TEST_F(KSplitGramMatmulTest, Verification4096x11008) {
    auto input = make_random_tensor(4096, 11008);
    auto output = ttml::metal::gram_matmul(input);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    check_tile(in_vec, out_vec, K, W, 2, 15, kRtol, kAtol, "G[2,15]");
}

TEST_F(KSplitGramMatmulTest, VerificationMirror) {
    auto input = make_random_tensor(640, 640);
    auto output = ttml::metal::gram_matmul(input, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    check_tile(in_vec, out_vec, K, W, 2, 4, kRtol, kAtol, "Upper G[2,4]");

    // Mirror: G[4,2] should equal G[2,4]^T
    auto ref_upper = compute_gram_tile(in_vec, K, 2, 4);
    auto dev_mirror = extract_output_tile(out_vec, W, 4, 2);
    std::vector<float> ref_mirror(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) ref_mirror[i * 32 + j] = ref_upper[j * 32 + i];
    auto ref_xt = xt::adapt(ref_mirror, {32u, 32u});
    auto dev_xt = xt::adapt(dev_mirror, {32u, 32u});
    EXPECT_TRUE(xt::allclose(ref_xt, dev_xt, kRtol, kAtol)) << "Mirror exceeded tolerance";
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

    check_tile(in_vec, out_vec, K, W, 2, 15, kRtol, kAtol, "Preallocated G[2,15]");
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

TEST_F(KSplitGramMatmulTest, NIGHTLY_StressTest8192x8192) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_random_tensor(8192, 8192);
    constexpr int N = 5;
    for (int i = 0; i < N; i++) {
        auto out = ttml::metal::gram_matmul(input);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        out.deallocate();
    }
    SUCCEED();
}
