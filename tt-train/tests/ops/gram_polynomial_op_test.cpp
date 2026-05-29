// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

class GramPolynomialTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

// Tolerances for BF16 matmul comparison (same as gram_matmul tests)
// Diagonal tiles G[i,i] scale linearly with M, accumulating large absolute errors despite <1% relative error.
constexpr float kRtol = 1e-2f;
constexpr float kAtol = 15.0f;

ttnn::Tensor make_random_tensor(uint32_t M, uint32_t K = 0, uint32_t seed = 42) {
    auto* device = &ttml::autograd::ctx().get_device();
    K = (K > 0) ? K : M;
    std::vector<float> data(M * K);
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0f, 1.0f); }, seed);
    auto shape = ttnn::Shape({1, 1, M, K});
    return ttnn::Tensor::from_vector(
        data,
        ttnn::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

// Compute G² = G @ G for a single tile on CPU
std::vector<float> compute_g_squared_tile(
    const std::vector<float>& g_vec, uint32_t M, uint32_t tile_r, uint32_t tile_c) {
    std::vector<float> result(32 * 32, 0.0f);
    uint32_t M_tiles = M / 32;
    for (uint32_t k_tile = 0; k_tile < M_tiles; k_tile++) {
        for (uint32_t i = 0; i < 32; i++) {
            for (uint32_t j = 0; j < 32; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < 32; k++) {
                    sum +=
                        g_vec[(tile_r * 32 + i) * M + k_tile * 32 + k] * g_vec[(tile_c * 32 + j) * M + k_tile * 32 + k];
                }
                result[i * 32 + j] += sum;
            }
        }
    }
    return result;
}

std::vector<float> extract_output_tile(
    const std::vector<float>& out_vec, uint32_t out_width, uint32_t tile_r, uint32_t tile_c) {
    std::vector<float> tile(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) tile[i * 32 + j] = out_vec[(tile_r * 32 + i) * out_width + tile_c * 32 + j];
    return tile;
}

void check_tile(
    const std::vector<float>& ref, const std::vector<float>& dev, const char* label, float atol_scale = 1.0f) {
    auto ref_xt = xt::adapt(ref, {32u, 32u});
    auto dev_xt = xt::adapt(dev, {32u, 32u});
    EXPECT_TRUE(xt::allclose(ref_xt, dev_xt, kRtol, kAtol * atol_scale)) << label << " exceeded tolerance";
}

}  // namespace

// Phase 1: G² = G @ G (b=0, c=1)
TEST_F(GramPolynomialTest, GSquared_2048x2048) {
    auto G = make_random_tensor(2048);
    auto output = ttml::metal::gram_polynomial(G, 0.0f, 1.0f, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto g_vec = G.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t M = G.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    auto ref = compute_g_squared_tile(g_vec, M, 2, 5);
    auto dev = extract_output_tile(out_vec, W, 2, 5);
    check_tile(ref, dev, "G²[2,5] off-diagonal");

    ref = compute_g_squared_tile(g_vec, M, 0, 0);
    dev = extract_output_tile(out_vec, W, 0, 0);
    check_tile(ref, dev, "G²[0,0] diagonal");
}

// Phase 2: cG² (b=0, c=2.5)
TEST_F(GramPolynomialTest, ScaledGSquared_2048x2048) {
    auto G = make_random_tensor(2048);
    constexpr float c = 2.5f;
    auto output = ttml::metal::gram_polynomial(G, 0.0f, c, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto g_vec = G.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t M = G.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    // CPU reference: c * (G @ G)
    auto ref = compute_g_squared_tile(g_vec, M, 2, 5);
    for (auto& v : ref) v *= c;
    auto dev = extract_output_tile(out_vec, W, 2, 5);
    check_tile(ref, dev, "cG²[2,5] off-diagonal", c);

    ref = compute_g_squared_tile(g_vec, M, 0, 0);
    for (auto& v : ref) v *= c;
    dev = extract_output_tile(out_vec, W, 0, 0);
    check_tile(ref, dev, "cG²[0,0] diagonal", c);
}

// Phase 3: bG + cG² (Muon coefficients)
TEST_F(GramPolynomialTest, BgPlusCgSquared_2048x2048) {
    auto G = make_random_tensor(2048);
    constexpr float b = -4.775f;
    constexpr float c = 2.0315f;
    auto output = ttml::metal::gram_polynomial(G, b, c, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto g_vec = G.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t M = G.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    // CPU reference: bG + c*(G@G), tile [2,5]
    auto g_tile = extract_output_tile(g_vec, M, 2, 5);
    auto g2_tile = compute_g_squared_tile(g_vec, M, 2, 5);
    std::vector<float> ref(32 * 32);
    for (size_t i = 0; i < ref.size(); i++) ref[i] = b * g_tile[i] + c * g2_tile[i];
    auto dev = extract_output_tile(out_vec, W, 2, 5);
    float max_scale = std::max(std::abs(b), std::abs(c));
    check_tile(ref, dev, "bG+cG²[2,5] off-diagonal", max_scale);

    // Diagonal tile [0,0]
    g_tile = extract_output_tile(g_vec, M, 0, 0);
    g2_tile = compute_g_squared_tile(g_vec, M, 0, 0);
    for (size_t i = 0; i < ref.size(); i++) ref[i] = b * g_tile[i] + c * g2_tile[i];
    dev = extract_output_tile(out_vec, W, 0, 0);
    check_tile(ref, dev, "bG+cG²[0,0] diagonal", max_scale);
}

TEST_F(GramPolynomialTest, BgPlusCgSquared_4096x4096) {
    auto G = make_random_tensor(4096);
    constexpr float b = -4.775f;
    constexpr float c = 2.0315f;
    auto output = ttml::metal::gram_polynomial(G, b, c, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto g_vec = G.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t M = G.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    auto g_tile = extract_output_tile(g_vec, M, 2, 5);
    auto g2_tile = compute_g_squared_tile(g_vec, M, 2, 5);
    std::vector<float> ref(32 * 32);
    for (size_t i = 0; i < ref.size(); i++) ref[i] = b * g_tile[i] + c * g2_tile[i];
    auto dev = extract_output_tile(out_vec, W, 2, 5);
    float max_scale = std::max(std::abs(b), std::abs(c));
    check_tile(ref, dev, "4096 bG+cG²[2,5]", max_scale);
}

TEST_F(GramPolynomialTest, BgPlusCgSquared_8192x8192) {
    auto G = make_random_tensor(8192);
    constexpr float b = -4.775f;
    constexpr float c = 2.0315f;
    auto output = ttml::metal::gram_polynomial(G, b, c, ttml::metal::OutputMode::Full);
    tt::tt_metal::distributed::Synchronize(&ttml::autograd::ctx().get_device(), std::nullopt);

    auto g_vec = G.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t M = G.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    auto g_tile = extract_output_tile(g_vec, M, 2, 5);
    auto g2_tile = compute_g_squared_tile(g_vec, M, 2, 5);
    std::vector<float> ref(32 * 32);
    for (size_t i = 0; i < ref.size(); i++) ref[i] = b * g_tile[i] + c * g2_tile[i];
    auto dev = extract_output_tile(out_vec, W, 2, 5);
    float max_scale = std::max(std::abs(b), std::abs(c));
    check_tile(ref, dev, "8192 bG+cG²[2,5]", max_scale);
}

// Compare gram_polynomial error against Newton-Schulz composite (ttnn::matmul + elementwise)
TEST_F(GramPolynomialTest, DISABLED_ErrorComparison) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto compute_kernel_config = ttml::core::ComputeKernelConfig::matmul();
    auto device_grid = device->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(device_grid.x, device_grid.y);

    constexpr float b = -4.775f;
    constexpr float c = 2.0315f;

    struct Shape {
        uint32_t M;
        const char* label;
    };
    Shape shapes[] = {{2048, "2048"}};

    std::cout << "\n  Shape | Tile  | gram_poly abs | composite abs | gram_poly rel | composite rel\n";
    std::cout << "  ------+-------+---------------+---------------+---------------+--------------\n";

    for (auto& s : shapes) {
        auto G = make_random_tensor(s.M);

        // gram_polynomial: fused bG + cG²
        auto fused = ttml::metal::gram_polynomial(G, b, c, ttml::metal::OutputMode::Full);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);

        // Composite: ttnn::matmul(G,G) then elementwise bG + cG²
        auto G_t = ttnn::transpose(G, -2, -1);
        auto G2 = ttnn::matmul(
            G,
            G_t,
            false,
            false,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            compute_kernel_config,
            core_grid,
            std::nullopt);
        auto bG = ttnn::multiply(G, b);
        auto composite = ttnn::addalpha(bG, G2, c);
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);

        auto g_vec = G.to_vector<float>();
        auto fused_vec = fused.to_vector<float>();
        auto comp_vec = composite.to_vector<float>();
        uint32_t M = s.M;
        uint32_t W_f = fused.logical_shape()[-1];
        uint32_t W_c = composite.logical_shape()[-1];

        uint32_t tiles[][2] = {{0, 0}, {2, 5}};
        for (auto& t : tiles) {
            // CPU reference
            auto g_tile = extract_output_tile(g_vec, M, t[0], t[1]);
            auto g2_tile = compute_g_squared_tile(g_vec, M, t[0], t[1]);
            std::vector<float> ref(32 * 32);
            float max_abs_ref = 1.0f;
            for (size_t i = 0; i < ref.size(); i++) {
                ref[i] = b * g_tile[i] + c * g2_tile[i];
                max_abs_ref = std::max(max_abs_ref, std::abs(ref[i]));
            }

            auto f_tile = extract_output_tile(fused_vec, W_f, t[0], t[1]);
            auto c_tile = extract_output_tile(comp_vec, W_c, t[0], t[1]);
            float f_abs = 0, c_abs = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                f_abs = std::max(f_abs, std::abs(ref[i] - f_tile[i]));
                c_abs = std::max(c_abs, std::abs(ref[i] - c_tile[i]));
            }

            char line[128];
            std::snprintf(
                line,
                sizeof(line),
                "  %5s | [%d,%d] | %13.4f | %13.4f | %13.6f | %13.6f",
                s.label,
                t[0],
                t[1],
                f_abs,
                c_abs,
                f_abs / max_abs_ref,
                c_abs / max_abs_ref);
            std::cout << line << "\n";
        }

        G.deallocate();
        fused.deallocate();
        G_t.deallocate();
        G2.deallocate();
        bG.deallocate();
        composite.deallocate();
    }
    std::cout << std::flush;
    SUCCEED();
}
