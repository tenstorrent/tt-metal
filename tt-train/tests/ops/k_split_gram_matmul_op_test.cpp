// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
    void SetUp() override {
        if (ttml::autograd::ctx().get_device().arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "KSplitGramMatmul is only supported on Blackhole.";
        }
    }
};

namespace {

constexpr float kRtol = 1e-2f;
// Diagonal G[i,i] = Σ X[i,k]^2 grows ~linearly with K (e.g. ~1365 for K=4096),
// so BF16 absolute error is much larger on the diagonal than off-diagonal.
constexpr float kDiagAtol = 15.0f;
constexpr float kOffDiagAtol = 0.5f;

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

// Compare the whole device output against a CPU reference G = X @ X^T, tile by tile.
// In UpperTriangle mode only tiles with tile_r <= tile_c are guaranteed written
// (diagonal Mpc-blocks also contain written below-diagonal tiles, but their extent
// depends on the device grid, so this check stays grid-agnostic); Full mode checks
// every tile including the mirror.
void check_full_gram(
    const std::vector<float>& in_vec,
    const std::vector<float>& out_vec,
    uint32_t M,
    uint32_t K,
    bool full_output,
    const char* label) {
    xt::xarray<float> x = xt::adapt(in_vec, std::array<size_t, 2>{M, K});
    xt::xarray<float> ref = xt::linalg::dot(x, xt::transpose(x));
    xt::xarray<float> dev = xt::adapt(out_vec, std::array<size_t, 2>{M, M});

    const uint32_t M_tiles = M / 32;
    uint32_t bad_tiles = 0;
    uint32_t first_bad_r = 0;
    uint32_t first_bad_c = 0;
    for (uint32_t tr = 0; tr < M_tiles; tr++) {
        for (uint32_t tc = 0; tc < M_tiles; tc++) {
            if (!full_output && tr > tc) {
                continue;
            }
            const float atol = (tr == tc) ? kDiagAtol : kOffDiagAtol;
            auto ref_t = xt::view(ref, xt::range(tr * 32, tr * 32 + 32), xt::range(tc * 32, tc * 32 + 32));
            auto dev_t = xt::view(dev, xt::range(tr * 32, tr * 32 + 32), xt::range(tc * 32, tc * 32 + 32));
            if (!xt::allclose(ref_t, dev_t, kRtol, atol)) {
                if (bad_tiles == 0) {
                    first_bad_r = tr;
                    first_bad_c = tc;
                }
                bad_tiles++;
            }
        }
    }
    EXPECT_EQ(bad_tiles, 0u) << label << ": " << bad_tiles << " tiles exceeded tolerance, first at tile ("
                             << first_bad_r << ", " << first_bad_c << ")";
}

}  // namespace

struct VerifyCase {
    uint32_t M;
    uint32_t K;
    uint32_t tile_r;
    uint32_t tile_c;
    const char* name;
};

class KSplitGramMatmulVerifyTest : public ::testing::TestWithParam<VerifyCase> {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
    void SetUp() override {
        if (ttml::autograd::ctx().get_device().arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "KSplitGramMatmul is only supported on Blackhole.";
        }
    }
};

TEST_P(KSplitGramMatmulVerifyTest, Tile) {
    const auto& c = GetParam();
    auto input = make_random_tensor(c.M, c.K);
    auto output = ttml::metal::gram_matmul(input);
    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t W = output.logical_shape()[-1];
    float atol = (c.tile_r == c.tile_c) ? kDiagAtol : kOffDiagAtol;
    check_tile(in_vec, out_vec, c.K, W, c.tile_r, c.tile_c, kRtol, atol, c.name);
}

static std::string CaseName(const ::testing::TestParamInfo<VerifyCase>& info) {
    return info.param.name;
}

// K must be multiple of 64 (op requires logical K_tiles even).
// tile_r/tile_c chosen to land fully inside logical M (no partial tiles).
static const VerifyCase kVerifyCases[] = {
    {2048, 2048, 2, 15, "Square2048"},
    {2048, 5632, 10, 40, "WideK_2048x5632"},
    {4096, 4096, 0, 0, "Square4096_Diag"},
    {4096, 4096, 2, 15, "Square4096_OffDiag"},
    {4096, 11008, 2, 15, "WideK_4096x11008"},
    {333, 384, 5, 5, "NonAligned_M333"},
    {2049, 2048, 30, 30, "NonAligned_M2049"},
    {8192, 28672, 100, 200, "Llama70B_8192x28672"},
};

INSTANTIATE_TEST_SUITE_P(AllShapes, KSplitGramMatmulVerifyTest, ::testing::ValuesIn(kVerifyCases), CaseName);

TEST_F(KSplitGramMatmulTest, VerificationMirror) {
    auto input = make_random_tensor(640, 640);
    auto output = ttml::metal::gram_matmul(input, ttml::metal::OutputMode::Full);

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    check_tile(in_vec, out_vec, K, W, 2, 4, kRtol, kOffDiagAtol, "Upper G[2,4]");

    // Mirror: G[4,2] should equal G[2,4]^T
    auto ref_upper = compute_gram_tile(in_vec, K, 2, 4);
    auto dev_mirror = extract_output_tile(out_vec, W, 4, 2);
    std::vector<float> ref_mirror(32 * 32);
    for (uint32_t i = 0; i < 32; i++)
        for (uint32_t j = 0; j < 32; j++) ref_mirror[i * 32 + j] = ref_upper[j * 32 + i];
    auto ref_xt = xt::adapt(ref_mirror, {32u, 32u});
    auto dev_xt = xt::adapt(dev_mirror, {32u, 32u});
    EXPECT_TRUE(xt::allclose(ref_xt, dev_xt, kRtol, kOffDiagAtol)) << "Mirror exceeded tolerance";
}

TEST_F(KSplitGramMatmulTest, PreallocatedOutput) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_random_tensor(2048, 2048);
    uint32_t M = input.logical_shape()[-2];

    auto output_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, M, M}),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    auto preallocated = ttnn::create_device_tensor(output_spec, device);

    auto output =
        ttml::metal::gram_matmul(input, ttml::metal::OutputMode::UpperTriangle, MathFidelity::HiFi4, preallocated);

    EXPECT_EQ(output.buffer()->address(), preallocated.buffer()->address());

    auto in_vec = input.to_vector<float>();
    auto out_vec = output.to_vector<float>();
    uint32_t K = input.logical_shape()[-1];
    uint32_t W = output.logical_shape()[-1];

    check_tile(in_vec, out_vec, K, W, 2, 15, kRtol, kOffDiagAtol, "Preallocated G[2,15]");
}

// Full-matrix verification of the cross-core reduction protocol in both output modes.
// The shapes are chosen for a 10-wide compute grid:
//   * M=8192 → Mpc=26 with M_block 13-14 → num_m_blocks=2: exercises the sub-block
//     ordering between reduce senders and receivers (every off-diagonal sub-block
//     pairing is order-sensitive).
//   * M=5440 → Mpc=17 (prime): any M_block < 17 leaves a partial edge block,
//     exercising the fixed-capacity CB protocol for partial blocks.
// These must pass on device before this op is wired into Muon / Newton-Schulz.
namespace {
void run_full_matrix_case(uint32_t M, uint32_t K, const char* label) {
    auto input = make_random_tensor(M, K);
    auto in_vec = input.to_vector<float>();
    {
        auto output = ttml::metal::gram_matmul(input);
        auto out_vec = output.to_vector<float>();
        check_full_gram(in_vec, out_vec, M, K, /*full_output=*/false, label);
    }
    {
        auto output = ttml::metal::gram_matmul(input, ttml::metal::OutputMode::Full);
        auto out_vec = output.to_vector<float>();
        check_full_gram(in_vec, out_vec, M, K, /*full_output=*/true, label);
    }
}
}  // namespace

TEST_F(KSplitGramMatmulTest, FullMatrixMultiBlock) {
    run_full_matrix_case(8192, 512, "FullMatrixMultiBlock");
}

TEST_F(KSplitGramMatmulTest, FullMatrixPartialEdgeBlock) {
    run_full_matrix_case(5440, 512, "FullMatrixPartialEdgeBlock");
}

// The kernels reduce whole K tiles and make no assumption about tile-padding contents
// (upstream ops may leave garbage there), so the op must reject non-tile-aligned logical K
// instead of silently accumulating out-of-shape columns. K_tiles must also be even.
TEST_F(KSplitGramMatmulTest, RejectsUnsupportedK) {
    // K=33 pads to 2 tiles (even), but logical K is not tile-aligned.
    EXPECT_THROW(ttml::metal::gram_matmul(make_random_tensor(320, 33)), std::exception);
    // K=96 is tile-aligned but K_tiles=3 is odd.
    EXPECT_THROW(ttml::metal::gram_matmul(make_random_tensor(320, 96)), std::exception);
}

TEST_F(KSplitGramMatmulTest, SmokeAllShapes) {
    struct Shape {
        uint32_t M, K;
    };
    Shape shapes[] = {{320, 320}, {2048, 2048}, {2048, 5632}, {4096, 4096}, {4096, 11008}, {8192, 8192}};
    for (auto& s : shapes) {
        auto input = make_random_tensor(s.M, s.K);
        auto output = ttml::metal::gram_matmul(input);
    }
    SUCCEED();
}

TEST_F(KSplitGramMatmulTest, NIGHTLY_StressTest8192x8192) {
    auto input = make_random_tensor(8192, 8192);
    constexpr int N = 5;
    for (int i = 0; i < N; i++) {
        auto out = ttml::metal::gram_matmul(input);
        out.deallocate();
    }
    SUCCEED();
}
