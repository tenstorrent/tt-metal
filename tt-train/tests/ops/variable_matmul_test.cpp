// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace {

using ttml::metal::VariableMatmulConfig;

float max_abs_error(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    auto a_vec = ttml::core::to_vector<float>(a);
    auto b_vec = ttml::core::to_vector<float>(b);

    float max_err = 0.0F;
    for (size_t i = 0; i < a_vec.size(); ++i) {
        max_err = std::max(max_err, std::abs(a_vec[i] - b_vec[i]));
    }
    return max_err;
}

ttnn::Tensor create_random_device_tensor(uint32_t M, uint32_t K, ttnn::distributed::MeshDevice* device) {
    xt::xarray<float> xt = xt::random::randn<float>({1U, 1U, M, K});
    return ttml::core::from_xtensor(xt, device);
}

const VariableMatmulConfig kConfig{
    .M_block_size = 2,
    .K_block_size = 4,
    .N_block_size = 4,
    .subblock_h = 2,
    .subblock_w = 2,
    .compute_with_storage_grid_size = {10, 10},
};

}  // namespace

class VariableMatmulTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(VariableMatmulTest, Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto result = ttml::metal::variable_matmul(input, weight, kConfig);
    auto ref = ttnn::matmul(input, weight, false, false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "Max absolute error: " << err;
}

TEST_F(VariableMatmulTest, VariableM_SmallM_32x128x512) {
    const uint32_t M = 32, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto result = ttml::metal::variable_matmul(input, weight, kConfig);
    auto ref = ttnn::matmul(input, weight, false, false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "Max absolute error: " << err;
}

// Multiple M values — simulates MoE dispatch. At most 2 compilations (transpose variants).
TEST_F(VariableMatmulTest, VariableM_MultipleShapes) {
    const uint32_t K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto weight = create_random_device_tensor(K, N, device);

    for (uint32_t M : {32U, 64U, 128U, 256U, 512U}) {
        auto input = create_random_device_tensor(M, K, device);

        auto result = ttml::metal::variable_matmul(input, weight, kConfig);
        auto ref = ttnn::matmul(input, weight, false, false);

        float err = max_abs_error(result, ref);
        EXPECT_LT(err, 2.0F) << "M=" << M << " max_abs_error: " << err;
    }
}

// transpose_b: weight is stored as [N, K] but interpreted as [K, N] for the matmul.
TEST_F(VariableMatmulTest, TransposeB_Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    auto cfg = kConfig;
    cfg.transpose_b = true;
    auto result = ttml::metal::variable_matmul(input, weight_nk, cfg);
    auto ref = ttnn::matmul(input, weight_nk, /*transpose_a=*/false, /*transpose_b=*/true);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_b max_abs_error: " << err;
}

TEST_F(VariableMatmulTest, TransposeB_VariableM) {
    const uint32_t K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    auto cfg = kConfig;
    cfg.transpose_b = true;

    for (uint32_t M : {32U, 64U, 128U, 256U, 512U}) {
        auto input = create_random_device_tensor(M, K, device);

        auto result = ttml::metal::variable_matmul(input, weight_nk, cfg);
        auto ref = ttnn::matmul(input, weight_nk, /*transpose_a=*/false, /*transpose_b=*/true);

        float err = max_abs_error(result, ref);
        EXPECT_LT(err, 2.0F) << "transpose_b M=" << M << " max_abs_error: " << err;
    }
}

// transpose_a: input is stored as [K, M] but interpreted as [M, K] for the matmul.
TEST_F(VariableMatmulTest, TransposeA_Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);  // stored [K, M]
    auto weight = create_random_device_tensor(K, N, device);

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_a max_abs_error: " << err;
}

// Config B (used by moe-ffn) + Mixtral dW_down backward shape.
// activated^T @ dY: [I, M] @ [M, H] = [I, H].
// matmul-M=I=14336, matmul-K=M=1024, matmul-N=H=4096.
// With transpose_a, input stored as [K=1024, M_for_matmul=14336].
TEST_F(VariableMatmulTest, ConfigB_TransposeA_MixtralDWDown) {
    const ttml::metal::VariableMatmulConfig cfg_b{
        .M_block_size = 4,
        .K_block_size = 8,
        .N_block_size = 8,
        .subblock_h = 2,
        .subblock_w = 2,
        .compute_with_storage_grid_size = {10, 10},
        .transpose_a = true,
    };
    const uint32_t M = 14336, K = 1024, N = 4096;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto result = ttml::metal::variable_matmul(input_km, weight, cfg_b);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 4.0F) << "ConfigB transpose_a Mixtral dW_down max_abs_error: " << err;
}

// h4096_i512 skewed dW_gate (transpose_a) — the shape that regressed in moe-ffn benchmark.
// Hot-expert: A stored [M_e=13120, H=4096], used as [H, M_e]. matmul-M=H=4096=128 tiles,
// matmul-K=M_e=410 tiles, matmul-N=I=512=16 tiles.
TEST_F(VariableMatmulTest, ConfigB_TransposeA_H4096I512_DWGate_Hot) {
    const ttml::metal::VariableMatmulConfig cfg_b{
        .M_block_size = 4,
        .K_block_size = 8,
        .N_block_size = 8,
        .subblock_h = 2,
        .subblock_w = 2,
        .compute_with_storage_grid_size = {10, 10},
        .transpose_a = true,
    };
    const uint32_t M_matmul = 4096, K_matmul = 13120, N_matmul = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K_matmul, M_matmul, device);
    auto weight = create_random_device_tensor(K_matmul, N_matmul, device);

    auto result = ttml::metal::variable_matmul(input_km, weight, cfg_b);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 64.0F) << "h4096_i512 dW_gate hot max_abs_error: " << err;
}

// Bisect: K_blocks > 1, N_blocks per core = 1. Tests the multi-K-iter path without
// the reuse path. (matmul-K=16 tiles / K_block=4 = 4 K_blocks; N_tiles_per_core=1.)
TEST_F(VariableMatmulTest, TransposeA_MultiKBlocks_NoNReuse) {
    const uint32_t M = 512, K = 512, N = 128;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_a multi-K no-N-reuse max_abs_error: " << err;
}

// Bisect: N_blocks per core > 1 (forces the reuse path) on small shape.
TEST_F(VariableMatmulTest, TransposeA_NReusePath) {
    const uint32_t M = 128, K = 128, N = 2048;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_a N-reuse-path max_abs_error: " << err;
}

// Small shape with M > N to flip transpose_core_grid to TRUE — same combination
// as the failing Mixtral case, but small enough to debug quickly.
TEST_F(VariableMatmulTest, TransposeA_MGreaterThanN_512x128x128) {
    const uint32_t M = 512, K = 128, N = 128;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_a M>N (512x128x128) max_abs_error: " << err;
}

// Same shape, but with the smaller default config that the existing tests use.
TEST_F(VariableMatmulTest, DefaultConfig_TransposeA_MixtralDWDown) {
    const uint32_t M = 14336, K = 1024, N = 4096;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto ref = ttnn::matmul(input_km, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 8.0F) << "DefaultConfig transpose_a Mixtral dW_down max_abs_error: " << err;
}

// Offset-read: treat input as a parent buffer, process only [offset, offset+effective_M).
// Reference is computed by slicing the parent first, then doing plain matmul.
TEST_F(VariableMatmulTest, OffsetRead_NoTranspose) {
    const uint32_t T_cap = 512, K = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(T_cap, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    // Take rows [128, 256) — 128 tile-rows, offset 128/32 = 4 tiles into the parent.
    constexpr uint32_t row_lo = 128;
    constexpr uint32_t row_hi = 256;
    constexpr uint32_t offset_tiles = row_lo / 32;
    constexpr uint32_t effective_M_tiles = (row_hi - row_lo) / 32;

    auto result = ttml::metal::variable_matmul(parent, weight, kConfig, std::nullopt, offset_tiles, effective_M_tiles);

    // Reference: slice the parent first, then plain matmul.
    auto sliced = ttnn::slice(
        parent,
        ttsl::SmallVector<uint32_t>{0, 0, row_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, row_hi, K},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight, false, false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "offset-read no-transpose max_abs_error: " << err;
}

// Multiple offsets into the same parent — simulates moe-ffn fwd dispatch.
TEST_F(VariableMatmulTest, OffsetRead_MultipleOffsets_SameParent) {
    const uint32_t T_cap = 512, K = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(T_cap, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    struct Range {
        uint32_t lo, hi;
    };
    const std::vector<Range> ranges = {{0, 64}, {64, 192}, {192, 256}, {256, 384}, {384, 512}};

    for (const auto& r : ranges) {
        const uint32_t offset_tiles = r.lo / 32;
        const uint32_t effective_M_tiles = (r.hi - r.lo) / 32;

        auto result =
            ttml::metal::variable_matmul(parent, weight, kConfig, std::nullopt, offset_tiles, effective_M_tiles);
        auto sliced = ttnn::slice(
            parent,
            ttsl::SmallVector<uint32_t>{0, 0, r.lo, 0},
            ttsl::SmallVector<uint32_t>{1, 1, r.hi, K},
            ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
        auto ref = ttnn::matmul(sliced, weight, false, false);

        float err = max_abs_error(result, ref);
        EXPECT_LT(err, 2.0F) << "offset M=[" << r.lo << "," << r.hi << ") err: " << err;
    }
}

// Offset-read with transpose_a: input stored as [K, M_parent]; offset is on the
// matmul-M axis (= input's stored col axis).
TEST_F(VariableMatmulTest, OffsetRead_TransposeA) {
    const uint32_t K = 128, M_parent = 512, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent_km = create_random_device_tensor(K, M_parent, device);  // stored [K, M_parent]
    auto weight = create_random_device_tensor(K, N, device);

    // Take M-cols [128, 320) of the stored tensor — offset 4 tiles, length 6 tiles.
    constexpr uint32_t m_lo = 128;
    constexpr uint32_t m_hi = 320;
    constexpr uint32_t offset_tiles = m_lo / 32;
    constexpr uint32_t effective_M_tiles = (m_hi - m_lo) / 32;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(parent_km, weight, cfg, std::nullopt, offset_tiles, effective_M_tiles);

    // Reference: slice columns of the [K, M_parent] tensor.
    auto sliced = ttnn::slice(
        parent_km,
        ttsl::SmallVector<uint32_t>{0, 0, 0, m_lo},
        ttsl::SmallVector<uint32_t>{1, 1, K, m_hi},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "offset-read transpose_a max_abs_error: " << err;
}

// Two consecutive offset-reads with transpose_b — gate_proj then up_proj on same parent.
TEST_F(VariableMatmulTest, OffsetRead_TransposeB_TwoCalls) {
    const uint32_t T_cap = 96, K = 64, N = 128;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(T_cap, K, device);
    auto w_gate = create_random_device_tensor(N, K, device);
    auto w_up = create_random_device_tensor(N, K, device);

    const VariableMatmulConfig cfg_moe{
        .M_block_size = 4,
        .K_block_size = 8,
        .N_block_size = 8,
        .subblock_h = 2,
        .subblock_w = 2,
        .compute_with_storage_grid_size = {10, 10},
        .transpose_a = false,
        .transpose_b = true,
    };

    constexpr uint32_t offset_tiles = 0;
    constexpr uint32_t effective_M_tiles = 2;
    auto gate = ttml::metal::variable_matmul(parent, w_gate, cfg_moe, std::nullopt, offset_tiles, effective_M_tiles);
    auto up = ttml::metal::variable_matmul(parent, w_up, cfg_moe, std::nullopt, offset_tiles, effective_M_tiles);

    auto sliced = ttnn::slice(
        parent,
        ttsl::SmallVector<uint32_t>{0, 0, 0, 0},
        ttsl::SmallVector<uint32_t>{1, 1, effective_M_tiles * 32, K},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref_gate = ttnn::matmul(sliced, w_gate, false, true);
    auto ref_up = ttnn::matmul(sliced, w_up, false, true);

    EXPECT_LT(max_abs_error(gate, ref_gate), 2.0F) << "gate two-call err";
    EXPECT_LT(max_abs_error(up, ref_up), 2.0F) << "up two-call err";
}

// Mimics moe-ffn fwd: large M_block relative to per-call M, transpose_b on weight,
// offset-read on input. Catches small-M-with-large-M_block bugs in this combination.
TEST_F(VariableMatmulTest, OffsetRead_TransposeB_SmallM_MoeFfnConfig) {
    const uint32_t T_cap = 96, K = 64, N = 128;  // matches moe_ffn Small_E2_H64_I128
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(T_cap, K, device);
    auto weight_nk = create_random_device_tensor(N, K, device);

    constexpr uint32_t row_lo = 64;
    constexpr uint32_t row_hi = 96;
    constexpr uint32_t offset_tiles = row_lo / 32;
    constexpr uint32_t effective_M_tiles = (row_hi - row_lo) / 32;

    const VariableMatmulConfig cfg_moe{
        .M_block_size = 4,
        .K_block_size = 8,
        .N_block_size = 8,
        .subblock_h = 2,
        .subblock_w = 2,
        .compute_with_storage_grid_size = {10, 10},
        .transpose_a = false,
        .transpose_b = true,
    };
    auto result =
        ttml::metal::variable_matmul(parent, weight_nk, cfg_moe, std::nullopt, offset_tiles, effective_M_tiles);

    auto sliced = ttnn::slice(
        parent,
        ttsl::SmallVector<uint32_t>{0, 0, row_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, row_hi, K},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight_nk, /*transpose_a=*/false, /*transpose_b=*/true);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "moe_ffn-config offset-read transpose_b err: " << err;
}

// Offset-read combined with transpose_b — the moe-ffn fwd pattern with [I, H] weights.
TEST_F(VariableMatmulTest, OffsetRead_TransposeB) {
    const uint32_t T_cap = 512, K = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(T_cap, K, device);
    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    constexpr uint32_t row_lo = 128;
    constexpr uint32_t row_hi = 256;
    constexpr uint32_t offset_tiles = row_lo / 32;
    constexpr uint32_t effective_M_tiles = (row_hi - row_lo) / 32;

    auto cfg = kConfig;
    cfg.transpose_b = true;
    auto result = ttml::metal::variable_matmul(parent, weight_nk, cfg, std::nullopt, offset_tiles, effective_M_tiles);

    auto sliced = ttnn::slice(
        parent,
        ttsl::SmallVector<uint32_t>{0, 0, row_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, row_hi, K},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight_nk, /*transpose_a=*/false, /*transpose_b=*/true);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "offset-read transpose_b max_abs_error: " << err;
}

// K-axis offset on in0: parent has larger K than weight; we slice [k_offset, k_offset+K_w).
TEST_F(VariableMatmulTest, KOffsetRead_NoTranspose) {
    const uint32_t M = 128, K_parent = 512, K_w = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(M, K_parent, device);
    auto weight = create_random_device_tensor(K_w, N, device);

    constexpr uint32_t k_lo = 128;
    constexpr uint32_t k_hi = 256;  // 128 cols = K_w
    constexpr uint32_t k_offset_tiles = k_lo / 32;

    auto result = ttml::metal::variable_matmul(
        parent,
        weight,
        kConfig,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/k_offset_tiles);

    auto sliced = ttnn::slice(
        parent,
        ttsl::SmallVector<uint32_t>{0, 0, 0, k_lo},
        ttsl::SmallVector<uint32_t>{1, 1, M, k_hi},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight, false, false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "k-offset no-transpose max_abs_error: " << err;
}

// Multiple K-offsets into the same parent (moe-ffn bw pattern).
TEST_F(VariableMatmulTest, KOffsetRead_MultipleOffsets_SameParent) {
    const uint32_t M = 128, K_parent = 512, K_w = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(M, K_parent, device);
    auto weight = create_random_device_tensor(K_w, N, device);

    const std::vector<uint32_t> k_los = {0, 128, 256, 384};
    for (uint32_t k_lo : k_los) {
        const uint32_t k_hi = k_lo + K_w;
        const uint32_t k_offset_tiles = k_lo / 32;

        auto result = ttml::metal::variable_matmul(
            parent,
            weight,
            kConfig,
            std::nullopt,
            /*in0_row_offset_tiles=*/0,
            /*effective_M_tiles=*/0,
            /*in0_k_offset_tiles=*/k_offset_tiles);
        auto sliced = ttnn::slice(
            parent,
            ttsl::SmallVector<uint32_t>{0, 0, 0, k_lo},
            ttsl::SmallVector<uint32_t>{1, 1, M, k_hi},
            ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
        auto ref = ttnn::matmul(sliced, weight, false, false);

        float err = max_abs_error(result, ref);
        EXPECT_LT(err, 2.0F) << "k-offset K=[" << k_lo << "," << k_hi << ") err: " << err;
    }
}

// K-offset with transpose_a: parent stored as [K_parent, M]; offset on stored row axis.
TEST_F(VariableMatmulTest, KOffsetRead_TransposeA) {
    const uint32_t K_parent = 512, K_w = 128, M = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent_km = create_random_device_tensor(K_parent, M, device);  // stored [K_parent, M]
    auto weight = create_random_device_tensor(K_w, N, device);

    constexpr uint32_t k_lo = 128;
    constexpr uint32_t k_hi = 256;
    constexpr uint32_t k_offset_tiles = k_lo / 32;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(
        parent_km,
        weight,
        cfg,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/k_offset_tiles);

    auto sliced = ttnn::slice(
        parent_km,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_hi, M},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "k-offset transpose_a max_abs_error: " << err;
}

// Combined M+K offset, transpose_a (moe-ffn dW_gate/dW_up pattern).
TEST_F(VariableMatmulTest, KOffsetRead_TransposeA_WithMOffset) {
    const uint32_t K_parent = 512, M_parent = 256, K_w = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent_km = create_random_device_tensor(K_parent, M_parent, device);
    auto weight = create_random_device_tensor(K_w, N, device);

    constexpr uint32_t k_lo = 64;
    constexpr uint32_t k_hi = k_lo + 128;  // K_w
    constexpr uint32_t m_lo = 96;
    constexpr uint32_t m_hi = 224;  // 128 M
    constexpr uint32_t k_offset_tiles = k_lo / 32;
    constexpr uint32_t m_offset_tiles = m_lo / 32;
    constexpr uint32_t effective_M_tiles = (m_hi - m_lo) / 32;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(
        parent_km,
        weight,
        cfg,
        std::nullopt,
        /*in0_row_offset_tiles=*/m_offset_tiles,
        /*effective_M_tiles=*/effective_M_tiles,
        /*in0_k_offset_tiles=*/k_offset_tiles);

    auto sliced = ttnn::slice(
        parent_km,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, m_lo},
        ttsl::SmallVector<uint32_t>{1, 1, k_hi, m_hi},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "k+m offset transpose_a max_abs_error: " << err;
}

// in1 K-axis offset: weight is the parent buffer; we slice its K rows.
TEST_F(VariableMatmulTest, In1KOffsetRead_NoTranspose) {
    const uint32_t M = 128, K_w_parent = 512, K_matmul = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K_matmul, device);
    auto weight_parent = create_random_device_tensor(K_w_parent, N, device);

    constexpr uint32_t k_lo = 128;
    constexpr uint32_t k_hi = 256;
    constexpr uint32_t k_offset_tiles = k_lo / 32;

    auto result = ttml::metal::variable_matmul(
        input,
        weight_parent,
        kConfig,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/k_offset_tiles);

    auto sliced = ttnn::slice(
        weight_parent,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_hi, N},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(input, sliced, false, false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "in1 k-offset no-transpose max_abs_error: " << err;
}

TEST_F(VariableMatmulTest, In1KOffsetRead_MultipleOffsets_SameParent) {
    const uint32_t M = 128, K_w_parent = 512, K_matmul = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K_matmul, device);
    auto weight_parent = create_random_device_tensor(K_w_parent, N, device);

    const std::vector<uint32_t> k_los = {0, 128, 256, 384};
    for (uint32_t k_lo : k_los) {
        const uint32_t k_hi = k_lo + K_matmul;
        const uint32_t k_offset_tiles = k_lo / 32;

        auto result =
            ttml::metal::variable_matmul(input, weight_parent, kConfig, std::nullopt, 0, 0, 0, k_offset_tiles);
        auto sliced = ttnn::slice(
            weight_parent,
            ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
            ttsl::SmallVector<uint32_t>{1, 1, k_hi, N},
            ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
        auto ref = ttnn::matmul(input, sliced, false, false);
        float err = max_abs_error(result, ref);
        EXPECT_LT(err, 2.0F) << "in1 k-offset K=[" << k_lo << "," << k_hi << ") err: " << err;
    }
}

// in1 K-offset with transpose_b: weight stored as [N, K_parent]; slice on K axis.
TEST_F(VariableMatmulTest, In1KOffsetRead_TransposeB) {
    const uint32_t M = 128, K_w_parent = 512, K_matmul = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K_matmul, device);
    auto weight_nk = create_random_device_tensor(N, K_w_parent, device);  // stored [N, K_parent]

    constexpr uint32_t k_lo = 128;
    constexpr uint32_t k_hi = 256;
    constexpr uint32_t k_offset_tiles = k_lo / 32;

    auto cfg = kConfig;
    cfg.transpose_b = true;
    auto result = ttml::metal::variable_matmul(input, weight_nk, cfg, std::nullopt, 0, 0, 0, k_offset_tiles);

    auto sliced = ttnn::slice(
        weight_nk,
        ttsl::SmallVector<uint32_t>{0, 0, 0, k_lo},
        ttsl::SmallVector<uint32_t>{1, 1, N, k_hi},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(input, sliced, /*transpose_a=*/false, /*transpose_b=*/true);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "in1 k-offset transpose_b max_abs_error: " << err;
}

// in1 K-offset paired with transpose_a on in0 (moe_ffn dW_down pattern).
// activated_e^T @ dY_e where dY is the parent (K-axis sliced).
TEST_F(VariableMatmulTest, In1KOffsetRead_TransposeA_DWDownPattern) {
    const uint32_t I = 256, M_e = 128, T_parent = 512, H = 128;
    auto* device = &ttml::autograd::ctx().get_device();

    // activated_e stored [M_e, I] used with transpose_a → matmul-K = M_e, matmul-M = I.
    auto activated = create_random_device_tensor(M_e, I, device);
    auto dY_parent = create_random_device_tensor(T_parent, H, device);  // parent: [T, H]

    constexpr uint32_t row_lo = 128;
    constexpr uint32_t row_hi = row_lo + 128;  // = M_e
    constexpr uint32_t k_offset_tiles = row_lo / 32;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(activated, dY_parent, cfg, std::nullopt, 0, 0, 0, k_offset_tiles);

    auto dY_sliced = ttnn::slice(
        dY_parent,
        ttsl::SmallVector<uint32_t>{0, 0, row_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, row_hi, H},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(activated, dY_sliced, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 4.0F) << "in1 k-offset dW_down pattern max_abs_error: " << err;
}

// Write-at-offset combined with transpose_b — exactly what moe-ffn fwd down_proj does.
TEST_F(VariableMatmulTest, WriteAtOffset_TransposeB) {
    const uint32_t M_e = 128, K = 128, N = 256, M_parent = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_e, K, device);
    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]
    auto parent_out = create_random_device_tensor(M_parent, N, device);

    constexpr uint32_t out_row_lo = 128;
    constexpr uint32_t out_row_hi = 256;
    constexpr uint32_t out_offset_tiles = out_row_lo / 32;

    auto cfg = kConfig;
    cfg.transpose_b = true;
    ttml::metal::variable_matmul(
        input,
        weight_nk,
        cfg,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/parent_out,
        /*out_row_offset_tiles=*/out_offset_tiles);

    auto ref_chunk = ttnn::matmul(input, weight_nk, /*transpose_a=*/false, /*transpose_b=*/true);
    auto ref_chunk_vec = ttml::core::to_vector<float>(ref_chunk);
    auto written_vec = ttml::core::to_vector<float>(parent_out);
    float written_err = 0.0F;
    for (uint32_t m = out_row_lo; m < out_row_hi; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            uint32_t idx_parent = m * N + n;
            uint32_t idx_chunk = (m - out_row_lo) * N + n;
            written_err = std::max(written_err, std::abs(written_vec[idx_parent] - ref_chunk_vec[idx_chunk]));
        }
    }
    EXPECT_LT(written_err, 2.0F) << "write-at-offset transpose_b err: " << written_err;
}

// Write-at-offset: caller provides parent output; matmul writes into a row-range.
// EP path: kernel reads the write-at-offset row from a device-side offsets tensor at
// runtime instead of a host-supplied scalar. Shape (M=128 > N=64) ensures
// transpose_core_grid=true so dm_in1_sender_out is the output writer.
TEST_F(VariableMatmulTest, OnDeviceOffsets_OutputRow) {
    const uint32_t M_e = 128, K = 128, N = 64, M_parent = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_e, K, device);
    auto weight = create_random_device_tensor(K, N, device);
    auto parent_out = create_random_device_tensor(M_parent, N, device);
    auto parent_orig_vec = ttml::core::to_vector<float>(parent_out);

    // Offsets [0, 128, 256, 384] (in rows). start_index=1 -> row 128 -> tile 4.
    std::vector<uint32_t> offsets_host = {0U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    ttml::metal::variable_matmul(
        input,
        weight,
        kConfig,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/parent_out,
        /*out_row_offset_tiles=*/0,  // overridden by device-side read
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::OutputRow,
        /*offsets_start_index=*/1U);

    auto ref_chunk = ttnn::matmul(input, weight, false, false);
    auto ref_chunk_vec = ttml::core::to_vector<float>(ref_chunk);
    auto written_vec = ttml::core::to_vector<float>(parent_out);

    // Verify rows [128, 256) match the matmul result.
    float written_err = 0.0F;
    for (uint32_t m = 128; m < 256; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            written_err = std::max(written_err, std::abs(written_vec[m * N + n] - ref_chunk_vec[(m - 128) * N + n]));
        }
    }
    EXPECT_LT(written_err, 2.0F) << "OutputRow on-device offsets err: " << written_err;

    // Verify untouched rows preserved.
    float untouched_err = 0.0F;
    for (uint32_t m = 0; m < M_parent; ++m) {
        if (m >= 128 && m < 256)
            continue;
        for (uint32_t n = 0; n < N; ++n) {
            untouched_err = std::max(untouched_err, std::abs(written_vec[m * N + n] - parent_orig_vec[m * N + n]));
        }
    }
    EXPECT_LT(untouched_err, 1e-3F) << "OutputRow on-device offsets touched wrong rows: " << untouched_err;
}

// Read-at-offset (InputRow): kernel reads offsets[start..start+2] from a device tensor
// and treats input rows [offsets[start], offsets[start+1]) as the matmul-M sub-range.
// Output is sized [effective_M_tiles*32 x N] (upper bound); only the first
// actual_eff_M_tiles*32 rows are valid. parent_M=320 > N=64 so transpose_core_grid=true
// (dm_in1_sender_out is the output writer).
TEST_F(VariableMatmulTest, OnDeviceOffsets_InputRow) {
    const uint32_t M_parent = 320, K = 128, N = 64;
    constexpr uint32_t effective_M_tiles_upper = 4U;  // 128 rows upper bound
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_parent, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    // Offsets [0, 32, 96, 160, 224, 288] (in rows). start_index=2 -> rows [96, 160) -> 2 tiles.
    std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 160U, 224U, 288U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::variable_matmul(
        input,
        weight,
        kConfig,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/effective_M_tiles_upper,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/std::nullopt,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputRow,
        /*offsets_start_index=*/2U);

    // Reference: matmul over the actual input sub-range [96, 160).
    auto input_slice = ttnn::slice(
        input,
        ttnn::SmallVector<uint32_t>{0U, 0U, 96U, 0U},
        ttnn::SmallVector<uint32_t>{1U, 1U, 160U, K},
        ttnn::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
    auto ref_chunk = ttnn::matmul(input_slice, weight, false, false);
    auto ref_chunk_vec = ttml::core::to_vector<float>(ref_chunk);
    auto result_vec = ttml::core::to_vector<float>(result);

    // Verify the first 64 output rows match the matmul result. Rows beyond actual_eff_M
    // are undefined (output sized to the upper bound).
    constexpr uint32_t actual_eff_M = 64U;
    float err = 0.0F;
    for (uint32_t m = 0; m < actual_eff_M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            err = std::max(err, std::abs(result_vec[m * N + n] - ref_chunk_vec[m * N + n]));
        }
    }
    EXPECT_LT(err, 2.0F) << "InputRow on-device offsets err: " << err;
}

// InputK (OffsetsRole::InputK): kernel reads offsets[start] and uses (value/32) as
// in0_k_offset_tiles. Parent-K mode: input is larger on the matmul-K axis; we read
// a K-aligned sub-range starting at the on-device offset. transpose_a=true matches
// the moe_ffn dW_down pattern.
TEST_F(VariableMatmulTest, OnDeviceOffsets_InputK_TransposeA) {
    const uint32_t K_parent = 512, K_w = 128, M = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent_km = create_random_device_tensor(K_parent, M, device);  // stored [K_parent, M]
    auto weight = create_random_device_tensor(K_w, N, device);

    std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    // start_index=2 -> offset=128 -> k_offset_tiles=4.
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t k_lo = 128;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(
        parent_km,
        weight,
        cfg,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,  // overridden by InputK
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/std::nullopt,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputK,
        /*offsets_start_index=*/kStart);

    auto sliced = ttnn::slice(
        parent_km,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_lo + K_w, M},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(sliced, weight, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "InputK transpose_a max_abs_error: " << err;
}

// WeightK (OffsetsRole::WeightK): kernel reads offsets[start] and uses (value/32) as
// in1_k_offset_tiles. Parent-K mode on the weight side: weight is the larger tensor
// on the matmul-K axis. transpose_a=true matches the moe_ffn dW_gate/dW_up pattern
// (grouped_value as the in1 K-parent).
TEST_F(VariableMatmulTest, OnDeviceOffsets_WeightK_TransposeA) {
    const uint32_t M = 128, K = 128, N = 256, K_w_parent = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);  // stored [K, M]
    auto weight_parent = create_random_device_tensor(K_w_parent, N, device);

    std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    // start_index=2 -> offset=128 -> in1_k_offset_tiles=4.
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t k_lo = 128;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(
        input_km,
        weight_parent,
        cfg,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,  // overridden by WeightK
        /*output_tensor=*/std::nullopt,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::WeightK,
        /*offsets_start_index=*/kStart);

    auto sliced = ttnn::slice(
        weight_parent,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_lo + K, N},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = ttnn::matmul(input_km, sliced, /*transpose_a=*/true, /*transpose_b=*/false);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "WeightK transpose_a max_abs_error: " << err;
}

TEST_F(VariableMatmulTest, WriteAtOffset_NoTranspose) {
    const uint32_t M_e = 128, K = 128, N = 256, M_parent = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_e, K, device);
    auto weight = create_random_device_tensor(K, N, device);
    auto parent_out = create_random_device_tensor(M_parent, N, device);
    // Capture original values for non-written-range verification later.
    auto parent_orig_vec = ttml::core::to_vector<float>(parent_out);

    constexpr uint32_t out_row_lo = 128;
    constexpr uint32_t out_row_hi = 256;
    constexpr uint32_t out_offset_tiles = out_row_lo / 32;

    ttml::metal::variable_matmul(
        input,
        weight,
        kConfig,
        std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/parent_out,
        /*out_row_offset_tiles=*/out_offset_tiles);

    // Reference: matmul into a fresh tensor, then verify parent's row-range matches it.
    auto ref_chunk = ttnn::matmul(input, weight, false, false);
    auto ref_chunk_vec = ttml::core::to_vector<float>(ref_chunk);

    auto written_vec = ttml::core::to_vector<float>(parent_out);
    // Verify written range
    float written_err = 0.0F;
    for (uint32_t m = out_row_lo; m < out_row_hi; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            uint32_t idx_parent = m * N + n;
            uint32_t idx_chunk = (m - out_row_lo) * N + n;
            written_err = std::max(written_err, std::abs(written_vec[idx_parent] - ref_chunk_vec[idx_chunk]));
        }
    }
    EXPECT_LT(written_err, 2.0F) << "write-at-offset written-range err: " << written_err;

    // Verify non-written ranges preserved
    float untouched_err = 0.0F;
    for (uint32_t m = 0; m < M_parent; ++m) {
        if (m >= out_row_lo && m < out_row_hi)
            continue;
        for (uint32_t n = 0; n < N; ++n) {
            uint32_t idx = m * N + n;
            untouched_err = std::max(untouched_err, std::abs(written_vec[idx] - parent_orig_vec[idx]));
        }
    }
    EXPECT_LT(untouched_err, 1e-3F) << "write-at-offset non-written rows changed: " << untouched_err;
}

// Multiple writes into same parent (moe-ffn fwd/bw concat pattern).
TEST_F(VariableMatmulTest, WriteAtOffset_MultipleRangesIntoSameParent) {
    const uint32_t K = 128, N = 256, M_parent = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto weight = create_random_device_tensor(K, N, device);
    auto parent_out = create_random_device_tensor(M_parent, N, device);

    struct Range {
        uint32_t lo, hi;
    };
    // Cover entire parent — full concat replacement.
    const std::vector<Range> ranges = {{0, 128}, {128, 256}, {256, 384}, {384, 512}};
    std::vector<ttnn::Tensor> chunks;
    chunks.reserve(ranges.size());
    for (const auto& r : ranges) {
        auto chunk_in = create_random_device_tensor(r.hi - r.lo, K, device);
        ttml::metal::variable_matmul(chunk_in, weight, kConfig, std::nullopt, 0, 0, 0, 0, parent_out, r.lo / 32);
        chunks.push_back(ttnn::matmul(chunk_in, weight, false, false));
    }

    // Reference: concat all chunks; compare with parent_out.
    auto ref = ttnn::concat(chunks, /*dim=*/2);
    float err = max_abs_error(parent_out, ref);
    EXPECT_LT(err, 2.0F) << "write-at-offset multi-range err: " << err;
}

// Both transposes simultaneously.
TEST_F(VariableMatmulTest, TransposeBoth_Correctness_128x128x512) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);   // stored [K, M]
    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    auto cfg = kConfig;
    cfg.transpose_a = true;
    cfg.transpose_b = true;
    auto result = ttml::metal::variable_matmul(input_km, weight_nk, cfg);
    auto ref = ttnn::matmul(input_km, weight_nk, /*transpose_a=*/true, /*transpose_b=*/true);

    float err = max_abs_error(result, ref);
    EXPECT_LT(err, 2.0F) << "transpose_a+b max_abs_error: " << err;
}
