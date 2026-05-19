// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"
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

// Run minimal_matmul with HiFi4 + fp32_acc + packer_l1_acc — matches variable_matmul's
// defaults — and the same block sizes / grid. Used as the BF16-precision reference.
ttnn::Tensor minimal_matmul_hifi4(
    const ttnn::Tensor& input, const ttnn::Tensor& weight, const VariableMatmulConfig& cfg) {
    const ttnn::experimental::prim::MinimalMatmulConfig mm_cfg{
        .M_block_size = cfg.M_block_size,
        .K_block_size = cfg.K_block_size,
        .N_block_size = cfg.N_block_size,
        .subblock_h = cfg.subblock_h,
        .subblock_w = cfg.subblock_w,
        .compute_with_storage_grid_size = cfg.compute_with_storage_grid_size,
    };
    return ttnn::experimental::minimal_matmul(
        input,
        weight,
        /*bias=*/std::nullopt,
        /*fused_activation=*/std::nullopt,
        /*config=*/mm_cfg,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        /*compute_kernel_config=*/ttml::core::ComputeKernelConfig::matmul());
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

// Debug repro for moe_ffn pattern: input M_parent > actual_eff_M (from offsets),
// output_tensor pre-zeroed. The OutputRow override should bound writes to actual
// rows, leaving the rest of parent_out at its pre-zero value.
TEST_F(VariableMatmulTest, OnDeviceOffsets_OutputRow_ActualLessThanParent) {
    const uint32_t M_input = 128, K = 128, N = 64, M_parent_out = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_input, K, device);
    auto weight = create_random_device_tensor(K, N, device);
    // Pre-zero the parent output — moe_ffn does this for y / dX_via_*.
    auto parent_out = ttml::core::zeros(ttnn::Shape({1U, 1U, M_parent_out, N}), device, ttnn::DataType::BFLOAT16);

    // Offsets {0, 32, 96, 192}. start_index=1 -> range [32, 96) = 64 rows = 2 tiles.
    // input is 128 rows (4 tiles); actual_eff_M = 2 tiles < M_input/32 = 4.
    std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 192U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    ttml::metal::variable_matmul(
        input,
        weight,
        kConfig,
        std::nullopt,
        0U,
        0U,
        0U,
        0U,
        parent_out,
        0U,
        offsets,
        ttml::metal::OffsetsRole::OutputRow,
        1U);

    auto written = ttml::core::to_vector<float>(parent_out);

    // Pad rows BEYOND the [32, 96) write range must remain zero.
    float pad_err = 0.0F;
    for (uint32_t m = 0; m < M_parent_out; ++m) {
        if (m >= 32 && m < 96)
            continue;
        for (uint32_t n = 0; n < N; ++n) {
            pad_err = std::max(pad_err, std::abs(written[m * N + n]));
        }
    }
    EXPECT_LT(pad_err, 1e-3F) << "OutputRow leaked into pad rows: max_abs=" << pad_err;

    // Written rows should match matmul of input[0:64] @ weight.
    auto input_slice = ttnn::slice(
        input,
        ttnn::SmallVector<uint32_t>{0U, 0U, 0U, 0U},
        ttnn::SmallVector<uint32_t>{1U, 1U, 64U, K},
        ttnn::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
    auto ref = ttnn::matmul(input_slice, weight, false, false);
    auto ref_vec = ttml::core::to_vector<float>(ref);
    float written_err = 0.0F;
    for (uint32_t m = 0; m < 64U; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            written_err = std::max(written_err, std::abs(written[(m + 32) * N + n] - ref_vec[m * N + n]));
        }
    }
    EXPECT_LT(written_err, 2.0F) << "OutputRow written rows err: " << written_err;
}

// Multiple OutputRow calls with different offsets_start_index to the SAME parent_out
// (moe_ffn down_proj pattern). Each call writes a per-expert range; pad/slack rows
// between/after expert ranges must stay zero.
TEST_F(VariableMatmulTest, OnDeviceOffsets_OutputRow_MultiExpert) {
    const uint32_t M_input = 128, K = 128, N = 64;
    const uint32_t T_cap = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_input, K, device);
    auto weight = create_random_device_tensor(K, N, device);
    auto parent_out = ttml::core::zeros(ttnn::Shape({1U, 1U, T_cap, N}), device, ttnn::DataType::BFLOAT16);

    // offsets = {0, 32, 96, 128}. Three experts:
    //   e=0: actual=1 tile, writes [0:32]
    //   e=1: actual=2 tiles, writes [32:96]
    //   e=2: actual=1 tile, writes [96:128]
    // Pad/slack rows: [128:256) must remain zero.
    std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 128U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    for (uint32_t e = 0; e < 3U; ++e) {
        ttml::metal::variable_matmul(
            input,
            weight,
            kConfig,
            std::nullopt,
            0U,
            0U,
            0U,
            0U,
            parent_out,
            0U,
            offsets,
            ttml::metal::OffsetsRole::OutputRow,
            e);
    }

    auto written = ttml::core::to_vector<float>(parent_out);

    // Trailing slack [128:256] must stay zero.
    float slack_err = 0.0F;
    for (uint32_t m = 128; m < T_cap; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            slack_err = std::max(slack_err, std::abs(written[m * N + n]));
        }
    }
    EXPECT_LT(slack_err, 1e-3F) << "OutputRow multi-expert leaked into slack rows: max_abs=" << slack_err;
}

// moe_ffn fwd gate_proj pattern: pre-zero output_tensor [upper*32, I], InputRow
// reads a sub-range of grouped, matmul writes to rows [0:actual*32] of output.
// Rows [actual*32 : upper*32] must remain zero (from pre-zero).
TEST_F(VariableMatmulTest, OnDeviceOffsets_InputRow_PreZeroOutput_PadRowsStayZero) {
    const uint32_t H_grouped = 128;  // 4 tiles
    const uint32_t K = 64;
    const uint32_t I = 64;
    constexpr uint32_t upper_M_tiles = 4U;
    auto* device = &ttml::autograd::ctx().get_device();

    auto grouped = create_random_device_tensor(H_grouped, K, device);
    auto w_gate = create_random_device_tensor(I, K, device);  // [I, K] used as [K, I] under transpose_b
    auto gate_proj = ttml::core::zeros(ttnn::Shape({1U, 1U, upper_M_tiles * 32U, I}), device, ttnn::DataType::BFLOAT16);

    // offsets = {0, 32, 64, 128}. start_index=1 -> range [32, 64) = 32 rows = 1 tile.
    // actual = 1 < upper = 4. Pad rows [32:128] of gate_proj must stay zero.
    std::vector<uint32_t> offsets_host = {0U, 32U, 64U, 128U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    auto cfg = kConfig;
    cfg.transpose_b = true;
    ttml::metal::variable_matmul(
        grouped,
        w_gate,
        cfg,
        std::nullopt,
        0U,
        /*effective_M_tiles=*/upper_M_tiles,
        0U,
        0U,
        gate_proj,
        0U,
        offsets,
        ttml::metal::OffsetsRole::InputRow,
        1U);

    auto written = ttml::core::to_vector<float>(gate_proj);

    // Rows [32, 128) of gate_proj must stay zero.
    float pad_err = 0.0F;
    for (uint32_t m = 32; m < upper_M_tiles * 32U; ++m) {
        for (uint32_t n = 0; n < I; ++n) {
            pad_err = std::max(pad_err, std::abs(written[m * I + n]));
        }
    }
    EXPECT_LT(pad_err, 1e-3F) << "InputRow w/ pre-zero output_tensor leaked into pad rows: max_abs=" << pad_err;
}

// moe-ffn fwd pattern with zero-input pad rows: grouped has counts<padded so trailing
// rows within an expert's range are zero. Matmul output for those rows should be zero.
// Mirrors expert 2 in MoeFfnSwigluBackwardTest (counts={32,16,48}): per-expert padded
// range is 64 rows but only first 48 have real data; rows 48..63 of the per-expert
// matmul output should be exactly zero (silu(0)*0 in fwd, propagating to dgrouped pad).
TEST_F(VariableMatmulTest, OnDeviceOffsets_InputRow_ZeroPadRowsProduceZeroOutput) {
    const uint32_t H_grouped = 128;
    const uint32_t K = 64;
    const uint32_t I = 64;
    constexpr uint32_t upper_M_tiles = 4U;
    auto* device = &ttml::autograd::ctx().get_device();

    // Build grouped with zero pad rows [112:128] (mimics expert 2's pad in the bwd test).
    auto grouped_h = xt::xarray<float>::from_shape({1U, 1U, H_grouped, K});
    for (uint32_t m = 0; m < H_grouped; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            grouped_h(0, 0, m, k) = (m < 112U) ? 0.5F : 0.0F;
        }
    }
    auto grouped = ttml::core::from_xtensor(grouped_h, device);
    auto w_gate = create_random_device_tensor(I, K, device);
    auto gate_proj = ttml::core::zeros(ttnn::Shape({1U, 1U, upper_M_tiles * 32U, I}), device, ttnn::DataType::BFLOAT16);

    // offsets = {0, 32, 64, 128}. start_index=2 -> range [64, 128) = 64 rows = 2 tiles.
    // grouped[64:112] = 0.5 (real), grouped[112:128] = 0 (pad).
    std::vector<uint32_t> offsets_host = {0U, 32U, 64U, 128U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    auto cfg = kConfig;
    cfg.transpose_b = true;
    ttml::metal::variable_matmul(
        grouped,
        w_gate,
        cfg,
        std::nullopt,
        0U,
        upper_M_tiles,
        0U,
        0U,
        gate_proj,
        0U,
        offsets,
        ttml::metal::OffsetsRole::InputRow,
        2U);

    auto out = ttml::core::to_vector<float>(gate_proj);

    // Rows [48:64] of gate_proj come from grouped[112:128]=0. Output must be 0.
    float zero_input_err = 0.0F;
    for (uint32_t m = 48; m < 64U; ++m) {
        for (uint32_t n = 0; n < I; ++n) {
            zero_input_err = std::max(zero_input_err, std::abs(out[m * I + n]));
        }
    }
    EXPECT_LT(zero_input_err, 1e-3F) << "InputRow zero-input rows produced non-zero output: max_abs=" << zero_input_err;
}

// Mimics moe_ffn bwd dX_via_gate matmul: in0 has real rows + trailing zero rows
// within the OutputRow-selected M range. Matmul output for zero-input rows MUST
// be zero — this test exposes whether OutputRow's M_tiles override correctly
// produces zero output for zero input across cores.
TEST_F(VariableMatmulTest, OnDeviceOffsets_OutputRow_ZeroInputRowsProduceZeroOutput) {
    const uint32_t M_input = 128, K = 128, N = 64;
    const uint32_t T_cap = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto in0_h = xt::xarray<float>::from_shape({1U, 1U, M_input, K});
    auto& rng = ttml::autograd::ctx().get_generator();
    std::mt19937 rng_local(rng());
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (uint32_t m = 0; m < M_input; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            in0_h(0, 0, m, k) = (m < 48U) ? dist(rng_local) : 0.0F;
        }
    }
    auto in0 = ttml::core::from_xtensor(in0_h, device);
    auto weight = create_random_device_tensor(K, N, device);
    auto parent_out = ttml::core::zeros(ttnn::Shape({1U, 1U, T_cap, N}), device, ttnn::DataType::BFLOAT16);

    std::vector<uint32_t> offsets_host = {0U, 64U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    ttml::metal::variable_matmul(
        in0,
        weight,
        kConfig,
        std::nullopt,
        0U,
        0U,
        0U,
        0U,
        parent_out,
        0U,
        offsets,
        ttml::metal::OffsetsRole::OutputRow,
        0U);

    auto written = ttml::core::to_vector<float>(parent_out);

    float zero_input_err = 0.0F;
    for (uint32_t m = 48U; m < 64U; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            zero_input_err = std::max(zero_input_err, std::abs(written[m * N + n]));
        }
    }
    EXPECT_LT(zero_input_err, 1.0F) << "OutputRow zero-input rows produced non-zero output: max_abs=" << zero_input_err;
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

// ---------------------------------------------------------------------------
// HiFi4-vs-HiFi4 minimal_matmul parity tests. variable_matmul is a superset of
// minimal_matmul; with matched fidelity (HiFi4 + fp32 dst + packer_l1_acc), matched
// block sizes / grid, and none of variable_matmul's extra features exercised (no
// on-device offsets, no write-at-offset), the two MUST produce bit-identical output —
// same algorithm, same reduction order, same FPU ops. Any deviation is a regression in
// variable_matmul's core path.
// ---------------------------------------------------------------------------

TEST_F(VariableMatmulTest, MinimalParity_NoTranspose) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    auto result = ttml::metal::variable_matmul(input, weight, kConfig);
    auto ref = minimal_matmul_hifi4(input, weight, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable vs minimal (no transpose) not bit-exact";
}

TEST_F(VariableMatmulTest, MinimalParity_TransposeB) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight_nk = create_random_device_tensor(N, K, device);  // stored [N, K]

    auto cfg = kConfig;
    cfg.transpose_b = true;
    auto result = ttml::metal::variable_matmul(input, weight_nk, cfg);
    // minimal_matmul has no transpose flag; pre-transpose the weight to compare apples-to-apples.
    auto weight_kn = ttnn::transpose(weight_nk, -2, -1);
    auto ref = minimal_matmul_hifi4(input, weight_kn, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(transpose_b) vs minimal not bit-exact";
}

TEST_F(VariableMatmulTest, MinimalParity_TransposeA) {
    const uint32_t M = 128, K = 128, N = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);  // stored [K, M]
    auto weight = create_random_device_tensor(K, N, device);

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto input_mk = ttnn::transpose(input_km, -2, -1);
    auto ref = minimal_matmul_hifi4(input_mk, weight, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(transpose_a) vs minimal not bit-exact";
}

// Production-scale parity at moe_ffn dW_down shape (transpose_a; K=1024).
// Even at K=1024, two HiFi4 reductions in the same order should agree well within rtol=1e-2.
TEST_F(VariableMatmulTest, MinimalParity_MixtralDWDown_TransposeA) {
    const VariableMatmulConfig cfg{
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

    auto result = ttml::metal::variable_matmul(input_km, weight, cfg);
    auto input_mk = ttnn::transpose(input_km, -2, -1);
    auto ref = minimal_matmul_hifi4(input_mk, weight, cfg);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(MixtralDWDown,tA) vs minimal not bit-exact";
}

// ---------------------------------------------------------------------------
// K-axis OffsetsRole bit-exact parity. variable_matmul with a K-offset reads only the
// K[a..b] tile range of the parent — that's the same math as pre-slicing the parent and
// matmul-ing the slice. With matched HiFi4 settings, results MUST be bit-identical to
// minimal_matmul on the slice. Covers both host-known offsets and on-device offsets.
// ---------------------------------------------------------------------------

TEST_F(VariableMatmulTest, MinimalParity_KOffset_In0_NoTranspose) {
    const uint32_t M = 128, K_parent = 512, K_w = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent = create_random_device_tensor(M, K_parent, device);
    auto weight = create_random_device_tensor(K_w, N, device);

    constexpr uint32_t k_lo = 128;
    constexpr uint32_t k_hi = k_lo + K_w;  // = 256
    constexpr uint32_t k_offset_tiles = k_lo / 32;

    auto result = ttml::metal::variable_matmul(
        parent,
        weight,
        kConfig,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/k_offset_tiles);

    auto sliced = ttnn::slice(
        parent,
        ttsl::SmallVector<uint32_t>{0, 0, 0, k_lo},
        ttsl::SmallVector<uint32_t>{1, 1, M, k_hi},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = minimal_matmul_hifi4(sliced, weight, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(K-offset in0) vs minimal not bit-exact";
}

TEST_F(VariableMatmulTest, MinimalParity_KOffset_In1_NoTranspose) {
    const uint32_t M = 128, K = 128, N = 256, K_w_parent = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M, K, device);
    auto weight_parent = create_random_device_tensor(K_w_parent, N, device);

    constexpr uint32_t k_lo = 128;
    constexpr uint32_t k_hi = k_lo + K;
    constexpr uint32_t k_offset_tiles = k_lo / 32;

    auto result = ttml::metal::variable_matmul(
        input,
        weight_parent,
        kConfig,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/k_offset_tiles);

    auto sliced = ttnn::slice(
        weight_parent,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_hi, N},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = minimal_matmul_hifi4(input, sliced, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(K-offset in1) vs minimal not bit-exact";
}

TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputK_TransposeA) {
    const uint32_t K_parent = 512, K_w = 128, M = 128, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto parent_km = create_random_device_tensor(K_parent, M, device);
    auto weight = create_random_device_tensor(K_w, N, device);

    const std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;  // offsets[2..4) = [128, 256) → K_w = 128
    constexpr uint32_t k_lo = 128;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(
        parent_km,
        weight,
        cfg,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/std::nullopt,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputK,
        /*offsets_start_index=*/kStart);

    auto sliced_km = ttnn::slice(
        parent_km,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_lo + K_w, M},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto sliced_mk = ttnn::transpose(sliced_km, -2, -1);
    auto ref = minimal_matmul_hifi4(sliced_mk, weight, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(InputK,tA) vs minimal not bit-exact";
}

TEST_F(VariableMatmulTest, MinimalParity_OnDeviceWeightK_TransposeA) {
    const uint32_t M = 128, K = 128, N = 256, K_w_parent = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input_km = create_random_device_tensor(K, M, device);
    auto weight_parent = create_random_device_tensor(K_w_parent, N, device);

    const std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;  // offsets[2..4) = [128, 256) → K = 128
    constexpr uint32_t k_lo = 128;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(
        input_km,
        weight_parent,
        cfg,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
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
    auto input_mk = ttnn::transpose(input_km, -2, -1);
    auto ref = minimal_matmul_hifi4(input_mk, sliced, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(WeightK,tA) vs minimal not bit-exact";
}

// InputAndWeightK: both in0_k and in1_k overridden from the same offsets entry. Mirrors
// the moe_ffn dW_down/dW_gate/dW_up backward call pattern.
TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndWeightK_TransposeA) {
    const uint32_t K_parent_in0 = 512, M = 128, K_parent_in1 = 512, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto in0_km = create_random_device_tensor(K_parent_in0, M, device);
    auto in1 = create_random_device_tensor(K_parent_in1, N, device);

    const std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t k_lo = 128;
    constexpr uint32_t K_active = 128;

    auto cfg = kConfig;
    cfg.transpose_a = true;
    auto result = ttml::metal::variable_matmul(
        in0_km,
        in1,
        cfg,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/std::nullopt,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputAndWeightK,
        /*offsets_start_index=*/kStart);

    auto in0_sliced_km = ttnn::slice(
        in0_km,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_lo + K_active, M},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto in0_sliced_mk = ttnn::transpose(in0_sliced_km, -2, -1);
    auto in1_sliced = ttnn::slice(
        in1,
        ttsl::SmallVector<uint32_t>{0, 0, k_lo, 0},
        ttsl::SmallVector<uint32_t>{1, 1, k_lo + K_active, N},
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
    auto ref = minimal_matmul_hifi4(in0_sliced_mk, in1_sliced, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(InputAndWeightK,tA) vs minimal not bit-exact";
}

// ---------------------------------------------------------------------------
// M-axis OffsetsRole bit-exact parity. variable_matmul with an M-axis offset reads only
// the M[a..b] row range of the input parent (InputRow), writes to that row range of the
// output parent (OutputRow), or both (InputAndOutputRow). With matched HiFi4 settings,
// the relevant sub-region of the output MUST be bit-identical to minimal_matmul on the
// corresponding sliced input.
// ---------------------------------------------------------------------------

namespace {
// Helper: compare a subregion `[m_lo, m_hi) × [0, N)` of `result_parent_vec` (M_parent×N
// flattened) against the entire `ref_vec` (M_e×N) computed from minimal_matmul on the
// same slice. Returns max abs diff.
float subregion_max_abs_error(
    const std::vector<float>& result_parent_vec,
    const std::vector<float>& ref_vec,
    uint32_t m_lo,
    uint32_t M_e,
    uint32_t N) {
    float err = 0.0F;
    for (uint32_t m = 0; m < M_e; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            err = std::max(err, std::abs(result_parent_vec[(m_lo + m) * N + n] - ref_vec[m * N + n]));
        }
    }
    return err;
}
}  // namespace

TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputRow) {
    // parent_M > N → transpose_core_grid = true (dm_in1 is writer).
    const uint32_t M_parent = 320, K = 128, N = 64;
    constexpr uint32_t effective_M_tiles_upper = 4U;  // 128 rows upper bound for output sizing
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_parent, K, device);
    auto weight = create_random_device_tensor(K, N, device);

    // start_index=2 → rows [96, 160) → actual_eff_M = 64 rows = 2 tiles.
    const std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 160U, 224U, 288U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t m_lo = 96U;
    constexpr uint32_t m_hi = 160U;
    constexpr uint32_t actual_M = m_hi - m_lo;  // = 64

    auto result = ttml::metal::variable_matmul(
        input,
        weight,
        kConfig,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/effective_M_tiles_upper,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/std::nullopt,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputRow,
        /*offsets_start_index=*/kStart);

    auto input_slice = ttnn::slice(
        input,
        ttnn::SmallVector<uint32_t>{0U, 0U, m_lo, 0U},
        ttnn::SmallVector<uint32_t>{1U, 1U, m_hi, K},
        ttnn::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
    auto ref = minimal_matmul_hifi4(input_slice, weight, kConfig);

    // result has shape [effective_M_tiles_upper*32, N] = [128, 64]. Only the first actual_M
    // rows are valid; the rest is whatever was left in the upper-bound buffer.
    auto result_vec = ttml::core::to_vector<float>(result);
    auto ref_vec = ttml::core::to_vector<float>(ref);
    EXPECT_EQ(subregion_max_abs_error(result_vec, ref_vec, /*m_lo=*/0U, actual_M, N), 0.0F)
        << "variable(InputRow) vs minimal not bit-exact on actual M sub-range";
}

TEST_F(VariableMatmulTest, MinimalParity_OnDeviceOutputRow) {
    // parent_M > N → transpose_core_grid = true; dm_in1 is the writer.
    const uint32_t M_e = 128, K = 128, N = 64, M_parent_out = 512;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_e, K, device);
    auto weight = create_random_device_tensor(K, N, device);
    // Pre-fill output with random values so we can verify untouched rows are preserved.
    auto parent_out = create_random_device_tensor(M_parent_out, N, device);
    const auto parent_orig_vec = ttml::core::to_vector<float>(parent_out);

    // start_index=1 → row 128 → out_row_offset_tiles = 4.
    const std::vector<uint32_t> offsets_host = {0U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 1U;
    constexpr uint32_t m_lo = 128U;

    ttml::metal::variable_matmul(
        input,
        weight,
        kConfig,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/0,
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/parent_out,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::OutputRow,
        /*offsets_start_index=*/kStart);

    auto ref = minimal_matmul_hifi4(input, weight, kConfig);
    const auto ref_vec = ttml::core::to_vector<float>(ref);
    const auto written_vec = ttml::core::to_vector<float>(parent_out);

    EXPECT_EQ(subregion_max_abs_error(written_vec, ref_vec, m_lo, M_e, N), 0.0F)
        << "variable(OutputRow) vs minimal not bit-exact at [" << m_lo << "," << m_lo + M_e << ")";

    // Verify untouched rows preserved exactly.
    float untouched_err = 0.0F;
    for (uint32_t m = 0; m < M_parent_out; ++m) {
        if (m >= m_lo && m < m_lo + M_e) {
            continue;
        }
        for (uint32_t n = 0; n < N; ++n) {
            untouched_err = std::max(untouched_err, std::abs(written_vec[m * N + n] - parent_orig_vec[m * N + n]));
        }
    }
    EXPECT_EQ(untouched_err, 0.0F) << "variable(OutputRow) corrupted untouched rows";
}

// InputAndOutputRow: read input rows [a, b), compute matmul, write into output rows [a, b)
// of the parent. This is the moe_ffn forward call pattern (gate_proj / up_proj /
// down_proj).
TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndOutputRow) {
    const uint32_t M_parent = 320, K = 128, N = 64;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_parent, K, device);
    auto weight = create_random_device_tensor(K, N, device);
    // Output parent is same shape as input M-axis (shared-tensor design).
    auto parent_out = create_random_device_tensor(M_parent, N, device);
    const auto parent_orig_vec = ttml::core::to_vector<float>(parent_out);

    // start_index=2 → rows [96, 160) → actual_M = 64.
    const std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 160U, 224U, 288U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t m_lo = 96U;
    constexpr uint32_t m_hi = 160U;
    constexpr uint32_t actual_M = m_hi - m_lo;

    ttml::metal::variable_matmul(
        input,
        weight,
        kConfig,
        /*bias=*/std::nullopt,
        /*in0_row_offset_tiles=*/0,
        /*effective_M_tiles=*/M_parent / 32U,  // upper bound = parent_M
        /*in0_k_offset_tiles=*/0,
        /*in1_k_offset_tiles=*/0,
        /*output_tensor=*/parent_out,
        /*out_row_offset_tiles=*/0,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputAndOutputRow,
        /*offsets_start_index=*/kStart);

    auto input_slice = ttnn::slice(
        input,
        ttnn::SmallVector<uint32_t>{0U, 0U, m_lo, 0U},
        ttnn::SmallVector<uint32_t>{1U, 1U, m_hi, K},
        ttnn::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
    auto ref = minimal_matmul_hifi4(input_slice, weight, kConfig);
    const auto ref_vec = ttml::core::to_vector<float>(ref);
    const auto written_vec = ttml::core::to_vector<float>(parent_out);

    EXPECT_EQ(subregion_max_abs_error(written_vec, ref_vec, m_lo, actual_M, N), 0.0F)
        << "variable(InputAndOutputRow) vs minimal not bit-exact at [" << m_lo << "," << m_hi << ")";

    // Untouched rows of parent_out preserved exactly.
    float untouched_err = 0.0F;
    for (uint32_t m = 0; m < M_parent; ++m) {
        if (m >= m_lo && m < m_hi) {
            continue;
        }
        for (uint32_t n = 0; n < N; ++n) {
            untouched_err = std::max(untouched_err, std::abs(written_vec[m * N + n] - parent_orig_vec[m * N + n]));
        }
    }
    EXPECT_EQ(untouched_err, 0.0F) << "variable(InputAndOutputRow) corrupted untouched rows";
}

// ---- Empty-expert probe: K-axis offset where count_e = 0 must produce all-zero output. ----

namespace {
const ttml::metal::VariableMatmulConfig kMoeFfnConfig{
    .M_block_size = 4,
    .K_block_size = 8,
    .N_block_size = 8,
    .subblock_h = 2,
    .subblock_w = 2,
    .compute_with_storage_grid_size = {10, 10},
};
}  // namespace

TEST_F(VariableMatmulTest, EmptyExpertProbe_InputAndWeightK_TransposeA) {
    auto cfg = kMoeFfnConfig;
    cfg.transpose_a = true;
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t H_tiles = 48, I_tiles = 24, T_cap_tiles = 64;
    const uint32_t H = H_tiles * 32, I = I_tiles * 32, T_cap = T_cap_tiles * 32;

    auto dY = create_random_device_tensor(T_cap, H, device);
    auto act = create_random_device_tensor(T_cap, I, device);

    // offsets where index 1 → count=0 (offsets[1]==offsets[2]).
    std::vector<uint32_t> offsets_host = {0U, 128U, 128U, 256U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::variable_matmul(
        dY,
        act,
        cfg,
        std::nullopt,
        0,
        0,
        0,
        0,
        std::nullopt,
        0,
        offsets,
        ttml::metal::OffsetsRole::InputAndWeightK,
        /*start=*/1U);

    const auto vec = ttml::core::to_vector<float>(result);
    float max_abs = 0.0F;
    size_t non_finite = 0;
    for (float v : vec) {
        if (!std::isfinite(v)) {
            ++non_finite;
        } else {
            max_abs = std::max(max_abs, std::abs(v));
        }
    }
    EXPECT_EQ(non_finite, 0U) << "empty-expert output has non-finite values";
    EXPECT_EQ(max_abs, 0.0F) << "empty-expert output non-zero; max_abs=" << max_abs;
}
