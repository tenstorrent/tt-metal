// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"
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

ttnn::Tensor create_random_device_tensor(uint32_t M, uint32_t K, ttnn::distributed::MeshDevice* device, uint32_t seed) {
    const auto data =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{1U, 1U, M, K}, -1.0F, 1.0F, seed);
    return ttml::core::from_xtensor(data, device);
}

// Build the 1-D UINT32 ROW_MAJOR offsets tensor the EP path requires.
ttnn::Tensor make_offsets(const std::vector<uint32_t>& offsets_host, ttnn::distributed::MeshDevice* device) {
    return ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
}

// Reference for the InputAndWeightK + transpose_a path: slice both operands to the K-range
// [k_lo, k_lo + K_active), transpose in0 ([K, M] -> [M, K]), and minimal_matmul -> [M, N].
ttnn::Tensor expert_k_reference(
    const ttnn::Tensor& in0_km,
    const ttnn::Tensor& in1,
    uint32_t k_lo,
    uint32_t K_active,
    uint32_t M,
    uint32_t N,
    const VariableMatmulConfig& cfg) {
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
    return minimal_matmul_hifi4(in0_sliced_mk, in1_sliced, cfg);
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
        if (ttml::autograd::ctx().get_device().arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "variable_matmul is only supported on Blackhole.";
        }
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

// ---------------------------------------------------------------------------
// minimal_matmul parity. variable_matmul is a superset of minimal_matmul: with matched fidelity
// (HiFi4 + fp32 dst + packer_l1_acc), block sizes, grid, and a trivial single offset range, the
// two must be bit-identical. Any deviation is a core-path regression.
// ---------------------------------------------------------------------------

TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndWeightK_TransposeA) {
    const uint32_t K_parent_in0 = 512, M = 128, K_parent_in1 = 512, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto in0_km = create_random_device_tensor(K_parent_in0, M, device, /*seed=*/42U);
    auto in1 = create_random_device_tensor(K_parent_in1, N, device, /*seed=*/43U);

    const std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = make_offsets(offsets_host, device);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t k_lo = 128;
    constexpr uint32_t K_active = 128;

    auto cfg = kConfig;
    auto result = ttml::metal::variable_matmul_k_sliced(
        /*input_tensor=*/in0_km,
        /*weight_tensor=*/in1,
        /*config=*/cfg,
        /*offsets_tensor=*/offsets,
        /*offsets_start_index=*/kStart,
        /*transpose_a=*/true,
        /*transpose_b=*/false);

    auto ref = expert_k_reference(in0_km, in1, k_lo, K_active, M, N, kConfig);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(InputAndWeightK,tA) vs minimal not bit-exact";
}

// Verifies the partial last M-tile (176 logical / 192 padded) is clipped correctly and its pad
// rows stay zero. M=176 is the DeepSeek-16B/TP=8 dW_gate/dW_up bwd shape.
TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndWeightK_TransposeA_NonTileAlignedM_176) {
    const uint32_t K_parent_in0 = 512, M = 176, K_parent_in1 = 512, N = 64;
    auto* device = &ttml::autograd::ctx().get_device();

    // in0 stored [K_parent, M=176]; physical [K_parent, 192] in TILE layout.
    auto in0_km = create_random_device_tensor(K_parent_in0, M, device, /*seed=*/44U);
    auto in1 = create_random_device_tensor(K_parent_in1, N, device, /*seed=*/45U);

    const std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = make_offsets(offsets_host, device);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t k_lo = 128;
    constexpr uint32_t K_active = 128;

    auto cfg = kConfig;
    auto result = ttml::metal::variable_matmul_k_sliced(
        /*input_tensor=*/in0_km,
        /*weight_tensor=*/in1,
        /*config=*/cfg,
        /*offsets_tensor=*/offsets,
        /*offsets_start_index=*/kStart,
        /*transpose_a=*/true,
        /*transpose_b=*/false);

    auto ref = expert_k_reference(in0_km, in1, k_lo, K_active, M, N, cfg);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(InputAndWeightK,tA,M=176) vs minimal not bit-exact";
}

// ---------------------------------------------------------------------------
// M-axis offset parity. InputAndOutputRow reads the input row range [a, b) and writes the same
// range of the output parent. With matched HiFi4 settings, that sub-region must be bit-identical
// to minimal_matmul on the corresponding sliced input.
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

TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndOutputRow) {
    const uint32_t M_parent = 320, K = 128, N = 64;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_parent, K, device, /*seed=*/46U);
    auto weight = create_random_device_tensor(K, N, device, /*seed=*/47U);
    // Output parent is same shape as input M-axis (shared-tensor design).
    auto parent_out = create_random_device_tensor(M_parent, N, device, /*seed=*/48U);
    const auto parent_orig_vec = ttml::core::to_vector<float>(parent_out);

    // start_index=2 → rows [96, 160) → actual_M = 64.
    const std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 160U, 224U, 288U};
    auto offsets = make_offsets(offsets_host, device);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t m_lo = 96U;
    constexpr uint32_t m_hi = 160U;
    constexpr uint32_t actual_M = m_hi - m_lo;

    ttml::metal::variable_matmul_into_rows(
        /*input_tensor=*/input,
        /*weight_tensor=*/weight,
        /*config=*/kConfig,
        /*offsets_tensor=*/offsets,
        /*output_tensor=*/parent_out,
        /*offsets_start_index=*/kStart,
        /*expected_M_tiles=*/M_parent / 32U,  // upper bound = parent_M
        /*transpose_a=*/false,
        /*transpose_b=*/false);

    auto input_slice = ttnn::slice(
        input,
        ttsl::SmallVector<uint32_t>{0U, 0U, m_lo, 0U},
        ttsl::SmallVector<uint32_t>{1U, 1U, m_hi, K},
        ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
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

TEST_F(VariableMatmulTest, InputAndOutputRow_DefaultExpectedMTiles_ReadsCorrectRows) {
    const uint32_t M_parent = 320, K = 128, N = 64;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_parent, K, device, /*seed=*/146U);
    auto weight = create_random_device_tensor(K, N, device, /*seed=*/147U);
    auto parent_out = create_random_device_tensor(M_parent, N, device, /*seed=*/148U);
    const auto parent_orig_vec = ttml::core::to_vector<float>(parent_out);

    // start_index=2 → rows [96, 160) → actual_M = 64. row_start=96>0 is what exposes the bug.
    const std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 160U, 224U, 288U};
    auto offsets = make_offsets(offsets_host, device);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t m_lo = 96U;
    constexpr uint32_t m_hi = 160U;
    constexpr uint32_t actual_M = m_hi - m_lo;

    ttml::metal::variable_matmul_into_rows(
        /*input_tensor=*/input,
        /*weight_tensor=*/weight,
        /*config=*/kConfig,
        /*offsets_tensor=*/offsets,
        /*output_tensor=*/parent_out,
        /*offsets_start_index=*/kStart,
        /*expected_M_tiles=*/0U,  // default: must still read the right rows
        /*transpose_a=*/false,
        /*transpose_b=*/false);

    auto input_slice = ttnn::slice(
        input,
        ttsl::SmallVector<uint32_t>{0U, 0U, m_lo, 0U},
        ttsl::SmallVector<uint32_t>{1U, 1U, m_hi, K},
        ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
    auto ref = minimal_matmul_hifi4(input_slice, weight, kConfig);
    const auto ref_vec = ttml::core::to_vector<float>(ref);
    const auto written_vec = ttml::core::to_vector<float>(parent_out);

    EXPECT_EQ(subregion_max_abs_error(written_vec, ref_vec, m_lo, actual_M, N), 0.0F)
        << "variable(InputAndOutputRow, expected_M_tiles=0) read the wrong input rows at [" << m_lo << "," << m_hi
        << ")";

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
    EXPECT_EQ(untouched_err, 0.0F) << "variable(InputAndOutputRow, expected_M_tiles=0) corrupted untouched rows";
}

TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndOutputRow_TransposeB) {
    const uint32_t M_parent = 320, K = 128, N = 64;
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = create_random_device_tensor(M_parent, K, device, /*seed=*/56U);
    auto weight_nk = create_random_device_tensor(N, K, device, /*seed=*/57U);  // stored [N, K]
    auto parent_out = create_random_device_tensor(M_parent, N, device, /*seed=*/58U);
    const auto parent_orig_vec = ttml::core::to_vector<float>(parent_out);

    // start_index=2 → rows [96, 160) → actual_M = 64.
    const std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 160U, 224U, 288U};
    auto offsets = make_offsets(offsets_host, device);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t m_lo = 96U;
    constexpr uint32_t m_hi = 160U;
    constexpr uint32_t actual_M = m_hi - m_lo;

    ttml::metal::variable_matmul_into_rows(
        /*input_tensor=*/input,
        /*weight_tensor=*/weight_nk,
        /*config=*/kConfig,
        /*offsets_tensor=*/offsets,
        /*output_tensor=*/parent_out,
        /*offsets_start_index=*/kStart,
        /*expected_M_tiles=*/M_parent / 32U,
        /*transpose_a=*/false,
        /*transpose_b=*/true);

    auto input_slice = ttnn::slice(
        input,
        ttsl::SmallVector<uint32_t>{0U, 0U, m_lo, 0U},
        ttsl::SmallVector<uint32_t>{1U, 1U, m_hi, K},
        ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
    auto weight_kn = ttnn::transpose(weight_nk, -2, -1);  // [N, K] -> [K, N]
    auto ref = minimal_matmul_hifi4(input_slice, weight_kn, kConfig);
    const auto ref_vec = ttml::core::to_vector<float>(ref);
    const auto written_vec = ttml::core::to_vector<float>(parent_out);

    EXPECT_EQ(subregion_max_abs_error(written_vec, ref_vec, m_lo, actual_M, N), 0.0F)
        << "variable(InputAndOutputRow, transpose_b) vs minimal not bit-exact at [" << m_lo << "," << m_hi << ")";

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
    EXPECT_EQ(untouched_err, 0.0F) << "variable(InputAndOutputRow, transpose_b) corrupted untouched rows";
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
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t H_tiles = 48, I_tiles = 24, T_cap_tiles = 64;
    const uint32_t H = H_tiles * 32, I = I_tiles * 32, T_cap = T_cap_tiles * 32;

    auto dY = create_random_device_tensor(T_cap, H, device, /*seed=*/49U);
    auto act = create_random_device_tensor(T_cap, I, device, /*seed=*/50U);

    // offsets where index 1 → count=0 (offsets[1]==offsets[2]).
    std::vector<uint32_t> offsets_host = {0U, 128U, 128U, 256U};
    auto offsets = make_offsets(offsets_host, device);

    auto result = ttml::metal::variable_matmul_k_sliced(
        /*input_tensor=*/dY,
        /*weight_tensor=*/act,
        /*config=*/cfg,
        /*offsets_tensor=*/offsets,
        /*offsets_start_index=*/1U,
        /*transpose_a=*/true,
        /*transpose_b=*/false);

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

// ---------------------------------------------------------------------------
// Program-cache reuse — the op's headline contract: one cached program serves any
// (M, K, offsets_start_index) within a transpose/role/grid variant, re-driven via
// override_runtime_arguments rather than recompiled. The parity tests above make a single call
// each (build path only); these enable the cache and call the same program with different
// offsets, asserting every result stays bit-exact AND no new program is compiled on the repeat
// calls (the override path, where the offsets-callback bug lived). Cache entries are sampled
// around the variable_matmul call only, so the reference ops don't perturb the count.
// ---------------------------------------------------------------------------

TEST_F(VariableMatmulTest, CacheHit_InputAndWeightK_VaryingK) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const uint32_t K_parent = 512, M = 128, N = 256;
    auto in0_km = create_random_device_tensor(K_parent, M, device, /*seed=*/51U);
    auto in1 = create_random_device_tensor(K_parent, N, device, /*seed=*/52U);

    // Expert K-ranges of different sizes (all tile-aligned): 64, 128, 64, 256.
    const std::vector<uint32_t> offsets_host = {0U, 64U, 192U, 256U, 512U};
    auto offsets = make_offsets(offsets_host, device);
    const uint32_t num_experts = static_cast<uint32_t>(offsets_host.size()) - 1U;

    auto cfg = kConfig;
    for (uint32_t s = 0; s < num_experts; ++s) {
        const uint32_t k_lo = offsets_host[s];
        const uint32_t K_active = offsets_host[s + 1U] - offsets_host[s];

        const auto entries_before = device->num_program_cache_entries();
        auto result = ttml::metal::variable_matmul_k_sliced(
            /*input_tensor=*/in0_km,
            /*weight_tensor=*/in1,
            /*config=*/cfg,
            /*offsets_tensor=*/offsets,
            /*offsets_start_index=*/s,
            /*transpose_a=*/true,
            /*transpose_b=*/false);
        const auto entries_after = device->num_program_cache_entries();

        if (s == 0U) {
            // Guard against a vacuous test: the program cache must actually be populated, else
            // the delta==0 checks below would pass trivially with the cache disabled.
            EXPECT_GT(entries_after, 0U) << "program cache not populated after first call";
        } else {
            EXPECT_EQ(entries_after, entries_before)
                << "variable_matmul compiled a new program on offsets_start_index=" << s
                << " (expected a cache hit through override_runtime_arguments)";
        }

        auto ref = expert_k_reference(in0_km, in1, k_lo, K_active, M, N, cfg);

        EXPECT_EQ(max_abs_error(result, ref), 0.0F)
            << "variable(InputAndWeightK) cache-hit result wrong at offsets_start_index=" << s
            << " (K_active=" << K_active << ")";
    }
}

TEST_F(VariableMatmulTest, CacheHit_InputAndOutputRow_VaryingM) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const uint32_t M_parent = 320, K = 128, N = 64;
    auto input = create_random_device_tensor(M_parent, K, device, /*seed=*/53U);
    auto weight = create_random_device_tensor(K, N, device, /*seed=*/54U);
    auto parent_out = create_random_device_tensor(M_parent, N, device, /*seed=*/55U);

    // Expert M-ranges of different sizes (all tile-aligned), tiling [0, M_parent): 32, 64, 64, 160.
    const std::vector<uint32_t> offsets_host = {0U, 32U, 96U, 160U, 320U};
    auto offsets = make_offsets(offsets_host, device);
    const uint32_t num_experts = static_cast<uint32_t>(offsets_host.size()) - 1U;

    for (uint32_t s = 0; s < num_experts; ++s) {
        const uint32_t m_lo = offsets_host[s];
        const uint32_t m_hi = offsets_host[s + 1U];
        const uint32_t actual_M = m_hi - m_lo;

        const auto entries_before = device->num_program_cache_entries();
        ttml::metal::variable_matmul_into_rows(
            /*input_tensor=*/input,
            /*weight_tensor=*/weight,
            /*config=*/kConfig,
            /*offsets_tensor=*/offsets,
            /*output_tensor=*/parent_out,
            /*offsets_start_index=*/s,
            /*expected_M_tiles=*/M_parent / 32U,
            /*transpose_a=*/false,
            /*transpose_b=*/false);
        const auto entries_after = device->num_program_cache_entries();

        if (s == 0U) {
            // Guard against a vacuous test: the program cache must actually be populated, else
            // the delta==0 checks below would pass trivially with the cache disabled.
            EXPECT_GT(entries_after, 0U) << "program cache not populated after first call";
        } else {
            EXPECT_EQ(entries_after, entries_before)
                << "variable_matmul compiled a new program on offsets_start_index=" << s
                << " (expected a cache hit through override_runtime_arguments)";
        }

        auto input_slice = ttnn::slice(
            input,
            ttsl::SmallVector<uint32_t>{0U, 0U, m_lo, 0U},
            ttsl::SmallVector<uint32_t>{1U, 1U, m_hi, K},
            ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
        auto ref = minimal_matmul_hifi4(input_slice, weight, kConfig);
        const auto ref_vec = ttml::core::to_vector<float>(ref);
        const auto written_vec = ttml::core::to_vector<float>(parent_out);

        EXPECT_EQ(subregion_max_abs_error(written_vec, ref_vec, m_lo, actual_M, N), 0.0F)
            << "variable(InputAndOutputRow) cache-hit result wrong at offsets_start_index=" << s << " rows [" << m_lo
            << "," << m_hi << ")";
    }
}
