// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

// ---------------------------------------------------------------------------
// HiFi4-vs-HiFi4 minimal_matmul parity tests. variable_matmul is a superset of
// minimal_matmul; with matched fidelity (HiFi4 + fp32 dst + packer_l1_acc), matched
// block sizes / grid, and the EP path's offsets reduced to a single trivial range, the
// two MUST produce bit-identical output — same algorithm, same reduction order, same
// FPU ops. Any deviation is a regression in variable_matmul's core path.
// ---------------------------------------------------------------------------

// InputAndWeightK: both in0 and in1 K-sliced from the same offsets entry. Mirrors the
// moe_ffn dW_down/dW_gate/dW_up backward call pattern.
TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndWeightK_TransposeA) {
    const uint32_t K_parent_in0 = 512, M = 128, K_parent_in1 = 512, N = 256;
    auto* device = &ttml::autograd::ctx().get_device();

    auto in0_km = create_random_device_tensor(K_parent_in0, M, device, /*seed=*/42U);
    auto in1 = create_random_device_tensor(K_parent_in1, N, device, /*seed=*/43U);

    const std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t k_lo = 128;
    constexpr uint32_t K_active = 128;

    auto cfg = kConfig;
    auto result = ttml::metal::variable_matmul(
        /*input_tensor=*/in0_km,
        /*weight_tensor=*/in1,
        /*config=*/cfg,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputAndWeightK,
        /*transpose_a=*/true,
        /*transpose_b=*/false,
        /*compute_kernel_config=*/std::nullopt,
        /*output_tensor=*/std::nullopt,
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

// Non-tile-aligned matmul-M on the InputAndWeightK + transpose_a path. This is the
// DeepSeek 16B / TP=8 moe_ffn bwd dW_gate (and dW_up) pattern: in0 = d_gate_proj
// stored [count, I/TP=176], read as [I/TP, count] under transpose_a. matmul-M = I/TP = 176
// (off-tile), matmul-N = H (tile-aligned), matmul-K = expert's count-tiles (tile-aligned,
// from offsets[start..start+2]). Output [176, H] gets logical M=176 / padded 192; the
// tail 16 padded rows are zero, downstream readers see only 176.
TEST_F(VariableMatmulTest, MinimalParity_OnDeviceInputAndWeightK_TransposeA_NonTileAlignedM_176) {
    const uint32_t K_parent_in0 = 512, M = 176, K_parent_in1 = 512, N = 64;
    auto* device = &ttml::autograd::ctx().get_device();

    // in0 stored [K_parent, M=176]; physical [K_parent, 192] in TILE layout.
    auto in0_km = create_random_device_tensor(K_parent_in0, M, device, /*seed=*/44U);
    auto in1 = create_random_device_tensor(K_parent_in1, N, device, /*seed=*/45U);

    const std::vector<uint32_t> offsets_host = {0U, 64U, 128U, 256U, 384U};
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t k_lo = 128;
    constexpr uint32_t K_active = 128;

    auto cfg = kConfig;
    auto result = ttml::metal::variable_matmul(
        /*input_tensor=*/in0_km,
        /*weight_tensor=*/in1,
        /*config=*/cfg,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputAndWeightK,
        /*transpose_a=*/true,
        /*transpose_b=*/false,
        /*compute_kernel_config=*/std::nullopt,
        /*output_tensor=*/std::nullopt,
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
    auto ref = minimal_matmul_hifi4(in0_sliced_mk, in1_sliced, cfg);

    EXPECT_EQ(max_abs_error(result, ref), 0.0F) << "variable(InputAndWeightK,tA,M=176) vs minimal not bit-exact";
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

// InputAndOutputRow: read input rows [a, b), compute matmul, write into output rows [a, b)
// of the parent. This is the moe_ffn forward call pattern (gate_proj / up_proj /
// down_proj).
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
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);
    constexpr uint32_t kStart = 2U;
    constexpr uint32_t m_lo = 96U;
    constexpr uint32_t m_hi = 160U;
    constexpr uint32_t actual_M = m_hi - m_lo;

    ttml::metal::variable_matmul(
        /*input_tensor=*/input,
        /*weight_tensor=*/weight,
        /*config=*/kConfig,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputAndOutputRow,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*compute_kernel_config=*/std::nullopt,
        /*output_tensor=*/parent_out,
        /*offsets_start_index=*/kStart,
        /*expected_M_tiles=*/M_parent / 32U);  // upper bound = parent_M

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
    auto offsets = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets_host, ttnn::Shape({static_cast<uint32_t>(offsets_host.size())}), device, ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::variable_matmul(
        /*input_tensor=*/dY,
        /*weight_tensor=*/act,
        /*config=*/cfg,
        /*offsets_tensor=*/offsets,
        /*offsets_role=*/ttml::metal::OffsetsRole::InputAndWeightK,
        /*transpose_a=*/true,
        /*transpose_b=*/false,
        /*compute_kernel_config=*/std::nullopt,
        /*output_tensor=*/std::nullopt,
        /*offsets_start_index=*/1U);

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
