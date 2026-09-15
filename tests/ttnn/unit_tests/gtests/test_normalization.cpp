// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Smoke tests for the ttnn normalization op family: one test per program
// factory, for every op under ttnn/cpp/ttnn/operations/normalization/
// (softmax, layernorm, rmsnorm, their distributed pre/post all-gather stages,
// groupnorm, batch_norm), plus tests for the front-end code paths that pick
// between factories. Known-broken factories are kept as DISABLED_ repros or
// excluded with an issue reference in the section comments: distributed
// pre-all-gather Welford (#51231, hang class), sharded groupnorm Welford
// (#53143/#52700), batch_norm one-sided running stats (#51230).
// Each test uses small deterministic inputs with exact
// closed-form expected outputs (or op-tolerance where accumulation forces it),
// so a kernel that runs but produces garbage fails.
//
// Each test states in a comment which program factory or code path it covers.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/shape.hpp>
#include "smoke_test_utils.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"
#include "ttnn/operations/normalization/batch_norm/device/running_statistics_device_operation.hpp"
#include "ttnn/operations/normalization/groupnorm/groupnorm.hpp"
#include "ttnn/operations/normalization/layernorm/layernorm.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/layernorm_post_all_gather.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/layernorm_pre_all_gather.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::normalization::test {

class NormalizationSmoke : public TTNNFixtureWithSuiteDevice<NormalizationSmoke> {};

namespace detail {

using ttnn::test_utils::expect_close;
using ttnn::test_utils::make_device_tensor;
using ttnn::test_utils::make_device_tensor_mc;
using ttnn::test_utils::to_float_vector;

}  // namespace detail

// ---------------------------------------------------------------------------
// SOFTMAX cells
// Entry points: ttnn/cpp/ttnn/operations/normalization/softmax/softmax.hpp
// Dispatch:     device/softmax_device_operation.cpp::select_program_factory
// ---------------------------------------------------------------------------

TEST_F(NormalizationSmoke, SoftmaxInterleavedUniform) {
    // SoftmaxProgramFactoryAttentionOptimized (interleaved attention path): ttnn::softmax(x, -1)
    // on a rank-4 TILE input. numeric_stable defaults to true on this entry point.
    auto& device = *device_;
    constexpr uint32_t H = 32, W = 32;
    const ttnn::Shape shape({1, 1, H, W});

    // Launch 1: all-zero rows -> uniform 1/32 (0.03125 is exact in bf16).
    const std::vector<float> zeros(H * W, 0.0f);
    const auto x0 = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto out0 = ttnn::softmax(x0, -1);
    detail::expect_close(detail::to_float_vector(out0), std::vector<float>(H * W, 1.0f / 32.0f), 0.0f, 1e-3f);

    // Launch 2: x[.,0] = ln(3), rest 0 -> closed form p[0] = 3/34, others 1/34 per row.
    // rtol 0.06: ln(3) quantizes in bf16 (rel err <= 2^-9) and, dominant, the default
    // approx-mode SFPU exp carries ~5% relative error at arguments near -1 (the numeric_stable
    // path evaluates exp(-ln 3) for the zero columns; measured 4.8% low on BH p100a). Still
    // discriminates: a wrong row normalization or factory mix-up is >> 6% off.
    std::vector<float> data(H * W, 0.0f);
    std::vector<float> expected(H * W, 1.0f / 34.0f);
    for (uint32_t h = 0; h < H; h++) {
        data[h * W] = std::log(3.0f);
        expected[h * W] = 3.0f / 34.0f;
    }
    const auto x1 = detail::make_device_tensor<float>(device, shape, data, DataType::BFLOAT16, Layout::TILE);
    const auto out1 = ttnn::softmax(x1, -1);
    detail::expect_close(detail::to_float_vector(out1), expected, 0.06f, 1e-3f);
}

TEST_F(NormalizationSmoke, SoftmaxNumericStableShift) {
    // SoftmaxProgramFactoryAttentionOptimized, numeric_stable shift invariance:
    // softmax(x) must equal softmax(x + 128) because the kernel subtracts the row max first.
    // Magnitudes chosen so BOTH launches see bit-identical operands after the max shift even
    // though the kernel's intermediate CBs quantize to bf16 when fp32 dest acc is off: the
    // pattern is integers 0..15 and the shift is +128, so every shifted value (128..143) is
    // bf16-exact (ulp at 128 is 1), while exp(128) ~ 3.9e55 overflows fp32 -- without the max
    // subtraction the shifted launch produces inf/nan and the non-finite pre-check fails.
    // (A +10000 shift is NOT bf16-exact -- ulp 64 -- and measurably destroys the pattern via
    // the bf16 intermediates; that is expected quantization, not a stability bug.)
    // #52131: softmax / softmax_in_place / scale_mask_softmax now default numeric_stable=true,
    // but scale_mask_softmax_in_place and scale_causal_mask_hw_dims_softmax_in_place still
    // default numeric_stable=false.
    auto& device = *device_;
    constexpr uint32_t H = 32, W = 64;
    const ttnn::Shape shape({1, 1, H, W});
    std::vector<float> base(H * W), shifted(H * W);
    for (uint32_t i = 0; i < H * W; i++) {
        base[i] = static_cast<float>((i * 7 + 3) % 16);  // fixed pseudo-random integers, bf16-exact
        shifted[i] = base[i] + 128.0f;                   // 128..143: bf16-exact; exp(128) overflows fp32
    }
    const auto x0 = detail::make_device_tensor<float>(device, shape, base, DataType::FLOAT32, Layout::TILE);
    const auto x1 = detail::make_device_tensor<float>(device, shape, shifted, DataType::FLOAT32, Layout::TILE);
    const auto out0 = ttnn::softmax(x0, -1, std::nullopt, std::nullopt, /*numeric_stable=*/true);
    const auto out1 = ttnn::softmax(x1, -1, std::nullopt, std::nullopt, /*numeric_stable=*/true);
    // Post-shift operands are bitwise identical after max subtraction -> outputs must match tightly.
    detail::expect_close(detail::to_float_vector(out0), detail::to_float_vector(out1), 1e-3f, 1e-6f, 0.9999f);
}

TEST_F(NormalizationSmoke, SoftmaxShardedInPlace) {
    // SoftmaxShardedProgramFactoryAttentionOptimized: softmax_in_place with
    // SoftmaxShardedMultiCoreProgramConfig. Validator requires: input sharded, inplace,
    // block_w * tile_width == shape[-1], block_h/block_w equal to the shard shape in tiles,
    // and #shards == #cores (here one [32,32] shard on a 1x1 grid).
    auto& device = *device_;
    constexpr uint32_t H = 32, W = 32;
    const ttnn::Shape shape({1, 1, H, W});
    const CoreRangeSet grid(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    const MemoryConfig sharded_mc(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(grid, {H, W}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    const std::vector<float> zeros(H * W, 0.0f);
    const auto x =
        detail::make_device_tensor_mc<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE, sharded_mc);
    const ttnn::SoftmaxShardedMultiCoreProgramConfig cfg{
        /*compute_with_storage_grid_size=*/CoreCoord{1, 1},
        /*subblock_w=*/1,
        /*block_h=*/1,
        /*block_w=*/1};
    const auto out = ttnn::softmax_in_place(x, -1, cfg);
    detail::expect_close(detail::to_float_vector(out), std::vector<float>(H * W, 1.0f / 32.0f), 0.0f, 1e-3f);
}

TEST_F(NormalizationSmoke, ScaleMaskSoftmax) {
    // SoftmaxProgramFactoryAttentionOptimized, fused scale+mask (out-of-place). Non-causal
    // interleaved masks are applied with add_tiles_bcast_rows -- only row 0 of each mask tile is
    // broadcast to every input row -- so a TILE mask [1,1,32,W] with identical rows is used here.
    // (A ROW_MAJOR mask must instead be the packed [B,1,W/32,32] form per the device validator.)
    auto& device = *device_;
    constexpr uint32_t H = 32, W = 64, KEEP = 16;
    const ttnn::Shape shape({1, 1, H, W});
    const std::vector<float> zeros(H * W, 0.0f);
    std::vector<float> mask_data(H * W, 0.0f);
    std::vector<float> expected(H * W, 0.0f);
    for (uint32_t h = 0; h < H; h++) {
        for (uint32_t w = KEEP; w < W; w++) {
            mask_data[h * W + w] = -1e9f;
        }
        for (uint32_t w = 0; w < KEEP; w++) {
            expected[h * W + w] = 1.0f / 16.0f;  // 0.0625, exact in bf16
        }
    }
    const auto x = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto mask = detail::make_device_tensor<float>(device, shape, mask_data, DataType::BFLOAT16, Layout::TILE);
    // scale multiplies only the input (2 * 0 = 0); the mask is added after: exp(-1e9) ~ 0,
    // leaving 16 unmasked exp(0)=1 terms per row -> 1/16 in the kept columns, ~0 elsewhere.
    // rtol 0.05: the default approx-mode SFPU exp does not flush exp(-1e9) fully to zero
    // (measured ~1e-3 per masked column on BH p100a -> the 48 masked columns leak ~3% of the
    // row's probability mass, shorting each kept column by the same factor). atol 2e-3 gives
    // the masked columns headroom over the same leak.
    const auto out = ttnn::scale_mask_softmax(x, 2.0f, mask);
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, 2e-3f);
}

TEST_F(NormalizationSmoke, ScaleMaskSoftmaxCausalInPlace) {
    // SoftmaxProgramFactoryAttentionOptimized via scale_mask_softmax_in_place with
    // is_causal_mask=true: causal TILE masks are applied elementwise (add_tiles), so the mask is
    // the full [1,1,H,W] of the input. numeric_stable passed EXPLICITLY: unlike softmax /
    // scale_mask_softmax, this in-place entry point still defaults numeric_stable=false (#52131).
    auto& device = *device_;
    constexpr uint32_t H = 32, W = 32;
    const ttnn::Shape shape({1, 1, H, W});
    const std::vector<float> zeros(H * W, 0.0f);
    std::vector<float> mask_data(H * W, 0.0f);
    std::vector<float> expected(H * W, 0.0f);
    for (uint32_t i = 0; i < H; i++) {
        for (uint32_t j = 0; j < W; j++) {
            mask_data[i * W + j] = (j <= i) ? 0.0f : -1e9f;
            if (j <= i) {
                expected[i * W + j] = 1.0f / static_cast<float>(i + 1);  // row i keeps i+1 zeros
            }
        }
    }
    const auto x = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto mask = detail::make_device_tensor<float>(device, shape, mask_data, DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::scale_mask_softmax_in_place(
        x,
        /*scale=*/1.0f,
        mask,
        ttnn::SoftmaxDefaultProgramConfig{},
        /*is_causal_mask=*/true,
        /*compute_kernel_config=*/std::nullopt,
        /*numeric_stable=*/true);
    // rtol 0.05: 1/(i+1) quantizes in bf16 (rel err <= 2^-9) and the approx-mode SFPU exp
    // leaks ~1e-3 per masked column (see ScaleMaskSoftmax) -- worst on the top rows, where few
    // kept columns absorb the whole leak (row 0: measured 0.969 vs exact 1.0 on BH p100a).
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, 2e-3f, 0.999f);
}

TEST_F(NormalizationSmoke, SoftmaxGeneralW) {
    // SoftmaxProgramFactoryGeneralWSmall: rank-5 input bypasses the rank-4 attention path.
    // General W/H/C factories JIT the moreh_softmax kernels (SOFTMAX_KERNEL_PATH_GENERAL ->
    // ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels).
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 1, 32, 32});
    const std::vector<float> zeros(32 * 32, 0.0f);
    const auto x = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::softmax(x, -1);
    // Zero rows -> uniform 1/32, exact in bf16.
    detail::expect_close(detail::to_float_vector(out), std::vector<float>(32 * 32, 1.0f / 32.0f), 0.0f, 1e-3f);
}

TEST_F(NormalizationSmoke, SoftmaxGeneralH) {
    // SoftmaxProgramFactoryGeneralHSmall: dim=-2 selects the H factory (moreh_softmax kernels).
    auto& device = *device_;
    constexpr uint32_t H = 64, W = 32;
    const ttnn::Shape shape({1, 1, H, W});
    const std::vector<float> zeros(H * W, 0.0f);
    const auto x = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::softmax(x, -2);
    // Softmax over H=64 zeros -> uniform 1/64 (2^-6, exact in bf16).
    detail::expect_close(detail::to_float_vector(out), std::vector<float>(H * W, 1.0f / 64.0f), 0.0f, 1e-3f);
}

TEST_F(NormalizationSmoke, SoftmaxGeneralWLarge) {
    // SoftmaxProgramFactoryGeneralWLarge: rank-5 keeps dim=-1 off the attention path, and
    // W=4096 (Wt=128) overflows the 512KB L1 small-path budget in
    // is_softmax_general_w_small_available, so the Large factory is selected.
    auto& device = *device_;
    constexpr uint32_t H = 32, W = 4096;
    const ttnn::Shape shape({1, 1, 1, H, W});
    const std::vector<float> zeros(H * W, 0.0f);
    const auto x = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::softmax(x, -1);
    // Zero rows -> uniform 1/4096 (2^-12, exact in bf16); rtol per the wide-row drift policy.
    detail::expect_close(detail::to_float_vector(out), std::vector<float>(H * W, 1.0f / 4096.0f), 0.2f, 1e-5f);
}

TEST_F(NormalizationSmoke, DISABLED_SoftmaxGeneralHLarge) {
    // KNOWN BUG (disabled; run with --gtest_also_run_disabled_tests): #53927 --
    // moreh_softmax_h_large trips a TRISC0 watcher assert on WH and BH, killing the shared
    // device. Results are correct without the watcher. Enable when #53927 closes.
    // SoftmaxProgramFactoryGeneralHLarge: dim=-2 with H=4096 (Ht=128) overflows the same
    // 512KB budget in is_softmax_general_h_small_available.
    auto& device = *device_;
    constexpr uint32_t H = 4096, W = 32;
    const ttnn::Shape shape({1, 1, H, W});
    const std::vector<float> zeros(H * W, 0.0f);
    const auto x = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::softmax(x, -2);
    detail::expect_close(detail::to_float_vector(out), std::vector<float>(H * W, 1.0f / 4096.0f), 0.2f, 1e-5f);
}

TEST_F(NormalizationSmoke, SoftmaxGeneralC) {
    // SoftmaxProgramFactoryGeneralCLarge: dim=1 on rank-4 (neither -1 nor -2) always routes to
    // the C factory -- there is no "small" C variant (moreh_softmax kernels).
    auto& device = *device_;
    constexpr uint32_t C = 4, H = 32, W = 32, HW = H * W;
    const ttnn::Shape shape({1, C, H, W});
    // Channel c holds the constant c everywhere -> channel softmax at every (h,w) is the same
    // 4-way closed form: p_c = e^c / (1 + e + e^2 + e^3).
    const float denom = 1.0f + std::numbers::e_v<float> + std::exp(2.0f) + std::exp(3.0f);
    std::vector<float> data(C * HW);
    std::vector<float> expected(C * HW);
    for (uint32_t c = 0; c < C; c++) {
        const float p_c = std::exp(static_cast<float>(c)) / denom;
        for (uint32_t i = 0; i < HW; i++) {
            data[c * HW + i] = static_cast<float>(c);  // 0..3, exact in bf16
            expected[c * HW + i] = p_c;
        }
    }
    const auto x = detail::make_device_tensor<float>(device, shape, data, DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::softmax(x, 1);
    // Device exp approximation + bf16 output quantization -> tolerance-based compare.
    detail::expect_close(detail::to_float_vector(out), expected, 0.03f, 5e-3f, 0.999f);
}

TEST_F(NormalizationSmoke, SoftmaxWideRowSum) {
    // SoftmaxProgramFactoryAttentionOptimized at wide W -- #52045 canary at smoke scale.
    // Each element of a zero row is 1/8192 (2^-13, exact in bf16), but the reconstructed row
    // *sum* drifts as W grows: #52045 measured ~0.865 at W=8192 on the old non-numeric-stable
    // default path. Elements get a generous relative band; the row-sum band [0.9, 1.1] is the
    // regression tripwire -- if drift at W=8192 worsens past ~10% this fails loudly.
    auto& device = *device_;
    constexpr uint32_t H = 32, W = 8192;
    const ttnn::Shape shape({1, 1, H, W});
    const std::vector<float> zeros(H * W, 0.0f);
    const auto x = detail::make_device_tensor<float>(device, shape, zeros, DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::softmax(x, -1);
    const auto result = detail::to_float_vector(out);
    // Per-element: rtol 0.2 tolerates the known uniform drift without masking a blowup.
    detail::expect_close(result, std::vector<float>(H * W, 1.0f / 8192.0f), 0.2f, 2.5e-5f);
    double row_sum = 0.0;
    for (uint32_t w = 0; w < W; w++) {
        row_sum += result[w];  // row 0
    }
    EXPECT_GT(row_sum, 0.9) << "wide-row softmax sum drift regressed (#52045, 0.865 @ W=8192)";
    EXPECT_LT(row_sum, 1.1) << "wide-row softmax sum drift regressed (#52045, 0.865 @ W=8192)";
}

// ---------------------------------------------------------------------------
// LAYERNORM + RMSNORM cells
// Op entry points: ttnn::layer_norm (layernorm/layernorm.hpp)
//                  ttnn::rms_norm  (rmsnorm/rmsnorm.hpp)
// Program configs: ttnn::prim::{LayerNormDefaultProgramConfig,
//                  LayerNormShardedMultiCoreProgramConfig}
//                  (field names/order verified in layernorm/device/layernorm_types.hpp)
// Default epsilon is 1e-12 in both signatures; sqrt(1 + 1e-12) deviates from 1 by ~5e-13,
// far below every tolerance used here, so closed-form goldens ignore it.
// ---------------------------------------------------------------------------

namespace detail {

// Alternating column pattern. Every tensor in this section has even row width, so
// flat-index parity equals column parity and a flat alternation is exactly a
// per-row alternating pattern (row mean 0 for +-v).
inline std::vector<float> norm_alternating(size_t n, float even_col, float odd_col) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = (i % 2 == 0) ? even_col : odd_col;
    }
    return v;
}

// L1-sharded MemoryConfig with a single shard on core (0,0) covering the whole tensor.
inline MemoryConfig norm_single_core_shard(
    tt::tt_metal::TensorMemoryLayout layout, uint32_t shard_h, uint32_t shard_w) {
    return MemoryConfig(
        layout,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})),
            {shard_h, shard_w},
            tt::tt_metal::ShardOrientation::ROW_MAJOR));
}

// Reciprocal LUT (1/1, 1/2, ..., 1/width) replicated on every core of `cores`:
// fp32, ROW_MAJOR, HEIGHT_SHARDED in L1 with one (1, width) shard per core.
// Mirrors ttnn.create_layer_norm_reciprocals (ttnn/ttnn/operations/normalization.py).
// Welford LayerNorm REQUIRES this tensor: both factories TT_FATAL without it
// (layernorm_op_multi_core.cpp, layernorm_op_multi_core_sharded.cpp).
inline Tensor norm_welford_recip_lut(
    tt::tt_metal::distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t width) {
    const uint32_t num_cores = cores.num_cores();
    std::vector<float> lut;
    lut.reserve(static_cast<size_t>(num_cores) * width);
    for (uint32_t c = 0; c < num_cores; ++c) {
        for (uint32_t i = 0; i < width; ++i) {
            lut.push_back(1.0f / static_cast<float>(i + 1));
        }
    }
    const MemoryConfig lut_cfg(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(cores, {1, width}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    return make_device_tensor_mc(
        device, ttnn::Shape({num_cores, width}), lut, DataType::FLOAT32, Layout::ROW_MAJOR, lut_cfg);
}

}  // namespace detail

// LayerNormMultiCoreProgramFactory, interleaved TILE bf16, small-tensor path
// (reader_unary_interleaved_ln.cpp + compute/layernorm.cpp), NO gamma/beta.
// Rows alternate -1,+1 -> mean 0 and population var 1 are exact in bf16 (all partial
// sums are small integers), so out = x/sqrt(1+eps) ~= +-1, which is the input pattern
// back unchanged. Tolerance covers device rsqrt only.
//
// The gamma/beta variant is LayerNormInterleavedGammaBeta below rather than a second
// ttnn::layer_norm call here: gamma/beta presence is a compile-time kernel variant, so
// bundling both doubled this cell's JIT bill (~4.0 s cold on BH p100a) against a 5 s
// per-test merge-gate ceiling (merge-gate.yaml). Same reasoning for RmsNormInterleaved
// and GroupNormShardedBlock1x1.
TEST_F(NormalizationSmoke, LayerNormInterleaved) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x_data = detail::norm_alternating(n, -1.0f, 1.0f);
    const auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);

    const auto out_plain = ttnn::layer_norm(x);
    detail::expect_close(detail::to_float_vector(out_plain), x_data, /*rtol=*/0.0f, /*atol=*/0.02f);
}

// LayerNormMultiCoreProgramFactory with TILE gamma/beta applied. Same input and normalization
// as LayerNormInterleaved, so out = +-1 * gamma + beta = +-2 + 0.5 = {-1.5, +2.5}; +-1, 1.5 and
// 2.5 are all bf16-exact, and the tolerance covers device rsqrt + per-op rounding only (one
// bf16 ulp at 2.5 is ~0.0156).
//
// TILE gamma/beta take logical shape [1,1,1,W]: the validator wants padded[-1] == W,
// logical[-1] >= W and padded[-2] == tile height (layernorm_device_operation.cpp);
// tilization pads rows 1..31 with zeros, which is safe because the compute kernel applies
// gamma/beta via mul/add_tiles_bcast_rows -- only tile row 0 is ever read
// (kernels/compute/layernorm.cpp). This matches the python convention
// (test_layernorm.py passes gamma as (1,1,1,K) TILE).
TEST_F(NormalizationSmoke, LayerNormInterleavedGammaBeta) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x = detail::make_device_tensor(
        device, shape, detail::norm_alternating(n, -1.0f, 1.0f), DataType::BFLOAT16, Layout::TILE);

    const ttnn::Shape gb_shape({1, 1, 1, 64});
    const auto gamma =
        detail::make_device_tensor(device, gb_shape, std::vector<float>(64, 2.0f), DataType::BFLOAT16, Layout::TILE);
    const auto beta =
        detail::make_device_tensor(device, gb_shape, std::vector<float>(64, 0.5f), DataType::BFLOAT16, Layout::TILE);
    const auto out_gb = ttnn::layer_norm(x, 1e-12f, gamma, beta);
    detail::expect_close(
        detail::to_float_vector(out_gb),
        detail::norm_alternating(n, -1.5f, 2.5f),
        /*rtol=*/0.01f,  // gamma=2 doubles the one-ulp normalization error
        /*atol=*/0.02f);
}

// LayerNormMultiCoreProgramFactory, fused pre-add path (residual_input_tensor present
// -> fuse_pre_add in layernorm_op_multi_core.cpp). x = (+-1 pattern) - 0.5 and
// residual = +0.5 constant, so x + residual is exactly the +-1 pattern of
// LayerNormInterleaved (-1.5, 0.5, 1.0 are all bf16-exact, and the adds are exact).
// A correct pre-add must therefore reproduce that cell's golden unchanged.
TEST_F(NormalizationSmoke, LayerNormResidualFused) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x = detail::make_device_tensor(
        device, shape, detail::norm_alternating(n, -1.5f, 0.5f), DataType::BFLOAT16, Layout::TILE);
    const auto residual =
        detail::make_device_tensor(device, shape, std::vector<float>(n, 0.5f), DataType::BFLOAT16, Layout::TILE);
    const auto out = ttnn::layer_norm(x, 1e-12f, std::nullopt, std::nullopt, residual);
    detail::expect_close(detail::to_float_vector(out), detail::norm_alternating(n, -1.0f, 1.0f), 0.0f, 0.02f);
}

// LayerNormMultiCoreProgramFactory, Welford variant (compute/layernorm_welford.cpp;
// reader is the standard reader_unary_interleaved_ln.cpp with the welford compile-time
// arg). LayerNormDefaultProgramConfig fields verified: {legacy_reduction, legacy_rsqrt,
// use_welford}. Welford is LN-only (validate_on_program_cache_miss rejects RMSNORM) and
// requires a reciprocal LUT tensor (TT_FATAL at layernorm_op_multi_core.cpp) --
// built here over the full compute grid with width = full W, exactly as
// ttnn.create_layer_norm_reciprocals does for non-sharded inputs.
// Same closed-form +-1 golden as LayerNormInterleaved.
TEST_F(NormalizationSmoke, LayerNormWelfordInterleaved) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x_data = detail::norm_alternating(n, -1.0f, 1.0f);
    const auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);

    const auto grid = device.compute_with_storage_grid_size();
    const CoreRangeSet full_grid(CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}));
    const auto recip = detail::norm_welford_recip_lut(device, full_grid, 64);

    const ttnn::prim::LayerNormProgramConfig cfg = ttnn::prim::LayerNormDefaultProgramConfig{.use_welford = true};
    const auto out =
        ttnn::layer_norm(x, 1e-12f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cfg, std::nullopt, recip);
    detail::expect_close(detail::to_float_vector(out), x_data, 0.0f, 0.02f);
}

// LayerNormShardedProgramFactory (compute/layernorm_sharded.cpp): single [32,64] L1
// shard on core (0,0), WIDTH_SHARDED. Config field order verified:
// {compute_with_storage_grid_size, subblock_w, block_h, block_w, inplace,
//  legacy_reduction=false, legacy_rsqrt=false, use_welford=false}.
// Validator (mcast-1d branch, M == block_h*tile_h): block_w == ceil(Kt/num_cores) == 2,
// block_h == Mt == 1, block dims * tile dims == shard shape, bbox < grid size, and
// sharded input requires a sharded output (inherited from the input's memory config).
// Golden: same closed-form +-1 as the interleaved cell.
TEST_F(NormalizationSmoke, LayerNormSharded1x1) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x_data = detail::norm_alternating(n, -1.0f, 1.0f);
    const auto shard_cfg = detail::norm_single_core_shard(tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, 32, 64);
    const auto x = detail::make_device_tensor_mc(device, shape, x_data, DataType::BFLOAT16, Layout::TILE, shard_cfg);
    const ttnn::prim::LayerNormProgramConfig cfg = ttnn::prim::LayerNormShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size = CoreCoord{1, 1},
        .subblock_w = 1,
        .block_h = 1,
        .block_w = 2,
        .inplace = false};
    const auto out = ttnn::layer_norm(x, 1e-12f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cfg);
    detail::expect_close(detail::to_float_vector(out), x_data, 0.0f, 0.02f);
}

// LayerNormShardedProgramFactory, Welford variant (compute/layernorm_sharded_welford.cpp).
TEST_F(NormalizationSmoke, LayerNormShardedWelford1x1) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x_data = detail::norm_alternating(n, -1.0f, 1.0f);
    const auto shard_cfg = detail::norm_single_core_shard(tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, 32, 64);
    const auto x = detail::make_device_tensor_mc(device, shape, x_data, DataType::BFLOAT16, Layout::TILE, shard_cfg);
    const auto recip = detail::norm_welford_recip_lut(device, shard_cfg.shard_spec()->grid, 64);
    const ttnn::prim::LayerNormProgramConfig cfg = ttnn::prim::LayerNormShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size = CoreCoord{1, 1},
        .subblock_w = 1,
        .block_h = 1,
        .block_w = 2,
        .inplace = false,
        .use_welford = true};
    const auto out =
        ttnn::layer_norm(x, 1e-12f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cfg, std::nullopt, recip);
    detail::expect_close(detail::to_float_vector(out), x_data, 0.0f, 0.02f);
}

// LayerNormMultiCoreProgramFactory with ROW_MAJOR gamma/beta -> selects the rm_gb reader
// (reader_unary_interleaved_ln_rm_gb.cpp). RM gamma/beta shape rule
// (layernorm_device_operation.cpp): padded[-1] == tile width and
// volume/tile_width == W/tile_width, i.e. [1, 1, W/32, 32] -- stick j covers columns
// j*32..j*32+31 (constant weights make ordering immaterial here).
// Same golden as LayerNormInterleaved: {-1.5, +2.5}.
TEST_F(NormalizationSmoke, LayerNormRowMajorGammaBeta) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x = detail::make_device_tensor(
        device, shape, detail::norm_alternating(n, -1.0f, 1.0f), DataType::BFLOAT16, Layout::TILE);
    const ttnn::Shape gb_shape({1, 1, 2, 32});  // W/32 x 32 sticks
    const auto gamma = detail::make_device_tensor(
        device, gb_shape, std::vector<float>(64, 2.0f), DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto beta = detail::make_device_tensor(
        device, gb_shape, std::vector<float>(64, 0.5f), DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto out = ttnn::layer_norm(x, 1e-12f, gamma, beta);
    detail::expect_close(detail::to_float_vector(out), detail::norm_alternating(n, -1.5f, 2.5f), 0.01f, 0.02f);
}

// RMSNorm through the shared LayerNormMultiCoreProgramFactory (norm_type=RMSNORM,
// interleaved TILE bf16; compute/layernorm.cpp with the RMSNORM define -- Welford is
// rejected for RMS by the validator, and rms_norm has no recip_tensor parameter).
// Rows constant 2 -> rms = sqrt(mean(x^2)+eps) = 2 exactly -> out = 1.0 everywhere.
// With weight = 0.5 (same TILE [1,1,1,W] rules as LN gamma) -> 0.5 everywhere.
// 1.0 and 0.5 are bf16-exact; atol covers device rsqrt only. pcc/frobenius skipped:
// both are degenerate for constant expected vectors.
TEST_F(NormalizationSmoke, RmsNormInterleaved) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x =
        detail::make_device_tensor(device, shape, std::vector<float>(n, 2.0f), DataType::BFLOAT16, Layout::TILE);

    const auto out_plain = ttnn::rms_norm(x);
    detail::expect_close(detail::to_float_vector(out_plain), std::vector<float>(n, 1.0f), 0.0f, 0.02f);
}

// Same factory with the TILE weight applied (split out for the JIT-cost reason given on
// LayerNormInterleaved): out = 1.0 * 0.5 = 0.5 everywhere, bf16-exact.
TEST_F(NormalizationSmoke, RmsNormInterleavedWeight) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x =
        detail::make_device_tensor(device, shape, std::vector<float>(n, 2.0f), DataType::BFLOAT16, Layout::TILE);

    const auto weight = detail::make_device_tensor(
        device, ttnn::Shape({1, 1, 1, 64}), std::vector<float>(64, 0.5f), DataType::BFLOAT16, Layout::TILE);
    const auto out_w = ttnn::rms_norm(x, 1e-12f, weight);
    detail::expect_close(detail::to_float_vector(out_w), std::vector<float>(n, 0.5f), 0.0f, 0.02f);
}

// RMSNorm through LayerNormShardedProgramFactory: same single-core [32,64] WIDTH_SHARDED
// setup and program config as LayerNormSharded1x1 (use_welford stays default-false --
// the validator rejects Welford for RMSNORM). Rows alternate -3,+3 -> mean(x^2) = 9
// exactly -> rms = 3 -> out = -+1 (bf16-exact inputs and golden).
TEST_F(NormalizationSmoke, RmsNormSharded1x1) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x_data = detail::norm_alternating(n, -3.0f, 3.0f);
    const auto shard_cfg = detail::norm_single_core_shard(tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, 32, 64);
    const auto x = detail::make_device_tensor_mc(device, shape, x_data, DataType::BFLOAT16, Layout::TILE, shard_cfg);
    const ttnn::prim::LayerNormProgramConfig cfg = ttnn::prim::LayerNormShardedMultiCoreProgramConfig{
        .compute_with_storage_grid_size = CoreCoord{1, 1},
        .subblock_w = 1,
        .block_h = 1,
        .block_w = 2,
        .inplace = false};
    const auto out = ttnn::rms_norm(x, 1e-12f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cfg);
    detail::expect_close(detail::to_float_vector(out), detail::norm_alternating(n, -1.0f, 1.0f), 0.0f, 0.02f);
}

// LayerNormMultiCoreProgramFactory with FLOAT32 input -- the fp32-accumulation precision
// class tracked in #48824. The default compute config (layernorm_default_compute_config:
// HiFi4, approx off, fp32_acc=true) enables fp32_dest_acc_en, so the whole reduction and
// float32_reduction path stay in fp32. Tight tolerance is the point of this cell: a
// silent fall-back to bf16 intermediates would miss atol=1e-3 by an order of magnitude
// (one bf16 ulp at 1.0 is 2^-7 ~= 0.0078). Same exact +-1 pattern: mean 0 / var 1 are
// exact in fp32, out = x/sqrt(1+1e-12) == +-1 to ~5e-13 in exact arithmetic.
TEST_F(NormalizationSmoke, LayerNormFp32) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const size_t n = shape.volume();
    const auto x_data = detail::norm_alternating(n, -1.0f, 1.0f);
    const auto x = detail::make_device_tensor(device, shape, x_data, DataType::FLOAT32, Layout::TILE);
    const auto out = ttnn::layer_norm(x);
    detail::expect_close(detail::to_float_vector(out), x_data, /*rtol=*/0.0f, /*atol=*/1e-3f);
}

// ---------------------------------------------------------------------------
// DISTRIBUTED LAYERNORM / RMSNORM cells (pre/post all-gather stages).
// With num_devices = 1 the pre-stage output IS the all-gathered stats tensor,
// so the two-stage pipeline is single-device-testable and must reproduce the
// single-stage op.
// ---------------------------------------------------------------------------

namespace detail {

inline std::vector<float> dist_norm_constant(size_t n, float value) { return std::vector<float>(n, value); }

}  // namespace detail

TEST_F(NormalizationSmoke, DistributedLayerNormStats) {
    // LayerNormPreAllGatherProgramFactory stats layout. Per the kernel header
    // (layernorm_pre_allgather.cpp) and the post kernel's consumption comment
    // (layernorm_post_allgather.cpp: "dfb::stats = [sum(x0**2), sum(x0), ...]"): the output is
    // two tiles wide per device with sum(x^2) in tile 0 and sum(x) in tile 1, meaningful values
    // in the left-most column of each tile only. The reader's reduce scaler is 1.0 (PoolType::SUM),
    // so these are SUMS over the local width, not means; post divides by logical_W * num_devices.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::dist_norm_constant(32 * 64, 2.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);

    auto stats = ttnn::layer_norm_pre_all_gather(x);
    ASSERT_EQ(stats.logical_shape()[-1], 64u);
    ASSERT_EQ(stats.logical_shape()[-2], 32u);
    const auto s = detail::to_float_vector(stats);

    // Exact in bf16: every partial subset sum is a multiple of 4 (resp. 2) bounded by 256 (128),
    // all representable with <= 8 significand bits regardless of accumulation order.
    for (uint32_t r = 0; r < 32; ++r) {
        EXPECT_FLOAT_EQ(s[r * 64 + 0], 256.0f) << "sum(x^2), col 0 of tile 0, row " << r;  // 64 * 2^2
        EXPECT_FLOAT_EQ(s[r * 64 + 32], 128.0f) << "sum(x), col 0 of tile 1, row " << r;   // 64 * 2
    }
    // Columns other than col 0 of each tile are reduce scratch (undefined) - deliberately unchecked.
}

TEST_F(NormalizationSmoke, DistributedLayerNormEndToEnd) {
    // LayerNormPreAllGatherProgramFactory + LayerNormPostAllGatherProgramFactory (1D interleaved,
    // non-Welford). With num_devices = 1 the pre output IS the all-gathered stats tensor
    // (post derives num_devices = stats_tiles_cols / 2), so the two-stage pipeline must
    // reproduce single-stage ttnn::layer_norm.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::norm_alternating(32 * 64, 1.0f, -1.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);
    const float eps = 1e-6f;

    auto stats = ttnn::layer_norm_pre_all_gather(x);  // bf16 default dtype
    ASSERT_EQ(stats.logical_shape()[-1], 64u);        // layernorm stats: 2 tiles wide
    auto out_two_stage = ttnn::layer_norm_post_all_gather(x, stats, eps);

    // Checked against the closed form only. A single-stage ttnn::layer_norm(x, eps) used to run
    // here as a second reference, but it adds the LayerNormDefaultProgramFactory kernels to this
    // test's JIT bill for a weaker check than the closed form -- and LayerNormInterleaved above
    // already covers that factory.
    // Closed form: each row is 32x(+1), 32x(-1) -> mean 0, var 1, out = x / sqrt(1 + eps) ~ +-1.
    // sum(x)=0 and sum(x^2)=64 are exact in bf16; the only error source is the SFPU rsqrt
    // approximation, hence atol 0.03 with rtol 0.
    detail::expect_close(detail::to_float_vector(out_two_stage), x_data, 0.0f, 0.03f);
}

TEST_F(NormalizationSmoke, DistributedRmsNormStats) {
    // RMSNormPreAllGatherProgramFactory stats layout, the RMS counterpart of
    // DistributedLayerNormStats. Per rmsnorm_pre_allgather.cpp the output is ONE tile wide
    // (vs layernorm's two) holding sum(x^2) in the left-most column; there is no sum(x) tile
    // because RMS needs no mean. Asserting it here also means DistributedRmsNormEndToEnd below
    // pays only for the post kernels -- declaration order decides which cell is charged for the
    // shared pre build, and splitting it keeps both under the 5 s per-test merge-gate ceiling.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::dist_norm_constant(32 * 64, 2.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);

    auto stats = ttnn::rms_norm_pre_all_gather(x);
    ASSERT_EQ(stats.logical_shape()[-1], 32u);  // one tile wide
    ASSERT_EQ(stats.logical_shape()[-2], 32u);
    const auto s = detail::to_float_vector(stats);

    // sum(x^2) = 64 * 2^2 = 256, exact in bf16 for any accumulation order (every partial sum is
    // a multiple of 4 bounded by 256, <= 8 significand bits).
    for (uint32_t r = 0; r < 32; ++r) {
        EXPECT_FLOAT_EQ(s[r * 32 + 0], 256.0f) << "sum(x^2), col 0, row " << r;
    }
    // Columns other than col 0 are reduce scratch (undefined) - deliberately unchecked.
}

TEST_F(NormalizationSmoke, DistributedRmsNormEndToEnd) {
    // RMS pre (rmsnorm_pre_allgather.cpp: one-tile stats = sum(x^2) in col 0) +
    // rmsnorm_post_allgather_metal2.cpp via LayerNormPostAllGatherProgramFactory, num_devices = 1.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::norm_alternating(32 * 64, 2.0f, -2.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);
    // TILE-layout weight [1,1,32,64], constant 3: validator wants padded height == tile height and
    // last dim == input's; constant fill makes the broadcast row irrelevant.
    const auto w_data = detail::dist_norm_constant(32 * 64, 3.0f);
    auto weight = detail::make_device_tensor(device, shape, w_data, DataType::BFLOAT16, Layout::TILE);
    const float eps = 1e-6f;

    auto stats = ttnn::rms_norm_pre_all_gather(x);
    ASSERT_EQ(stats.logical_shape()[-1], 32u);  // rmsnorm stats: 1 tile wide
    auto out_two_stage = ttnn::rms_norm_post_all_gather(x, stats, eps, weight);

    // Closed form only, for the reason given in DistributedLayerNormEndToEnd: the single-stage
    // ttnn::rms_norm reference that used to run here is covered by RmsNormInterleaved.
    // E(x^2) = 4 exactly (sum 256 / 64), rms = sqrt(4 + eps) ~ 2, out = 3 * x / 2 = +-3.
    // Tolerance covers the SFPU sqrt/recip approximation only.
    const auto v2 = detail::to_float_vector(out_two_stage);
    const auto expected = detail::norm_alternating(32 * 64, 3.0f, -3.0f);
    detail::expect_close(v2, expected, 0.02f, 0.05f, 0.999f);
}

// LayerNormPreAllGather2DProgramFactory alone (worker cores + merge-core column reduce).
// Split from DistributedRmsNorm2DGrid below so that cell pays only for the 2D post kernels;
// same declaration-order reasoning as DistributedRmsNormStats.
TEST_F(NormalizationSmoke, DistributedRmsNorm2DGridStats) {
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::dist_norm_constant(32 * 64, 2.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);

    auto stats = ttnn::rms_norm_pre_all_gather(
        x,
        DataType::BFLOAT16,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        /*use_2d_core_grid=*/true);
    ASSERT_EQ(stats.logical_shape()[-1], 32u);
    // The 2D merge-core reduce must land on the same sum(x^2) = 64 * 2^2 = 256 as the 1D path.
    const auto s = detail::to_float_vector(stats);
    for (uint32_t r = 0; r < 32; ++r) {
        EXPECT_FLOAT_EQ(s[r * 32 + 0], 256.0f) << "sum(x^2), col 0, row " << r;
    }
}

TEST_F(NormalizationSmoke, DistributedRmsNorm2DGrid) {
    // The 2D work split inside LayerNormPostAllGatherProgramFactory (use_2d_core_grid = true);
    // the 2D pre factory is asserted by DistributedRmsNorm2DGridStats above.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::norm_alternating(32 * 64, 2.0f, -2.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);
    const auto w_data = detail::dist_norm_constant(32 * 64, 3.0f);
    auto weight = detail::make_device_tensor(device, shape, w_data, DataType::BFLOAT16, Layout::TILE);
    const float eps = 1e-6f;

    auto stats = ttnn::rms_norm_pre_all_gather(
        x,
        DataType::BFLOAT16,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        /*use_2d_core_grid=*/true);
    ASSERT_EQ(stats.logical_shape()[-1], 32u);
    auto out = ttnn::rms_norm_post_all_gather(
        x,
        stats,
        eps,
        weight,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        /*use_2d_core_grid=*/true);

    // Same +-3 closed form as the 1D cell; tolerance rationale identical.
    const auto v = detail::to_float_vector(out);
    const auto expected = detail::norm_alternating(32 * 64, 3.0f, -3.0f);
    detail::expect_close(v, expected, 0.02f, 0.05f, 0.999f);
}

TEST_F(NormalizationSmoke, DistributedPreResidualFused) {
    // LayerNormPreAllGatherProgramFactory with FUSE_PRE_ADD: the residual b is added to a inside
    // the compute kernel (pre_add::one_row) before the stats pass. Reference: a second pre launch
    // on host-precomputed (x + res); both must agree exactly at the meaningful stats positions.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::norm_alternating(32 * 64, 1.0f, 3.0f);
    const auto res_data = detail::dist_norm_constant(32 * 64, 1.0f);
    std::vector<float> sum_data(32 * 64);
    for (size_t i = 0; i < sum_data.size(); ++i) {
        sum_data[i] = x_data[i] + res_data[i];  // alternating 2, 4 - exact in bf16
    }
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);
    auto res = detail::make_device_tensor(device, shape, res_data, DataType::BFLOAT16, Layout::TILE);
    auto xr = detail::make_device_tensor(device, shape, sum_data, DataType::BFLOAT16, Layout::TILE);

    auto stats_fused = ttnn::layer_norm_pre_all_gather(x, DataType::BFLOAT16, res);
    auto stats_ref = ttnn::layer_norm_pre_all_gather(xr);

    const auto sf = detail::to_float_vector(stats_fused);
    const auto sr = detail::to_float_vector(stats_ref);
    // Exact in bf16: every partial subset sum of {4,16} (resp. {2,4}) is a multiple of 4 (2)
    // bounded by 640 (192), all within 8 significand bits, so accumulation order cannot round.
    for (uint32_t r = 0; r < 32; ++r) {
        EXPECT_FLOAT_EQ(sf[r * 64 + 0], sr[r * 64 + 0]) << "sum((x+res)^2) mismatch, row " << r;
        EXPECT_FLOAT_EQ(sf[r * 64 + 32], sr[r * 64 + 32]) << "sum(x+res) mismatch, row " << r;
        EXPECT_FLOAT_EQ(sf[r * 64 + 0], 640.0f) << "row " << r;   // 32*(2^2) + 32*(4^2)
        EXPECT_FLOAT_EQ(sf[r * 64 + 32], 192.0f) << "row " << r;  // 32*2 + 32*4
    }
    // Only col 0 of each stats tile is meaningful; remaining columns are reduce scratch.
}

TEST_F(NormalizationSmoke, DistributedPostGammaBeta) {
    // LayerNormPostAllGatherProgramFactory with FUSE_GAMMA / FUSE_BETA defines. The post validator
    // accepts TILE-layout gamma/beta with padded height == tile height (32) and last dim equal to
    // the input's (layernorm_post_all_gather_device_operation.cpp); constant fill sidesteps which
    // broadcast row the kernel reads. RM gamma/beta (tile_width sticks) also validates - not
    // exercised here.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::norm_alternating(32 * 64, 1.0f, -1.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);
    const auto g_data = detail::dist_norm_constant(32 * 64, 2.0f);
    const auto b_data = detail::dist_norm_constant(32 * 64, 1.0f);
    auto gamma = detail::make_device_tensor(device, shape, g_data, DataType::BFLOAT16, Layout::TILE);
    auto beta = detail::make_device_tensor(device, shape, b_data, DataType::BFLOAT16, Layout::TILE);
    const float eps = 1e-6f;

    auto stats = ttnn::layer_norm_pre_all_gather(x);
    auto out = ttnn::layer_norm_post_all_gather(x, stats, eps, gamma, beta);

    // Hand-checkable golden: mean 0, var 1 (both exact in bf16), xhat ~ x, out = 2*x + 1 in {3, -1}.
    // atol 0.06 = the +-1 rsqrt-approximation budget from the end-to-end cell scaled by gamma = 2.
    const auto v = detail::to_float_vector(out);
    const auto expected = detail::norm_alternating(32 * 64, 3.0f, -1.0f);
    detail::expect_close(v, expected, 0.0f, 0.06f);
}

TEST_F(NormalizationSmoke, DISABLED_DistributedLayerNormPostWelford) {
    // KNOWN BUG (disabled; run with --gtest_also_run_disabled_tests): #51231 --
    // LayerNormPostAllGatherWelfordProgramFactory returns garbage (measured -64512 where the
    // golden is -1, every element, BH p100a 2026-08-21). use_welford=true on the post stage,
    // fed by the plain pre stage (num_devices = 1); same +-1 golden as the end-to-end cell.
    // Enable when #51231 closes. The pre-all-gather Welford factory stays untested: same
    // issue, hang class.
    auto& device = *device_;
    const ttnn::Shape shape({1, 1, 32, 64});
    const auto x_data = detail::norm_alternating(32 * 64, 1.0f, -1.0f);
    auto x = detail::make_device_tensor(device, shape, x_data, DataType::BFLOAT16, Layout::TILE);

    auto stats = ttnn::layer_norm_pre_all_gather(x);
    const ttnn::prim::LayerNormProgramConfig cfg = ttnn::prim::LayerNormDefaultProgramConfig{.use_welford = true};
    auto out =
        ttnn::layer_norm_post_all_gather(x, stats, 1e-6f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, cfg);
    detail::expect_close(detail::to_float_vector(out), x_data, 0.0f, 0.03f);
}

// ---------------------------------------------------------------------------
// GROUPNORM cells
// Entry point: ttnn::group_norm (groupnorm/groupnorm.hpp) -- note num_groups
// comes BEFORE epsilon. Factory selection (device/groupnorm_device_operation.cpp):
// sharded -> GroupNormShardedProgramFactory; else num_virtual_rows =
// (grid_x / num_virtual_cols) * grid_y vs batch: batch >= nvr -> NoMcast, else Mcast.
// Memory interpretation is [N, 1, H*W, C]: channels last, data[row * C + ch].
// ---------------------------------------------------------------------------

namespace detail {

// Shared closed-form group-norm golden, eps fixed at 1e-5 across all groupnorm cells.
constexpr float kGnEps = 1e-5f;

// Input for the reusable groupnorm golden. Memory interpretation: [N,1,H*W,C] puts channels in
// the LAST dim, so "channel ch is constant down all rows" means data[row * 64 + ch].
// two-group variant (num_groups=2, 32 ch/group):
//   ch  0..31: 1 if ch even, 3 if ch odd -> group mean 2, biased var 1 -> out -/+ 1/sqrt(1+eps)
//   ch 32..63: constant 7                -> var 0                      -> out 0
// four-group variant (num_groups=4, 16 ch/group) keeps the SAME expected output but gives every
// group distinct statistics so wrong group boundaries are detected, not masked:
//   g0 (0..15): 1/3 (mean 2, var 1),  g1 (16..31): 5/7 (mean 6, var 1),
//   g2 (32..47): const 7,             g3 (48..63): const 9.
// All values are small integers -> exact in bf16.
inline std::vector<float> gn_golden_input(size_t rows, bool four_groups) {
    std::vector<float> x(rows * 64);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t ch = 0; ch < 64; ++ch) {
            float v = 0.0f;
            if (ch < 32) {
                const float base = (four_groups && ch >= 16) ? 6.0f : 2.0f;
                v = base + ((ch % 2 == 0) ? -1.0f : 1.0f);
            } else {
                v = (four_groups && ch >= 48) ? 9.0f : 7.0f;
            }
            x[r * 64 + ch] = v;
        }
    }
    return x;
}

// Expected output for BOTH variants above: ch<32 -> -/+ gamma/sqrt(1+eps) + beta, ch>=32 -> beta
// (zero-variance groups normalize to exactly 0: mean of a constant bf16 group is exact, so
// x - mean == 0 regardless of the rsqrt(eps) approximation).
inline std::vector<float> gn_golden_expected(size_t rows, float gamma, float beta) {
    const float inv = 1.0f / std::sqrt(1.0f + kGnEps);
    std::vector<float> e(rows * 64);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t ch = 0; ch < 64; ++ch) {
            e[r * 64 + ch] = (ch < 32) ? (((ch % 2 == 0) ? -inv : inv) * gamma + beta) : beta;
        }
    }
    return e;
}

// Host-side port of ttnn.create_group_norm_weight_bias_rm (python ttnn/operations/normalization.py)
// for num_cores_x=1 and C=64 (already a multiple of 32): the per-channel vector reshaped to
// [1, 1, C/32, 32] ROW_MAJOR. The groupnorm validation requires RM gamma/beta with dim3 == tile
// width (groupnorm_device_operation.cpp). Constant-valued here, so chunk order is moot.
inline Tensor gn_make_gamma_beta(tt::tt_metal::distributed::MeshDevice& device, float value) {
    std::vector<float> v(64, value);
    return make_device_tensor(device, ttnn::Shape({1, 1, 2, 32}), v, DataType::BFLOAT16, Layout::ROW_MAJOR);
}

// Port of create_group_norm_input_mask (groupnorm_input_mask.cpp) for C=64, num_groups=4,
// num_cores_across_channel=1: block_wt = worst-case tile span of a 16-channel group = 1, so the
// mask is [1, 4, 32, 32] TILE with group g selecting columns [(g*16)%32, (g*16)%32 + 16).
// (The reference start-stride recurrence reduces to (g*group_w) % 32 for group_w=16.)
inline Tensor gn_make_input_mask_c64_g4(tt::tt_metal::distributed::MeshDevice& device) {
    constexpr size_t kGroups = 4, kTile = 32, kGroupW = 16;
    std::vector<float> m(kGroups * kTile * kTile, 0.0f);
    for (size_t g = 0; g < kGroups; ++g) {
        const size_t start = (g * kGroupW) % kTile;
        for (size_t h = 0; h < kTile; ++h) {
            for (size_t w = start; w < start + kGroupW; ++w) {
                m[(g * kTile + h) * kTile + w] = 1.0f;
            }
        }
    }
    return make_device_tensor(device, ttnn::Shape({1, 4, 32, 32}), m, DataType::BFLOAT16, Layout::TILE);
}

}  // namespace detail

TEST_F(NormalizationSmoke, GroupNormNoMcastInterleaved) {
    // GroupNormNoMcastProgramFactory, legacy two-pass compute kernel (kernels/compute/groupnorm.cpp):
    // interleaved input, grid {1,1} -> num_virtual_cols=1, num_virtual_rows=1, batch=1 >= 1
    // (groupnorm_device_operation.cpp). TILE layout: interleaved RM is Wormhole-only (#52279).
    auto& device = *device_;
    auto input = detail::make_device_tensor(
        device, ttnn::Shape({1, 1, 32, 64}), detail::gn_golden_input(32, false), DataType::BFLOAT16, Layout::TILE);
    auto out = ttnn::group_norm(
        input,
        /*num_groups=*/2,
        detail::kGnEps,
        /*input_mask=*/std::nullopt,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        ttnn::CoreGrid(1, 1));
    // atol 0.03: mean/var accumulate 32x32 elements through bf16 intermediate CBs.
    detail::expect_close(detail::to_float_vector(out), detail::gn_golden_expected(32, 1.0f, 0.0f), 0.0f, 0.03f);
}

TEST_F(NormalizationSmoke, GroupNormMcastInterleaved) {
    // GroupNormMcastProgramFactory: grid {1,2} -> num_virtual_rows=2 > batch=1
    // (groupnorm_device_operation.cpp), so two cores split one batch's 64 rows and multicast
    // partial mean/variance. C is kept at ONE tile (32 channels, one group) deliberately:
    // interleaved TILE groupnorm returns wrong values in the second channel tile whenever
    // H*W spans more than one row tile -- bug #53846, repro in
    // DISABLED_GroupNormMultiRowTileWideC below.
    // Data alternates 1/3 by channel -> mean 2, biased var 1 -> out = -/+ 1/sqrt(1+eps).
    auto& device = *device_;
    constexpr size_t kRows = 64, kC = 32;
    std::vector<float> data(kRows * kC);
    std::vector<float> expected(kRows * kC);
    const float inv = 1.0f / std::sqrt(1.0f + detail::kGnEps);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (i % 2 == 0) ? 1.0f : 3.0f;
        expected[i] = (i % 2 == 0) ? -inv : inv;
    }
    auto input =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, kRows, kC}), data, DataType::BFLOAT16, Layout::TILE);
    auto out = ttnn::group_norm(
        input,
        /*num_groups=*/1,
        detail::kGnEps,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        ttnn::CoreGrid(1, 2));
    detail::expect_close(detail::to_float_vector(out), expected, 0.0f, 0.03f);
}

TEST_F(NormalizationSmoke, DISABLED_GroupNormMultiRowTileWideC) {
    // KNOWN BUG #53846 (disabled; run with --gtest_also_run_disabled_tests): the legacy
    // two-pass statistics path computes an inexact group mean once a group spans more than one
    // tile, so a group whose variance is at or below eps normalizes to about +-1 instead of 0
    // (the division amplifies the mean error by 1/sqrt(eps)).
    // Repro confirmed against a torch reference via python ttnn on BH p100a (2026-08-19):
    // with [1,1,64,64], num_groups=2, channels 32..63 constant 7 (their group golden is exactly
    // 0), the device returns a constant -0.992 for every element of channels 32..63 on every
    // row, on NoMcast grid {1,1} and Mcast grid {1,2} alike.
    // The identical 32-row input (single row tile) passes -- see GroupNormNoMcastInterleaved --
    // and #53846 records that use_welford=true is bit-exact at every tile count tested, which is
    // why GroupNormWelfordInterleaved below stays green. Nightly's interleaved groupnorm coverage
    // is ROW_MAJOR-input only, so no python test reaches this path.
    // Enable this cell when #53846 is fixed.
    auto& device = *device_;
    auto input = detail::make_device_tensor(
        device, ttnn::Shape({1, 1, 64, 64}), detail::gn_golden_input(64, false), DataType::BFLOAT16, Layout::TILE);
    auto out = ttnn::group_norm(
        input,
        /*num_groups=*/2,
        detail::kGnEps,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        ttnn::CoreGrid(1, 1));
    detail::expect_close(detail::to_float_vector(out), detail::gn_golden_expected(64, 1.0f, 0.0f), 0.0f, 0.03f);
}

TEST_F(NormalizationSmoke, GroupNormShardedBlock1x1) {
    // GroupNormShardedProgramFactory (kernels/compute/groupnorm_sharded_v2.cpp): L1 BLOCK-sharded
    // ROW_MAJOR input on a 1x1 grid (in-kernel tilize/untilize; output layout defaults to the
    // input's RM). Grid must match the shard bounding box (groupnorm_sharded_program_factory.cpp).
    auto& device = *device_;
    const CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    const MemoryConfig sharded_cfg(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(grid, {32, 64}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    auto input = detail::make_device_tensor_mc(
        device,
        ttnn::Shape({1, 1, 32, 64}),
        detail::gn_golden_input(32, false),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        sharded_cfg);
    auto out = ttnn::group_norm(
        input,
        /*num_groups=*/2,
        detail::kGnEps,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        sharded_cfg,
        std::nullopt,
        ttnn::CoreGrid(1, 1));
    detail::expect_close(detail::to_float_vector(out), detail::gn_golden_expected(32, 1.0f, 0.0f), 0.0f, 0.03f);
}

// Same sharded factory and shard spec with gamma/beta applied (split out for the JIT-cost reason
// given on LayerNormInterleaved). gamma=2 / beta=1: the zero-variance group lands on exactly
// 0*2 + 1 = 1.
TEST_F(NormalizationSmoke, GroupNormShardedBlock1x1GammaBeta) {
    auto& device = *device_;
    const CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    const MemoryConfig sharded_cfg(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(grid, {32, 64}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    auto input = detail::make_device_tensor_mc(
        device,
        ttnn::Shape({1, 1, 32, 64}),
        detail::gn_golden_input(32, false),
        DataType::BFLOAT16,
        Layout::ROW_MAJOR,
        sharded_cfg);
    auto gamma = detail::gn_make_gamma_beta(device, 2.0f);
    auto beta = detail::gn_make_gamma_beta(device, 1.0f);
    auto out_gb = ttnn::group_norm(
        input,
        /*num_groups=*/2,
        detail::kGnEps,
        std::nullopt,
        gamma,
        beta,
        sharded_cfg,
        std::nullopt,
        ttnn::CoreGrid(1, 1));
    detail::expect_close(detail::to_float_vector(out_gb), detail::gn_golden_expected(32, 2.0f, 1.0f), 0.0f, 0.03f);
}

TEST_F(NormalizationSmoke, GroupNormWelfordInterleaved) {
    // GroupNormNoMcastProgramFactory with use_welford=true (kernels/compute/welford_groupnorm.cpp).
    // Welford interleaved requires TILE in/out (groupnorm_device_operation.cpp)
    auto& device = *device_;
    auto input = detail::make_device_tensor(
        device, ttnn::Shape({1, 1, 32, 64}), detail::gn_golden_input(32, false), DataType::BFLOAT16, Layout::TILE);
    auto out = ttnn::group_norm(
        input,
        /*num_groups=*/2,
        detail::kGnEps,
        /*input_mask=*/std::nullopt,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt,
        ttnn::CoreGrid(1, 1),
        /*inplace=*/std::nullopt,
        /*output_layout=*/std::nullopt,
        /*num_out_blocks=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt,
        /*negative_mask=*/std::nullopt,
        /*use_welford=*/true);
    detail::expect_close(detail::to_float_vector(out), detail::gn_golden_expected(32, 1.0f, 0.0f), 0.0f, 0.03f);
}

TEST_F(NormalizationSmoke, DISABLED_GroupNormInputMask) {
    // KNOWN BUG #53803 (disabled; run with --gtest_also_run_disabled_tests): the legacy two-pass
    // statistics path corrupts sub-tile group widths, i.e. whenever groups are narrower than
    // TILE_WIDTH so input_mask performs column selection inside a tile -- escape class #43826 /
    // #51231-7, now confirmed on silicon. GroupNormNoMcastProgramFactory with num_groups=4 on
    // C=64: 16 channels/group. On BH p100a (2026-08-19) the constant groups g2/g3 (goldens
    // exactly 0) come back at -0.984/-0.992 and the varying groups' means are visibly biased
    // (-1.008/0.977, -1.023/0.961 instead of -/+1). Confirmed independently via python ttnn with
    // the wrapper's auto-generated mask -- the hand-built mask below is element-identical to
    // create_group_norm_input_mask(64, 4, 1), so the bug is in the kernel's masked-stats path,
    // not the mask; #53803 confirms use_welford=true computes the same cases exactly, which
    // localizes it to the two-pass reduction rather than to addressing or masking. #53803 also
    // measures the error growing with group count, up to 1.73 absolute on a normalized output of
    // true magnitude 1.0 at an SD-UNet-like [1,1,1024,320] / 32-group shape.
    // Enable this cell when #53803 is fixed.
    // Groups get distinct stats (means 2/6/7/9) so a wrong group boundary shifts a mean and
    // fails the closed-form golden.
    auto& device = *device_;
    auto input = detail::make_device_tensor(
        device, ttnn::Shape({1, 1, 32, 64}), detail::gn_golden_input(32, true), DataType::BFLOAT16, Layout::TILE);
    auto mask = detail::gn_make_input_mask_c64_g4(device);
    auto out = ttnn::group_norm(
        input,
        /*num_groups=*/4,
        detail::kGnEps,
        mask,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        std::nullopt,
        std::nullopt,
        ttnn::CoreGrid(1, 1));
    // Same expected vector as the two-group golden by construction (each varying group has
    // mean +/-1 pattern with var 1; constant groups normalize to 0).
    detail::expect_close(detail::to_float_vector(out), detail::gn_golden_expected(32, 1.0f, 0.0f), 0.0f, 0.03f);
}

// ---------------------------------------------------------------------------
// BATCH_NORM cells
// Op under test: ttnn::batch_norm (batch_norm/batch_norm.hpp)
//   Tensor batch_norm(input, running_mean, running_var, training, eps, momentum, weight, bias,
//                     output, memory_config, compute_kernel_config)
// Validation (batch_norm_device_operation.cpp): input/stat/weight/bias/output must be rank-4 TILE
// on device, BFLOAT16 or FLOAT32, INTERLEAVED, with shape[1] == C; stats/weight/bias are logical
// [1, C, 1, 1] (padded to tiles). running_mean/running_var are REQUIRED in inference mode.
// Kernel choice (batch_norm_program_factory.cpp): use_sfpu_kernel = fp32_dest_acc_en || any_float32;
// the op's default compute config (batch_norm_utils.cpp) sets fp32_dest_acc_en = true, so the FPU
// kernel is reachable only with an explicit config + all-bf16 tensors.
// ---------------------------------------------------------------------------

// Covers: BatchNormOperation::BatchNormFactory with the FPU compute kernel
// (device/kernels/compute/batch_norm_kernel.cpp). Explicit fp32_dest_acc_en=false plus all-BFLOAT16
// tensors pins use_sfpu_kernel = fp32_dest_acc_en || any_float32 to false (default config would
// force the SFPU kernel).
TEST_F(NormalizationSmoke, BatchNormInferenceFpu) {
    auto& device = *device_;
    const ttnn::Shape x_shape({2, 1, 32, 32});
    const ttnn::Shape stat_shape({1, 1, 1, 1});  // logical [1, C, 1, 1] with C = 1; padded to one tile
    auto x = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 3.0f), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

    const DeviceComputeKernelConfig fpu_cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,  // pins the FPU kernel (no float32 tensors involved)
        .packer_l1_acc = false,
        .dst_full_sync_en = false};
    auto y = ttnn::batch_norm(
        x,
        running_mean,
        running_var,
        /*training=*/false,
        /*eps=*/0.0f,
        /*momentum=*/0.1f,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*output=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        fpu_cfg);

    // y = (3 - 1) / sqrt(4 + 0) = 1.0; every operand and intermediate (2, 4, 0.5, 1) is a small
    // integer or power of two, exact in bf16 -> exact golden.
    detail::expect_close(detail::to_float_vector(y), std::vector<float>(2 * 32 * 32, 1.0f), 0.0f, 0.0f);
}

// Covers: BatchNormOperation::BatchNormFactory with the SFPU compute kernel
// (device/kernels/compute/batch_norm_sfpu_kernel.cpp): both triggers at once -- fp32_dest_acc_en=true
// AND FLOAT32 tensors (any_float32). HiFi3 per Wormhole HW bug #38306 (HiFi4 + fp32 acc can be
// inaccurate; matches the op's own fp32-acc default). Also covers the weight/bias affine stage.
TEST_F(NormalizationSmoke, BatchNormInferenceSfpu) {
    auto& device = *device_;
    const ttnn::Shape x_shape({2, 1, 32, 32});
    const ttnn::Shape stat_shape({1, 1, 1, 1});
    auto x = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 3.0f), ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    // weight/bias share the params' dtype family (all params must have identical dtype).
    auto weight = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{3.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);
    auto bias = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{-1.0f}, ttnn::DataType::FLOAT32, ttnn::Layout::TILE);

    const DeviceComputeKernelConfig sfpu_cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,  // SFPU kernel; float32 tensors would force it regardless
        .packer_l1_acc = false,
        .dst_full_sync_en = false};
    auto y = ttnn::batch_norm(
        x,
        running_mean,
        running_var,
        /*training=*/false,
        /*eps=*/0.0f,
        /*momentum=*/0.1f,
        weight,
        bias,
        /*output=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        sfpu_cfg);

    // y = ((3 - 1) / sqrt(4)) * 3 + (-1) = 2.0 -- all steps exact in fp32.
    detail::expect_close(detail::to_float_vector(y), std::vector<float>(2 * 32 * 32, 2.0f), 0.0f, 0.0f);
}

// Covers: RunningStatistics::RunningStatisticsProgramFactory
// (running_statistics_program_factory.cpp) updating running_mean/running_var IN PLACE:
//   new = (1 - momentum) * old + momentum * batch.
// The prim is driven DIRECTLY rather than through ttnn::batch_norm(training=true). The training
// composite (batch_norm.cpp) reaches this prim only after mean_NHW -- two chained ttnn::mean
// reductions -- plus ttnn::subtract and ttnn::square, and those are reduction/eltwise factories
// owned by ReductionSmoke, not normalization factories. Routing through them cost 51 JIT kernel
// builds (13.6 s cold on BH p100a) against this file's 13 (1.7 s), for no normalization coverage
// this file does not already have: BatchNormOperation::BatchNormFactory is covered by the four
// inference cells above. Merge-gate rule is 5 s per test case (merge-gate.yaml), so the composite
// wiring -- that batch_norm(training=true) feeds mean_NHW's output into this prim -- is left to
// the post-merge suites, where the budget is 10-25x larger.
//
// bf16 with momentum 0.25: every term below is exact in bf16, so the readbacks are exact rather
// than tolerance-based. Values are chosen so both terms of both accumulators are distinct
// (1.5 != 8.0, and neither equals any input), which a swapped mean/var or a swapped
// momentum/(1-momentum) would break:
//   running_mean = 0.75 * 1 + 0.25 * 3 = 0.75 + 0.75 = 1.5
//   running_var  = 0.75 * 9 + 0.25 * 5 = 6.75 + 1.25 = 8.0
TEST_F(NormalizationSmoke, BatchNormRunningStatsUpdate) {
    auto& device = *device_;
    const ttnn::Shape stat_shape({1, 1, 1, 1});
    auto make = [&](float v) {
        return detail::make_device_tensor(
            device, stat_shape, std::vector<float>{v}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    };
    const auto batch_mean = make(3.0f);
    const auto batch_var = make(5.0f);
    auto running_mean = make(1.0f);
    auto running_var = make(9.0f);

    ttnn::prim::running_statistics(batch_mean, batch_var, /*momentum=*/0.25f, running_mean, running_var);

    // Read back from the SAME tensor handles the prim wrote in place.
    const auto rm = detail::to_float_vector(running_mean);
    const auto rv = detail::to_float_vector(running_var);
    ASSERT_EQ(rm.size(), 1u);
    ASSERT_EQ(rv.size(), 1u);
    EXPECT_FLOAT_EQ(rm[0], 1.5f);
    EXPECT_FLOAT_EQ(rv[0], 8.0f);
}

// Covers: per-channel parameter indexing in BatchNormOperation::BatchNormFactory -- C = 2, so
// mean/var/weight/bias each carry one tile per channel ([1, C, 1, 1] logical) and the reader must
// pair the right stat tile with each input channel. Distinct exact goldens per channel:
//   ch0: (5 - 4)/sqrt(1) * 1 + 0 = 1.0        ch1: (1 - 0)/sqrt(4) * 2 + 1 = 2.0
// Any cross-channel mix-up of mean/var/weight/bias produces a different (still exact) value.
TEST_F(NormalizationSmoke, BatchNormMultiChannel) {
    auto& device = *device_;
    const ttnn::Shape x_shape({1, 2, 32, 32});
    const ttnn::Shape stat_shape({1, 2, 1, 1});
    std::vector<float> x_data(2 * 32 * 32, 5.0f);             // channel 0 = 5.0
    std::fill(x_data.begin() + 32 * 32, x_data.end(), 1.0f);  // channel 1 = 1.0 (NCHW row-major)
    auto x = detail::make_device_tensor(device, x_shape, x_data, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f, 0.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f, 4.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto weight = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f, 2.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto bias = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{0.0f, 1.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

    auto y = ttnn::batch_norm(
        x, running_mean, running_var, /*training=*/false, /*eps=*/0.0f, /*momentum=*/0.1f, weight, bias);

    std::vector<float> expected(2 * 32 * 32, 1.0f);
    std::fill(expected.begin() + 32 * 32, expected.end(), 2.0f);
    detail::expect_close(detail::to_float_vector(y), expected, 0.0f, 0.0f);
}

// Covers: preallocated-output path (BatchNormOperation::create_output_tensors returns the provided
// tensor instead of allocating). Output must match input dtype; result is read back from the
// PREALLOCATED handle, proving the kernel wrote into the caller's buffer.
TEST_F(NormalizationSmoke, BatchNormPreallocatedOutput) {
    auto& device = *device_;
    const ttnn::Shape x_shape({2, 1, 32, 32});
    const ttnn::Shape stat_shape({1, 1, 1, 1});
    auto x = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 3.0f), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_mean = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{1.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto running_var = detail::make_device_tensor(
        device, stat_shape, std::vector<float>{4.0f}, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
    auto out = detail::make_device_tensor(
        device, x_shape, std::vector<float>(2 * 32 * 32, 0.0f), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

    auto ret = ttnn::batch_norm(
        x,
        running_mean,
        running_var,
        /*training=*/false,
        /*eps=*/0.0f,
        /*momentum=*/0.1f,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*output=*/out);

    // Same golden as BatchNormInferenceFpu: (3 - 1)/sqrt(4) = 1.0 exact; check the caller's tensor.
    const std::vector<float> expected(2 * 32 * 32, 1.0f);
    detail::expect_close(detail::to_float_vector(out), expected, 0.0f, 0.0f);
    detail::expect_close(detail::to_float_vector(ret), expected, 0.0f, 0.0f);
}

}  // namespace ttnn::operations::normalization::test
