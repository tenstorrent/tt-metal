// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Smoke tests for ttnn matmul: one test per program factory under
// ttnn/cpp/ttnn/operations/matmul/device/factory/ -- with one documented
// exclusion, listed below -- plus tests for the front-end code paths that pick
// between factories (auto-dispatch routing, bias fuse-vs-post-process,
// activation fusion, dtype/compute-config plumbing). Each test uses small
// deterministic inputs checked against a host fp32 reference (or a closed-form
// golden), so a kernel that runs but produces garbage fails.
//
// Each test states in a comment which program factory or code path it covers.
//
// Covered factories: matmul_multicore, matmul_multicore_reuse_optimized,
// matmul_multicore_reuse_mcast_1d, matmul_multicore_reuse_mcast_2d,
// matmul_multicore_reuse_mcast_dram_sharded.
//
// DELIBERATELY EXCLUDED: matmul_multicore_reuse_batched_hs_dram_sharded. It is
// never auto-selected (create_matmul_program_config excludes the config type,
// matmul_program_config.cpp), so only an explicit
// MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig reaches it, and
// its validation (validate_matmul_batched_dram_sharded_config,
// matmul_device_operation.cpp) demands A height-sharded in L1, B height-sharded
// across all DRAM banks, and a matching height-sharded output. The bank count is
// arch-specific, so the shapes cannot be a single set of constants shared by
// Wormhole and Blackhole -- more setup than a smoke test should carry. Coverage
// stays with tests/ttnn/unit_tests/operations/matmul/test_matmul_deepseek.py
// (post-merge, not the merge gate); this factory can still regress into main.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "common_test_utils.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::matmul::test {

class MatmulSmoke : public TTNNFixtureWithSuiteDevice<MatmulSmoke> {};

namespace detail {

inline MemoryConfig dram_interleaved() {
    return MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
}

template <typename T>
Tensor make_device_tensor_mc(
    tt::tt_metal::distributed::MeshDevice& device,
    const ttnn::Shape& shape,
    const std::vector<T>& data,
    DataType dtype,
    Layout layout,
    const MemoryConfig& mem_cfg) {
    const TensorLayout tensor_layout(dtype, PageConfig(layout), mem_cfg);
    const tt::tt_metal::TensorSpec tensor_spec(shape, tensor_layout);
    return Tensor::from_vector(data, tensor_spec).to_device(&device, mem_cfg, ttnn::QueueId(0));
}

template <typename T>
Tensor make_device_tensor(
    tt::tt_metal::distributed::MeshDevice& device,
    const ttnn::Shape& shape,
    const std::vector<T>& data,
    DataType dtype,
    Layout layout) {
    return make_device_tensor_mc(device, shape, data, dtype, layout, dram_interleaved());
}

// Device tensor -> host float vector. bf16 needs an explicit element-wise
// conversion; block-float and fp32 tensors decode directly via to_vector<float>.
inline std::vector<float> to_float_vector(const Tensor& t) {
    if (t.dtype() == DataType::BFLOAT16) {
        const auto v = t.to_vector<bfloat16>();
        std::vector<float> out(v.size());
        std::transform(v.begin(), v.end(), out.begin(), [](bfloat16 x) { return static_cast<float>(x); });
        return out;
    }
    return t.to_vector<float>();
}

// Tolerance-based comparison built on ttnn::test_utils: non-finite positions
// must match, every element must satisfy allclose(rtol, atol), and optionally
// PCC >= pcc_min and relative Frobenius error <= frob_max (pass -1 to skip).
inline void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float rtol,
    float atol,
    float pcc_min = -1.0f,
    float frob_max = -1.0f) {
    ASSERT_EQ(actual.size(), expected.size());
    const ttnn::test_utils::NonfiniteReport nf = ttnn::test_utils::check_nonfinite_positions(actual, expected);
    ASSERT_TRUE(nf.positions_match) << "non-finite mismatch at flat index " << nf.first_mismatch_index
                                    << ": device=" << nf.first_mismatch_actual << " ref=" << nf.first_mismatch_expected;
    const ttnn::test_utils::AllcloseReport report = ttnn::test_utils::allclose_report(actual, expected, rtol, atol);
    EXPECT_EQ(report.failures, 0u) << report.failures << " element(s) failed allclose(rtol=" << rtol
                                   << ", atol=" << atol << "); worst: flat index " << report.worst_margin_index
                                   << " device=" << report.worst_margin_actual
                                   << " ref=" << report.worst_margin_expected << " diff=" << report.worst_margin_diff
                                   << " tol=" << report.worst_margin_tol;
    if (nf.any_nonfinite) {
        return;  // pcc / relative_frobenius NaN-poison on non-finite inputs
    }
    if (pcc_min >= 0.0f) {
        const float p = ttnn::test_utils::pcc(actual, expected);
        EXPECT_GE(p, pcc_min);
    }
    if (frob_max >= 0.0f) {
        bool expected_norm_is_zero = false;
        const float f = ttnn::test_utils::relative_frobenius(actual, expected, expected_norm_is_zero);
        EXPECT_LE(f, frob_max) << (expected_norm_is_zero ? "absolute" : "relative") << " Frobenius error over limit";
    }
}

// Host fp32 reference: C = A * B per batch, row-major, b not transposed.
// i-k-j loop order for cache locality on the innermost j loop.
inline std::vector<float> cpu_matmul(
    const std::vector<float>& a, const std::vector<float>& b, int batch, int M, int K, int N) {
    std::vector<float> c(static_cast<size_t>(batch) * M * N, 0.0f);
    for (int bi = 0; bi < batch; ++bi) {
        const float* pa = a.data() + static_cast<size_t>(bi) * M * K;
        const float* pb = b.data() + static_cast<size_t>(bi) * K * N;
        float* pc = c.data() + static_cast<size_t>(bi) * M * N;
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                const float aik = pa[static_cast<size_t>(i) * K + k];
                for (int j = 0; j < N; ++j) {
                    pc[static_cast<size_t>(i) * N + j] += aik * pb[static_cast<size_t>(k) * N + j];
                }
            }
        }
    }
    return c;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Auto-dispatch defaults at single-tile and multi-tile scale. Integrated from
// the original test_matmul.cpp (SingleTileMatmul / MultiTileMatmul): ones x
// constant with an exact closed-form golden (every partial sum is a small
// multiple of 0.5 / 0.25, exact in bf16).
// ---------------------------------------------------------------------------
TEST_F(MatmulSmoke, SingleAndMultiTileOnes) {
    auto& device = *device_;
    struct Case {
        uint32_t hw;
        float b_fill;
    };
    for (const auto& c : {Case{32, 0.5f}, Case{64, 0.25f}}) {
        const ttnn::Shape shape({1, 1, c.hw, c.hw});
        const auto a = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto b = ttnn::full(shape, c.b_fill, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto out = matmul(a, b);
        // Each element = hw * (1.0 * b_fill) = 16.0 exactly.
        detail::expect_close(
            detail::to_float_vector(out),
            std::vector<float>(static_cast<size_t>(c.hw) * c.hw, 16.0f),
            /*rtol=*/0.0f,
            /*atol=*/0.0f);
    }
}

// ---------------------------------------------------------------------------
// Core factories and compute-config plumbing: MultiCore fallback, ReuseOptimized
// bmm, the auto-dispatch 2D mcast default, transpose, fp32 dest-acc grid, and
// dtype plumbing (bfp8 weights, output-dtype override).
// ---------------------------------------------------------------------------

namespace detail {

// Section-local helper for the Fp32DestAccConfigGrid cells: accumulation-order-insensitive expected
// values (double accumulation, single cast to float at the end).
inline std::vector<float> cpu_matmul_f64_acc(
    const std::vector<float>& a, const std::vector<float>& b, int M, int K, int N) {
    std::vector<float> c(static_cast<size_t>(M) * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(a[i * K + k]) * static_cast<double>(b[k * N + j]);
            }
            c[i * N + j] = static_cast<float>(acc);
        }
    }
    return c;
}

// Section-local helper for the Fp32DestAccConfigGrid cells that need a well-conditioned reference:
// k/64 for k in [-64, 63], which needs <= 7 significant bits and so is exact in fp32, bf16 and
// tf32 alike. No input quantization anywhere, no cancellation across K -- the only divergence
// left is dest-register accumulation, which is what the non-fp32 half of the grid measures.
// (detail::mm_rand_bf16 is the same idea but is declared further down, after this section.)
inline std::vector<float> grid_rand_exact(std::size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-64, 63);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = static_cast<float>(dist(gen)) * 0.015625f;  // 1/64: exact power-of-two scale
    }
    return v;
}

}  // namespace detail

// MatmulMultiCoreProgramFactory ("fallback of last resort"), pinned explicitly via
// MatmulMultiCoreProgramConfig{} to bypass auto-selection. Two shapes, replicating
// test_matmul_multicore.cpp's split_work_to_cores coverage at smoke scale:
//   * {512,512,512}: 16*16 = 256 output tiles -- divides the common 8x8 grid evenly, so
//     core_group_2 stays empty and the primary CreateKernel path runs alone.
//   * {2080,64,64}: 65*2 = 130 output tiles (2080 = 65*32 forces an odd Mt; 130 = 2*5*13)
//     cannot evenly divide grid sizes like 64 or 80, forcing a non-empty core_group_2 and the
//     second CreateKernel call with a different per-core tile count. (Caveat inherited from the
//     parent test: a 130-core grid, e.g. Blackhole 13x10, divides evenly.) K=64 keeps the CPU
//     reference cheap.
TEST_F(MatmulSmoke, MultiCoreExplicit) {
    auto& device = *device_;
    struct ShapeMKN {
        uint32_t M, K, N;
    };
    for (const auto& s : {ShapeMKN{512, 512, 512}, ShapeMKN{2080, 64, 64}}) {
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> a(static_cast<size_t>(s.M) * s.K);
        std::vector<float> b(static_cast<size_t>(s.K) * s.N);
        for (auto& v : a) {
            v = dist(rng);
        }
        for (auto& v : b) {
            v = dist(rng);
        }
        const auto ta =
            detail::make_device_tensor<float>(device, ttnn::Shape({s.M, s.K}), a, DataType::FLOAT32, Layout::TILE);
        const auto tb =
            detail::make_device_tensor<float>(device, ttnn::Shape({s.K, s.N}), b, DataType::FLOAT32, Layout::TILE);

        // HiFi3 + fp32_dest_acc_en to avoid known Wormhole HW bug #38306 (HiFi4 + fp32 acc).
        const ttnn::ComputeKernelConfig compute_cfg{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
            .math_approx_mode = false,
            .fp32_dest_acc_en = true,
        };
        const MatmulProgramConfig program_cfg = MatmulMultiCoreProgramConfig{};
        const auto out = ttnn::matmul(
            ta, tb, false, false, detail::dram_interleaved(), std::nullopt, program_cfg, std::nullopt, compute_cfg);

        // fp32 tolerances proven in the former test_matmul_multicore.cpp at K=2048 (rtol 0.1 /
        // atol ~0.11 / pcc 0.9999 / frob 0.002); N(0,1) inputs give smaller output magnitudes at
        // K <= 512, so the same bounds are conservative here.
        detail::expect_close(
            detail::to_float_vector(out), detail::cpu_matmul(a, b, 1, s.M, s.K, s.N), 0.1f, 0.11f, 0.9999f, 0.002f);
    }
}

// MatmulMultiCoreProgramFactory with logical dims not tile-aligned: [1,1,60,60] x [60,60]
// exercises the host-side tile padding path (K pads 60->64 with zeros so sums are unaffected;
// M/N padding is stripped by to_vector). Escape class: M-padding correctness (#26707).
// bf16 tolerance rationale (shared by all bf16 cells below): the reference operands are read
// back from the device tensors, so input quantization is excluded from the error budget; what
// remains is HiFi2 product truncation (~2^-8 relative) plus 16-bit dest accumulation over K
// partial sums, whose absolute error grows with K -- hence atol = max(1.0, 0.02*K); rtol 0.05,
// pcc 0.999 and frob 0.02 catch broad corruption.
TEST_F(MatmulSmoke, MultiCoreUnaligned) {
    auto& device = *device_;
    constexpr uint32_t M = 60, K = 60, N = 60;
    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(M * K), b(K * N);
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    const auto ta =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto tb = detail::make_device_tensor<float>(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    // Explicit program configs default the fidelity to LoFi; pass HiFi2 explicitly to match the
    // bf16 auto-path default so one tolerance rationale covers all bf16 cells.
    const ttnn::ComputeKernelConfig compute_cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
        .math_approx_mode = false,
    };
    const MatmulProgramConfig program_cfg = MatmulMultiCoreProgramConfig{};
    const auto out = ttnn::matmul(
        ta, tb, false, false, detail::dram_interleaved(), std::nullopt, program_cfg, std::nullopt, compute_cfg);

    const std::vector<float> a_q = detail::to_float_vector(ta);
    const std::vector<float> b_q = detail::to_float_vector(tb);
    // atol = max(1.0, 0.02*K) = 1.2 at K=60 (see rationale above).
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_q, b_q, 1, M, K, N), 0.05f, 1.2f, 0.999f, 0.02f);
}

// MatmulMultiCoreReuseOptimizedProgramFactory: batched bmm via explicit
// MatmulMultiCoreReuseProgramConfig. Both inputs batched (padded batch of B is 2), so
// get_broadcast_batch yields bcast_batch=false (non-bcast bmm path). per_core_M = Mt = 2 and
// per_core_N = Nt = 2, so each core owns whole [64,64] batch blocks:
// (B*Mt/per_core_M)*(Nt/per_core_N) = 2 output blocks over the 1x2 grid -> one batch per core.
// NOTE (#51550, missing-validation canary): per_core_M > Mt is NOT validated today and running
// it corrupts memory -- deliberately not exercised here; recorded as a gap only.
TEST_F(MatmulSmoke, ReuseBmm) {
    auto& device = *device_;
    constexpr uint32_t B = 2, M = 64, K = 64, N = 64;
    std::mt19937 rng(2);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(B * M * K), b(B * K * N);
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    const auto ta =
        detail::make_device_tensor<float>(device, ttnn::Shape({B, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto tb =
        detail::make_device_tensor<float>(device, ttnn::Shape({B, 1, K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const MatmulProgramConfig program_cfg = MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size = {1, 2},  // x=1, y=2: two cores, fits any device grid
        .in0_block_w = 2,                          // Kt = 2 -> single K block per batch
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .per_core_M = 2,  // = Mt (whole per-batch output block per core)
        .per_core_N = 2,  // = Nt
    };
    // HiFi2 to match the bf16 auto default (explicit configs otherwise downgrade to LoFi).
    const ttnn::ComputeKernelConfig compute_cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
        .math_approx_mode = false,
    };
    const auto out = ttnn::matmul(
        ta, tb, false, false, detail::dram_interleaved(), std::nullopt, program_cfg, std::nullopt, compute_cfg);

    const std::vector<float> a_q = detail::to_float_vector(ta);
    const std::vector<float> b_q = detail::to_float_vector(tb);
    const std::vector<float> result = detail::to_float_vector(out);
    for (uint32_t bi = 0; bi < B; ++bi) {
        const std::vector<float> a_b(a_q.begin() + bi * M * K, a_q.begin() + (bi + 1) * M * K);
        const std::vector<float> b_b(b_q.begin() + bi * K * N, b_q.begin() + (bi + 1) * K * N);
        const std::vector<float> r_b(result.begin() + bi * M * N, result.begin() + (bi + 1) * M * N);
        // atol = max(1.0, 0.02*K) = 1.28 at K=64 (rationale in MultiCoreUnaligned).
        detail::expect_close(r_b, detail::cpu_matmul(a_b, b_b, 1, M, K, N), 0.05f, 1.28f, 0.999f, 0.02f);
    }
}

// Batch-broadcast bmm, a=[2,1,64,64] x b=[1,1,64,64] (escape class #47640). Auto-dispatch:
// get_broadcast_batch (matmul_device_operation.cpp) sets bcast_batch = (padded batch of B == 1),
// and create_simple_matmul_program_config routes all-DRAM-interleaved non-narrow shapes to
// MatmulMultiCoreReuseMultiCastProgramConfig (2D mcast factory, fuse_batch=false batch loop).
// An explicit MatmulMultiCoreReuseProgramConfig would also accept this (bcast kept when
// batch_a > 1), but the auto path is what models actually hit.
TEST_F(MatmulSmoke, ReuseBmmBatchBroadcast) {
    auto& device = *device_;
    constexpr uint32_t B = 2, M = 64, K = 64, N = 64;
    std::mt19937 rng(3);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(B * M * K), b(K * N);
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    const auto ta =
        detail::make_device_tensor<float>(device, ttnn::Shape({B, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto tb =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const auto out = ttnn::matmul(ta, tb);  // pure auto-dispatch (defaults: HiFi2, packer_l1_acc)

    const std::vector<float> a_q = detail::to_float_vector(ta);
    const std::vector<float> b_q = detail::to_float_vector(tb);
    const std::vector<float> result = detail::to_float_vector(out);
    for (uint32_t bi = 0; bi < B; ++bi) {
        const std::vector<float> a_b(a_q.begin() + bi * M * K, a_q.begin() + (bi + 1) * M * K);
        const std::vector<float> r_b(result.begin() + bi * M * N, result.begin() + (bi + 1) * M * N);
        // Same b for every batch (the broadcast under test). atol = max(1.0, 0.02*64) = 1.28.
        detail::expect_close(r_b, detail::cpu_matmul(a_b, b_q, 1, M, K, N), 0.05f, 1.28f, 0.999f, 0.02f);
    }
}

// MatmulMultiCoreReuseMcast2DProgramFactory via THE default path: plain ttnn::matmul(a, b).
// All-DRAM-interleaved, tile-aligned, non-narrow [1,1,128,128] x [128,128] makes
// create_simple_matmul_program_config pick MatmulMultiCoreReuseMultiCastProgramConfig
// (use_mcast_2d_config = all_dram_interleaved). Auto defaults: HiFi2, packer_l1_acc=true.
TEST_F(MatmulSmoke, Auto2DMcastDefault) {
    auto& device = *device_;
    constexpr uint32_t M = 128, K = 128, N = 128;
    std::mt19937 rng(4);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(M * K), b(K * N);
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    const auto ta =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto tb = detail::make_device_tensor<float>(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const auto out = ttnn::matmul(ta, tb);

    const std::vector<float> a_q = detail::to_float_vector(ta);
    const std::vector<float> b_q = detail::to_float_vector(tb);
    // atol = max(1.0, 0.02*K) = 2.56 at K=128 (rationale in MultiCoreUnaligned).
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_q, b_q, 1, M, K, N), 0.05f, 2.56f, 0.999f, 0.02f);
}

// 2D mcast auto path with transpose_b=true: [1,1,64,96] x [64,96]^T -> [1,1,64,64] (K=96).
// The device transposes in1; the reference transposes host-side, so a mis-wired transpose flag
// produces a shape/value mismatch immediately.
TEST_F(MatmulSmoke, TransposeB) {
    auto& device = *device_;
    constexpr uint32_t M = 64, K = 96, N = 64;
    std::mt19937 rng(5);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(M * K), b(N * K);  // b is [N=64, K=96] pre-transpose
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    const auto ta =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto tb = detail::make_device_tensor<float>(device, ttnn::Shape({N, K}), b, DataType::BFLOAT16, Layout::TILE);

    const auto out = ttnn::matmul(ta, tb, /*transpose_a=*/false, /*transpose_b=*/true);

    const std::vector<float> a_q = detail::to_float_vector(ta);
    const std::vector<float> b_q = detail::to_float_vector(tb);
    std::vector<float> b_t(K * N);  // host-side transpose: b_t[k][j] = b[j][k]
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t k = 0; k < K; ++k) {
            b_t[k * N + j] = b_q[j * K + k];
        }
    }
    // atol = max(1.0, 0.02*K) = 1.92 at K=96 (rationale in MultiCoreUnaligned).
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_q, b_t, 1, M, K, N), 0.05f, 1.92f, 0.999f, 0.02f);
}

// fp32 inputs across the full {fp32_dest_acc_en} x {packer_l1_acc} compute-config grid on the
// auto 2D path -- 4 cells. (untilize_out would be a third axis, but it is a field of
// MatmulMultiCoreReuseMultiCast1DProgramConfig only, so the 2D path cannot set it.)
// Every cell gets a value check that a zeroed or corrupted output fails, but the two halves
// of the grid need different inputs to get one, because a 16-bit dest cannot hold the fp32
// cell's reference at all:
//
//   * fp32_dest_acc_en=true -> CANCELLING inputs, exact golden. Row pairs (+1024, -1023)
//     drive partial sums to ~1024 while the true row sum is only 32, so dest precision alone
//     decides the answer. The operands have <= 11-bit mantissas (exact in src regs at HiFi3)
//     and every partial sum is an integer <= 2^15, exactly representable in the fp32 dest, so
//     the device must return exactly 32; atol 0.02 only absorbs pack rounding. This is the
//     cell that would catch a silent loss of fp32 accumulation.
//
//   * fp32_dest_acc_en=false -> WELL-CONDITIONED inputs, ordinary tolerances. Feeding the
//     cancelling inputs here is a known precision cliff (ulp(1024) = 8 in bf16, so the
//     +-(1024,1023) operands quantize during unpack/accumulate; measured: uniform 128 on
//     Blackhole p100a). The only tolerance wide enough to accept that -- atol ~1024, the
//     partial-sum scale -- also accepts an all-zero output, so it verifies nothing beyond
//     "finite". Using detail::grid_rand_exact inputs instead keeps these two cells on the
//     same bf16-accumulation budget as the rest of the file (atol = max(1.0, 0.02*K) = 1.28
//     at K=64, rationale in MultiCoreUnaligned) plus pcc/frob, so they now verify the
//     arithmetic and not just the absence of a crash. Measured on Blackhole p100a, identical
//     for both packer_l1_acc values: worst |diff| 0.090 (atol 1.28), relative Frobenius
//     0.0043 (limit 0.02), pcc > 0.99999 (limit 0.999) -- so the tightest of the three bounds
//     still has ~4.7x headroom. An all-zero output, which the old atol 1024 accepted, now
//     fails on all three: output sd is ~2.7 so most elements exceed atol outright, and
//     relative Frobenius goes to 1.0 with pcc undefined.
//
// HiFi3 (not HiFi4) with fp32 acc: Wormhole HW bug #38306.
TEST_F(MatmulSmoke, Fp32DestAccConfigGrid) {
    auto& device = *device_;
    constexpr uint32_t M = 64, K = 64, N = 64;

    // Cancelling pair, for the fp32-dest cells: true row sum 32, partial sums ~1024.
    std::vector<float> a_cancel(M * K);
    const std::vector<float> b_cancel(K * N, 1.0f);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t k = 0; k < K; ++k) {
            a_cancel[i * K + k] = (k % 2 == 0) ? 1024.0f : -1023.0f;
        }
    }
    // Well-conditioned pair, for the 16-bit-dest cells.
    const std::vector<float> a_plain = detail::grid_rand_exact(static_cast<std::size_t>(M) * K, 801);
    const std::vector<float> b_plain = detail::grid_rand_exact(static_cast<std::size_t>(K) * N, 802);

    // Accumulation-order-insensitive fp64 references. The cancelling one is 32.0f everywhere
    // by construction; asserting that documents the intent and pins the input generator.
    const std::vector<float> expected_cancel = detail::cpu_matmul_f64_acc(a_cancel, b_cancel, M, K, N);
    const std::vector<float> expected_plain = detail::cpu_matmul_f64_acc(a_plain, b_plain, M, K, N);
    ASSERT_TRUE(std::all_of(expected_cancel.begin(), expected_cancel.end(), [](float v) { return v == 32.0f; }))
        << "cancelling input pair no longer sums to exactly 32";

    const auto ta_cancel =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, M, K}), a_cancel, DataType::FLOAT32, Layout::TILE);
    const auto tb_cancel =
        detail::make_device_tensor<float>(device, ttnn::Shape({K, N}), b_cancel, DataType::FLOAT32, Layout::TILE);
    const auto ta_plain =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, M, K}), a_plain, DataType::FLOAT32, Layout::TILE);
    const auto tb_plain =
        detail::make_device_tensor<float>(device, ttnn::Shape({K, N}), b_plain, DataType::FLOAT32, Layout::TILE);

    for (const bool fp32_acc : {false, true}) {
        for (const bool l1_acc : {false, true}) {
            SCOPED_TRACE("fp32_dest_acc_en=" + std::to_string(fp32_acc) + " packer_l1_acc=" + std::to_string(l1_acc));
            const ttnn::ComputeKernelConfig compute_cfg{
                .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
                .math_approx_mode = false,
                .fp32_dest_acc_en = fp32_acc,
                .packer_l1_acc = l1_acc,
            };
            const auto out = ttnn::matmul(
                fp32_acc ? ta_cancel : ta_plain,
                fp32_acc ? tb_cancel : tb_plain,
                false,
                false,
                detail::dram_interleaved(),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                compute_cfg);
            if (fp32_acc) {
                // pcc/frob skipped (defaults): expected is constant, so pcc's variance is zero.
                detail::expect_close(detail::to_float_vector(out), expected_cancel, /*rtol=*/0.0f, /*atol=*/0.02f);
            } else {
                detail::expect_close(detail::to_float_vector(out), expected_plain, 0.05f, 1.28f, 0.999f, 0.02f);
            }
        }
    }
}

// Mixed dtype, the model default: bf16 activations x BFLOAT8_B weights, [1,1,32,256] x [256,256].
// M = one tile, so is_narrow_shape routes the auto path to the 1D mcast factory
// (MatmulMultiCoreReuseMcast1DProgramFactory, mcast_in0). Weights are converted host-side by
// Tensor::from_vector<float> with a BFLOAT8_B/TILE spec (encode-then-to_dtype path in
// host_tensor_factory.cpp). The reference reads BOTH operands back from the device tensors --
// b via Tensor::to_vector<float>, which decodes the exact bfp8 bytes the device computes with --
// so bfp8 quantization (~2-3 decimal digits) is excluded from the error budget and the standard
// bf16 accumulation tolerances apply: atol = max(1.0, 0.02*K) = 5.12 at K=256.
TEST_F(MatmulSmoke, MixedDtypeBfp8Weights) {
    auto& device = *device_;
    constexpr uint32_t M = 32, K = 256, N = 256;
    std::mt19937 rng(6);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(M * K), b(K * N);
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    const auto ta =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto tb =
        detail::make_device_tensor<float>(device, ttnn::Shape({K, N}), b, DataType::BFLOAT8_B, Layout::TILE);

    const auto out = ttnn::matmul(ta, tb);

    const std::vector<float> a_q = detail::to_float_vector(ta);
    const std::vector<float> b_q = tb.to_vector<float>();  // decodes bfp8 (block floats need T=float)
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_q, b_q, 1, M, K, N), 0.05f, 5.12f, 0.999f, 0.02f);
}

// Output dtype override on the 2D mcast auto path: bf16 x bf16 with dtype=BFLOAT8_B output.
// Verifies the dtype plumbs through to the output spec and that values survive the pack.
// Tolerances looser than the bf16 cells: bfp8_b shares one 8-bit exponent across each 16-value
// block, so outputs co-blocked with a larger value lose mantissa bits -- quantization step is
// ~blockmax * 2^-7 (~0.2 for N(0,1) data at K=64, output sigma ~8). rtol 0.1 / atol 1.28 absorb
// that comfortably; pcc 0.995 / frob 0.05 are relaxed for the added quantization noise.
TEST_F(MatmulSmoke, OutputDtypeOverride) {
    auto& device = *device_;
    constexpr uint32_t M = 64, K = 64, N = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(M * K), b(K * N);
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : b) {
        v = dist(rng);
    }
    const auto ta =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto tb = detail::make_device_tensor<float>(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const auto out = ttnn::matmul(ta, tb, false, false, std::nullopt, /*dtype=*/DataType::BFLOAT8_B);

    ASSERT_EQ(out.dtype(), DataType::BFLOAT8_B);
    const std::vector<float> a_q = detail::to_float_vector(ta);
    const std::vector<float> b_q = detail::to_float_vector(tb);
    const std::vector<float> result = out.to_vector<float>();  // bfp8 output: decode via T=float
    detail::expect_close(result, detail::cpu_matmul(a_q, b_q, 1, M, K, N), 0.1f, 1.28f, 0.995f, 0.05f);
}

// ---------------------------------------------------------------------------
// 1D mcast factories, bias paths (kernel-fused vs post-processed), fused and
// post-op activations, and the multi-block-per-core 2D loop nest.
// ---------------------------------------------------------------------------

namespace detail {

// Deterministic bf16-exact random data: k/64 for k in [-64, 63] needs <= 7 significant
// bits, so every value round-trips bfloat16 exactly. Host reference and device inputs
// therefore see identical values; the only remaining divergence is device-side
// accumulation order / fidelity and SFPU activation approximation.
inline std::vector<float> mm_rand_bf16(std::size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-64, 63);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = static_cast<float>(dist(gen)) * 0.015625f;  // 1/64: exact power-of-two scale
    }
    return v;
}

// Host tanh-approximate GELU. Matches UnaryWithParam(GELU, 1.0f): param0 is consumed as
// the fast/approximate-mode flag (matmul_utilities.hpp, get_activation_params).
inline float cpu_gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

// Host SiLU reference: x * sigmoid(x).
inline float cpu_silu(float x) { return x / (1.0f + std::exp(-x)); }

}  // namespace detail

// Program factory: MatmulMultiCoreReuseMultiCast1D, mcast_in0 = true, AUTO-routed.
// [1,1,32,1024] x [1024,512], all DRAM interleaved, no config/core_grid, so
// create_simple_matmul_program_config picks the config (matmul_program_config.cpp):
//   height/width = a[-2]=32 vs b[-1]=512, ratio 16 > NARROW_SHAPE_RATIO_THRESHOLD=8
//   => is_narrow; all-interleaved && width > height => is_wide => use_mcast_1d_in0_config
//   => get_mcast_1d_config(..., true /* mcast_in0 */).
// The chosen factory cannot be introspected from the public API; this comment documents
// the routing rationale instead of asserting it.
TEST_F(MatmulSmoke, Mcast1DIn0Wide) {
    auto& device = *device_;
    constexpr int M = 32, K = 1024, N = 512;
    const auto a = detail::mm_rand_bf16(static_cast<std::size_t>(M) * K, 101);
    const auto b = detail::mm_rand_bf16(static_cast<std::size_t>(K) * N, 102);
    const auto a_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto b_dev = detail::make_device_tensor(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const auto out = matmul(a_dev, b_dev);

    const auto expected = detail::cpu_matmul(a, b, 1, M, K, N);
    // Random data + bf16 accumulation over K=1024 forces tolerance-based verification
    // (no closed-form golden possible): rtol 0.05, atol max(1.0, K*0.02), pcc 0.999.
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, std::max(1.0f, K * 0.02f), 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast1D, mcast_in0 = false, AUTO-routed.
// [1,1,1024,32] x [32,64], all DRAM interleaved: height/width = 1024 vs 64, ratio 16 > 8
// => is_narrow; width < height => is_tall => use_mcast_1d_in1_config
// => get_mcast_1d_config(..., false /* mcast_in0 */). Factory choice is not
// introspectable; comment documents the routing.
TEST_F(MatmulSmoke, Mcast1DIn1Tall) {
    auto& device = *device_;
    constexpr int M = 1024, K = 32, N = 64;
    const auto a = detail::mm_rand_bf16(static_cast<std::size_t>(M) * K, 201);
    const auto b = detail::mm_rand_bf16(static_cast<std::size_t>(K) * N, 202);
    const auto a_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto b_dev = detail::make_device_tensor(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const auto out = matmul(a_dev, b_dev);

    const auto expected = detail::cpu_matmul(a, b, 1, M, K, N);
    // Random data: tolerances per bf16 accumulation over K=32.
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, std::max(1.0f, K * 0.02f), 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast1D (explicit config), KERNEL-FUSED bias.
// Bias [1,256] in TILE pads to [32,256]: padded[-2] == tile height, and the config is not
// MatmulMultiCoreProgramConfig, so get_post_process_bias() returns false (matmul.cpp)
// and the bias is passed into prim::matmul. Config math: Mt=1, Kt=8, Nt=8 on grid 2x1 =>
// per_core_M=1 (mcast_in0 requires num_blocks_y==1), per_core_N=4 (2 blocks over 2
// cores), in0_block_w=2 divides Kt, out_block == per_core, subblock 1x4 (<=4 dest tiles,
// valid even with fp32 dest acc). Field order verified against
// matmul_program_config_types.hpp.
TEST_F(MatmulSmoke, Mcast1DExplicitFusedBias) {
    auto& device = *device_;
    constexpr int M = 32, K = 256, N = 256;
    const auto a = detail::mm_rand_bf16(static_cast<std::size_t>(M) * K, 301);
    const auto b = detail::mm_rand_bf16(static_cast<std::size_t>(K) * N, 302);
    const auto bias = detail::mm_rand_bf16(N, 303);
    const auto a_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto b_dev = detail::make_device_tensor(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);
    const auto bias_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, N}), bias, DataType::BFLOAT16, Layout::TILE);

    const MatmulMultiCoreReuseMultiCast1DProgramConfig cfg{
        .compute_with_storage_grid_size = {2, 1},
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 4,
        .out_block_h = 1,
        .out_block_w = 4,
        .per_core_M = 1,
        .per_core_N = 4,
        .fuse_batch = true,
        .fused_activation = std::nullopt,
        .mcast_in0 = true,
        // gather_in0=false, hop_cores={}, num_global_cb_receivers=0 (validated >0 only
        // when gather_in0), untilize_out=false.
    };
    const auto out = linear(a_dev, b_dev, bias_dev, false, false, std::nullopt, std::nullopt, cfg);

    auto expected = detail::cpu_matmul(a, b, 1, M, K, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            expected[static_cast<std::size_t>(i) * N + j] += bias[j];
        }
    }
    // Random data: tolerances per bf16 accumulation over K=256.
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, std::max(1.0f, K * 0.02f), 0.999f);
}

// Program factory: MatmulMultiCore (explicit MatmulMultiCoreProgramConfig{}) + bias =>
// POST-PROCESSED bias path. MultiCore never fuses bias: get_post_process_bias() returns
// true whenever the config holds MatmulMultiCoreProgramConfig (matmul.cpp), so the
// prim runs bias-less and the bias is applied afterwards via ttnn::add. Contrast with
// Mcast1DExplicitFusedBias above, where the same bias rides into the kernel. Same shapes
// as that cell so the two paths are comparable.
TEST_F(MatmulSmoke, MultiCorePostProcessedBias) {
    auto& device = *device_;
    constexpr int M = 32, K = 256, N = 256;
    const auto a = detail::mm_rand_bf16(static_cast<std::size_t>(M) * K, 401);
    const auto b = detail::mm_rand_bf16(static_cast<std::size_t>(K) * N, 402);
    const auto bias = detail::mm_rand_bf16(N, 403);
    const auto a_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto b_dev = detail::make_device_tensor(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);
    const auto bias_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, N}), bias, DataType::BFLOAT16, Layout::TILE);

    const auto out =
        linear(a_dev, b_dev, bias_dev, false, false, std::nullopt, std::nullopt, MatmulMultiCoreProgramConfig{});

    auto expected = detail::cpu_matmul(a, b, 1, M, K, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            expected[static_cast<std::size_t>(i) * N + j] += bias[j];
        }
    }
    // Random data: tolerances per bf16 accumulation over K=256 (+ one bf16 add).
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, std::max(1.0f, K * 0.02f), 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast (2D) with KERNEL-FUSED GELU -- encoder
// FFN archetype. UnaryWithParam(GELU, 1.0f) selects fast/approximate (tanh) mode: param0
// is consumed as the fast-mode flag (matmul_utilities.hpp); python passes the same as
// ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0) "# fast_and_approximate mode"
// (tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_activations.py), and the
// "gelu_approx" string maps to the identical param (unary_op_utils.cpp).
// Config math: Mt=2, Kt=8, Nt=8 on grid 2x2 => per_core_M=1/per_core_N=4 gives
// num_blocks_y=2<=grid.y, num_blocks_x=2<=grid.x.
TEST_F(MatmulSmoke, FusedActivationGelu2D) {
    auto& device = *device_;
    constexpr int M = 64, K = 256, N = 256;
    const auto a = detail::mm_rand_bf16(static_cast<std::size_t>(M) * K, 501);
    const auto b = detail::mm_rand_bf16(static_cast<std::size_t>(K) * N, 502);
    const auto a_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto b_dev = detail::make_device_tensor(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const MatmulMultiCoreReuseMultiCastProgramConfig cfg{
        .compute_with_storage_grid_size = {2, 2},
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 4,
        .out_block_h = 1,
        .out_block_w = 4,
        .per_core_M = 1,
        .per_core_N = 4,
        .transpose_mcast = false,
        .fused_activation = unary::UnaryWithParam(unary::UnaryOpType::GELU, 1.0f),
        // fuse_batch defaults to true (batch is 1 here anyway).
    };
    const auto out = matmul(a_dev, b_dev, false, false, std::nullopt, std::nullopt, cfg);

    auto expected = detail::cpu_matmul(a, b, 1, M, K, N);
    for (auto& x : expected) {
        x = detail::cpu_gelu_tanh(x);
    }
    // Random data + SFPU fast-gelu approximation: tolerance/pcc-based verification only.
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, std::max(1.0f, K * 0.02f), 0.999f);
}

// Code path: string activation. matmul(..., activation="silu") converts via
// string_to_unary_with_param ("silu" => UnaryWithParam(SILU), unary_op_utils.cpp)
// and -- with no sharded inputs and no user core_grid -- is applied as a POST-OP
// ttnn::unary_chain on the matmul output (matmul.cpp), not fused in the kernel.
TEST_F(MatmulSmoke, PostOpActivationSiluString) {
    auto& device = *device_;
    constexpr int M = 32, K = 64, N = 64;
    const auto a = detail::mm_rand_bf16(static_cast<std::size_t>(M) * K, 601);
    const auto b = detail::mm_rand_bf16(static_cast<std::size_t>(K) * N, 602);
    const auto a_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto b_dev = detail::make_device_tensor(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const auto out = matmul(a_dev, b_dev, false, false, std::nullopt, std::nullopt, std::nullopt, Activation("silu"));

    auto expected = detail::cpu_matmul(a, b, 1, M, K, N);
    for (auto& x : expected) {
        x = detail::cpu_silu(x);
    }
    // Random data + SFPU sigmoid approximation in silu: tolerance/pcc-based verification.
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, std::max(1.0f, K * 0.02f), 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast (2D), escape class -- multiple output
// blocks per core in BOTH dimensions (issues #50975 / #51550 family).
// Block math on [1,1,256,256] x [256,256], grid 2x2: Mt = Nt = Kt = 8.
//   per_core_M = per_core_N = 4  => num_blocks_y = num_blocks_x = 8/4 = 2 == grid dims.
//   out_block_h = out_block_w = 2 => per core: num_blocks_h = per_core_M/out_block_h = 2 > 1
//   AND num_blocks_w = per_core_N/out_block_w = 2 > 1 (the under-tested loop nest).
//   out_subblock 1x2 divides out_block 2x2, product 2 <= 4 dest tiles so valid even with
//   fp32 dest acc; in0_block_w = 2 divides Kt = 8.
TEST_F(MatmulSmoke, MultiBlockPerCore2D) {
    auto& device = *device_;
    constexpr int M = 256, K = 256, N = 256;
    const auto a = detail::mm_rand_bf16(static_cast<std::size_t>(M) * K, 701);
    const auto b = detail::mm_rand_bf16(static_cast<std::size_t>(K) * N, 702);
    const auto a_dev =
        detail::make_device_tensor(device, ttnn::Shape({1, 1, M, K}), a, DataType::BFLOAT16, Layout::TILE);
    const auto b_dev = detail::make_device_tensor(device, ttnn::Shape({K, N}), b, DataType::BFLOAT16, Layout::TILE);

    const MatmulMultiCoreReuseMultiCastProgramConfig cfg{
        .compute_with_storage_grid_size = {2, 2},
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 2,
        .out_block_h = 2,
        .out_block_w = 2,
        .per_core_M = 4,
        .per_core_N = 4,
        .transpose_mcast = false,
        .fused_activation = std::nullopt,
    };
    const auto out = matmul(a_dev, b_dev, false, false, std::nullopt, std::nullopt, cfg);

    const auto expected = detail::cpu_matmul(a, b, 1, M, K, N);
    // Random data: tolerances per bf16 accumulation over K=256.
    detail::expect_close(detail::to_float_vector(out), expected, 0.05f, std::max(1.0f, K * 0.02f), 0.999f);
}

// ---------------------------------------------------------------------------
// Sharded-input factories: DRAM-sharded decode projection, height/width/block
// sharded auto-dispatch (1D and 2D mcast, incl. transpose mcast).
// ---------------------------------------------------------------------------

namespace detail {

// Snap a float to bf16 precision (truncate to the top 16 bits of the fp32 encoding) so the
// fp32 CPU reference consumes exactly the values the device sees; remaining mismatch then
// comes only from device-side accumulation/fidelity.
inline float sharded_snap_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits &= 0xFFFF0000u;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Deterministic xorshift32 stream of bf16-exact values in [-1, 1).
inline std::vector<float> sharded_random_bf16(size_t n, uint32_t seed) {
    std::vector<float> out(n);
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        out[i] = sharded_snap_bf16(static_cast<float>(static_cast<int32_t>(s)) / 2147483648.0f);
    }
    return out;
}

// Port of models/tt_transformers/tt/model_config.py::find_largest_divisor (max divisor 8).
inline uint32_t sharded_largest_divisor(uint32_t n, uint32_t max_divisor = 8) {
    for (uint32_t i = max_divisor; i > 1; --i) {
        if (n % i == 0) {
            return i;
        }
    }
    return 1;
}

}  // namespace detail

// Program factory: MatmulMultiCoreReuseMultiCastDRAMSharded
// (device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp) -- THE model
// decode projection archetype (runs every token): in0 [1,1,32,K] bf16 WIDTH_SHARDED in L1,
// in1 [K,N] bf16 WIDTH_SHARDED across the DRAM banks, explicit DRAMSharded program config,
// output L1 WIDTH_SHARDED (shard spec computed by the op). Config math ports
// models/tt_transformers/tt/model_config.py::{create_dram_sharded_mem_config,
// dram_matmul_config}. Missing coverage (deliberate): dram-sharded + bias is known broken --
// models apply bias as a separate add (models/tt_transformers/tt/attention.py
// "FIXME: File bug against dram-sharded matmuls with bias") -- so bias is NOT tested here.
// Tiny tiles (tile_h < 16) are rejected on this path per #42927; this test uses 32x32 tiles.
TEST_F(MatmulSmoke, DramShardedDecodeProjection) {
    auto& device = *device_;
    constexpr uint32_t M = 32, K = 512, N = 256, kTile = 32;
    constexpr uint32_t num_in0_cores = 4;

    // in0: WIDTH_SHARDED in L1 over 4 cores -> shard [32, 128]; validator requires M == 1 tile
    // and ROW_MAJOR orientation (validate_matmul_dram_sharded_config).
    const MemoryConfig in0_mem_cfg(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_in0_cores - 1, 0))),
            {M, K / num_in0_cores},
            tt::tt_metal::ShardOrientation::ROW_MAJOR));

    // in1: WIDTH_SHARDED across the DRAM bank grid -- bank-count agnostic (12 banks on WH,
    // 8 on BH p150, 7 on p100). Per-bank width = ceil(N_tiles / num_banks) tiles; the factory
    // drops banks that hold only padding. Grid must be a single row (validator bbox check);
    // dram_grid_size().y == 1 on all current parts.
    const CoreCoord dram_grid = device.dram_grid_size();
    const uint32_t num_banks = static_cast<uint32_t>(dram_grid.x * dram_grid.y);
    ASSERT_GT(num_banks, 0u);
    const uint32_t per_bank_n = ((N / kTile + num_banks - 1) / num_banks) * kTile;
    const MemoryConfig in1_mem_cfg(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::DRAM,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(dram_grid.x - 1, dram_grid.y - 1))),
            {K, per_bank_n},
            tt::tt_metal::ShardOrientation::ROW_MAJOR));

    const auto a_data = detail::sharded_random_bf16(M * K, 0xA0010001u);
    const auto b_data = detail::sharded_random_bf16(K * N, 0xB0010002u);
    const auto a = detail::make_device_tensor_mc<float>(
        device, ttnn::Shape({1, 1, M, K}), a_data, DataType::BFLOAT16, Layout::TILE, in0_mem_cfg);
    const auto b = detail::make_device_tensor_mc<float>(
        device, ttnn::Shape({1, 1, K, N}), b_data, DataType::BFLOAT16, Layout::TILE, in1_mem_cfg);

    // dram_matmul_config math: in0_block_w = largest divisor (<= 8) of K_tiles / num_in0_cores;
    // per_core_M = M_tiles; per_core_N = ceil(N_tiles / num_in0_cores) (output storage width).
    const MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig program_config{
        .in0_block_w = detail::sharded_largest_divisor((K / kTile) / num_in0_cores),  // = 4
        .per_core_M = M / kTile,                                                      // = 1
        .per_core_N = (N / kTile + num_in0_cores - 1) / num_in0_cores,                // = 2
        .fused_activation = std::nullopt};
    // Output: sharded layout without an explicit shard spec -- compute_output_specs derives it
    // (mirrors ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG usage in the models).
    const MemoryConfig out_mem_cfg(tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1);

    const auto out = matmul(a, b, false, false, out_mem_cfg, std::nullopt, program_config);
    EXPECT_EQ(out.memory_config().memory_layout(), tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED);
    EXPECT_EQ(out.memory_config().buffer_type(), BufferType::L1);

    // Random bf16 inputs force tolerances: K=512-term bf16-product accumulation vs fp32
    // reference; PCC 0.999 is the sharp check, rtol/atol per bf16 tolerance policy.
    const float atol = K * 0.02f > 1.0f ? K * 0.02f : 1.0f;  // = 10.24
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_data, b_data, 1, M, K, N), 0.05f, atol, 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast1D, mcast_in0=false (in1 multicast) -- auto
// dispatch: HEIGHT_SHARDED L1 in0 + interleaved DRAM in1 routes through
// get_matmul_program_config (device/config/matmul_program_config.cpp) to the 1D config with
// mcast_in0=false, per_core_M = shard height (1 tile), in0_block_w = K.
TEST_F(MatmulSmoke, HeightSharded1DAuto) {
    auto& device = *device_;
    constexpr uint32_t M = 128, K = 64, N = 64;
    // 4-core column; ROW_MAJOR orientation walks it top-down, one [32, 64] shard per core.
    const MemoryConfig in0_mem_cfg(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 3))),
            {M / 4, K},
            tt::tt_metal::ShardOrientation::ROW_MAJOR));

    const auto a_data = detail::sharded_random_bf16(M * K, 0xA0020001u);
    const auto b_data = detail::sharded_random_bf16(K * N, 0xB0020002u);
    const auto a = detail::make_device_tensor_mc<float>(
        device, ttnn::Shape({1, 1, M, K}), a_data, DataType::BFLOAT16, Layout::TILE, in0_mem_cfg);
    const auto b =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, K, N}), b_data, DataType::BFLOAT16, Layout::TILE);

    // No program/memory config: output defaults to DRAM interleaved, avoiding the sharded-output
    // per_core_N constraints of the auto path.
    const auto out = matmul(a, b);

    // Random bf16 data vs fp32 reference; tolerance covers K=64 bf16 accumulation, PCC is sharp.
    const float atol = K * 0.02f > 1.0f ? K * 0.02f : 1.0f;  // = 1.28
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_data, b_data, 1, M, K, N), 0.05f, atol, 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast1D, mcast_in0=true (in0 gathered + multicast) --
// auto dispatch: WIDTH_SHARDED L1 in0 + interleaved DRAM in1 -> 1D config with mcast_in0=true,
// per_core_N = ceil(N_tiles / num_cores) = 1. Only 2 of the 4 shard cores produce output tiles;
// the factory keeps the remaining shard cores as in0 mcast senders without work (covered
// explicitly by the "cores_without_work_and_not_in_receiver_grid" paths of the 1D factory).
TEST_F(MatmulSmoke, WidthSharded1DAuto) {
    auto& device = *device_;
    constexpr uint32_t M = 32, K = 256, N = 64;
    const MemoryConfig in0_mem_cfg(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(3, 0))),
            {M, K / 4},
            tt::tt_metal::ShardOrientation::ROW_MAJOR));

    const auto a_data = detail::sharded_random_bf16(M * K, 0xA0030001u);
    const auto b_data = detail::sharded_random_bf16(K * N, 0xB0030002u);
    const auto a = detail::make_device_tensor_mc<float>(
        device, ttnn::Shape({1, 1, M, K}), a_data, DataType::BFLOAT16, Layout::TILE, in0_mem_cfg);
    const auto b =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, K, N}), b_data, DataType::BFLOAT16, Layout::TILE);

    const auto out = matmul(a, b);  // output defaults to DRAM interleaved

    const float atol = K * 0.02f > 1.0f ? K * 0.02f : 1.0f;  // = 5.12
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_data, b_data, 1, M, K, N), 0.05f, atol, 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast (2D mcast) -- auto dispatch: BLOCK_SHARDED L1
// in0 on a 2x2 grid with ROW_MAJOR orientation + interleaved DRAM in1 -> 2D config with
// transpose_mcast=false (per_core_M = per_core_N = in0_block_w = 1 tile; both grid axes match
// the M/K tiling exactly, so the auto path takes the no-padding branch).
TEST_F(MatmulSmoke, BlockSharded2DAuto) {
    auto& device = *device_;
    constexpr uint32_t M = 64, K = 64, N = 64;
    const MemoryConfig in0_mem_cfg(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 1))),
            {M / 2, K / 2},
            tt::tt_metal::ShardOrientation::ROW_MAJOR));

    const auto a_data = detail::sharded_random_bf16(M * K, 0xA0040001u);
    const auto b_data = detail::sharded_random_bf16(K * N, 0xB0040002u);
    const auto a = detail::make_device_tensor_mc<float>(
        device, ttnn::Shape({1, 1, M, K}), a_data, DataType::BFLOAT16, Layout::TILE, in0_mem_cfg);
    const auto b =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, K, N}), b_data, DataType::BFLOAT16, Layout::TILE);

    const auto out = matmul(a, b);  // output defaults to DRAM interleaved

    const float atol = K * 0.02f > 1.0f ? K * 0.02f : 1.0f;  // = 1.28
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_data, b_data, 1, M, K, N), 0.05f, atol, 0.999f);
}

// Program factory: MatmulMultiCoreReuseMultiCast (2D mcast), transpose_mcast=true -- auto
// dispatch: BLOCK_SHARDED in0 with COL_MAJOR shard orientation flips the mcast axes
// (get_matmul_program_config sets transpose_mcast from the orientation; the mcast2d validator
// requires COL_MAJOR in0 for transpose mcast and permits it here since on the square 2x2 grid
// both virtual axes still match the M/K tiling). in1 must stay interleaved: the validator
// rejects transpose mcast with sharded in1.
TEST_F(MatmulSmoke, BlockSharded2DTransposeAuto) {
    auto& device = *device_;
    constexpr uint32_t M = 64, K = 64, N = 64;
    const MemoryConfig in0_mem_cfg(
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 1))),
            {M / 2, K / 2},
            tt::tt_metal::ShardOrientation::COL_MAJOR));

    const auto a_data = detail::sharded_random_bf16(M * K, 0xA0050001u);
    const auto b_data = detail::sharded_random_bf16(K * N, 0xB0050002u);
    const auto a = detail::make_device_tensor_mc<float>(
        device, ttnn::Shape({1, 1, M, K}), a_data, DataType::BFLOAT16, Layout::TILE, in0_mem_cfg);
    const auto b =
        detail::make_device_tensor<float>(device, ttnn::Shape({1, 1, K, N}), b_data, DataType::BFLOAT16, Layout::TILE);

    const auto out = matmul(a, b);  // output defaults to DRAM interleaved

    const float atol = K * 0.02f > 1.0f ? K * 0.02f : 1.0f;  // = 1.28
    detail::expect_close(
        detail::to_float_vector(out), detail::cpu_matmul(a_data, b_data, 1, M, K, N), 0.05f, atol, 0.999f);
}

}  // namespace ttnn::operations::matmul::test
