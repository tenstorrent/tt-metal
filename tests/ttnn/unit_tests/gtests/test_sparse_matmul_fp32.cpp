// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

#include <tt-metalium/tile.hpp>

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

class SparseMatmulFp32Test : public TTNNFixtureWithDevice {};

TEST_F(SparseMatmulFp32Test, PartialReloadPreservesFp32) {
    constexpr std::size_t M = 32;
    constexpr std::size_t K = 128;
    constexpr std::size_t N = 32;

    constexpr std::size_t TILE_H = 32;
    constexpr std::size_t TILE_W = 32;

    // Four K blocks:
    // K / TILE_W / in0_block_w = 128 / 32 / 1 = 4.
    constexpr std::size_t IN0_BLOCK_W = 1;

    //
    // A is [1, 1, M, K].
    // B is [1, 1, K, N].
    //
    // We deliberately arrange the four 32-wide K blocks so that the
    // first block establishes a partial near 1.0 and later blocks add
    // small FP32-representable increments.
    //
    // If an FP32 partial is spilled to L1 and reloaded through a TF32
    // unpack path, those small increments can be lost at block
    // boundaries.
    //

    std::vector<float> a(M * K, 0.0f);
    std::vector<float> b(K * N, 0.0f);

    constexpr float large = 1.0f / 32.0f;
    constexpr float small = 1.0f / 131072.0f;

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t k = 0; k < 32; ++k) {
            a[m * K + k] = large;
        }

        for (std::size_t k = 32; k < K; ++k) {
            a[m * K + k] = small;
        }
    }

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t n = 0; n < N; ++n) {
            b[k * N + n] = 1.0f;
        }
    }

    // One active sparse group.
    std::vector<::bfloat16> sparsity_data(1, ::bfloat16(1.0f));

    auto* dev_ptr = device_;

    auto input_a = ttnn::Tensor::from_vector(
        a,
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, M, K}),
            tt::tt_metal::TensorLayout(ttnn::DataType::FLOAT32, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        dev_ptr);

    auto input_b = ttnn::Tensor::from_vector(
        b,
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, K, N}),
            tt::tt_metal::TensorLayout(ttnn::DataType::FLOAT32, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        dev_ptr);

    auto sparsity = ttnn::Tensor::from_vector(
        sparsity_data,
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, 1, 1}),
            tt::tt_metal::TensorLayout(
                ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR, ttnn::DRAM_MEMORY_CONFIG)),
        dev_ptr);

    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig program_config{
        .compute_with_storage_grid_size = CoreCoord(1, 1),
        .in0_block_w = IN0_BLOCK_W,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .out_block_h = 1,
        .out_block_w = 1,
        .per_core_M = 1,
        .per_core_N = 1,
        .fuse_batch = false,
        .fused_activation = std::nullopt,
        .mcast_in0 = true,
    };

    const tt::tt_metal::Tile output_tile({TILE_H, TILE_W});

    auto result = ttnn::sparse_matmul(
        input_a,
        input_b,
        sparsity,
        program_config,
        std::nullopt,  // nnz -- infer it
        false,         // is_input_a_sparse
        true,          // is_input_b_sparse
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::DataType::FLOAT32,
        std::nullopt,  // compute_kernel_config
        std::nullopt,  // core_grid
        output_tile,
        std::nullopt,   // optional_output_tensor
        std::nullopt,   // global_cb
        std::nullopt,   // sub_device_id
        std::nullopt);  // indices

    auto output_cpu = ttnn::from_device(result);
    auto output = output_cpu.to_vector<float>();

    ASSERT_GE(output.size(), M * N);

    //
    // CPU reference. Use double so our reference itself does not reproduce
    // the precision loss we're trying to detect.
    //
    double max_abs_error = -1.0;
    std::size_t worst_index = 0;
    double worst_expected = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        worst_expected += static_cast<double>(a[k]) * static_cast<double>(b[k * N]);
    }
    double worst_actual = static_cast<double>(output[0]);

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            double expected = 0.0;

            for (std::size_t k = 0; k < K; ++k) {
                expected += static_cast<double>(a[m * K + k]) * static_cast<double>(b[k * N + n]);
            }

            const std::size_t output_index = m * N + n;
            const double actual = static_cast<double>(output[output_index]);
            const double abs_error = std::abs(actual - expected);

            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
                worst_index = output_index;
                worst_expected = expected;
                worst_actual = actual;
            }
        }
    }

    std::cout << std::setprecision(17) << "\nSparse FP32 partial-reload precision:\n"
              << "  K blocks       : " << (K / TILE_W / IN0_BLOCK_W) << '\n'
              << "  max abs error  : " << max_abs_error << '\n'
              << "  worst index    : " << worst_index << '\n'
              << "  expected       : " << worst_expected << '\n'
              << "  actual         : " << worst_actual << '\n'
              << "  delta          : " << (worst_actual - worst_expected) << '\n';

    EXPECT_LT(max_abs_error, 1.0e-5) << "FP32 sparse matmul appears to lose precision across K-block "
                                        "partial spill/reload boundaries";
}

TEST_F(SparseMatmulFp32Test, Fp32PartialWithBfloat16OutputUsesCorrectIntermediateSize) {
    constexpr std::size_t M = 32;
    constexpr std::size_t K = 128;
    constexpr std::size_t N = 32;
    constexpr std::size_t IN0_BLOCK_W = 1;

    std::vector<float> a(M * K, 1.0f / 32.0f);
    std::vector<float> b(K * N, 1.0f);

    // One active sparse group.
    std::vector<::bfloat16> sparsity_data(1, ::bfloat16(1.0f));

    auto* dev_ptr = device_;

    auto input_a = ttnn::Tensor::from_vector(
        a,
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, M, K}),
            tt::tt_metal::TensorLayout(ttnn::DataType::FLOAT32, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        dev_ptr);

    auto input_b = ttnn::Tensor::from_vector(
        b,
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, K, N}),
            tt::tt_metal::TensorLayout(ttnn::DataType::FLOAT32, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        dev_ptr);

    auto sparsity = ttnn::Tensor::from_vector(
        sparsity_data,
        tt::tt_metal::TensorSpec(
            ttnn::Shape({1, 1, 1, 1}),
            tt::tt_metal::TensorLayout(
                ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR, ttnn::DRAM_MEMORY_CONFIG)),
        dev_ptr);

    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig program_config{
        .compute_with_storage_grid_size = CoreCoord(1, 1),
        .in0_block_w = IN0_BLOCK_W,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .out_block_h = 1,
        .out_block_w = 1,
        .per_core_M = 1,
        .per_core_N = 1,
        .fuse_batch = false,
        .fused_activation = std::nullopt,
        .mcast_in0 = true,
    };

    // Force FP32 destination accumulation while requesting BF16 output.
    // With four K blocks, intermediate partials spill/reload through c_5.
    const ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
    };

    const tt::tt_metal::Tile output_tile({32, 32});

    auto result = ttnn::sparse_matmul(
        input_a,
        input_b,
        sparsity,
        program_config,
        std::nullopt,  // nnz -- infer it
        false,         // is_input_a_sparse
        true,          // is_input_b_sparse
        ttnn::DRAM_MEMORY_CONFIG,
        ttnn::DataType::BFLOAT16,
        compute_config,
        std::nullopt,  // core_grid
        output_tile,
        std::nullopt,   // optional_output_tensor
        std::nullopt,   // global_cb
        std::nullopt,   // sub_device_id
        std::nullopt);  // indices

    auto output_cpu = ttnn::from_device(result);
    auto output = output_cpu.to_vector<::bfloat16>();

    ASSERT_GE(output.size(), M * N);

    // Each output element is 128 * (1/32) * 1 = 4, exactly
    // representable in BF16. This also catches gross corruption from an
    // undersized FP32 intermediate circular-buffer page.
    for (std::size_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(output[i]), 4.0f) << "Mismatch at output index " << i;
    }
}

}  // namespace ttnn::test
