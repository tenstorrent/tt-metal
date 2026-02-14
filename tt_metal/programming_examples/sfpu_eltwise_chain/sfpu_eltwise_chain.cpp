// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

/**
 * @brief Computes the softplus activation function element-wise on input vector
 *
 * The softplus function is defined as: softplus(x) = log(1 + exp(x))
 * This is a smooth approximation to the ReLU function that outputs positive values.
 *
 * @param src_vec Input vector containing bfloat16 values to apply softplus to
 * @param result_vec Output vector where computed softplus values will be stored
 *                   Must be the same size as src_vec
 *
 * @throws TT_FATAL if input and output vectors have different sizes
 *
 * @note This is a reference implementation used for validation/testing purposes
 * @note The function uses std::log1p for numerical stability
 */
void golden_softplus(const std::vector<bfloat16>& src_vec, std::vector<bfloat16>& result_vec) {
    TT_FATAL(src_vec.size() == result_vec.size(), "Input and output vectors must be the same size");
    for (size_t i = 0; i < src_vec.size(); ++i) {
        result_vec[i] = bfloat16(std::log1p(std::exp(static_cast<float>(src_vec[i]))));  // Softplus function
    }
}

/**
 * @brief Calculates the Pearson Correlation Coefficient (PCC) between two bfloat16 vectors.
 *
 * This function computes the linear correlation coefficient between two vectors of bfloat16 values,
 * which measures the strength and direction of the linear relationship between the two datasets.
 * The PCC value ranges from -1 to 1, where:
 * - 1 indicates a perfect positive linear relationship
 * - 0 indicates no linear relationship
 * - -1 indicates a perfect negative linear relationship
 *
 * @param vec_a First input vector of bfloat16 values
 * @param vec_b Second input vector of bfloat16 values (must be same size as vec_a)
 *
 * @return float The Pearson correlation coefficient between the two input vectors
 *
 * @note The function assumes both input vectors have the same size
 * @note bfloat16 values are converted to float for calculations to maintain precision
 */
inline float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
    // Calculate the mean of x and y values
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += static_cast<float>(vec_a[i]);
        y_mean += static_cast<float>(vec_b[i]);
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        float x_diff = static_cast<float>(vec_a[i]) - x_mean;
        float y_diff = static_cast<float>(vec_b[i]) - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= vec_a.size();
    x_stddev /= vec_a.size();
    y_stddev /= vec_b.size();

    // Calculate the correlation coefficient
    float correlation_coefficient = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient;
}

int main() {
    DeviceContext ctx(0);

    constexpr CoreCoord core = {0, 0};

    // Input data preparation — fill one tile with random values in [0, 1).
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);

    std::vector<bfloat16> src_vec(constants::TILE_HW);
    for (bfloat16& v : src_vec) {
        v = bfloat16(dist(rng));
    }

    // Calculate golden function results on CPU for later comparison.
    std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
    golden_softplus(src_vec, golden_vec);

    // Tilize the input vectors to match the expected tiled layout for the device.
    // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
    // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by
    // the device. This transformation groups elements into 32x32 blocks and reorders them in memory so
    // that each tile (32x32 elements) is stored contiguously. This matches the native data access
    // patterns of the matrix engine and enables efficient operations on the accelerator.
    src_vec = tilize_nfaces(src_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // DRAM buffer setup — allocate tile buffers for input and output.
    auto src_dram_buffer = ctx.dram_tile_buffer(1);
    auto dst_dram_buffer = ctx.dram_tile_buffer(1);

    // Upload data from host to device DRAM.
    ctx.write(src_dram_buffer, src_vec);

    // Build the program with 3 kernels forming a pipeline:
    //   Reader  → reads tile from DRAM into cb_0, also fills cb_1 with ones (for softplus computation)
    //   Compute → applies softplus (exp → add_ones → log) via SFPU, writes result to cb_2
    //   Writer  → writes result tile from cb_2 back to DRAM
    //
    // The CB indices are passed as compile-time args before the auto-generated TensorAccessorArgs.
    // The compute kernel receives CB indices but does NOT need TensorAccessorArgs since it only
    // operates on L1 circular buffers, not DRAM.
    constexpr uint32_t src_cb_index = CBIndex::c_0;
    constexpr uint32_t ones_cb_index = CBIndex::c_1;
    constexpr uint32_t result_cb_index = CBIndex::c_2;

    auto program =
        ProgramBuilder(core)
            .cb(tt::CBIndex::c_0, /*num_tiles=*/1)
            .cb(tt::CBIndex::c_1, /*num_tiles=*/1)
            .cb(tt::CBIndex::c_2, /*num_tiles=*/1)
            .reader(
                OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/reader.cpp",
                {src_dram_buffer},
                {src_cb_index, ones_cb_index})
            .runtime_args({src_dram_buffer->address()})
            .writer(
                OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/writer.cpp",
                {dst_dram_buffer},
                {result_cb_index})
            .runtime_args({dst_dram_buffer->address()})
            .compute(
                OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/compute/compute.cpp",
                MathFidelity::HiFi4,
                {src_cb_index, ones_cb_index, result_cb_index})
            .build();

    // Execute program and read result back to host.
    ctx.run(std::move(program));
    auto result_vec = ctx.read<bfloat16>(dst_dram_buffer);

    // Reverse the tilization to get the result in the row-major format that the CPU expects.
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector.
    // This is a measure of how similar the two vectors are.
    // A PCC close to 1 indicates that the two vectors are very similar.
    const float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);
}
