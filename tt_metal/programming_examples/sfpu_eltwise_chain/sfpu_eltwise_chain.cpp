// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>

#include <cmath>
#include <random>
#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

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
        result_vec[i] = bfloat16(std::log1p(std::exp(src_vec[i].to_float())));  // Softplus function
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
        x_mean += vec_a[i].to_float();
        y_mean += vec_b[i].to_float();
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        float x_diff = vec_a[i].to_float() - x_mean;
        float y_diff = vec_b[i].to_float() - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= vec_a.size();
    x_stddev /= vec_a.size();
    y_stddev /= vec_b.size();

    // Calculate the correlation coefficient
    float correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

int main() {
    // Device setup
    IDevice* device = CreateDevice(0);

    // Device command queue and program setup
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Core range setup
    constexpr CoreCoord core = {0, 0};

    // Input data preparation
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);

    // Fill the source vector with random values
    std::vector<bfloat16> src_vec(constants::TILE_HW);
    for (bfloat16& v : src_vec) {
        v = bfloat16(dist(rng));
    }

    // Calculate golden function results on CPU
    std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
    golden_softplus(src_vec, golden_vec);

    // Tilize the input vectors to match the expected tiled layout for the device
    // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
    // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
    // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
    // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
    // and enables efficient operations on the accelerator.
    src_vec = tilize_nfaces(src_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Dram buffer config
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = sizeof(bfloat16) * src_vec.size(),
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);  // Input buffer
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);  // Output buffer

    // DRAM transfer
    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);

    // L1 circular buffer setup
    constexpr uint32_t src_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src_config =
        CircularBufferConfig(single_tile_size, {{src_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src_config);

    constexpr uint32_t ones_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_ones_config =
        CircularBufferConfig(single_tile_size, {{ones_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ones_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_ones_config);

    constexpr uint32_t result_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_result_config =
        CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(result_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_result_config);

    // Kernels setup
    // Data movement kernels
    std::vector<uint32_t> reader_compile_time_args = {src_cb_index, ones_cb_index};
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
    std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/writer.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    // Compute kernel
    std::vector<uint32_t> compute_compile_time_args = {src_cb_index, ones_cb_index, result_cb_index};
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/compute/compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Runtime args setup
    SetRuntimeArgs(program, reader_kernel_id, core, {src_dram_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});

    // Program enqueue
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Data transfer back to host machine
    std::vector<bfloat16> result_vec(constants::TILE_HW, 0);
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);  // Blocking call to ensure data is read before proceeding

    // Reverse the tilization to get the result in the row-major format that the CPU expects
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
    // This is a measure of how similar the two vectors are.
    // A PCC close to 1 indicates that the two vectors are very similar.
    const float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    CloseDevice(device);
}
