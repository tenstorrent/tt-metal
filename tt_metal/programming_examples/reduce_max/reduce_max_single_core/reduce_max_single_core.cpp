// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-wise maximum reduction example.
//
// This example computes, for each row of an M×N matrix, the maximum value across
// all N columns.  The result is an M-element vector.
//
// The accelerator kernel uses the hardware reduce engine configured for
// PoolType::MAX, ReduceDim::REDUCE_ROW which, for each 32-row group of tiles,
// accumulates the per-row maximum across all Nt = N/32 column tiles into a single
// output tile (column 0 of that tile holds the 32 row maxima).

#include <fmt/base.h>
#include <sys/types.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// CPU reference: for each row i, output[i] = max(input[i*N .. i*N+N-1]).
void golden_row_max(const vector<bfloat16>& input, vector<bfloat16>& output, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < N; j++) {
            float val = static_cast<float>(input[i * N + j]);
            row_max = std::max(row_max, val);
        }
        output[i] = bfloat16(row_max);
    }
}

// Row-wise max on the accelerator.
//
// Parameters:
//   input      - Tilized M×N input matrix (Mt×Nt tiles, each 32×32 bfloat16).
//   output     - Destination for Mt output tiles (each 32×32 bfloat16).
//                After the call, column 0 of each untilized output tile row holds
//                the row maxima for the corresponding 32 input rows.
//   M, N       - Matrix dimensions (both must be multiples of TILE_HEIGHT/TILE_WIDTH = 32).
void reduce_max_single_core(
    const vector<bfloat16>& input,
    vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    const shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    // Single Tensix core at {0, 0}.
    CoreCoord core({0, 0});

    uint32_t Mt = M / TILE_HEIGHT;  // Number of tile rows.
    uint32_t Nt = N / TILE_WIDTH;   // Number of tile columns (reduction dimension).

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // -------------------------------------------------------------------------
    // DRAM buffers
    // -------------------------------------------------------------------------
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    // Input matrix: Mt×Nt tiles.
    distributed::ReplicatedBufferConfig input_buf_config{.size = single_tile_size * Mt * Nt};
    auto src_dram_buffer = distributed::MeshBuffer::create(input_buf_config, dram_config, mesh_device.get());

    // Scaler tile: one 32×32 tile, all values 1.0.
    // For MAX reduction, the hardware multiplies each element by the scaler before
    // comparing.  A scaler of 1.0 preserves the true maximum.
    distributed::ReplicatedBufferConfig scaler_buf_config{.size = single_tile_size};
    auto scaler_dram_buffer = distributed::MeshBuffer::create(scaler_buf_config, dram_config, mesh_device.get());

    // Output: Mt tiles (one per row group), each 32×32.
    // Column 0 of each tile holds the row maxima; other columns are zero.
    distributed::ReplicatedBufferConfig output_buf_config{.size = single_tile_size * Mt};
    auto dst_dram_buffer = distributed::MeshBuffer::create(output_buf_config, dram_config, mesh_device.get());

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // cb_in (c_0): Double-buffered input (2 tiles).
    // The compute kernel processes one tile at a time and pops it immediately,
    // so 2 tiles is sufficient for the reader and compute to overlap.
    uint32_t src_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src_config =
        CircularBufferConfig(2 * single_tile_size, {{src_cb_index, cb_data_format}})
            .set_page_size(src_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src_config);

    // cb_scaler (c_1): Holds the single scaler tile (sent once, never popped).
    uint32_t scaler_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_scaler_config =
        CircularBufferConfig(single_tile_size, {{scaler_cb_index, cb_data_format}})
            .set_page_size(scaler_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_scaler_config);

    // cb_out (c_16): Double-buffered output (2 tiles).
    uint32_t output_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(2 * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // -------------------------------------------------------------------------
    // Kernels
    // -------------------------------------------------------------------------
    // Reader (RISCV_1): loads input tiles and the scaler tile from DRAM into L1 CBs.
    vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*scaler_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "reduce_max/reduce_max_single_core/kernels/dataflow/reader_reduce_max.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Writer (RISCV_0): writes reduced output tiles from L1 to DRAM.
    vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "reduce_max/reduce_max_single_core/kernels/dataflow/writer_reduce_max.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compute kernel: performs the reduce_tile<MAX, REDUCE_ROW> accumulation loop.
    // Mt and Nt are passed as compile-time arguments so the inner loop bounds can
    // be unrolled or constant-folded by the compiler.
    vector<uint32_t> compute_compile_time_args = {Mt, Nt};
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "reduce_max/reduce_max_single_core/kernels/compute/reduce_max.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    // -------------------------------------------------------------------------
    // Runtime arguments
    // -------------------------------------------------------------------------
    uint32_t src_addr = src_dram_buffer->address();
    uint32_t scaler_addr = scaler_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    tt_metal::SetRuntimeArgs(program, reader_id, core, {src_addr, scaler_addr, Mt, Nt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt});
    // Compute kernel uses only compile-time arguments; no runtime args needed.

    // -------------------------------------------------------------------------
    // Upload inputs, execute, and read back
    // -------------------------------------------------------------------------
    // Build the scaler tile: all 1.0 values in bfloat16.
    vector<bfloat16> scaler_tile(TILE_HEIGHT * TILE_WIDTH, bfloat16(1.0f));

    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, input, false);
    distributed::EnqueueWriteMeshBuffer(cq, scaler_dram_buffer, scaler_tile, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
}

int main() {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Matrix dimensions — both must be divisible by the tile size (32).
        constexpr uint32_t M = 640;
        constexpr uint32_t N = 640;

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT (32)");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH (32)");

        // Generate a random M×N input matrix.
        mt19937 rng(42);
        uniform_real_distribution<float> dist(-10.0f, -1.0f);
        vector<bfloat16> src_vec(M * N);
        for (auto& v : src_vec) {
            v = bfloat16(dist(rng));
        }

        // Compute the golden row-wise maximum on the CPU (from the original untilized layout).
        vector<bfloat16> golden_vec(M);
        golden_row_max(src_vec, golden_vec, M, N);

        // Tilize the input so its memory layout matches what the device expects.
        // The hardware operates on 32×32 tiles; tilize_nfaces() reorders the data
        // from row-major into contiguous 32×32 blocks.
        vector<bfloat16> src_tilized = tilize_nfaces(src_vec, M, N);

        // Allocate the output: Mt tiles × 1 tile column, each 32×32 bfloat16.
        uint32_t Mt = M / TILE_HEIGHT;
        vector<bfloat16> result_tilized(Mt * TILE_HEIGHT * TILE_WIDTH, bfloat16(0.0f));
        reduce_max_single_core(src_tilized, result_tilized, M, N, mesh_device);

        // Convert the output back to row-major.  The output is Mt×1 tiles
        // (Mt tile-rows, 1 tile-column wide = 32 element columns), so untilize
        // gives an M×32 matrix.  The row maxima sit in column 0 of that matrix.
        vector<bfloat16> result_untilized = untilize_nfaces(result_tilized, M, TILE_WIDTH);

        // Extract column 0 from each row: result_untilized[i * TILE_WIDTH + 0].
        vector<bfloat16> result_max(M);
        for (uint32_t i = 0; i < M; i++) {
            result_max[i] = result_untilized[i * TILE_WIDTH];
            // fmt::print("Max at row {:3d}: {:.6f}\n", i, static_cast<float>(result_max[i]));
        }

        // Validate against the golden reference.
        // The max operation involves no floating-point arithmetic (only comparisons),
        // so results should match the golden values to within bfloat16 rounding.
        constexpr float kMaxAllowedError = 0.01f;
        float max_abs_err = 0.0f;
        uint32_t num_mismatches = 0;
        for (uint32_t i = 0; i < M; i++) {
            float err = std::abs(static_cast<float>(golden_vec[i]) - static_cast<float>(result_max[i]));
            if (err > kMaxAllowedError) {
                num_mismatches++;
                if (num_mismatches <= 5) {
                    fmt::print(
                        "Mismatch at row {:3d}: golden={:.6f}  result={:.6f}  diff={:.6f}\n",
                        i,
                        static_cast<float>(golden_vec[i]),
                        static_cast<float>(result_max[i]),
                        err);
                }
            }
            max_abs_err = std::max(max_abs_err, err);
        }

        fmt::print("Row-wise max on {}×{} matrix\n", M, N);
        fmt::print("Max absolute error : {:.6f}\n", max_abs_err);
        fmt::print("Number of mismatches: {} / {}\n", num_mismatches, M);

        TT_FATAL(num_mismatches == 0, "Row-wise max result does not match the golden reference ({} mismatches)", num_mismatches);

        pass &= mesh_device->close();

    } catch (const exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
