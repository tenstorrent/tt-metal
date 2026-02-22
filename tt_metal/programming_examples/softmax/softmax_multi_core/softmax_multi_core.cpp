// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-wise softmax example (multi-core).
//
// Extends softmax_single_core by distributing the Mt tile-row groups across
// all available Tensix cores.  Each core independently runs the full four-pass
// softmax (max → exp → sum → normalise) on its assigned subset of rows, so no
// inter-core communication or intermediate buffers are required.
//
// Work split: Mt tile-row groups are divided across cores using split_work_to_cores.
// Core i receives runtime args (mt_start, mt_count) and processes the mt_count
// rows starting at tile-row group mt_start.
//
// Each core runs the same four passes as the single-core version:
//
//   Pass 1 — Row-wise max  (1 tile → cb_max[0])
//   Pass 2 — exp(x − max)  (Nt tiles → cb_max[1..Nt])
//   Pass 3 — Row sum → 1/sum  (1 tile → cb_sum)
//   Pass 4 — Normalise  (Nt tiles → cb_out)

#include <fmt/base.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// CPU reference: numerically stable row-wise softmax.
void golden_softmax(const vector<bfloat16>& input, vector<bfloat16>& output, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < N; j++) {
            row_max = std::max(row_max, static_cast<float>(input[i * N + j]));
        }
        float sum = 0.0f;
        for (uint32_t j = 0; j < N; j++) {
            float val = std::exp(static_cast<float>(input[i * N + j]) - row_max);
            output[i * N + j] = bfloat16(val);
            sum += val;
        }
        for (uint32_t j = 0; j < N; j++) {
            output[i * N + j] = bfloat16(static_cast<float>(output[i * N + j]) / sum);
        }
    }
}

// Row-wise softmax on the accelerator (multi-core).
//
// Parameters:
//   input      - Tilized M×N input matrix (Mt×Nt tiles, each 32×32 bfloat16).
//   output     - Destination for the softmax result (same Mt×Nt tile layout).
//   M, N       - Matrix dimensions; both must be multiples of TILE_HEIGHT/TILE_WIDTH (32).
void softmax_multi_core(
    const vector<bfloat16>& input,
    vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    const shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};

    uint32_t Mt = M / TILE_HEIGHT;  // Number of tile rows.
    uint32_t Nt = N / TILE_WIDTH;   // Number of tile columns.

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // -------------------------------------------------------------------------
    // Work distribution
    // -------------------------------------------------------------------------
    // Split Mt tile-row groups across all available Tensix cores.
    // split_work_to_cores returns two groups:
    //   core_group_1: cores that each handle work_per_core_1 row groups.
    //   core_group_2: cores that each handle work_per_core_2 row groups (or 0 if Mt
    //                 divides evenly).
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        split_work_to_cores(core_grid, Mt);

    // -------------------------------------------------------------------------
    // DRAM buffers
    // -------------------------------------------------------------------------
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    // Input matrix: Mt×Nt tiles.
    distributed::ReplicatedBufferConfig input_buf_config{.size = single_tile_size * Mt * Nt};
    auto src_dram_buffer = distributed::MeshBuffer::create(input_buf_config, dram_config, mesh_device.get());

    // Scaler tile: one 32×32 tile, all values 1.0.
    distributed::ReplicatedBufferConfig scaler_buf_config{.size = single_tile_size};
    auto scaler_dram_buffer = distributed::MeshBuffer::create(scaler_buf_config, dram_config, mesh_device.get());

    // Output matrix: same shape as input (Mt×Nt tiles), holds the final softmax values.
    distributed::ReplicatedBufferConfig output_buf_config{.size = single_tile_size * Mt * Nt};
    auto dst_dram_buffer = distributed::MeshBuffer::create(output_buf_config, dram_config, mesh_device.get());

    // -------------------------------------------------------------------------
    // Circular buffers (created on all_cores)
    // -------------------------------------------------------------------------
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // cb_in (c_0): Double-buffered input (2 tiles).
    uint32_t in_cb_index = CBIndex::c_0;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * single_tile_size, {{in_cb_index, cb_data_format}})
            .set_page_size(in_cb_index, single_tile_size));

    // cb_scaler (c_1): Constant 1.0 tile — pushed once, never popped.
    uint32_t scaler_cb_index = CBIndex::c_1;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(single_tile_size, {{scaler_cb_index, cb_data_format}})
            .set_page_size(scaler_cb_index, single_tile_size));

    // cb_max (c_2): Row-max tile + Nt exp(x−max) tiles.
    // Capacity: Nt+1 to hold the row-max and all Nt exp tiles simultaneously.
    uint32_t max_cb_index = CBIndex::c_2;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig((Nt + 1) * single_tile_size, {{max_cb_index, cb_data_format}})
            .set_page_size(max_cb_index, single_tile_size));

    // cb_sum (c_3): 1/sum tile — one tile per mt, produced by pass 3, consumed by pass 4.
    uint32_t sum_cb_index = CBIndex::c_3;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(single_tile_size, {{sum_cb_index, cb_data_format}})
            .set_page_size(sum_cb_index, single_tile_size));

    // cb_out (c_16): Softmax output — Nt tiles capacity (one full row at a time).
    // The compute kernel fills one row of Nt tiles; the writer drains them in parallel.
    uint32_t out_cb_index = CBIndex::c_16;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Nt * single_tile_size, {{out_cb_index, cb_data_format}})
            .set_page_size(out_cb_index, single_tile_size));

    // -------------------------------------------------------------------------
    // Kernels (created on all_cores)
    // -------------------------------------------------------------------------
    // Reader (RISCV_1): loads input tiles (twice per row: pass 1 and pass 2) and
    // the scaler tile from DRAM, starting from the assigned tile-row range.
    vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*scaler_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "softmax/softmax_multi_core/kernels/dataflow/reader_softmax_multi_core.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Writer (RISCV_0): writes mt_count×Nt softmax tiles from cb_out to DRAM,
    // at global tile indices [mt_start*Nt .. (mt_start+mt_count)*Nt).
    vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "softmax/softmax_multi_core/kernels/dataflow/writer_softmax_multi_core.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compute kernel: four-pass row-wise softmax.
    // Nt is compile-time so inner loop bounds can be constant-folded.
    // mt_count is runtime so each core can receive a different share of rows.
    vector<uint32_t> compute_compile_time_args = {Nt};
    auto compute_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "softmax/softmax_multi_core/kernels/compute/softmax.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    // -------------------------------------------------------------------------
    // Per-core runtime arguments
    // -------------------------------------------------------------------------
    // Iterate over core_group_1 and core_group_2 in tile-row order, assigning each
    // core its [mt_start, mt_start + mt_count) slice of the Mt row groups.
    uint32_t src_addr    = src_dram_buffer->address();
    uint32_t scaler_addr = scaler_dram_buffer->address();
    uint32_t dst_addr    = dst_dram_buffer->address();

    uint32_t mt_start = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core_1), std::make_pair(core_group_2, work_per_core_2)};

    for (const auto& [ranges, mt_count] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                tt_metal::SetRuntimeArgs(
                    program, reader_id, core, {src_addr, scaler_addr, mt_start, mt_count, Nt});
                tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, mt_start, mt_count, Nt});
                tt_metal::SetRuntimeArgs(program, compute_id, core, {mt_count});

                mt_start += mt_count;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Upload inputs, execute, and read back
    // -------------------------------------------------------------------------
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
        uniform_real_distribution<float> dist(-1.0f, 1.0f);
        vector<bfloat16> src_vec(M * N);
        for (auto& v : src_vec) {
            v = bfloat16(dist(rng));
        }

        // Compute the golden row-wise softmax on the CPU (original row-major layout).
        vector<bfloat16> golden_vec(M * N);
        golden_softmax(src_vec, golden_vec, M, N);

        // Tilize the input so the device sees the expected 32×32 tile layout.
        vector<bfloat16> src_tilized = tilize_nfaces(src_vec, M, N);

        // Allocate the output buffer: same tile layout as the input (Mt×Nt tiles).
        vector<bfloat16> result_tilized(M * N, bfloat16(0.0f));
        softmax_multi_core(src_tilized, result_tilized, M, N, mesh_device);

        // Untilize the result back to row-major for validation.
        vector<bfloat16> result = untilize_nfaces(result_tilized, M, N);

        constexpr float kMaxAllowedError = 0.01f;
        float max_abs_err = 0.0f;
        uint32_t num_mismatches = 0;
        for (uint32_t i = 0; i < M * N; i++) {
            float err = std::abs(static_cast<float>(golden_vec[i]) - static_cast<float>(result[i]));
            if (err > kMaxAllowedError) {
                num_mismatches++;
                if (num_mismatches <= 5) {
                    uint32_t row = i / N, col = i % N;
                    fmt::print(
                        "Mismatch at ({:3d},{:3d}): golden={:.6f}  result={:.6f}  diff={:.6f}\n",
                        row,
                        col,
                        static_cast<float>(golden_vec[i]),
                        static_cast<float>(result[i]),
                        err);
                }
            }
            max_abs_err = std::max(max_abs_err, err);
        }

        fmt::print("Row-wise softmax on {}×{} matrix\n", M, N);
        fmt::print("Max absolute error : {:.6f}\n", max_abs_err);
        fmt::print("Number of mismatches: {} / {}\n", num_mismatches, M * N);

        TT_FATAL(
            num_mismatches == 0,
            "Softmax result does not match the golden reference ({} mismatches)",
            num_mismatches);

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
