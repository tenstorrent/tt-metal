// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-wise softmax example (single core).
//
// For each row i of an M×N input matrix, computes:
//
//   softmax(x)[i,j] = exp(x[i,j] − max_i) / Σ_k exp(x[i,k] − max_i)
//
// where max_i = max_k x[i,k] is subtracted for numerical stability.
//
// The device implementation uses four compute passes per tile-row (mt):
//
//   Pass 1 — Row-wise max  (1 tile → cb_max[0])
//     Reduce Nt input tiles; reduce_tile<MAX, REDUCE_ROW> accumulates the
//     per-row maximum into DST.  Result tile packed to cb_max (col 0 = max).
//
//   Pass 2 — exp(x − max)  (Nt tiles → cb_max[1..Nt])
//     For each of the Nt input tiles, subtract cb_max[0] (broadcast across
//     columns) and apply exp.  Each result is pushed to the back of cb_max,
//     leaving [row-max | exp[0] | … | exp[Nt-1]].  The row-max is then popped.
//
//   Pass 3 — Row sum → 1/sum  (1 tile → cb_sum)
//     reduce_tile<SUM, REDUCE_ROW> accumulates the Nt exp tiles in cb_max
//     (indexed by tile number, none popped).  recip_tile converts the sums to
//     1/sum, which is packed to cb_sum (col 0 = 1/sum_r per row r).
//
//   Pass 4 — Normalise  (Nt tiles → cb_out)
//     Each exp tile is popped from cb_max and multiplied by 1/sum (col 0 of
//     cb_sum, broadcast across columns), producing the final softmax output.

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
        // Pass 1: row max for numerical stability.
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < N; j++) {
            row_max = std::max(row_max, static_cast<float>(input[i * N + j]));
        }
        // Pass 2: exp(x − max) and accumulate sum.
        float sum = 0.0f;
        for (uint32_t j = 0; j < N; j++) {
            float val = std::exp(static_cast<float>(input[i * N + j]) - row_max);
            output[i * N + j] = bfloat16(val);
            sum += val;
        }
        // Pass 3: normalize.
        for (uint32_t j = 0; j < N; j++) {
            output[i * N + j] = bfloat16(static_cast<float>(output[i * N + j]) / sum);
        }
    }
}

// Row-wise softmax on the accelerator.
//
// Parameters:
//   input      - Tilized M×N input matrix (Mt×Nt tiles, each 32×32 bfloat16).
//   output     - Destination for the softmax result (same Mt×Nt tile layout).
//   M, N       - Matrix dimensions; both must be multiples of TILE_HEIGHT/TILE_WIDTH (32).
void softmax_single_core(
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
    uint32_t Nt = N / TILE_WIDTH;   // Number of tile columns.

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
    // The hardware reduce engine multiplies each input element by the scaler before
    // accumulating; a value of 1.0 leaves inputs unscaled.
    distributed::ReplicatedBufferConfig scaler_buf_config{.size = single_tile_size};
    auto scaler_dram_buffer = distributed::MeshBuffer::create(scaler_buf_config, dram_config, mesh_device.get());

    // Output matrix: same shape as input (Mt×Nt tiles), holds the final softmax values.
    distributed::ReplicatedBufferConfig output_buf_config{.size = single_tile_size * Mt * Nt};
    auto dst_dram_buffer = distributed::MeshBuffer::create(output_buf_config, dram_config, mesh_device.get());

    // -------------------------------------------------------------------------
    // Circular buffers
    // -------------------------------------------------------------------------
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // cb_in (c_0): Double-buffered input (2 tiles).
    // The reader streams tiles here; the compute kernel consumes them one at a time.
    // The small capacity allows the reader and compute to overlap.
    uint32_t in_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(2 * single_tile_size, {{in_cb_index, cb_data_format}})
            .set_page_size(in_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_in_config);

    // cb_scaler (c_1): Holds the constant scaler tile (all 1.0 values).
    // Pushed once by the reader before any pass; never popped by the compute kernel,
    // so it stays available throughout all three passes.
    uint32_t scaler_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_scaler_config =
        CircularBufferConfig(single_tile_size, {{scaler_cb_index, cb_data_format}})
            .set_page_size(scaler_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_scaler_config);

    // cb_max (c_2): Dual-purpose buffer shared by passes 1–4.
    //   Pass 1 writes 1 row-max tile (col 0 = per-row maxima).
    //   Pass 2 appends Nt exp(x − max) tiles at the back while keeping the
    //   row-max tile at index 0 for bcast-sub; the row-max is popped at the
    //   end of pass 2, leaving Nt exp tiles.
    //   Passes 3 and 4 consume those Nt tiles.
    //   Capacity: Nt + 1 to hold the row-max + Nt exp tiles simultaneously.
    uint32_t max_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_max_config =
        CircularBufferConfig((Nt + 1) * single_tile_size, {{max_cb_index, cb_data_format}})
            .set_page_size(max_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_max_config);

    // cb_sum (c_3): One tile per mt.  Pass 3 packs the per-row reciprocal sum
    // (1/Σ exp(x − max)) here; pass 4 multiplies each exp tile by this value.
    uint32_t sum_cb_index = CBIndex::c_3;
    CircularBufferConfig cb_sum_config =
        CircularBufferConfig(single_tile_size, {{sum_cb_index, cb_data_format}})
            .set_page_size(sum_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_sum_config);

    // cb_out (c_16): Softmax output — same shape as the input (Mt×Nt tiles).
    // Filled by the compute kernel; drained by the writer.
    // Total size: Mt×Nt = (M/32)×(N/32) tiles (400 tiles for M=N=640).
    uint32_t out_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(Mt * Nt * single_tile_size, {{out_cb_index, cb_data_format}})
            .set_page_size(out_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // -------------------------------------------------------------------------
    // Kernels
    // -------------------------------------------------------------------------
    // Reader (RISCV_1): loads input tiles and the scaler tile from DRAM into CBs.
    // Must feed the input to cb_in three times (once per pass); between passes it
    // re-reads the same Mt×Nt tile block from DRAM in row-major tile order.
    vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*scaler_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "softmax/softmax_single_core/kernels/dataflow/reader_softmax.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Writer (RISCV_0): writes the Mt×Nt final softmax tiles from cb_out to DRAM.
    // Tiles arrive in row-major tile order from the compute kernel.
    vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "softmax/softmax_single_core/kernels/dataflow/writer_softmax.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compute kernel: three-pass row-wise softmax.
    //   Pass 1 — row max  → cb_max  (Mt tiles)
    //   Pass 2 — exp-sum  → cb_sum  (Mt tiles)
    //   Pass 3 — normalize → cb_out (Mt×Nt tiles)
    // Mt and Nt are compile-time so inner loop bounds can be constant-folded.
    vector<uint32_t> compute_compile_time_args = {Mt, Nt};
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "softmax/softmax_single_core/kernels/compute/softmax.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    // -------------------------------------------------------------------------
    // Runtime arguments
    // -------------------------------------------------------------------------
    uint32_t src_addr = src_dram_buffer->address();
    uint32_t scaler_addr = scaler_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    tt_metal::SetRuntimeArgs(program, reader_id, core, {src_addr, scaler_addr, Mt, Nt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt, Nt});
    // Compute kernel uses only compile-time arguments; no runtime args needed.

    // -------------------------------------------------------------------------
    // Upload inputs, execute, and read back
    // -------------------------------------------------------------------------
    // Scaler tile: all 1.0 in bfloat16.
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
        constexpr uint32_t M = 32;
        constexpr uint32_t N = 32;

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
        softmax_single_core(src_tilized, result_tilized, M, N, mesh_device);

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