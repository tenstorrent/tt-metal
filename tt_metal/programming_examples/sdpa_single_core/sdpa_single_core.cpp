// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <fstream>
#include <filesystem>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// ---------------------------------------------------------------------------
// File I/O helpers for test mode
// ---------------------------------------------------------------------------

std::vector<bfloat16> read_bf16_bin(const std::string& path, size_t num_elements) {
    std::ifstream ifs(path, std::ios::binary);
    TT_FATAL(ifs.good(), "Cannot open file for reading: {}", path);

    std::vector<uint16_t> raw(num_elements);
    ifs.read(reinterpret_cast<char*>(raw.data()), num_elements * sizeof(uint16_t));
    TT_FATAL(ifs.good(), "Failed to read {} elements from {}", num_elements, path);

    std::vector<bfloat16> result(num_elements);
    std::memcpy(result.data(), raw.data(), num_elements * sizeof(uint16_t));
    return result;
}

void write_bf16_bin(const std::string& path, const std::vector<bfloat16>& data) {
    std::ofstream ofs(path, std::ios::binary);
    TT_FATAL(ofs.good(), "Cannot open file for writing: {}", path);

    // bfloat16 stores uint16_t internally, same layout
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint16_t));
    TT_FATAL(ofs.good(), "Failed to write {} elements to {}", data.size(), path);
}

void sdpa_single_core(
    const uint32_t Sq_chunk_t,
    const uint32_t Sk_chunk_t,
    const uint32_t Sv_chunk_t,
    const uint32_t head_dim_t,
    const uint32_t subblock_h,
    const uint32_t num_q_chunks,
    const uint32_t num_k_chunks,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const uint32_t mm_throttle_level = 0) {
    TT_FATAL(subblock_h == 1 || subblock_h == 2, "subblock_h must be 1 or 2. Got {}.", subblock_h);
    TT_FATAL(mm_throttle_level <= 5, "mm_throttle_level must be 0-5. Got {}.", mm_throttle_level);
    TT_FATAL(
        Sq_chunk_t % subblock_h == 0, "Sq_chunk_t ({}) must be divisible by subblock_h ({}).", Sq_chunk_t, subblock_h);

    // Set up mesh command queue, workload, device range, and program. This is a single-core example using core {0,0}.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    CoreCoord core({0, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    const MathFidelity math_fidelity = MathFidelity::HiFi2;
    const uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Per-chunk tile counts
    const uint32_t q_chunk_tiles = Sq_chunk_t * head_dim_t;
    const uint32_t k_chunk_tiles = Sk_chunk_t * head_dim_t;
    const uint32_t v_chunk_tiles = Sv_chunk_t * head_dim_t;
    const uint32_t out_chunk_tiles = Sq_chunk_t * head_dim_t;

    // DRAM data sizes: Q/Out sized for num_q_chunks, K/V sized for num_k_chunks
    const uint32_t q_num_tiles = num_q_chunks * q_chunk_tiles;
    const uint32_t k_num_tiles = num_k_chunks * k_chunk_tiles;
    const uint32_t v_num_tiles = num_k_chunks * v_chunk_tiles;
    const uint32_t out_num_tiles = num_q_chunks * out_chunk_tiles;

    const uint32_t q_data_size = q_num_tiles * single_tile_size;
    const uint32_t k_data_size = k_num_tiles * single_tile_size;
    const uint32_t v_data_size = v_num_tiles * single_tile_size;
    const uint32_t out_data_size = out_num_tiles * single_tile_size;

    // CB sizes still based on per-chunk (double-buffered for pipelining)
    const uint32_t q_chunk_data_size = q_chunk_tiles * single_tile_size;
    const uint32_t out_chunk_data_size = out_chunk_tiles * single_tile_size;

    // CB sizes (double-buffered per chunk for Q, KT, V)
    const uint32_t q_cb_size = 2 * q_chunk_data_size;
    const uint32_t kt_cb_size = 2 * k_chunk_tiles * single_tile_size;
    const uint32_t v_cb_size = 2 * v_chunk_tiles * single_tile_size;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_Q{.size = q_data_size};
    distributed::ReplicatedBufferConfig buffer_config_K{.size = k_data_size};
    distributed::ReplicatedBufferConfig buffer_config_V{.size = v_data_size};
    distributed::ReplicatedBufferConfig buffer_config_out{.size = out_data_size};

    auto q_dram_buffer = distributed::MeshBuffer::create(buffer_config_Q, dram_config, mesh_device.get());
    auto k_dram_buffer = distributed::MeshBuffer::create(buffer_config_K, dram_config, mesh_device.get());
    auto v_dram_buffer = distributed::MeshBuffer::create(buffer_config_V, dram_config, mesh_device.get());
    auto out_dram_buffer = distributed::MeshBuffer::create(buffer_config_out, dram_config, mesh_device.get());

    const uint32_t q_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_q_config =
        CircularBufferConfig(q_cb_size, {{q_cb_index, cb_data_format}}).set_page_size(q_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_q_config);

    const uint32_t kt_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_kt_config =
        CircularBufferConfig(kt_cb_size, {{kt_cb_index, cb_data_format}}).set_page_size(kt_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_kt_config);

    const uint32_t qkt_cb_index = CBIndex::c_2;
    const uint32_t qkt_tiles = Sq_chunk_t * Sk_chunk_t;
    auto cb_qkt_config = CircularBufferConfig(qkt_tiles * single_tile_size, {{qkt_cb_index, cb_data_format}})
                             .set_page_size(qkt_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_qkt_config);

    const uint32_t v_cb_index = CBIndex::c_3;
    CircularBufferConfig cb_v_config =
        CircularBufferConfig(v_cb_size, {{v_cb_index, cb_data_format}}).set_page_size(v_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_v_config);

    // cb_qkt_row_A — ping buffer for raw matmul output of one QKT row
    const uint32_t qkt_row_A_cb_index = CBIndex::c_4;
    auto cb_qkt_row_A_config =
        CircularBufferConfig(subblock_h * Sk_chunk_t * single_tile_size, {{qkt_row_A_cb_index, cb_data_format}})
            .set_page_size(qkt_row_A_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_qkt_row_A_config);

    // cb_qkt_row_B — pong buffer for raw matmul output of one QKT row
    const uint32_t qkt_row_B_cb_index = CBIndex::c_6;
    auto cb_qkt_row_B_config =
        CircularBufferConfig(subblock_h * Sk_chunk_t * single_tile_size, {{qkt_row_B_cb_index, cb_data_format}})
            .set_page_size(qkt_row_B_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_qkt_row_B_config);

    // cb_neginf — helper tile filled with -inf by writer, consumed by compute for prev_max init
    const uint32_t neginf_cb_index = CBIndex::c_7;
    auto cb_neginf_config = CircularBufferConfig(single_tile_size, {{neginf_cb_index, cb_data_format}})
                                .set_page_size(neginf_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_neginf_config);

    const uint32_t identity_scalar_cb_index = CBIndex::c_5;
    auto c_identity_scalar_config = CircularBufferConfig(single_tile_size, {{identity_scalar_cb_index, cb_data_format}})
                                        .set_page_size(tt::CBIndex::c_5, single_tile_size);
    CreateCircularBuffer(program, core, c_identity_scalar_config);

    // cb_col_identity — tile with 1.0 in column 0, zeros elsewhere; used for matmul_reduce normalization
    const uint32_t col_identity_cb_index = CBIndex::c_8;
    auto cb_col_identity_config = CircularBufferConfig(single_tile_size, {{col_identity_cb_index, cb_data_format}})
                                      .set_page_size(col_identity_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_col_identity_config);

    // cb_normalized_out — 1-tile streaming CB for normalized output (decoupled from ping-pong out CBs)
    const uint32_t normalized_out_cb_index = CBIndex::c_9;
    auto cb_normalized_out_config = CircularBufferConfig(single_tile_size, {{normalized_out_cb_index, cb_data_format}})
                                        .set_page_size(normalized_out_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_normalized_out_config);

    const uint32_t cb_prev_out_index = CBIndex::c_25;
    CircularBufferConfig cb_prev_out_config =
        CircularBufferConfig(out_chunk_data_size, {{cb_prev_out_index, cb_data_format}})
            .set_page_size(cb_prev_out_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_prev_out_config);

    const uint32_t cb_curr_out_index = CBIndex::c_26;
    CircularBufferConfig cb_curr_out_config =
        CircularBufferConfig(out_chunk_data_size, {{cb_curr_out_index, cb_data_format}})
            .set_page_size(cb_curr_out_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_curr_out_config);

    // cb_prev_max
    auto cb_prev_max_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_27, cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_27, single_tile_size);
    CreateCircularBuffer(program, core, cb_prev_max_config);

    // cb_cur_max
    auto cb_cur_max_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_28, cb_data_format}})
                                 .set_page_size(tt::CBIndex::c_28, single_tile_size);
    CreateCircularBuffer(program, core, cb_cur_max_config);

    // cb_prev_sum
    auto cb_prev_sum_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_29, cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_29, single_tile_size);
    CreateCircularBuffer(program, core, cb_prev_sum_config);

    // cb_cur_sum
    auto cb_curr_sum_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_30, cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_30, single_tile_size);
    CreateCircularBuffer(program, core, cb_curr_sum_config);

    // cb_exp_max_diff
    auto c_exp_max_diff_config =
        CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_31, cb_data_format}})
            .set_page_size(tt::CBIndex::c_31, single_tile_size);
    CreateCircularBuffer(program, core, c_exp_max_diff_config);

    // Determine granularity for statistics computation
    const uint32_t dst_size = 8;
    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_FATAL(
        sub_exp_granularity == (1 << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    std::map<std::string, std::string> defines;
    constexpr bool exp_approx_mode = true;
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    if (mm_throttle_level > 0) {
        defines["MM_THROTTLE"] = std::to_string(mm_throttle_level);
    }

    // Create the data movement kernels and the compute kernel
    std::vector<uint32_t> reader_compile_time_args = {
        Sq_chunk_t,
        Sk_chunk_t,
        Sv_chunk_t,
        head_dim_t,
        num_q_chunks,
        num_k_chunks,
    };
    TensorAccessorArgs(*q_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*k_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*v_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/dataflow/reader.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = defines});

    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = 1.0f / std::sqrt(static_cast<float>(head_dim_t * TILE_WIDTH));

    std::vector<uint32_t> writer_compile_time_args = {
        Sq_chunk_t, Sk_chunk_t, Sv_chunk_t, head_dim_t, num_q_chunks, num_k_chunks, packed_identity_scalar};
    TensorAccessorArgs(*out_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/dataflow/writer.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = defines});

    std::vector<uint32_t> compute_compile_time_args = {
        Sq_chunk_t, Sk_chunk_t, Sv_chunk_t, head_dim_t, num_q_chunks, num_k_chunks, scale_union.u, subblock_h};
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/compute/sdpa.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .compile_args = compute_compile_time_args, .defines = defines});

    // Set kernel arguments
    uint32_t q_addr = q_dram_buffer->address();
    uint32_t k_addr = k_dram_buffer->address();
    uint32_t v_addr = v_dram_buffer->address();
    uint32_t out_addr = out_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {q_addr, k_addr, v_addr});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {out_addr});
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Create zero-filled input data, tilize, and upload to DRAM
    const uint32_t q_rows = num_q_chunks * Sq_chunk_t * TILE_HEIGHT;
    const uint32_t q_cols = head_dim_t * TILE_WIDTH;
    std::vector<bfloat16> q_data(q_rows * q_cols, bfloat16(0.0f));
    q_data = tilize_nfaces(q_data, q_rows, q_cols);

    const uint32_t k_rows = num_k_chunks * Sk_chunk_t * TILE_HEIGHT;
    const uint32_t k_cols = head_dim_t * TILE_WIDTH;
    std::vector<bfloat16> k_data(k_rows * k_cols, bfloat16(0.0f));
    k_data = tilize_nfaces(k_data, k_rows, k_cols);

    const uint32_t v_rows = num_k_chunks * Sv_chunk_t * TILE_HEIGHT;
    const uint32_t v_cols = head_dim_t * TILE_WIDTH;
    std::vector<bfloat16> v_data(v_rows * v_cols, bfloat16(0.0f));
    v_data = tilize_nfaces(v_data, v_rows, v_cols);

    distributed::EnqueueWriteMeshBuffer(cq, q_dram_buffer, q_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, k_dram_buffer, k_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, v_dram_buffer, v_data, false);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read final SDPA output back from DRAM (blocking call waits for workload to finish).
    std::vector<bfloat16> result(out_num_tiles * TILE_HEIGHT * TILE_WIDTH);
    distributed::EnqueueReadMeshBuffer(cq, result, out_dram_buffer, true);
    fmt::print("Output read from DRAM successfully ({} tiles, {} elements)\n", out_num_tiles, result.size());
}

// ---------------------------------------------------------------------------
// Test mode: reads Q/K/V from files, runs kernel, writes output to file
// ---------------------------------------------------------------------------

void sdpa_single_core_test(
    const uint32_t Sq_chunk_t,
    const uint32_t Sk_chunk_t,
    const uint32_t Sv_chunk_t,
    const uint32_t head_dim_t,
    const uint32_t subblock_h,
    const uint32_t num_q_chunks,
    const uint32_t num_k_chunks,
    const std::string& test_dir,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const uint32_t mm_throttle_level = 0) {
    TT_FATAL(subblock_h == 1 || subblock_h == 2, "subblock_h must be 1 or 2. Got {}.", subblock_h);
    TT_FATAL(mm_throttle_level <= 5, "mm_throttle_level must be 0-5. Got {}.", mm_throttle_level);
    TT_FATAL(
        Sq_chunk_t % subblock_h == 0, "Sq_chunk_t ({}) must be divisible by subblock_h ({}).", Sq_chunk_t, subblock_h);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    CoreCoord core({0, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    const MathFidelity math_fidelity = MathFidelity::HiFi2;
    const uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    const uint32_t q_chunk_tiles = Sq_chunk_t * head_dim_t;
    const uint32_t k_chunk_tiles = Sk_chunk_t * head_dim_t;
    const uint32_t v_chunk_tiles = Sv_chunk_t * head_dim_t;
    const uint32_t out_chunk_tiles = Sq_chunk_t * head_dim_t;

    const uint32_t q_num_tiles = num_q_chunks * q_chunk_tiles;
    const uint32_t k_num_tiles = num_k_chunks * k_chunk_tiles;
    const uint32_t v_num_tiles = num_k_chunks * v_chunk_tiles;
    const uint32_t out_num_tiles = num_q_chunks * out_chunk_tiles;

    const uint32_t q_data_size = q_num_tiles * single_tile_size;
    const uint32_t k_data_size = k_num_tiles * single_tile_size;
    const uint32_t v_data_size = v_num_tiles * single_tile_size;
    const uint32_t out_data_size = out_num_tiles * single_tile_size;

    const uint32_t q_chunk_data_size = q_chunk_tiles * single_tile_size;
    const uint32_t out_chunk_data_size = out_chunk_tiles * single_tile_size;

    const uint32_t q_cb_size = 2 * q_chunk_data_size;
    const uint32_t kt_cb_size = 2 * k_chunk_tiles * single_tile_size;
    const uint32_t v_cb_size = 2 * v_chunk_tiles * single_tile_size;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_Q{.size = q_data_size};
    distributed::ReplicatedBufferConfig buffer_config_K{.size = k_data_size};
    distributed::ReplicatedBufferConfig buffer_config_V{.size = v_data_size};
    distributed::ReplicatedBufferConfig buffer_config_out{.size = out_data_size};

    auto q_dram_buffer = distributed::MeshBuffer::create(buffer_config_Q, dram_config, mesh_device.get());
    auto k_dram_buffer = distributed::MeshBuffer::create(buffer_config_K, dram_config, mesh_device.get());
    auto v_dram_buffer = distributed::MeshBuffer::create(buffer_config_V, dram_config, mesh_device.get());
    auto out_dram_buffer = distributed::MeshBuffer::create(buffer_config_out, dram_config, mesh_device.get());

    // --- Circular Buffers (identical to sdpa_single_core) ---
    const uint32_t q_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_q_config =
        CircularBufferConfig(q_cb_size, {{q_cb_index, cb_data_format}}).set_page_size(q_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_q_config);

    const uint32_t kt_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_kt_config =
        CircularBufferConfig(kt_cb_size, {{kt_cb_index, cb_data_format}}).set_page_size(kt_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_kt_config);

    const uint32_t qkt_cb_index = CBIndex::c_2;
    const uint32_t qkt_tiles = Sq_chunk_t * Sk_chunk_t;
    auto cb_qkt_config = CircularBufferConfig(qkt_tiles * single_tile_size, {{qkt_cb_index, cb_data_format}})
                             .set_page_size(qkt_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_qkt_config);

    const uint32_t v_cb_index = CBIndex::c_3;
    CircularBufferConfig cb_v_config =
        CircularBufferConfig(v_cb_size, {{v_cb_index, cb_data_format}}).set_page_size(v_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_v_config);

    const uint32_t qkt_row_A_cb_index = CBIndex::c_4;
    auto cb_qkt_row_A_config =
        CircularBufferConfig(subblock_h * Sk_chunk_t * single_tile_size, {{qkt_row_A_cb_index, cb_data_format}})
            .set_page_size(qkt_row_A_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_qkt_row_A_config);

    const uint32_t qkt_row_B_cb_index = CBIndex::c_6;
    auto cb_qkt_row_B_config =
        CircularBufferConfig(subblock_h * Sk_chunk_t * single_tile_size, {{qkt_row_B_cb_index, cb_data_format}})
            .set_page_size(qkt_row_B_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_qkt_row_B_config);

    const uint32_t neginf_cb_index = CBIndex::c_7;
    auto cb_neginf_config = CircularBufferConfig(single_tile_size, {{neginf_cb_index, cb_data_format}})
                                .set_page_size(neginf_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_neginf_config);

    const uint32_t identity_scalar_cb_index = CBIndex::c_5;
    auto c_identity_scalar_config = CircularBufferConfig(single_tile_size, {{identity_scalar_cb_index, cb_data_format}})
                                        .set_page_size(tt::CBIndex::c_5, single_tile_size);
    CreateCircularBuffer(program, core, c_identity_scalar_config);

    const uint32_t col_identity_cb_index = CBIndex::c_8;
    auto cb_col_identity_config = CircularBufferConfig(single_tile_size, {{col_identity_cb_index, cb_data_format}})
                                      .set_page_size(col_identity_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_col_identity_config);

    const uint32_t normalized_out_cb_index = CBIndex::c_9;
    auto cb_normalized_out_config = CircularBufferConfig(single_tile_size, {{normalized_out_cb_index, cb_data_format}})
                                        .set_page_size(normalized_out_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_normalized_out_config);

    const uint32_t cb_prev_out_index = CBIndex::c_25;
    CircularBufferConfig cb_prev_out_config =
        CircularBufferConfig(out_chunk_data_size, {{cb_prev_out_index, cb_data_format}})
            .set_page_size(cb_prev_out_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_prev_out_config);

    const uint32_t cb_curr_out_index = CBIndex::c_26;
    CircularBufferConfig cb_curr_out_config =
        CircularBufferConfig(out_chunk_data_size, {{cb_curr_out_index, cb_data_format}})
            .set_page_size(cb_curr_out_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_curr_out_config);

    auto cb_prev_max_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_27, cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_27, single_tile_size);
    CreateCircularBuffer(program, core, cb_prev_max_config);

    auto cb_cur_max_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_28, cb_data_format}})
                                 .set_page_size(tt::CBIndex::c_28, single_tile_size);
    CreateCircularBuffer(program, core, cb_cur_max_config);

    auto cb_prev_sum_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_29, cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_29, single_tile_size);
    CreateCircularBuffer(program, core, cb_prev_sum_config);

    auto cb_curr_sum_config = CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_30, cb_data_format}})
                                  .set_page_size(tt::CBIndex::c_30, single_tile_size);
    CreateCircularBuffer(program, core, cb_curr_sum_config);

    auto c_exp_max_diff_config =
        CircularBufferConfig(Sq_chunk_t * single_tile_size, {{tt::CBIndex::c_31, cb_data_format}})
            .set_page_size(tt::CBIndex::c_31, single_tile_size);
    CreateCircularBuffer(program, core, c_exp_max_diff_config);

    const uint32_t dst_size = 8;
    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_FATAL(
        sub_exp_granularity == (1 << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    std::map<std::string, std::string> defines;
    constexpr bool exp_approx_mode = true;
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    if (mm_throttle_level > 0) {
        defines["MM_THROTTLE"] = std::to_string(mm_throttle_level);
    }

    // --- Kernels (identical to sdpa_single_core) ---
    std::vector<uint32_t> reader_compile_time_args = {
        Sq_chunk_t,
        Sk_chunk_t,
        Sv_chunk_t,
        head_dim_t,
        num_q_chunks,
        num_k_chunks,
    };
    TensorAccessorArgs(*q_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*k_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*v_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/dataflow/reader.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .defines = defines});

    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = 1.0f / std::sqrt(static_cast<float>(head_dim_t * TILE_WIDTH));

    std::vector<uint32_t> writer_compile_time_args = {
        Sq_chunk_t, Sk_chunk_t, Sv_chunk_t, head_dim_t, num_q_chunks, num_k_chunks, packed_identity_scalar};
    TensorAccessorArgs(*out_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/dataflow/writer.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = defines});

    std::vector<uint32_t> compute_compile_time_args = {
        Sq_chunk_t, Sk_chunk_t, Sv_chunk_t, head_dim_t, num_q_chunks, num_k_chunks, scale_union.u, subblock_h};
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/compute/sdpa.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .compile_args = compute_compile_time_args, .defines = defines});

    // --- Set kernel runtime args ---
    uint32_t q_addr = q_dram_buffer->address();
    uint32_t k_addr = k_dram_buffer->address();
    uint32_t v_addr = v_dram_buffer->address();
    uint32_t out_addr = out_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {q_addr, k_addr, v_addr});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {out_addr});

    // --- Read input data from files, tilize, and upload ---
    const uint32_t q_rows = num_q_chunks * Sq_chunk_t * TILE_HEIGHT;
    const uint32_t q_cols = head_dim_t * TILE_WIDTH;
    const uint32_t k_rows = num_k_chunks * Sk_chunk_t * TILE_HEIGHT;
    const uint32_t k_cols = head_dim_t * TILE_WIDTH;
    const uint32_t v_rows = num_k_chunks * Sv_chunk_t * TILE_HEIGHT;
    const uint32_t v_cols = head_dim_t * TILE_WIDTH;

    fmt::print("Reading Q [{}, {}] from {}/q.bin\n", q_rows, q_cols, test_dir);
    auto q_data = read_bf16_bin(test_dir + "/q.bin", q_rows * q_cols);
    q_data = tilize_nfaces(q_data, q_rows, q_cols);

    fmt::print("Reading K [{}, {}] from {}/k.bin\n", k_rows, k_cols, test_dir);
    auto k_data = read_bf16_bin(test_dir + "/k.bin", k_rows * k_cols);
    k_data = tilize_nfaces(k_data, k_rows, k_cols);

    fmt::print("Reading V [{}, {}] from {}/v.bin\n", v_rows, v_cols, test_dir);
    auto v_data = read_bf16_bin(test_dir + "/v.bin", v_rows * v_cols);
    v_data = tilize_nfaces(v_data, v_rows, v_cols);

    distributed::EnqueueWriteMeshBuffer(cq, q_dram_buffer, q_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, k_dram_buffer, k_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, v_dram_buffer, v_data, false);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // --- Read output, untilize, and write to file ---
    std::vector<bfloat16> result(out_num_tiles * TILE_HEIGHT * TILE_WIDTH);
    distributed::EnqueueReadMeshBuffer(cq, result, out_dram_buffer, true);

    const uint32_t out_rows = num_q_chunks * Sq_chunk_t * TILE_HEIGHT;
    const uint32_t out_cols = head_dim_t * TILE_WIDTH;
    auto untilized_result = untilize_nfaces(result, out_rows, out_cols);

    std::string output_path = test_dir + "/device_output.bin";
    write_bf16_bin(output_path, untilized_result);
    fmt::print("Device output written to {} ({} elements)\n", output_path, untilized_result.size());
}

///////////////////////////////////////

// Simple CLI arg parser: returns value for --key, or default_val if not found
static uint32_t get_arg_uint(int argc, char* argv[], const std::string& key, uint32_t default_val) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == key) {
            return static_cast<uint32_t>(std::stoul(argv[i + 1]));
        }
    }
    return default_val;
}

int main(int argc, char* argv[]) {
    bool pass = true;
    bool test_mode = false;
    std::string test_dir;

    // Check for --test <dir> mode
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--test" && i + 1 < argc) {
            test_mode = true;
            test_dir = argv[i + 1];
            break;
        }
    }

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        if (test_mode) {
            // Parse parameters from CLI (with defaults matching the benchmark path)
            uint32_t Sq_chunk_t = get_arg_uint(argc, argv, "--Sq_chunk_t", 7);
            uint32_t Sk_chunk_t = get_arg_uint(argc, argv, "--Sk_chunk_t", 16);
            uint32_t head_dim_t = get_arg_uint(argc, argv, "--head_dim_t", 4);
            uint32_t num_q_chunks = get_arg_uint(argc, argv, "--num_q_chunks", 3);
            uint32_t num_k_chunks = get_arg_uint(argc, argv, "--num_k_chunks", 5);
            uint32_t subblock_h = get_arg_uint(argc, argv, "--subblock_h", 1);
            uint32_t mm_throttle_level = get_arg_uint(argc, argv, "--mm_throttle_level", 0);
            uint32_t Sv_chunk_t = Sk_chunk_t;  // Always equal

            fmt::print(
                "Test mode: dir={}, Sq_chunk_t={}, Sk_chunk_t={}, head_dim_t={}, "
                "num_q_chunks={}, num_k_chunks={}, subblock_h={}, mm_throttle={}\n",
                test_dir,
                Sq_chunk_t,
                Sk_chunk_t,
                head_dim_t,
                num_q_chunks,
                num_k_chunks,
                subblock_h,
                mm_throttle_level);

            sdpa_single_core_test(
                Sq_chunk_t,
                Sk_chunk_t,
                Sv_chunk_t,
                head_dim_t,
                subblock_h,
                num_q_chunks,
                num_k_chunks,
                test_dir,
                mesh_device,
                mm_throttle_level);
        } else {
            // Existing benchmark path (zero data)
            constexpr uint32_t Sq_chunk_t = 7;
            constexpr uint32_t Sk_chunk_t = 16;
            constexpr uint32_t Sv_chunk_t = 16;
            constexpr uint32_t head_dim_t = 128 / TILE_WIDTH;
            constexpr uint32_t subblock_h = 1;
            constexpr uint32_t num_q_chunks = 3;
            constexpr uint32_t num_k_chunks = 5;
            constexpr uint32_t mm_throttle_level = 1;

            sdpa_single_core(
                Sq_chunk_t,
                Sk_chunk_t,
                Sv_chunk_t,
                head_dim_t,
                subblock_h,
                num_q_chunks,
                num_k_chunks,
                mesh_device,
                mm_throttle_level);
        }

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
