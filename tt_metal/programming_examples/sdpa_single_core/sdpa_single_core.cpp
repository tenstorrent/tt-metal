// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
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

void sdpa_single_core(
    const uint32_t Sq_chunk_t,
    const uint32_t Sk_chunk_t,
    const uint32_t Sv_chunk_t,
    const uint32_t head_dim_t,
    const uint32_t subblock_h,
    const uint32_t num_q_chunks,
    const uint32_t num_k_chunks,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    TT_FATAL(subblock_h == 1 || subblock_h == 2, "subblock_h must be 1 or 2. Got {}.", subblock_h);
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
    const uint32_t kt_chunk_tiles = head_dim_t * Sk_chunk_t;
    const uint32_t v_chunk_tiles = Sv_chunk_t * head_dim_t;
    const uint32_t out_chunk_tiles = Sq_chunk_t * head_dim_t;

    // DRAM data sizes: Q/Out sized for num_q_chunks, KT/V sized for num_k_chunks
    const uint32_t q_num_tiles = num_q_chunks * q_chunk_tiles;
    const uint32_t kt_num_tiles = num_k_chunks * kt_chunk_tiles;
    const uint32_t v_num_tiles = num_k_chunks * v_chunk_tiles;
    const uint32_t out_num_tiles = num_q_chunks * out_chunk_tiles;

    const uint32_t q_data_size = q_num_tiles * single_tile_size;
    const uint32_t kt_data_size = kt_num_tiles * single_tile_size;
    const uint32_t v_data_size = v_num_tiles * single_tile_size;
    const uint32_t out_data_size = out_num_tiles * single_tile_size;

    // CB sizes still based on per-chunk (double-buffered for pipelining)
    const uint32_t q_chunk_data_size = q_chunk_tiles * single_tile_size;
    const uint32_t out_chunk_data_size = out_chunk_tiles * single_tile_size;

    // CB sizes (double-buffered per chunk for Q, KT, V)
    const uint32_t q_cb_size = 2 * q_chunk_data_size;
    const uint32_t kt_cb_size = 2 * kt_chunk_tiles * single_tile_size;
    const uint32_t v_cb_size = 2 * v_chunk_tiles * single_tile_size;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_Q{.size = q_data_size};
    distributed::ReplicatedBufferConfig buffer_config_KT{.size = kt_data_size};
    distributed::ReplicatedBufferConfig buffer_config_V{.size = v_data_size};
    distributed::ReplicatedBufferConfig buffer_config_out{.size = out_data_size};

    auto q_dram_buffer = distributed::MeshBuffer::create(buffer_config_Q, dram_config, mesh_device.get());
    auto kt_dram_buffer = distributed::MeshBuffer::create(buffer_config_KT, dram_config, mesh_device.get());
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
    TensorAccessorArgs(*kt_dram_buffer).append_to(reader_compile_time_args);
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
    uint32_t kt_addr = kt_dram_buffer->address();
    uint32_t v_addr = v_dram_buffer->address();
    uint32_t out_addr = out_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {q_addr, kt_addr, v_addr});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {out_addr});
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Create zero-filled input data, tilize, and upload to DRAM
    const uint32_t q_rows = num_q_chunks * Sq_chunk_t * TILE_HEIGHT;
    const uint32_t q_cols = head_dim_t * TILE_WIDTH;
    std::vector<bfloat16> q_data(q_rows * q_cols, bfloat16(0.0f));
    q_data = tilize_nfaces(q_data, q_rows, q_cols);

    const uint32_t kt_rows = head_dim_t * TILE_HEIGHT;
    const uint32_t kt_cols = num_k_chunks * Sk_chunk_t * TILE_WIDTH;
    std::vector<bfloat16> kt_data(kt_rows * kt_cols, bfloat16(0.0f));
    kt_data = tilize_nfaces(kt_data, kt_rows, kt_cols);

    const uint32_t v_rows = num_k_chunks * Sv_chunk_t * TILE_HEIGHT;
    const uint32_t v_cols = head_dim_t * TILE_WIDTH;
    std::vector<bfloat16> v_data(v_rows * v_cols, bfloat16(0.0f));
    v_data = tilize_nfaces(v_data, v_rows, v_cols);

    distributed::EnqueueWriteMeshBuffer(cq, q_dram_buffer, q_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, kt_dram_buffer, kt_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, v_dram_buffer, v_data, false);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read final SDPA output back from DRAM (blocking call waits for workload to finish).
    std::vector<bfloat16> result(out_num_tiles * TILE_HEIGHT * TILE_WIDTH);
    distributed::EnqueueReadMeshBuffer(cq, result, out_dram_buffer, true);
    fmt::print("Output read from DRAM successfully ({} tiles, {} elements)\n", out_num_tiles, result.size());
}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        // Open device
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr uint32_t Sq_chunk_t = 7;
        constexpr uint32_t Sk_chunk_t = 16;
        constexpr uint32_t Sv_chunk_t = 16;
        constexpr uint32_t head_dim_t = 128 / TILE_WIDTH;
        constexpr uint32_t subblock_h = 1;
        constexpr uint32_t num_q_chunks = 2;
        constexpr uint32_t num_k_chunks = 3;

        sdpa_single_core(
            Sq_chunk_t, Sk_chunk_t, Sv_chunk_t, head_dim_t, subblock_h, num_q_chunks, num_k_chunks, mesh_device);

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
