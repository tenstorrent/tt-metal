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

// ---------------------------------------------------------------------------
// Helper: create a CB with a single data format and uniform page size
// ---------------------------------------------------------------------------
static void create_cb(
    Program& program,
    const CoreCoord& core,
    uint32_t cb_index,
    uint32_t num_tiles,
    uint32_t tile_size,
    tt::DataFormat fmt) {
    CircularBufferConfig cfg =
        CircularBufferConfig(num_tiles * tile_size, {{cb_index, fmt}}).set_page_size(cb_index, tile_size);
    CreateCircularBuffer(program, core, cfg);
}

// ---------------------------------------------------------------------------
// Unified SDPA single-core runner
//   test_dir empty  → benchmark mode (zero-filled inputs, no file output)
//   test_dir set    → test mode (reads Q/K/V from files, writes output)
// ---------------------------------------------------------------------------

void sdpa_single_core(
    const uint32_t Sq_chunk_t,
    const uint32_t Sk_chunk_t,
    const uint32_t Sv_chunk_t,
    const uint32_t head_dim_t,
    const uint32_t subblock_h,
    const uint32_t num_q_chunks,
    const uint32_t num_k_chunks,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const uint32_t mm_throttle_level = 0,
    const bool exp_approx_mode = false,
    const uint32_t padded_k_tiles = 0,
    const std::string& test_dir = "") {
    const bool test_mode = !test_dir.empty();

    TT_FATAL(subblock_h == 1 || subblock_h == 2, "subblock_h must be 1 or 2. Got {}.", subblock_h);
    TT_FATAL(mm_throttle_level <= 5, "mm_throttle_level must be 0-5. Got {}.", mm_throttle_level);
    TT_FATAL(
        Sq_chunk_t % subblock_h == 0, "Sq_chunk_t ({}) must be divisible by subblock_h ({}).", Sq_chunk_t, subblock_h);
    TT_FATAL(head_dim_t <= 8, "head_dim_t ({}) must fit in DST (max 8 tiles with fp16b double-buffer).", head_dim_t);
    TT_FATAL(
        padded_k_tiles < Sk_chunk_t,
        "padded_k_tiles ({}) must be less than Sk_chunk_t ({}).",
        padded_k_tiles,
        Sk_chunk_t);

    // ---- Device / program setup ----
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    CoreCoord core({0, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    const MathFidelity math_fidelity = MathFidelity::HiFi2;
    const uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // ---- Tile counts ----
    const uint32_t q_chunk_tiles = Sq_chunk_t * head_dim_t;
    const uint32_t k_chunk_tiles = Sk_chunk_t * head_dim_t;
    const uint32_t v_chunk_tiles = Sv_chunk_t * head_dim_t;
    const uint32_t out_chunk_tiles = Sq_chunk_t * head_dim_t;

    const uint32_t q_num_tiles = num_q_chunks * q_chunk_tiles;
    const uint32_t k_num_tiles = num_k_chunks * k_chunk_tiles;
    const uint32_t v_num_tiles = num_k_chunks * v_chunk_tiles;
    const uint32_t out_num_tiles = num_q_chunks * out_chunk_tiles;

    // ---- DRAM buffers ----
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    auto make_dram_buf = [&](uint32_t num_tiles) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = num_tiles * single_tile_size}, dram_config, mesh_device.get());
    };
    auto q_dram_buffer = make_dram_buf(q_num_tiles);
    auto k_dram_buffer = make_dram_buf(k_num_tiles);
    auto v_dram_buffer = make_dram_buf(v_num_tiles);
    auto out_dram_buffer = make_dram_buf(out_num_tiles);

    // ---- Circular buffers ----
    const uint32_t qkt_tiles = Sq_chunk_t * Sk_chunk_t;

    create_cb(program, core, CBIndex::c_0, 2 * q_chunk_tiles, single_tile_size, cb_data_format);  // Q (double-buf)
    create_cb(program, core, CBIndex::c_1, 2 * k_chunk_tiles, single_tile_size, cb_data_format);  // KT (double-buf)
    create_cb(program, core, CBIndex::c_2, qkt_tiles, single_tile_size, cb_data_format);          // QKT
    create_cb(program, core, CBIndex::c_3, 2 * v_chunk_tiles, single_tile_size, cb_data_format);  // V (double-buf)
    create_cb(
        program, core, CBIndex::c_4, subblock_h * Sk_chunk_t, single_tile_size, cb_data_format);  // qkt_row_A (ping)
    create_cb(program, core, CBIndex::c_5, 1, single_tile_size, cb_data_format);                  // identity_scalar
    create_cb(
        program, core, CBIndex::c_6, subblock_h * Sk_chunk_t, single_tile_size, cb_data_format);  // qkt_row_B (pong)
    if (padded_k_tiles > 0) {
        create_cb(program, core, CBIndex::c_7, 1, single_tile_size, cb_data_format);  // neginf tile for padded mask
    }
    create_cb(program, core, CBIndex::c_8, 1, single_tile_size, cb_data_format);                  // col_identity
    create_cb(program, core, CBIndex::c_9, head_dim_t, single_tile_size, cb_data_format);         // normalized_out
    create_cb(program, core, CBIndex::c_10, 1, single_tile_size, cb_data_format);                 // recip_scratch
    create_cb(program, core, CBIndex::c_25, out_chunk_tiles, single_tile_size, cb_data_format);   // prev_out
    create_cb(program, core, CBIndex::c_26, out_chunk_tiles, single_tile_size, cb_data_format);   // curr_out
    create_cb(program, core, CBIndex::c_27, Sq_chunk_t, single_tile_size, cb_data_format);        // prev_max
    create_cb(program, core, CBIndex::c_28, Sq_chunk_t, single_tile_size, cb_data_format);        // cur_max
    create_cb(program, core, CBIndex::c_29, Sq_chunk_t, single_tile_size, cb_data_format);        // prev_sum
    create_cb(program, core, CBIndex::c_30, Sq_chunk_t, single_tile_size, cb_data_format);        // cur_sum
    create_cb(program, core, CBIndex::c_31, Sq_chunk_t, single_tile_size, cb_data_format);        // exp_max_diff

    // ---- Granularity check ----
    const uint32_t dst_size = 8;
    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_FATAL(
        sub_exp_granularity == (1 << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    // ---- Defines ----
    std::map<std::string, std::string> defines;
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    if (mm_throttle_level > 0) {
        defines["MM_THROTTLE"] = std::to_string(mm_throttle_level);
    }

    // ---- Reader kernel ----
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

    // ---- Writer kernel ----
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    std::vector<uint32_t> writer_compile_time_args = {
        Sq_chunk_t,
        Sk_chunk_t,
        Sv_chunk_t,
        head_dim_t,
        num_q_chunks,
        num_k_chunks,
        packed_identity_scalar,
        padded_k_tiles};
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

    // ---- Compute kernel ----
    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = 1.0f / std::sqrt(static_cast<float>(head_dim_t * TILE_WIDTH));

    std::vector<uint32_t> compute_compile_time_args = {
        Sq_chunk_t,
        Sk_chunk_t,
        Sv_chunk_t,
        head_dim_t,
        num_q_chunks,
        num_k_chunks,
        scale_union.u,
        subblock_h,
        padded_k_tiles};
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/compute/sdpa.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .compile_args = compute_compile_time_args, .defines = defines});

    // ---- Runtime args ----
    tt_metal::SetRuntimeArgs(
        program, reader_id, core, {q_dram_buffer->address(), k_dram_buffer->address(), v_dram_buffer->address()});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {out_dram_buffer->address()});

    // ---- Prepare and upload input data ----
    const uint32_t q_rows = num_q_chunks * Sq_chunk_t * TILE_HEIGHT;
    const uint32_t q_cols = head_dim_t * TILE_WIDTH;
    const uint32_t k_rows = num_k_chunks * Sk_chunk_t * TILE_HEIGHT;
    const uint32_t k_cols = head_dim_t * TILE_WIDTH;
    const uint32_t v_rows = num_k_chunks * Sv_chunk_t * TILE_HEIGHT;
    const uint32_t v_cols = head_dim_t * TILE_WIDTH;

    std::vector<bfloat16> q_data, k_data, v_data;

    if (test_mode) {
        fmt::print("Reading Q [{}, {}] from {}/q.bin\n", q_rows, q_cols, test_dir);
        q_data = read_bf16_bin(test_dir + "/q.bin", q_rows * q_cols);

        fmt::print("Reading K [{}, {}] from {}/k.bin\n", k_rows, k_cols, test_dir);
        k_data = read_bf16_bin(test_dir + "/k.bin", k_rows * k_cols);

        fmt::print("Reading V [{}, {}] from {}/v.bin\n", v_rows, v_cols, test_dir);
        v_data = read_bf16_bin(test_dir + "/v.bin", v_rows * v_cols);
    } else {
        q_data.assign(q_rows * q_cols, bfloat16(0.0f));
        k_data.assign(k_rows * k_cols, bfloat16(0.0f));
        v_data.assign(v_rows * v_cols, bfloat16(0.0f));
    }

    q_data = tilize_nfaces(q_data, q_rows, q_cols);
    k_data = tilize_nfaces(k_data, k_rows, k_cols);
    v_data = tilize_nfaces(v_data, v_rows, v_cols);

    distributed::EnqueueWriteMeshBuffer(cq, q_dram_buffer, q_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, k_dram_buffer, k_data, false);
    distributed::EnqueueWriteMeshBuffer(cq, v_dram_buffer, v_data, false);

    // ---- Execute ----
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // ---- Read output (blocking) ----
    std::vector<bfloat16> result(out_num_tiles * TILE_HEIGHT * TILE_WIDTH);
    distributed::EnqueueReadMeshBuffer(cq, result, out_dram_buffer, true);

    if (test_mode) {
        const uint32_t out_rows = num_q_chunks * Sq_chunk_t * TILE_HEIGHT;
        const uint32_t out_cols = head_dim_t * TILE_WIDTH;
        auto untilized_result = untilize_nfaces(result, out_rows, out_cols);

        std::string output_path = test_dir + "/device_output.bin";
        write_bf16_bin(output_path, untilized_result);
        fmt::print("Device output written to {} ({} elements)\n", output_path, untilized_result.size());
    } else {
        fmt::print("Output read from DRAM successfully ({} tiles, {} elements)\n", out_num_tiles, result.size());
    }
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
    std::string test_dir;

    // Check for --test <dir> mode
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--test" && i + 1 < argc) {
            test_dir = argv[i + 1];
            break;
        }
    }

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        uint32_t Sq_chunk_t = get_arg_uint(argc, argv, "--Sq_chunk_t", 7);
        uint32_t Sk_chunk_t = get_arg_uint(argc, argv, "--Sk_chunk_t", 16);
        uint32_t head_dim_t = get_arg_uint(argc, argv, "--head_dim_t", 4);
        uint32_t num_q_chunks = get_arg_uint(argc, argv, "--num_q_chunks", test_dir.empty() ? 2 : 3);
        uint32_t num_k_chunks = get_arg_uint(argc, argv, "--num_k_chunks", test_dir.empty() ? 3 : 5);
        uint32_t subblock_h = get_arg_uint(argc, argv, "--subblock_h", 1);
        uint32_t mm_throttle_level = get_arg_uint(argc, argv, "--mm_throttle_level", 0);
        bool exp_approx_mode = get_arg_uint(argc, argv, "--exp_approx_mode", 0) != 0;
        uint32_t padded_k_tiles = get_arg_uint(argc, argv, "--padded_k_tiles", 0);
        uint32_t Sv_chunk_t = Sk_chunk_t;  // Always equal

        if (!test_dir.empty()) {
            fmt::print(
                "Test mode: dir={}, Sq_chunk_t={}, Sk_chunk_t={}, head_dim_t={}, "
                "num_q_chunks={}, num_k_chunks={}, subblock_h={}, mm_throttle={}, exp_approx_mode={}, "
                "padded_k_tiles={}\n",
                test_dir,
                Sq_chunk_t,
                Sk_chunk_t,
                head_dim_t,
                num_q_chunks,
                num_k_chunks,
                subblock_h,
                mm_throttle_level,
                exp_approx_mode,
                padded_k_tiles);
        }

        sdpa_single_core(
            Sq_chunk_t,
            Sk_chunk_t,
            Sv_chunk_t,
            head_dim_t,
            subblock_h,
            num_q_chunks,
            num_k_chunks,
            mesh_device,
            mm_throttle_level,
            exp_approx_mode,
            padded_k_tiles,
            test_dir);

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
