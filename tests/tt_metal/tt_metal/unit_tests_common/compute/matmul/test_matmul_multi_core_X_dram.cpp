// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tests/tt_metal/test_utils/tilization.hpp"
#include "tests/tt_metal/test_utils/print_helpers.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/compute/matmul/matmul_utils.hpp"
using namespace tt;

namespace unit_tests_common::matmul::test_matmul_multi_core_X_dram {

struct MatmulConfig{
    int multi_dram;
    std::string reader_kernel;
    std::string writer_kernel;
};

std::tuple<tt_metal::Program, tt_metal::KernelHandle , tt_metal::KernelHandle> create_program(
    tt_metal::Device *device,
    const MatmulConfig &cfg,
    int num_cores_r, int num_cores_c,
    int per_core_M, int per_core_N, int K,
    int in0_block_w, int out_subblock_h, int out_subblock_w) {

    tt_metal::Program program = tt_metal::CreateProgram();
    uint32_t single_tile_size = 2 * 1024;
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size;  // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size;  // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;
    TT_FATAL(in0_CB_size <= 130*1024, "in0_CB_size {} too large", in0_CB_size);
    TT_FATAL(in1_CB_size <= 130*1024, "in1_CB_size {} too large", in1_CB_size);
    TT_FATAL(out_CB_size <= 540*1024, "out_CB_size {} too large", out_CB_size);

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};
    const CoreRange all_cores(start_core, end_core);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {ouput_cb_index, tt::DataFormat::Float16_b},
        {interm0_cb_index, tt::DataFormat::Float16_b}
    };
    int cores_r = cfg.multi_dram ? 1 : num_cores_r;
    int cores_c = cfg.multi_dram ? 1 : num_cores_c;
    for(int i = 0; i < cores_r; i++) {
        for(int j = 0; j < cores_c; j++) {
            uint32_t src0_cb_index = 0;
            uint32_t cb0_tiles = in0_block_tiles * 2;  // double buffer
            tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);

            uint32_t src1_cb_index = 1;
            uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
            tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);

            tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(out_CB_size, partials_and_out_data_format_spec)
                .set_page_size(ouput_cb_index, single_tile_size)
                .set_page_size(interm0_cb_index, single_tile_size);
            if (cfg.multi_dram){
                auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
                auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
                auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);
            }else{
                CoreCoord core = {(std::size_t) j, (std::size_t) i};
                CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});
                auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
                auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
                auto cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_output_config);
            }
        }
    }

    auto mm_reader_kernel = tt_metal::CreateKernel(
        program,
        cfg.reader_kernel,
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        cfg.writer_kernel,
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    int num_blocks = (K / in0_block_w);

    int in0_num_subblocks = (per_core_M / out_subblock_h);
    int in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    int in1_num_subblocks = (per_core_N / out_subblock_w);
    int in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    int in1_per_core_w = out_subblock_w * in1_num_subblocks;

    int out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        uint(in0_block_w),
        uint(in0_num_subblocks),
        uint(in0_block_num_tiles),
        uint(in0_subblock_num_tiles),

        uint(in1_num_subblocks),
        uint(in1_block_num_tiles),
        uint(in1_per_core_w),

        uint(num_blocks),

        uint(out_subblock_h),
        uint(out_subblock_w),
        uint(out_subblock_num_tiles)
    };

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    return {std::move(program), mm_reader_kernel, unary_writer_kernel};
}

bool matmul_multi_core_single_dram(tt_metal::Device *device){
    bool pass = true;
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    int num_cores_r = compute_with_storage_grid_size.y;
    int num_cores_c = compute_with_storage_grid_size.x;
    log_info(LogTest, "Num cores r = {}, Num cores c = {}", num_cores_r, num_cores_c);
    uint32_t M = 16 * num_cores_r;
    uint32_t K = 16 * 12;
    uint32_t N = 16 * num_cores_c;
    int out_subblock_h = 4;
    int out_subblock_w = 2;
    int in0_block_w = 2;
    int per_core_M = M / num_cores_r;
    int per_core_N = N / num_cores_c;
    uint32_t single_tile_size = 2 * 1024;
    uint32_t dram_unreserved_base = device->get_base_allocator_addr(HalMemType::DRAM);
    log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
    log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
    log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
    log_info(LogTest, "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_h, in0_block_w, K / in0_block_w, M / out_subblock_h);
    log_info(LogTest, "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}", out_subblock_w, in0_block_w, K / in0_block_w, N / out_subblock_w);
    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 identity
    auto golden = select_columns(tensor.get_values(), M, K, N);

    MatmulConfig matmul_cfg = {
        .multi_dram = 0,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        .writer_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp"
    };

    auto [program, mm_reader_kernel, unary_writer_kernel] = unit_tests_common::matmul::test_matmul_multi_core_X_dram::create_program (
        device,
        matmul_cfg, // not multi dram
        num_cores_r, num_cores_c,
        per_core_M, per_core_N, K,
        in0_block_w, out_subblock_h, out_subblock_w);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    log_debug(LogTest, "Slicing input tensors and copying them to dram along with sending runtime args to device");
    for(int i = 0; i < num_cores_r; i++) {
        std::vector<bfloat16> activation_slice = get_row_slice(tensor.get_values(), num_cores_r, i, M * 32, K * 32);
        for(int j = 0; j < num_cores_c; j++) {
            std::vector<bfloat16> weights_slice = get_col_slice(identity, num_cores_c, j, K * 32, N * 32);
            int core_index = i * num_cores_c + j;
            CoreCoord core = {(std::size_t) j, (std::size_t) i};

            uint32_t dram_buffer_src0_addr = (  core_index * per_core_M * K * single_tile_size) + dram_unreserved_base;
            int dram_src0_channel_id = 0;
            uint32_t dram_buffer_src1_addr = (core_index * K * per_core_N * single_tile_size) + dram_unreserved_base;
            int dram_src1_channel_id = 1;
            uint32_t dram_buffer_dst_addr = (core_index * per_core_M * per_core_N * single_tile_size) + dram_unreserved_base;
            int dram_dst_channel_id = 2;

            uint32_t dram_buffer_size_act = single_tile_size * per_core_M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
            uint32_t dram_buffer_size_weights = single_tile_size * K * per_core_N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
            uint32_t dram_buffer_size_out = single_tile_size * per_core_M * per_core_N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

            TT_FATAL(
                dram_buffer_src0_addr + dram_buffer_size_act < 1024 * 1024 * 1024,
                "addr {} + size {} too large",
                dram_buffer_src0_addr,
                dram_buffer_size_act);
            TT_FATAL(
                dram_buffer_src1_addr + dram_buffer_size_weights < 1024 * 1024 * 1024,
                "addr {} + size {} too large",
                dram_buffer_src1_addr,
                dram_buffer_size_weights);
            TT_FATAL(
                dram_buffer_dst_addr + dram_buffer_size_out < 1024 * 1024 * 1024,
                "addr {} + size {} too large",
                dram_buffer_dst_addr,
                dram_buffer_size_out);

            auto dram_src0_noc_xy = device->dram_core_from_dram_channel(dram_src0_channel_id);
            auto dram_src1_noc_xy = device->dram_core_from_dram_channel(dram_src1_channel_id);
            auto dram_dst_noc_xy = device->dram_core_from_dram_channel(dram_dst_channel_id);

            auto activations_tilized = test_utils::tilize(activation_slice, per_core_M * 32, K * 32);
            auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
            auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
            auto activations_tile_transposed = transpose_tiles(activations, per_core_M, K, in0_block_w);
            pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_src0_channel_id, dram_buffer_src0_addr, activations_tile_transposed);

            auto identity_tilized = test_utils::tilize(weights_slice, K * 32, per_core_N * 32);
            auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
            auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
            pass &= tt_metal::detail::WriteToDeviceDRAMChannel(device, dram_src1_channel_id, dram_buffer_src1_addr, weights);

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t) dram_buffer_src0_addr,
                (std::uint32_t) dram_src0_noc_xy.x,
                (std::uint32_t) dram_src0_noc_xy.y,
                (std::uint32_t) dram_buffer_src1_addr,
                (std::uint32_t) dram_src1_noc_xy.x,
                (std::uint32_t) dram_src1_noc_xy.y,
                (std::uint32_t) (K/in0_block_w), // num_blocks
                (std::uint32_t) per_core_M * in0_block_w, // input 0 block num tiles
                (std::uint32_t) per_core_N * in0_block_w, // input 1 block num tiles
                (std::uint32_t) per_core_M * in0_block_w * single_tile_size, // input 0 block bytes
                (std::uint32_t) per_core_N * in0_block_w * single_tile_size};

            std::vector<uint32_t> writer_args = {
                (std::uint32_t) dram_buffer_dst_addr,
                (std::uint32_t) dram_dst_noc_xy.x,
                (std::uint32_t) dram_dst_noc_xy.y,
                (std::uint32_t) out_subblock_h, // num tiles per sub block m
                (std::uint32_t) out_subblock_w, // num tiles per sub block n
                (std::uint32_t) per_core_M/out_subblock_h, // num sub blocks m
                (std::uint32_t) per_core_N/out_subblock_w, // num sub blocks n
                (std::uint32_t) out_subblock_w * single_tile_size * (per_core_N/out_subblock_w), // bytes offset to next row within sub-block
                (std::uint32_t) out_subblock_h * out_subblock_w * single_tile_size * (per_core_N/out_subblock_w), // bytes offset to next row of sub-blocks
                (std::uint32_t) out_subblock_w * single_tile_size};

            tt_metal::SetRuntimeArgs(program, mm_reader_kernel, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
        }
    }

    log_debug(LogTest, "Copying inputs to dram and runtime args to cores complete");

    log_debug(LogTest, "Running Matmul {} core test", num_cores_c * num_cores_r);

    tt_metal::detail::LaunchProgram(device, program);
    log_debug(LogTest, "Matmul test done");
    log_debug(LogTest, "Gathering data back from dram and checking against golden");
    for(int i = 0; i < num_cores_r; i++) {
        auto golden_row = get_row_slice(golden, num_cores_r, i, M * 32, N * 32);
        for(int j = 0; j < num_cores_c; j++) {
            auto per_core_golden = get_col_slice(golden_row, num_cores_c, j, per_core_M * 32, N * 32);
            std::vector<uint32_t> result_vec;
            int core_index = i * num_cores_c + j;
            uint32_t dram_buffer_dst_addr = (core_index * per_core_M * per_core_N * single_tile_size) + dram_unreserved_base;
            int dram_dst_channel_id = 2;
            tt_metal::detail::ReadFromDeviceDRAMChannel(device, dram_dst_channel_id, dram_buffer_dst_addr, per_core_M * per_core_N * single_tile_size, result_vec);
            auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
            auto result_flat_layout = convert_to_flat_layout(result_bfp16);
            auto result_untilized = test_utils::untilize(result_flat_layout, per_core_M*32, per_core_N*32);
            pass &= (per_core_golden == result_untilized);
        }
    }
    log_debug(LogTest, "Golden check complete");
    return pass;
}

bool assign_runtime_args_to_program(
    tt_metal::Device *device,
    tt_metal::Program &program,
    int num_cores_r,
    int num_cores_c,
    tt_metal::KernelHandle mm_reader_kernel,
    tt_metal::KernelHandle unary_writer_kernel,
    int M,
    int N,
    int K,
    int in0_block_w,
    int out_subblock_h,
    int out_subblock_w,
    int per_core_M,
    int per_core_N,
    uint32_t in0_dram_addr,
    uint32_t in1_dram_addr,
    uint32_t out_dram_addr) {
    bool pass = true;
    uint32_t single_tile_size = 2 * 1024;

    uint32_t dram_buffer_size_act =
        single_tile_size * M * K;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_weights =
        single_tile_size * K * N;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_out =
        single_tile_size * M * N;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    TT_FATAL(in0_dram_addr + dram_buffer_size_act < 1024 * 1024 * 1024, "addr {} + size {} too large", in0_dram_addr, dram_buffer_size_act);
    TT_FATAL(in1_dram_addr + dram_buffer_size_weights < 1024 * 1024 * 1024, "addr {} + size {} too large", in1_dram_addr, dram_buffer_size_weights);
    TT_FATAL(out_dram_addr + dram_buffer_size_out < 1024 * 1024 * 1024, "addr {} + size {} too large", out_dram_addr, dram_buffer_size_out);

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

            std::vector<uint32_t> mm_reader_args = {
                (std::uint32_t)in0_dram_addr,                // in0_tensor_addr
                (std::uint32_t)K * per_core_M * core_idx_y,  // in0_tensor_start_tile_id
                (std::uint32_t)1,                            // in0_tensor_stride_w
                (std::uint32_t)K,                            // in0_tensor_stride_h
                (std::uint32_t)in0_block_w,                  // in0_tensor_next_block_stride

                (std::uint32_t)in0_block_w,                  // in0_block_w
                (std::uint32_t)per_core_M,                   // in0_block_h
                (std::uint32_t)in0_block_w * per_core_M,     // in0_block_num_tiles

                (std::uint32_t)in1_dram_addr,                // in1_tensor_addr
                (std::uint32_t)per_core_N * core_idx_x,      // in1_tensor_start_tile_id
                (std::uint32_t)1,                            // in1_tensor_stride_w
                (std::uint32_t)N,                            // in1_tensor_stride_h
                (std::uint32_t)in0_block_w * N,              // in1_tensor_next_block_stride

                (std::uint32_t)per_core_N,                   // in1_block_w
                (std::uint32_t)in0_block_w,                  // in1_block_h
                (std::uint32_t)per_core_N * in0_block_w,     // in1_block_num_tiles

                (std::uint32_t)K / in0_block_w               // num_blocks
            };

            std::vector<uint32_t> writer_args = {
                (std::uint32_t)out_dram_addr,                                          // out_tensor_addr
                (std::uint32_t)core_idx_x * per_core_N + core_idx_y * per_core_M * N,  // out_tensor_start_tile_id
                (std::uint32_t)1,                                                      // out_tensor_stride_w
                (std::uint32_t)N,                                                      // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,                     // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * N,                 // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_N / out_subblock_w),      // out_num_subblocks_w
                (std::uint32_t)(per_core_M / out_subblock_h),      // out_num_subblocks_h
            };

            tt_metal::SetRuntimeArgs(program, mm_reader_kernel, core, mm_reader_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
        }
    }
    return pass;
}

bool matmul_multi_core_multi_dram(CommonFixture *fixture, tt_metal::Device *device){
    bool pass = true;
    int num_cores_r = device->compute_with_storage_grid_size().y;
    int num_cores_c = device->compute_with_storage_grid_size().x;
    uint32_t M = 16 * num_cores_r;
    uint32_t K = 16 * 12;
    uint32_t N = 16 * num_cores_c;
    int out_subblock_h = 4;
    int out_subblock_w = 2;
    int in0_block_w = 2;
    int per_core_M = M / num_cores_r;
    int per_core_N = N / num_cores_c;
    uint32_t single_tile_size = 2 * 1024;
    log_info(LogTest, "num_cores_r={}, num_cores_c={}", num_cores_r, num_cores_c);
    log_info(LogTest, "M = {}, N = {}, K = {}", M, N, K);
    log_info(LogTest, "Activation = {}x{}", M * 32, K * 32);
    log_info(LogTest, "Weights = {}x{}", K * 32, N * 32);
    log_info(
        LogTest,
        "Activation block = {}x{}, #blocks = {}, #sub-blocks = {}",
        per_core_M,
        in0_block_w,
        K / in0_block_w,
        per_core_M / out_subblock_h);
    log_info(
        LogTest,
        "Weights block = {}x{}, #blocks = {}, #sub-blocks = {}",
        in0_block_w,
        per_core_N,
        K / in0_block_w,
        per_core_N / out_subblock_w);
    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape,
        tt::deprecated::Initialize::RANDOM,
        100,
        10 /* seed */);
    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32);  // bfloat16 identity

    MatmulConfig matmul_cfg = {
        .multi_dram = 1,
        .reader_kernel =  "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_tile_layout.cpp",
        .writer_kernel =  "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_matmul_tile_layout.cpp"
    };

    auto [program, mm_reader_kernel, unary_writer_kernel] = unit_tests_common::matmul::test_matmul_multi_core_X_dram::create_program(
        device,
        matmul_cfg,
        num_cores_r, num_cores_c,
        per_core_M, per_core_N, K,
        in0_block_w, out_subblock_h, out_subblock_w
        );

    // CommandQueue& cq = device->command_queue();

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    log_debug(LogTest, "Scattering inputs (activation & weights) to dram channels using tiled layout");
    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);

    auto activation_buffer = std::make_shared<Buffer>(device, activations.size() * sizeof(uint32_t), 1024 * 2, BufferType::DRAM);
    pass &= move_tiles_to_dram(device, activations, M, K, activation_buffer);
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);

    auto weight_buffer = std::make_shared<Buffer>(device, weights.size() * sizeof(uint32_t), 1024 * 2, BufferType::DRAM);
    pass &= move_tiles_to_dram(device, weights, K, N, weight_buffer);
    log_debug(LogTest, "Copying inputs to dram complete");

    auto out_buffer = std::make_shared<Buffer>(device, M * N * sizeof(uint32_t) * 32 * 32, 1024 * 2, BufferType::DRAM);
    uint32_t out_dram_addr = out_buffer->address();

    log_debug(LogTest, "Writing kernel runtime args to device");
    pass &= unit_tests_common::matmul::test_matmul_multi_core_X_dram::assign_runtime_args_to_program(
        device,
        program,
        num_cores_r,
        num_cores_c,
        mm_reader_kernel,
        unary_writer_kernel,
        M,
        N,
        K,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        activation_buffer->address(),
        weight_buffer->address(),
        out_dram_addr);
    log_debug(LogTest, "Writing kernel runtime args to device complete");

    log_debug(LogTest, "Running Matmul {} core test", num_cores_r * num_cores_c);
    fixture->RunProgram(device, program);

    log_debug(LogTest, "Matmul test done");
    log_debug(LogTest, "Gathering data back from dram and checking against golden");

    vector<uint32_t> result;
    fixture->ReadBuffer(device, out_buffer, result);
    auto golden = select_columns(tensor.get_values(), M, K, N);

    // Keeping this old code because took me too long to decipher. Matmul
    // owner can refactor at a later time
    auto result_iter = result.begin();
    for(int i = 0; i < M; i++) {
        auto row = get_row_slice(golden, M, i, M * 32, N * 32);
        for(int j = 0; j < N; j++) {
            auto golden_tile = get_col_slice(row, N, j, 32, N * 32);
            std::vector<uint32_t> result_vec;
            result_vec.insert(result_vec.end(), result_iter, result_iter + 512);
            result_iter += 512;
            auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
            auto result_flat_layout = convert_to_flat_layout(result_bfp16);

            pass &= (golden_tile == result_flat_layout);
        }
    }
    log_debug(LogTest, "Golden check complete");
    return pass;
}

}

TEST_F(CommonFixture, MatmulMultiCoreSingleDRAM){
    const char* arch = getenv("ARCH_NAME");
    if (!getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        log_info(LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    } else if ( strcasecmp(arch,"wormhole_b0") == 0 ) {
        tt::log_info("This test is disabled in WH B0");
        GTEST_SKIP();
    }
    for(unsigned int id=0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_multi_core_X_dram::matmul_multi_core_single_dram(devices_.at(id)));
    }
}

TEST_F(CommonFixture, MatmulMultiCoreMultiDRAM){
    // need to update move_tiles_to_dram to support both slow and fast
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE")){
        log_info(LogTest, "This test is not supported in slow dispatch mode, need to update move_tiles_to_dram..");
        GTEST_SKIP();
    }
    for (unsigned int id = 0; id < devices_.size(); id++) {
        if (this->devices_.at(id)->arch() == tt::ARCH::BLACKHOLE) {
            GTEST_SKIP();
        }
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_multi_core_X_dram::matmul_multi_core_multi_dram(this, devices_.at(id)));
    }
}
