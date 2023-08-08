#include <algorithm>
#include <functional>
#include <random>

#include "common/test_tiles.hpp"  // FIXME: Remove dependency on this or move to test_utils like tilize/untilize
#include "doctest.h"
#include "single_device_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/tilization.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::matmul {

void create_CBs_for_fused_matmul(
    tt_metal::Program& program,
    tt_metal::Device* device,
    CoreCoord core,
    bool activations_rm,
    bool output_rm,
    uint32_t M,
    uint32_t N,
    uint32_t in0_block_w,
    uint32_t out_subblock_h) {
    uint32_t num_bytes_for_df = 2;
    uint32_t in0_cb = 0;
    uint32_t in1_cb = 1;
    uint32_t tilize_mode_tilized_in0_cb = 24;
    uint32_t matmul_partials_cb = 25;
    uint32_t untilize_mode_final_matmul_partials_cb = 26;
    uint32_t untilize_mode_reblock_cb = 27;
    uint32_t out0_cb = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t src0_cb_addr = 120 * 1024;
    uint32_t cb0_tiles = M * in0_block_w * 2;
    auto cb_in0 = tt_metal::CreateCircularBuffer(
        program,
        in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b,
        src0_cb_addr);

    uint32_t src1_cb_addr = 250 * 1024;
    uint32_t cb1_tiles = N * in0_block_w * 2;
    auto cb_in1 = tt_metal::CreateCircularBuffer(
        program,
        in1_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        tt::DataFormat::Float16_b,
        src1_cb_addr);

    if (not activations_rm and not output_rm) {  // no tilize, no untilize
        uint32_t matmul_partials_addr = 400 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            matmul_partials_addr);

        // Partials share same L1 address space as output
        uint32_t output_cb_addr = matmul_partials_addr;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr);

    } else if (not activations_rm and output_rm) {  // no tilize, just untilize

        uint32_t matmul_partials_addr = 450 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            matmul_partials_addr);

        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        uint32_t temp_addr = 600 * 1024;
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            temp_addr);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_addr = 750 * 1024;
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            reblock_cb_addr);

        uint32_t output_cb_addr = 800 * 1024;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr);

    } else if (activations_rm and not output_rm) {  // just tilize, no untilize

        uint32_t tilized_cb_addr = 400 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            tilize_mode_tilized_in0_cb,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            tilized_cb_addr);

        uint32_t matmul_partials_addr = 550 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            matmul_partials_addr);

        uint32_t output_cb_addr = matmul_partials_addr;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            matmul_partials_addr);

    } else {  // tilize activations and untilize output

        // Used for placing tilized activations
        uint32_t tilized_cb_addr = 300 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            tilize_mode_tilized_in0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            tilized_cb_addr);

        uint32_t cb_matmul_partials_addr = 440 * 1024;
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            cb_matmul_partials_addr);

        // Shares same address space as matmul partials
        uint32_t temp_addr = 580 * 1024;
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            temp_addr);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_addr = 720 * 1024;
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            reblock_cb_addr);

        uint32_t output_cb_addr = 860 * 1024;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr);
    }
}
struct SingleCoreMatmulConfig {
    bool activations_rm = false;
    bool outputs_rm = false;
    size_t M = 0;
    size_t K = 0;
    size_t N = 0;
    size_t out_subblock_h = 0;
    size_t out_subblock_w = 0;
    size_t in0_block_w = 0;
    size_t input0_dram_channel = 0;
    size_t input0_dram_byte_address = 0;
    size_t input1_dram_channel = 0;
    size_t input1_dram_byte_address = 0;
    size_t output_dram_channel = 0;
    size_t output_dram_byte_address = 0;
    CoreCoord core = {};
};

bool single_core_matmul(tt_metal::Device* device, const SingleCoreMatmulConfig& cfg) {
    // Since running in slow dispatch mode
    tt::tt_metal::detail::GLOBAL_CQ.reset();
    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    uint32_t single_tile_size = 2 * 1024;
    tt::log_assert(
        cfg.M * cfg.in0_block_w * single_tile_size * 2 <= 150 * 1024, "input0 block must fit within 150kB of L1");
    tt::log_assert(
        cfg.N * cfg.in0_block_w * single_tile_size * 2 <= 100 * 1024, "input1 block must fit within 100kB of L1");
    tt::log_assert(cfg.M * cfg.N * single_tile_size <= 600 * 1024, "output block must fit within 600kB of L1");
    uint32_t dram_buffer_size_input0 =
        single_tile_size * cfg.M * cfg.K;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_input1 =
        single_tile_size * cfg.K * cfg.N;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_output =
        single_tile_size * cfg.M * cfg.N;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    auto input0_dram_buffer = tt_metal::Buffer(
        device,
        dram_buffer_size_input0,
        cfg.input0_dram_byte_address,
        dram_buffer_size_input0,
        tt_metal::BufferType::DRAM);
    auto input1_dram_buffer = tt_metal::Buffer(
        device,
        dram_buffer_size_input1,
        cfg.input1_dram_byte_address,
        dram_buffer_size_input1,
        tt_metal::BufferType::DRAM);
    auto output_dram_buffer = tt_metal::Buffer(
        device,
        dram_buffer_size_output,
        cfg.output_dram_byte_address,
        dram_buffer_size_output,
        tt_metal::BufferType::DRAM);

    auto input0_dram_noc_xy = input0_dram_buffer.noc_coordinates();
    auto input1_dram_noc_xy = input1_dram_buffer.noc_coordinates();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    std::vector<uint32_t> reader_rt_args{
        (std::uint32_t)cfg.input0_dram_byte_address,
        (std::uint32_t)input0_dram_noc_xy.x,
        (std::uint32_t)input0_dram_noc_xy.y,
        (std::uint32_t)cfg.input1_dram_byte_address,
        (std::uint32_t)input1_dram_noc_xy.x,
        (std::uint32_t)input1_dram_noc_xy.y,
        (std::uint32_t)(cfg.K / cfg.in0_block_w),                      // num_blocks
        (std::uint32_t)(cfg.M * cfg.in0_block_w),                      // input 0 block num tiles
        (std::uint32_t)(cfg.N * cfg.in0_block_w),                      // input 1 block num tiles
        (std::uint32_t)(cfg.M * cfg.in0_block_w * single_tile_size),   // input 0 block bytes
        (std::uint32_t)(cfg.N * cfg.in0_block_w * single_tile_size)};  // input 1 block bytes
    std::vector<uint32_t> writer_rt_args;
    string writer_kernel_name;
    if (cfg.outputs_rm) {
        writer_kernel_name = "tt_metal/kernels/dataflow/writer_unary.cpp";
        writer_rt_args = {
            (std::uint32_t)cfg.output_dram_byte_address,
            (std::uint32_t)output_dram_noc_xy.x,
            (std::uint32_t)output_dram_noc_xy.y,
            uint(cfg.M * cfg.N)};
    } else {
        writer_kernel_name = "tt_metal/kernels/dataflow/writer_unswizzle.cpp";
        writer_rt_args = {
            (std::uint32_t)cfg.output_dram_byte_address,
            (std::uint32_t)output_dram_noc_xy.x,
            (std::uint32_t)output_dram_noc_xy.y,
            (std::uint32_t)cfg.out_subblock_h,            // num tiles per sub block m
            (std::uint32_t)cfg.out_subblock_w,            // num tiles per sub block n
            (std::uint32_t)(cfg.M / cfg.out_subblock_h),  // num sub blocks m
            (std::uint32_t)(cfg.N / cfg.out_subblock_w),  // num sub blocks n
            (std::uint32_t)(
                cfg.out_subblock_w * single_tile_size *
                (cfg.N / cfg.out_subblock_w)),  // bytes offset to next row within sub-block
            (std::uint32_t)(
                cfg.out_subblock_h * cfg.out_subblock_w * single_tile_size *
                (cfg.N / cfg.out_subblock_w)),                        // bytes offset to next row of sub-blocks
            (std::uint32_t)(cfg.out_subblock_w * single_tile_size)};  // bytes offset to next sub-block
    }
    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        writer_kernel_name,
        cfg.core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_matmul_blocked.cpp",
        cfg.core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    int num_blocks = (cfg.K / cfg.in0_block_w);
    int in0_num_subblocks = (cfg.M / cfg.out_subblock_h);
    int in0_block_num_tiles = cfg.out_subblock_h * cfg.in0_block_w * in0_num_subblocks;
    int in0_subblock_num_tiles = cfg.out_subblock_h * cfg.in0_block_w;
    int in1_num_subblocks = (cfg.N / cfg.out_subblock_w);
    int in1_block_num_tiles = cfg.out_subblock_w * cfg.in0_block_w * in1_num_subblocks;
    int in1_per_core_w = cfg.out_subblock_w * in1_num_subblocks;
    int out_subblock_num_tiles = cfg.out_subblock_h * cfg.out_subblock_w;
    int in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / cfg.in0_block_w;

    create_CBs_for_fused_matmul(
        program,
        device,
        cfg.core,
        cfg.activations_rm,
        cfg.outputs_rm,
        cfg.M,
        cfg.N,
        cfg.in0_block_w,
        cfg.out_subblock_h);

    tt::log_assert(
        in0_subblock_h * cfg.in0_block_w * in0_num_subblocks == in0_block_num_tiles,
        "in0_subblock_h * cfg.in0_block_w * in0_num_subblocks == in0_block_num_tiles");
    tt::log_assert(cfg.in0_block_w == cfg.K, "Must match k tiles");

    vector<uint32_t> compute_kernel_args = {
        uint(cfg.in0_block_w),
        uint(in0_num_subblocks),
        uint(in0_block_num_tiles),
        uint(in0_subblock_num_tiles),
        uint(in0_subblock_h),

        uint(in1_num_subblocks),
        uint(in1_block_num_tiles),
        uint(in1_per_core_w),

        uint(num_blocks),

        uint(cfg.out_subblock_h),
        uint(cfg.out_subblock_w),
        uint(out_subblock_num_tiles),

        uint(cfg.activations_rm),
        uint(cfg.outputs_rm)};

    auto matmul_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/matmul_large_block.cpp",
        cfg.core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_identity = {};
    std::vector<uint32_t> packed_activation = {};
    auto activation = generate_uniform_random_vector<bfloat16>(
        1.0f,
        1.0f,
        dram_buffer_size_input0 / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    if (cfg.activations_rm) {
        packed_activation = pack_vector<uint32_t, bfloat16>(activation);
    } else {
        auto activations_tilized = tilize<bfloat16, 32, 32>(activation, cfg.M * 32, cfg.K * 32);
        auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
        packed_activation = pack_vector<uint32_t, bfloat16>(activations_tile_layout);
    }
    auto identity =
        generate_strided_vector<bfloat16>(0.0f, 1.0f, cfg.N * 32 + 1, 0, dram_buffer_size_input1 / bfloat16::SIZEOF);
    auto identity_tilized = tilize<bfloat16, 32, 32>(identity, cfg.K * 32, cfg.N * 32);
    auto identity_tile_layout = convert_to_tile_layout(identity_tilized);
    packed_identity = pack_vector<uint32_t, bfloat16>(identity_tile_layout);
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = packed_activation;  //

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::StartDebugPrintServer(device, {{1,1}} );
    pass &= tt_metal::CompileProgram(device, program);
    tt_metal::WriteToBuffer(input0_dram_buffer, packed_activation);
    tt_metal::WriteToBuffer(input1_dram_buffer, packed_identity);
    std::vector<uint32_t> input0_dram_readback_packed;
    tt_metal::ReadFromBuffer(input0_dram_buffer, input0_dram_readback_packed);
    REQUIRE(input0_dram_readback_packed == packed_activation);
    print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(input0_dram_readback_packed), 32);
    std::vector<uint32_t> input1_dram_readback_packed;
    tt_metal::ReadFromBuffer(input1_dram_buffer, input1_dram_readback_packed);
    REQUIRE(input1_dram_readback_packed == packed_identity);
    print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(input1_dram_readback_packed), 32);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::SetRuntimeArgs(program, reader_kernel, cfg.core, reader_rt_args);
    tt_metal::SetRuntimeArgs(program, writer_kernel, cfg.core, writer_rt_args);

    tt_metal::WriteRuntimeArgsToDevice(device, program);
    pass &= tt_metal::LaunchKernels(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> input0_l1_readback_packed;
    tt_metal::detail::ReadFromDeviceL1(device, cfg.core, 120 * 1024, 2 * single_tile_size, input0_l1_readback_packed);
    REQUIRE(input0_l1_readback_packed == packed_activation);
    std::vector<uint32_t> input1_l1_readback_packed;
    tt_metal::detail::ReadFromDeviceL1(device, cfg.core, 250 * 1024, 2 * single_tile_size, input1_l1_readback_packed);
    REQUIRE(input1_l1_readback_packed == packed_identity);

    // std::vector<uint32_t> input1_l1_readback_packed;
    // std::vector<uint32_t> output_l1_readback_packed;
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    auto dest_buffer_data_unpacked = unpack_vector<bfloat16, uint32_t>(dest_buffer_data);
    if (not cfg.outputs_rm) {
        dest_buffer_data_unpacked = convert_to_flat_layout(dest_buffer_data_unpacked);
        dest_buffer_data_unpacked = untilize<bfloat16, 32, 32>(dest_buffer_data_unpacked, cfg.M * 32, cfg.N * 32);
    }
    pass &=
        is_close_vectors<bfloat16>(activation, dest_buffer_data_unpacked, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.15f);
        });

    return pass;
}
bool single_tile_matmul(tt_metal::Device* device) {

    // Once this test is uplifted to use fast dispatch, this can be removed.
    tt::tt_metal::detail::GLOBAL_CQ.reset();
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core = {.x = 0, .y = 0};
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const uint32_t in0_l1_addr = 150 * 1024;
    const uint32_t in1_l1_addr = 200 * 1024;
    const uint32_t out_l1_addr = 250 * 1024;
    const uint32_t in0_dram_addr = 150 * 1024 * 1024;
    const uint32_t in1_dram_addr = 200 * 1024 * 1024;
    const uint32_t out_dram_addr = 250 * 1024 * 1024;
    const size_t byte_size = 1 * 2 * 32 * 32;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    auto input0_dram_buffer =
        tt_metal::Buffer(device, byte_size, in0_dram_addr, byte_size, tt_metal::BufferType::DRAM);
    auto input0_dram_noc_xy = input0_dram_buffer.noc_coordinates();
    auto input1_dram_buffer =
        tt_metal::Buffer(device, byte_size, in1_dram_addr, byte_size, tt_metal::BufferType::DRAM);
    auto input1_dram_noc_xy = input1_dram_buffer.noc_coordinates();
    auto output_dram_buffer =
        tt_metal::Buffer(device, byte_size, out_dram_addr, byte_size, tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input0_cb = tt_metal::CreateCircularBuffer(
        program, in0_cb_index, core, 1, byte_size, tt::DataFormat::Float16_b, in0_l1_addr);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(
        program, in1_cb_index, core, 1, byte_size, tt::DataFormat::Float16_b, in1_l1_addr);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program, out_cb_index, core, 1, byte_size, tt::DataFormat::Float16_b, out_l1_addr);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = {out_cb_index}});

    auto simple_matmul_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/single_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {in0_cb_index, in1_cb_index, out_cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f / 32.0f,
        1.0f / 32.0f,
        byte_size / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    // Setup the weights such that final result is the original input.

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = packed_input0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    tt_metal::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::WriteToBuffer(input1_dram_buffer, packed_input1);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)in1_dram_addr,
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)1,  // num_tiles
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)1,
        });

    tt_metal::WriteRuntimeArgsToDevice(device, program);
    pass &= tt_metal::LaunchKernels(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); });
    return pass;
}
// blocked matmul has blocking, but still fits within dst, so no spill/reloads or intermediates
bool single_block_matmul(tt_metal::Device* device, uint32_t M, uint32_t K, uint32_t N) {
    // Once this test is uplifted to use fast dispatch, this can be removed.
    tt::tt_metal::detail::GLOBAL_CQ.reset();
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core = {.x = 0, .y = 0};
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const uint32_t in0_l1_addr = 256 * 1024;
    const uint32_t in1_l1_addr = 512 * 1024;
    const uint32_t out_l1_addr = 768 * 1024;
    const uint32_t in0_dram_addr = 256 * 1024 * 1024;
    const uint32_t in1_dram_addr = 512 * 1024 * 1024;
    const uint32_t out_dram_addr = 768 * 1024 * 1024;
    const size_t in0_byte_size = M * K * 2 * 32 * 32;
    const size_t in1_byte_size = K * N * 2 * 32 * 32;
    const size_t out_byte_size = M * N * 2 * 32 * 32;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    auto input0_dram_buffer =
        tt_metal::Buffer(device, in0_byte_size, in0_dram_addr, in0_byte_size, tt_metal::BufferType::DRAM);
    auto input0_dram_noc_xy = input0_dram_buffer.noc_coordinates();
    auto input1_dram_buffer =
        tt_metal::Buffer(device, in1_byte_size, in1_dram_addr, in1_byte_size, tt_metal::BufferType::DRAM);
    auto input1_dram_noc_xy = input1_dram_buffer.noc_coordinates();
    auto output_dram_buffer =
        tt_metal::Buffer(device, out_byte_size, out_dram_addr, out_byte_size, tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input0_cb = tt_metal::CreateCircularBuffer(
        program, in0_cb_index, core, M * K, in0_byte_size, tt::DataFormat::Float16_b, in0_l1_addr);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(
        program, in1_cb_index, core, K * N, in1_byte_size, tt::DataFormat::Float16_b, in1_l1_addr);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program, out_cb_index, core, M * N, out_byte_size, tt::DataFormat::Float16_b, out_l1_addr);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = {out_cb_index}});

    auto simple_matmul_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/multi_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {in0_cb_index, in1_cb_index, out_cb_index, M * K, K * N, M * N, M, N, K}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f, 1.0f, in0_byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.03125f,
        0.03125f,
        in1_byte_size / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f * K,
        1.0f * K,
        (out_byte_size) / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::StartDebugPrintServer(device, {{1,1}} );
    pass &= tt_metal::CompileProgram(device, program);
    tt_metal::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::WriteToBuffer(input1_dram_buffer, packed_input1);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)in1_dram_addr,
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)1,              // num_blocks
            (uint32_t)M * K,          // in0_block_tile_cnt
            (uint32_t)K * N,          // in1_block_tile_cnt
            (uint32_t)in0_byte_size,  // in0_block_size_bytes
            (uint32_t)in1_byte_size,  // in1_block_size_bytes
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)M * N,
        });
    tt_metal::WriteRuntimeArgsToDevice(device, program);
    pass &= tt_metal::LaunchKernels(device, program);
    sleep(1);
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    int failed_index;
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); },
        &failed_index);
    if (not pass) {
        log_info("Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(dest_buffer_data), 32);
    }
    return pass;
}
// blocked matmul has blocking on output, spill/reloads using intermediate
bool blocked_matmul(tt_metal::Device* device, uint32_t M, uint32_t K, uint32_t N) {
    // Once this test is uplifted to use fast dispatch, this can be removed.
    tt::tt_metal::detail::GLOBAL_CQ.reset();
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core = {.x = 0, .y = 0};
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const uint32_t partials_cb_index = 24;
    const uint32_t in0_l1_addr = 128 * 1024;
    const uint32_t in1_l1_addr = 256 * 1024;
    const uint32_t out_l1_addr = 512 * 1024;
    const uint32_t partials_l1_addr = 768 * 1024;
    const uint32_t in0_dram_addr = 256 * 1024 * 1024;
    const uint32_t in1_dram_addr = 512 * 1024 * 1024;
    const uint32_t out_dram_addr = 768 * 1024 * 1024;
    const size_t in0_byte_size = M * K * 2 * 32 * 32;
    const size_t in1_byte_size = K * N * 2 * 32 * 32;
    const size_t out_byte_size = M * N * 2 * 32 * 32;
    const size_t num_blocks = 1;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    auto input0_dram_buffer =
        tt_metal::Buffer(device, in0_byte_size, in0_dram_addr, in0_byte_size, tt_metal::BufferType::DRAM);
    auto input0_dram_noc_xy = input0_dram_buffer.noc_coordinates();
    auto input1_dram_buffer =
        tt_metal::Buffer(device, in1_byte_size, in1_dram_addr, in1_byte_size, tt_metal::BufferType::DRAM);
    auto input1_dram_noc_xy = input1_dram_buffer.noc_coordinates();
    auto output_dram_buffer =
        tt_metal::Buffer(device, out_byte_size, out_dram_addr, out_byte_size, tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input0_cb = tt_metal::CreateCircularBuffer(
        program, in0_cb_index, core, M * K, in0_byte_size, tt::DataFormat::Float16_b, in0_l1_addr);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(
        program, in1_cb_index, core, K * N, in1_byte_size, tt::DataFormat::Float16_b, in1_l1_addr);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program, out_cb_index, core, M * N, out_byte_size, tt::DataFormat::Float16_b, out_l1_addr);
    auto l1_partials_cb = tt_metal::CreateCircularBuffer(
        program, partials_cb_index, core, M * N, out_byte_size, tt::DataFormat::Float16_b, partials_l1_addr);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = {out_cb_index}});

    auto simple_matmul_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {in0_cb_index, in1_cb_index, out_cb_index, partials_cb_index, M * K, K * N, M * N, M, N, K, num_blocks}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f, 1.0f, in0_byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.03125f,
        0.03125f,
        in1_byte_size / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f * K,
        1.0f * K,
        (out_byte_size) / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::StartDebugPrintServer(device, {{1,1}} );
    pass &= tt_metal::CompileProgram(device, program);
    tt_metal::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::WriteToBuffer(input1_dram_buffer, packed_input1);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)in1_dram_addr,
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)1,              // num_blocks
            (uint32_t)M * K,          // in0_block_tile_cnt
            (uint32_t)K * N,          // in1_block_tile_cnt
            (uint32_t)in0_byte_size,  // in0_block_size_bytes
            (uint32_t)in1_byte_size,  // in1_block_size_bytes
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)M * N,
        });
    tt_metal::WriteRuntimeArgsToDevice(device, program);
    pass &= tt_metal::LaunchKernels(device, program);
    sleep(1);
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    int failed_index;
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); },
        &failed_index);
    if (not pass) {
        log_info("Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(dest_buffer_data), 32);
    }
    return pass;
}
}  // namespace unit_tests::compute::matmul
TEST_SUITE("MatmulCompute") {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "SingleCore") {
        SUBCASE("SingleTile") { REQUIRE(unit_tests::compute::matmul::single_tile_matmul(this->device_)); }
        SUBCASE("SingleBlockSingleTile") {
            REQUIRE(unit_tests::compute::matmul::single_block_matmul(this->device_, 1, 1, 1));
        }
        SUBCASE("SingleBlockSingleTileAccumulation") {
            REQUIRE(unit_tests::compute::matmul::single_block_matmul(this->device_, 1, 2, 1));
        }
        SUBCASE("SingleBlockNoAccumulation") {
            REQUIRE(unit_tests::compute::matmul::single_block_matmul(this->device_, 2, 1, 2));
        }
        // SUBCASE("BlockedSingleTile") {
        //     unit_tests::compute::matmul::SingleCoreMatmulConfig config = {
        //         .activations_rm = false,
        //         .outputs_rm = false,
        //         .input0_dram_channel = 0,
        //         .input0_dram_byte_address = 0,
        //         .input1_dram_channel = 0,
        //         .input1_dram_byte_address = 256 * 1024 * 1024,  // 256 MB
        //         .output_dram_channel = 0,
        //         .output_dram_byte_address = 512 * 1024 * 1024,  // 512 MB
        //         .core = {.x = 0, .y = 0}};
        //     config.M = 1;
        //     config.K = 1;
        //     config.N = 1;
        //     config.out_subblock_h = 1;
        //     config.out_subblock_w = 1;
        //     config.in0_block_w = 1;
        //     config.activations_rm = true;
        //     config.outputs_rm = true;
        //     REQUIRE(unit_tests::compute::matmul::single_core_matmul(this->device_, config));
        // }
    }
}
