#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "common/core_coord.h"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

std::tuple<tt_metal::Program, tt_metal::KernelID, tt_metal::KernelID> create_program(
    tt_metal::Device *device,
    uint32_t single_tile_size,
    const CoreRange &all_cores,
    const std::vector<uint32_t> &eltwise_unary_args) {
    tt_metal::Program program = tt_metal::Program();

    CoreCoord start_core = all_cores.start;
    CoreCoord end_core = all_cores.end;
    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto core = CoreCoord{x, y};
            uint32_t src0_cb_index = 0;
            uint32_t src0_cb_addr = 200 * 1024;
            uint32_t num_input_tiles = 8;
            auto cb_src0 = tt_metal::CreateCircularBuffer(
                program,
                src0_cb_index,
                core,
                num_input_tiles,
                num_input_tiles * single_tile_size,
                tt::DataFormat::Float16_b,
                src0_cb_addr
            );

            uint32_t ouput_cb_index = 16; // output operands start at index 16
            uint32_t output_cb_addr = 300 * 1024;
            uint32_t num_output_tiles = 1;
            auto cb_output = tt_metal::CreateCircularBuffer(
                program,
                ouput_cb_index,
                core,
                num_output_tiles,
                num_output_tiles * single_tile_size,
                tt::DataFormat::Float16_b,
                output_cb_addr
            );
        }
    }

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_push_4.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = eltwise_unary_args}
    );

    return {std::move(program), reader_kernel, writer_kernel};
}

void compile_and_configure_program(
    tt_metal::Device *device,
    tt_metal::Program &program,
    std::vector<uint32_t> &src_vec,
    tt_metal::Buffer &src_dram_buffer) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::WriteToBuffer(src_dram_buffer, src_vec);

    tt_metal::ConfigureDeviceWithProgram(device, program);
}

void set_rt_args(tt_metal::Program &program, tt_metal::KernelID kernel, const CoreRange &core_range, const std::vector<uint32_t> &rt_args) {
    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            CoreCoord core = CoreCoord(x, y);
            tt_metal::SetRuntimeArgs(program, kernel, core, rt_args);
        }
    }
}

void write_same_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program &program,
    tt_metal::KernelID reader_kernel_id,
    tt_metal::KernelID writer_kernel_id,
    const CoreRange &core_range,
    int32_t num_tiles,
    tt_metal::Buffer &src_dram_buffer,
    tt_metal::Buffer &dst_dram_buffer)
{
    auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

    std::vector<uint32_t> unary_reader_args{
    (std::uint32_t)src_dram_buffer.address(),
    (std::uint32_t)dram_src_noc_xy.x,
    (std::uint32_t)dram_src_noc_xy.y,
    (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args{
    (std::uint32_t)dst_dram_buffer.address(),
    (std::uint32_t)dram_dst_noc_xy.x,
    (std::uint32_t)dram_dst_noc_xy.y,
    (std::uint32_t)num_tiles};

    set_rt_args(program, reader_kernel_id, core_range, unary_reader_args);
    set_rt_args(program, writer_kernel_id, core_range, unary_writer_args);
    tt_metal::WriteRuntimeArgsToDevice(device, program);
}

void write_unique_writer_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program &program,
    tt_metal::KernelID reader_kernel_id,
    tt_metal::KernelID writer_kernel_id,
    const CoreRange &core_range,
    const CoreRangeSet &core_blocks,
    int32_t num_tiles,
    tt_metal::Buffer &src_dram_buffer,
    tt_metal::Buffer &dst_dram_buffer_1,
    tt_metal::Buffer &dst_dram_buffer_2,
    tt_metal::Buffer &dst_dram_buffer_3
) {
    auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
    // All dst buffers use the same DRAM channel
    auto dram_dst_noc_xy = dst_dram_buffer_1.noc_coordinates();

    // Same readers args because all kernels read from same src
    std::vector<uint32_t> unary_reader_args{
        (std::uint32_t)src_dram_buffer.address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args_1{
        dst_dram_buffer_1.address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args_2{
        dst_dram_buffer_2.address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    std::vector<uint32_t> unary_writer_args_3{
        dst_dram_buffer_3.address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    set_rt_args(program, reader_kernel_id, core_range, unary_reader_args);
    int core_range_idx = 0;
    std::vector<std::vector<uint32_t>> rt_args = {unary_writer_args_1, unary_writer_args_2, unary_writer_args_3};
    for (auto core_range : core_blocks.ranges()) {
        set_rt_args(program, writer_kernel_id, core_range, rt_args.at(core_range_idx++));
    }
    tt_metal::WriteRuntimeArgsToDevice(device, program);
}

bool test_multi_core_kernel_same_runtime_args(tt_metal::Device *device) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Buffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {2, 2};

    CoreRange all_cores{.start=start_core, .end=end_core};

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_src_addr = 0;

    uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

    auto src_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, dram_buffer_size, tt_metal::BufferType::DRAM);
    auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, dram_buffer_size, tt_metal::BufferType::DRAM);

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile Time Args Setup
    ////////////////////////////////////////////////////////////////////////////
    // Same compile time args for all cores
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile and Execute Program
    ////////////////////////////////////////////////////////////////////////////
    auto [program, reader_kernel_id, writer_kernel_id] = create_program(device, single_tile_size, all_cores, compute_kernel_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer.size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    compile_and_configure_program(device, program, src_vec, src_dram_buffer);

    write_same_runtime_args_to_device(device, program, reader_kernel_id, writer_kernel_id, all_cores, num_tiles, src_dram_buffer, dst_dram_buffer);

    tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> result_vec;
    tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

    ////////////////////////////////////////////////////////////////////////////
    //                          Validation
    ////////////////////////////////////////////////////////////////////////////
    pass &= (src_vec == result_vec);

    DeallocateBuffer(src_dram_buffer);
    DeallocateBuffer(dst_dram_buffer);

    return pass;
}

bool test_multi_core_kernel_unique_runtime_args(tt_metal::Device *device) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Buffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {1, 1};
    CoreRange start_core_range = {.start=start_core, .end=start_core};
    CoreRange core_group{.start={0, 1}, .end={1, 1}};
    CoreRange single_core = {.start={1, 0}, .end={1, 0}};
    CoreRange all_cores{.start=start_core, .end=end_core};
    CoreRangeSet core_blocks = CoreRangeSet({start_core_range, single_core, core_group});

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t dram_buffer_src_addr = 0;

    uint32_t dram_buffer_dst_addr_1 = 512 * 1024 * 1024; // 512 MB (upper half)
    uint32_t dram_buffer_dst_addr_2 = dram_buffer_dst_addr_1 + dram_buffer_size;
    uint32_t dram_buffer_dst_addr_3 = dram_buffer_dst_addr_2 + dram_buffer_size;

    auto src_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, dram_buffer_size, tt_metal::BufferType::DRAM);
    auto dst_dram_buffer_1 = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr_1, dram_buffer_size, tt_metal::BufferType::DRAM);
    auto dst_dram_buffer_2 = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr_2, dram_buffer_size, tt_metal::BufferType::DRAM);
    auto dst_dram_buffer_3 = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr_3, dram_buffer_size, tt_metal::BufferType::DRAM);

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile Time Args Setup
    ////////////////////////////////////////////////////////////////////////////
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile and Execute Program
    ////////////////////////////////////////////////////////////////////////////
    auto [program, reader_kernel_id, writer_kernel_id] = create_program(device, single_tile_size, all_cores, compute_kernel_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer.size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    compile_and_configure_program(device, program, src_vec, src_dram_buffer);

    write_unique_writer_runtime_args_to_device(
        device, program, reader_kernel_id, writer_kernel_id, all_cores, core_blocks, num_tiles, src_dram_buffer, dst_dram_buffer_1, dst_dram_buffer_2, dst_dram_buffer_3);

    tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> result_vec_1;
    tt_metal::ReadFromBuffer(dst_dram_buffer_1, result_vec_1);

    std::vector<uint32_t> result_vec_2;
    tt_metal::ReadFromBuffer(dst_dram_buffer_2, result_vec_2);

    std::vector<uint32_t> result_vec_3;
    tt_metal::ReadFromBuffer(dst_dram_buffer_3, result_vec_3);


    ////////////////////////////////////////////////////////////////////////////
    //                          Validation
    ////////////////////////////////////////////////////////////////////////////
    pass &= (src_vec == result_vec_1);
    pass &= (src_vec == result_vec_2);
    pass &= (src_vec == result_vec_3);

    DeallocateBuffer(src_dram_buffer);
    DeallocateBuffer(dst_dram_buffer_1);
    DeallocateBuffer(dst_dram_buffer_2);
    DeallocateBuffer(dst_dram_buffer_3);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;


    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        string arch_name = "";
        try {
            std::tie(arch_name, input_args) =
                test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        } catch (const std::exception& e) {
            log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
        }
        const tt::ARCH arch = tt::get_arch_from_string(arch_name);
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);
        tt_metal::InitializeDevice(device);

        pass &= test_multi_core_kernel_same_runtime_args(device);

        pass &= test_multi_core_kernel_unique_runtime_args(device);

        ////////////////////////////////////////////////////////////////////////////
        //                          Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
