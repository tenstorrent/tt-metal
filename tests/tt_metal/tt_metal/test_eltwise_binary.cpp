#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "llrt/tt_debug_print_server.hpp"
#include "test_gold_impls.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    bool pass = true;
    bool multibank = true;

    const char* op_id_to_op_define[] = {"add_tiles", "sub_tiles", "mul_tiles"};
    const char* op_id_to_op_code_define[] = {"0", "1", "2"};
    const char* op_id_to_op_name[] = {"ADD", "SUB", "MUL"};

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
    tt_metal::Device* device = tt_metal::CreateDevice(arch, pci_express_slot);

    pass &= tt_metal::InitializeDevice(device);
    tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}, {1, 11}});
    CommandQueue cq(device);

    Program programs[] = {tt_metal::Program(), tt_metal::Program(), tt_metal::Program()};

    auto ops = EltwiseOp::all();
    for (auto eltwise_op : ops) {
        log_info(LogTest, "====================================================================");
        log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);

        try {


            ////////////////////////////////////////////////////////////////////////////
            //                      Application Setup
            ////////////////////////////////////////////////////////////////////////////
            // tt_metal::Program program = tt_metal::Program();
            tt_metal::Program& program = programs[eltwise_op];

            CoreCoord core = {0, 0};

            uint32_t single_tile_size = 2 * 1024;
            uint32_t num_tiles = 2048;
            uint32_t dram_buffer_size =
                single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

            uint32_t dram_buffer_src0_addr = 0;
            int dram_src0_channel_id = 0;
            uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024;
            int dram_src1_channel_id = 1;
            uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024;  // 512 MB (upper half)
            int dram_dst_channel_id = 0;

            uint32_t page_size = single_tile_size;
            if (not multibank) {
                page_size = dram_buffer_size;
            }
            auto src0_dram_buffer = tt_metal::Buffer(
                device,
                dram_buffer_size,
                dram_buffer_src0_addr,
                dram_src0_channel_id,
                page_size,
                tt_metal::BufferType::DRAM);
            auto src1_dram_buffer = tt_metal::Buffer(
                device,
                dram_buffer_size,
                dram_buffer_src1_addr,
                dram_src1_channel_id,
                page_size,
                tt_metal::BufferType::DRAM);
            auto dst_dram_buffer = tt_metal::Buffer(
                device,
                dram_buffer_size,
                dram_buffer_dst_addr,
                dram_dst_channel_id,
                page_size,
                tt_metal::BufferType::DRAM);

            auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
            auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
            auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

            uint32_t src0_cb_index = 0;
            uint32_t src0_cb_addr = 200 * 1024;
            uint32_t num_input_tiles = 2;
            auto cb_src0 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src0_cb_index,
                core,
                num_input_tiles,
                num_input_tiles * single_tile_size,
                src0_cb_addr,
                tt::DataFormat::Float16_b);

            uint32_t src1_cb_index = 1;
            uint32_t src1_cb_addr = 300 * 1024;
            auto cb_src1 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src1_cb_index,
                core,
                num_input_tiles,
                num_input_tiles * single_tile_size,
                src1_cb_addr,
                tt::DataFormat::Float16_b);

            uint32_t ouput_cb_index = 16;  // output operands start at index 16
            uint32_t output_cb_addr = 400 * 1024;
            uint32_t num_output_tiles = 2;
            auto cb_output = tt_metal::CreateCircularBuffer(
                program,
                device,
                ouput_cb_index,
                core,
                num_output_tiles,
                num_output_tiles * single_tile_size,
                output_cb_addr,
                tt::DataFormat::Float16_b);

            auto binary_reader_kernel = tt_metal::CreateDataMovementKernel(
                program,
                multibank ? "tt_metal/kernels/dataflow/reader_dual_8bank.cpp"
                          : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
                core,
                tt_metal::DataMovementProcessor::RISCV_1,
                tt_metal::NOC::RISCV_1_default);

            auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
                program,
                multibank ? "tt_metal/kernels/dataflow/writer_unary_8bank.cpp"
                          : "tt_metal/kernels/dataflow/writer_unary.cpp",
                core,
                tt_metal::DataMovementProcessor::RISCV_0,
                tt_metal::NOC::RISCV_0_default);
            unary_writer_kernel->add_define("DEVICE_DISPATCH_MODE", "1");

            vector<uint32_t> compute_kernel_args = {
                2048,  // per_core_block_cnt
                1,     // per_core_block_size
            };

            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
                program,
                "tt_metal/kernels/compute/eltwise_binary.cpp",
                core,
                compute_kernel_args,
                MathFidelity::HiFi4,
                fp32_dest_acc_en,
                math_approx_mode);

            eltwise_binary_kernel->add_define("ELTWISE_OP", op_id_to_op_define[eltwise_op]);
            eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", op_id_to_op_code_define[eltwise_op]);

            ////////////////////////////////////////////////////////////////////////////
            //                      Compile Application
            ////////////////////////////////////////////////////////////////////////////
            pass &= tt_metal::CompileProgram(device, program);

            ////////////////////////////////////////////////////////////////////////////
            //                      Execute Application
            ////////////////////////////////////////////////////////////////////////////
            std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
                dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

            EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);

            std::vector<uint32_t> src1_vec;
            if (eltwise_op == EltwiseOp::MUL)
                // TODO(AP): this doesn't provide very good coverage
                // switch to a better test with different values like in reduce
                src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1.0f);
            else
                src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);

            EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

            vector<u32> reader_args = {
                dram_buffer_src0_addr,
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                num_tiles,
                dram_buffer_src1_addr,
                (std::uint32_t)dram_src1_noc_xy.x,
                (std::uint32_t)dram_src1_noc_xy.y,
                num_tiles,
                0};

            vector<u32> writer_args = {
                dram_buffer_dst_addr, (std::uint32_t)dram_dst_noc_xy.x, (std::uint32_t)dram_dst_noc_xy.y, num_tiles};

            SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
            SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

            EnqueueProgram(cq, program, false);
            std::vector<uint32_t> result_vec;
            EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

            ////////////////////////////////////////////////////////////////////////////
            //                      Validation & Teardown
            ////////////////////////////////////////////////////////////////////////////

            pass &= (src0_vec == result_vec);

        } catch (const std::exception& e) {
            pass = false;
            // Capture the exception error message
            log_error(LogTest, "{}", e.what());
            // Capture system call errors that may have returned from driver/kernel
            log_error(LogTest, "System error message: {}", std::strerror(errno));
        }
    }  // for EltwiseOp::all()

    pass &= tt_metal::CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
