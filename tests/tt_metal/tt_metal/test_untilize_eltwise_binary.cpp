#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"
#include "tt_metal/llrt/test_libs/debug_mailbox.hpp"

using namespace tt;

inline vector<uint32_t> gold_standard_untilize(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {
    vector<uint32_t> dst_vec;

    int num_rows = shape.at(0);
    int num_cols = shape.at(1) / 2;

    int num_tile_rows = num_rows / 32;
    int num_tile_cols = num_cols / 16;

    int face_size = 16 * 8;
    int tile_size = face_size * 4;

    std::set<int> ind;

    // Iterate over tile rows
    for (int t = 0; t < num_tile_rows; t++) {

        int tile_start_index = t * num_tile_cols;

        int physical_start_for_tile_row = tile_start_index * 32 * 16;

        // Iterate over tile columns 32 times (naive, but simple for validation)
        for (int x = 0; x < 2; x++) {
            for (int i = 0; i < 16; i++) { // num rows in a face
                for (int j = 0; j < num_tile_cols; j++) { // num columns top two faces
                    // Left face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + i * 8 + k + j * tile_size;
                        TT_ASSERT(ind.find(idx) == ind.end(), t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }

                    // Right face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + i * 8 + k + face_size + j * tile_size;
                        TT_ASSERT(ind.find(idx) == ind.end(), t);
                        ind.insert(idx);
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            physical_start_for_tile_row += 2 * face_size; // Move to bottom faces
        }
    }

    return dst_vec;
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;
    bool multibank = true;

    const char* op_id_to_op_define[] = {"add_tiles"};
    const char* op_id_to_op_name[] = {"ADD"};

    auto eltwise_op = EltwiseOp::ADD;
    log_info(LogTest, "====================================================================");
    log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);
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

        pass &= tt_metal::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t num_blocks = 1;
        uint32_t num_tiles_r = 2;
        uint32_t num_tiles_c = 2;

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = num_blocks * num_tiles_r * num_tiles_c;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        uint32_t page_size = single_tile_size;
        if (not multibank) {
            page_size = dram_buffer_size;
        }

        auto src0_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src0_addr, page_size, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src1_addr, page_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, page_size, tt_metal::BufferType::DRAM);

        auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = num_tiles_c;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src0_cb_addr
        );

        uint32_t untilized_src0_cb_index = 24;
        uint32_t untilized_src0_cb_addr = 300 * 1024;
        auto cb_untilized_src0 = tt_metal::CreateCircularBuffer(
            program,
            untilized_src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            untilized_src0_cb_addr
        );


        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 400 * 1024;
        auto cb_src1 = tt_metal::CreateCircularBuffer(
            program,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src1_cb_addr
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 500 * 1024;
        uint32_t num_output_tiles = num_tiles_c;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        auto binary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ? "tt_metal/kernels/dataflow/reader_dual_8bank.cpp"
                      : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ? "tt_metal/kernels/dataflow/writer_unary_8bank.cpp"
                      : "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            num_blocks,
            num_tiles_r,
            num_tiles_c
        };

        auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/untilA_elwbin_3m.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = {{"ELTWISE_OP", op_id_to_op_define[eltwise_op]}}}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::WriteToBuffer(src0_dram_buffer, src0_vec);

        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);

        tt_metal::WriteToBuffer(src1_dram_buffer, src1_vec);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel,
            core,
            {dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tiles,
            dram_buffer_src1_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            num_tiles, 0});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles});

        read_trisc_debug_mailbox(device->cluster(), 0, {1, 1}, 0);
        tt_metal::WriteRuntimeArgsToDevice(device, program);
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        vector<uint32_t> golden = gold_standard_untilize(src0_vec, {num_tiles_r * 32, num_tiles_c * 32});

        print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles, "result");
        print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles, "golden");

        pass &= (golden == result_vec);

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
