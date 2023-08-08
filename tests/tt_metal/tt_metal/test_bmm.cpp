#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"
#include "llrt/tt_debug_print_server.hpp"

using namespace tt;
using namespace tt::tt_metal;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    log_info(LogTest, "====================================================================");
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

        tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t Mt = 4, Kt = 2, Nt = 3, B = 2;
        uint32_t num_tilesA = Mt*Kt*B;
        uint32_t num_tilesB = Mt*Kt*B;
        uint32_t num_tilesC = Mt*Nt*B;
        uint32_t bytesA = single_tile_size * num_tilesA;
        uint32_t bytesB = single_tile_size * num_tilesB;
        uint32_t bytesC = single_tile_size * num_tilesC;

        uint32_t dram_buffer_src0_addr = 0;
        uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        auto src0_dram_buffer = tt_metal::Buffer(device, bytesA, dram_buffer_src0_addr, single_tile_size, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = tt_metal::Buffer(device, bytesB, dram_buffer_src1_addr, single_tile_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, bytesC, dram_buffer_dst_addr, single_tile_size, tt_metal::BufferType::DRAM);

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 2;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src0_cb_addr
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
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
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = 2;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        bool src0_is_dram = true;
        bool src1_is_dram = true;
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

        bool dst_is_dram = true;
        std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};
        auto reader = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_bmm_8bank.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

        auto writer = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_bmm_8bank.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

        vector<uint32_t> compute_kernel_args = {
            B, // batch
            Mt, // Mt
            Kt, // Kt
            Nt // Nt
        };

        auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/bmm.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        pass &= tt_metal::CompileProgram(device, program);

        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
        tt_metal::WriteToBuffer(src0_dram_buffer, src0_vec);
        tt_metal::WriteToBuffer(src1_dram_buffer, src1_vec);
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        uint32_t do_bcast = 0;
        tt_metal::SetRuntimeArgs(
            program, reader, core,
            {dram_buffer_src0_addr, dram_buffer_src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, do_bcast}
        );
        tt_metal::SetRuntimeArgs(
            program, writer, core,
            {dram_buffer_dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
        );
        tt_metal::WriteRuntimeArgsToDevice(device, program);
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        {
            // Read the result back from device DRAM and ref comparisone
            int argfail = -1;
            auto comparison_function = [](float a, float b) {
                const float rtol = 0.05f; // TODO(AP): need a spec for reference
                const float atol = 0.05f;
                float maxabs = fmaxf(fabsf(a), fabsf(b));
                float absdiff = fabsf(a - b);
                auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
                return result;
            };

            // recover a linear view of input vector for consumption by gold_ function
            vector<uint32_t> shapeA = {1, B, Mt*32, Kt*32};
            vector<uint32_t> shapeB = {1, B, Kt*32, Nt*32};
            vector<uint32_t> shapeC = {1, B, Mt*32, Nt*32};
            auto u16_src0_vec = u16_from_u32_vector(src0_vec);
            auto u16_src1_vec = u16_from_u32_vector(src1_vec);
            vector<u16> src0_linear = convert_layout<u16>(u16_src0_vec, shapeA, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
            vector<u16> src1_linear = convert_layout<u16>(u16_src1_vec, shapeB, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
            vector<u16> ref_bmm = gold_bmm(shapeA, src0_linear, shapeB, src1_linear);

            // Tilize gold from row major and convert to pairs (u32)
            auto gold_4f_u32 = u32_from_u16_vector( convert_layout<u16>(
                ref_bmm, shapeC, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES));

            pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
            if (!pass)
                log_error(LogTest, "Failure position={}", argfail);

        }
        //pass &= (src0_vec == result_vec);
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
