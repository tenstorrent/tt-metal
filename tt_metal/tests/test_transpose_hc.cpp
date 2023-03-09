#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"

#include "test_tiles.hpp"

#include "llrt/tt_debug_print_server.hpp"

using namespace tt;

using u32 = std::uint32_t;
using u16 = std::uint16_t;


//////////////////////////////////////////////////////////////////////////////////////////
// Reference CPU implementation of transpose_HC
//////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: tests transpose kernel for HC dimensions
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;
    bool multibank = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);;

        // Also tests that the debug print server terminates cleanly with new tt_metal APIs
        // (it was previously crashing due to different termination sequence)
        tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program *program = new tt_metal::Program();

        tt_xy_pair core = {0, 0};

        //vector<uint32_t> shape = {1, 96, 32*4, 32*5};
        vector<uint32_t> shape = {2, 32*3, 32*5, 32*2};
        uint32_t num_tensor_tiles = shape.at(0) * shape.at(1) * shape.at(2) * shape.at(3) / (32*32);

        uint32_t single_tile_bytes = 2 * 1024;
        uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = tt_metal::CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_bytes, dram_buffer_src0_addr);
        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dst_dram_buffer = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_bytes, dram_buffer_dst_addr);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_buffer_tiles = 2;
        // this buffer is used in transpose_hc.cpp NCRISC kernel
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_buffer_tiles,
            num_buffer_tiles * single_tile_bytes,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        // this buffer is used in writer_unary.cpp BRISC kernel
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_buffer_tiles,
            num_buffer_tiles * single_tile_bytes,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        u32 W = shape[3], H = shape[2], C = shape[1], N = shape[0];
        u32 HW = H*W;
        u32 CHW = C*H*W;

        auto reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ?
                "kernels/dataflow/transpose_hc_8bank.cpp" :
                "kernels/dataflow/transpose_hc.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ?
                "kernels/dataflow/writer_unary_8bank.cpp" :
                "kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        vector<uint32_t> compute_kernel_args = {
            uint(num_tensor_tiles)
        };
        tt_metal::ComputeKernelArgs *kernel_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto blank_binary_kernel = tt_metal::CreateComputeKernel(
            program,
            "kernels/compute/eltwise_copy.cpp",
            core,
            kernel_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= tt_metal::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 100, 0x1234);
        auto src_4f_16 = u16_from_u32_vector(src0_vec);
        if (multibank)
            pass &= tt_metal::WriteToDeviceDRAMChannelsInterleavedTiles(device, src0_vec, src0_dram_buffer->address());
        else
            pass &= tt_metal::WriteToDeviceDRAM(src0_dram_buffer, src0_vec);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            reader_kernel,
            core,
            {
                dram_buffer_src0_addr,
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                W, H, C, HW, N, CHW
            }
        );

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {
                dram_buffer_dst_addr,
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tensor_tiles
            }
        );

        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        if (multibank)
            tt_metal::ReadFromDeviceDRAMChannelsInterleavedTiles(
                device, dst_dram_buffer->address(), result_vec, dst_dram_buffer->size());
        else
            tt_metal::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        int argfail = -1;
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.001f;
            const float atol = 1e-3f;
            float maxabs = fmaxf(fabsf(a), fabsf(b));
            float absdiff = fabsf(a - b);
            auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
            if (!result)
                absdiff *= 1.0f; // breakpoint spot
            return result;
        };

        // recover a linear view of input vector for consumption by gold_ function
        vector<u16> src_linear = convert_layout<u16>(src_4f_16, shape, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
        vector<u16> gold_reduced = gold_transpose_hc(src_linear, shape); // result is u16 untilized

        // Tilize from row major and convert to pairs (u32)
        vector<uint32_t> shapeR{shape[0], shape[2], shape[1], shape[3]};
        auto gold_16_4f = convert_layout<u16>(gold_reduced, shapeR, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
        auto gold_4f_u32 = u32_from_u16_vector(gold_16_4f);
        auto u16_result = u16_from_u32_vector(result_vec);

        pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
        if (!pass)
            log_error(LogTest, "Failure position={}", argfail);

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
