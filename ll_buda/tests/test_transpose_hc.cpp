#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"

#include "test_tiles.hpp"

#include "llrt/tt_debug_print_server.hpp"

using namespace tt;

namespace hlk_copy_binary {
// clone of hlk args from "kernels/compute/eltwise_copy.cpp"
// FIXME:copy pasted the args here from the blank kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
};
}

using u32 = std::uint32_t;
using u16 = std::uint16_t;


//////////////////////////////////////////////////////////////////////////////////////////
// Reference CPU implementation of transpose_HC
//////////////////////////////////////////////////////////////////////////////////////////

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
// TODO(AP) - move to gold header
inline vector<u16> gold_transpose_hc(std::vector<u16> src_vec, vector<uint32_t> shape) {
    struct TensLinAddr {
        vector<uint32_t> sh;
        TensLinAddr(vector<uint32_t> shape) : sh(shape) {}
        int offs(int n, int c, int h, int w) {
            TT_ASSERT(u32(n) < sh[0] && u32(c) < sh[1] && u32(h) < sh[2] && u32(w) < sh[3]);
            return w + sh[3]*h + sh[2]*sh[3]*c + sh[1]*sh[2]*sh[3]*n;
        }
    };

    vector<uint32_t> shapeT{shape[0], shape[2], shape[1], shape[3]};
    TensLinAddr addr(shape);
    TensLinAddr addrt(shapeT);

    vector<u16> transposed(src_vec.size());
    for (int n = 0; n < shape[0]; n++)
    for (int c = 0; c < shape[1]; c++)
    for (int h = 0; h < shape[2]; h++)
    for (int w = 0; w < shape[3]; w++) {
        auto toffs = addrt.offs(n, h, c, w);
        auto offs = addr.offs(n, c, h, w);
        TT_ASSERT(toffs < transposed.size() && offs < src_vec.size());
        transposed[toffs] = src_vec[offs];
    }
    //log_info(tt::LogVerif, "Prior size = {}", transposed.size());

    return transposed;
};

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: tests transpose kernel for HC dimensions
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        // Also tests that the debug print server terminates cleanly with new ll_buda APIs
        // (it was previously crashing due to different termination sequence)
        //tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair core = {0, 0};

        vector<uint32_t> shape = {1, 96, 32*4, 32*5};
        uint32_t num_tensor_tiles = shape.at(0) * shape.at(1) * shape.at(2) * shape.at(3) / (32*32);

        uint32_t single_tile_bytes = 2 * 1024;
        uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        
        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = ll_buda::CreateDramBuffer(dram_src0_channel_id, dram_buffer_bytes, dram_buffer_src0_addr);
        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates(device);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(dram_dst_channel_id, dram_buffer_bytes, dram_buffer_dst_addr);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates(device);

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_buffer_tiles = 2;
        // this buffer is used in transpose_hc.cpp NCRISC kernel
        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
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
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_buffer_tiles,
            num_buffer_tiles * single_tile_bytes,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        u32 W = shape[3], H = shape[2], C = shape[1];
        u32 HW = H*W;        
        
        auto reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/transpose_hc.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);
        
        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new hlk_copy_binary::hlk_args_t{ .per_core_tile_cnt = 96*4*5 };
        ll_buda::ComputeKernelArgs *kernel_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(hlk_copy_binary::hlk_args_t));
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto blank_binary_kernel = ll_buda::CreateComputeKernel(
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
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= ll_buda::WriteToDeviceDRAM(device, src0_dram_buffer, src0_vec);
        
        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            reader_kernel,
            core,
            {
                dram_buffer_src0_addr,
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                W, H, C, HW
            }
        );

        ll_buda::WriteRuntimeArgsToDevice(
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

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_vec, dst_dram_buffer->size());

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        // untilize input vector for consumption by gold_transpose_hc
        vector<u16> src_untilized16 = untilize_nchw(u16_from_u32_vector(src0_vec), shape);
        auto gold_transposed = gold_transpose_hc(src_untilized16, shape); // result is u16 untilized

        // Tilize from row major and convert to pairs (u32)
        vector<uint32_t> shapeT{shape[0], shape[2], shape[1], shape[3]};
        auto expected32 = u32_from_u16_vector(tilize_nchw(gold_transposed, shapeT));

        auto comparison_function = [](float a, float b) { return a == b; };
        int argfail = -1;
        bool pass = packed_uint32_t_vector_comparison(result_vec, expected32, comparison_function, &argfail);
        if (!pass)
            log_error(LogTest, "Failure position={}", argfail);

        pass &= ll_buda::CloseDevice(device);;

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
