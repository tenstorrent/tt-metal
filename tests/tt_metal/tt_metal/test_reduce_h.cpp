#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include "test_tiles.hpp"
#include "test_gold_impls.hpp"

#include "llrt/tt_debug_print_server.hpp"
#include "constants.hpp"

using namespace tt;

using u32 = std::uint32_t;
using u16 = std::uint16_t;
using std::vector;
using namespace constants;


//////////////////////////////////////////////////////////////////////////////////////////
// Tests reduce_h kernel in H dimension (NCHW->NC1W)
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;
    bool multibank = true;

    for (int do_max = 0; do_max <= 1; do_max++) {
    log_info(LogTest, "Running reduce test for max={}", do_max);
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);;

        // Also tests that the debug print server terminates cleanly with new tt_metal APIs
        // (it was previously crashing due to different termination sequence)
        tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program *program = new tt_metal::Program();

        tt_xy_pair core = {0, 0};

        vector<uint32_t> shape = {1, 3, 19*TILE_HEIGHT, 17*TILE_WIDTH};
        //std::cout << "++ A" << std::endl;
        //vector<uint32_t> shape = {1, 1, 1*TILE_HEIGHT, 1*TILE_WIDTH};
        u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
        u32 HW = H*W;
        u32 N = shape[0]*shape[1];
        TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
        TT_ASSERT(H > 0 && W > 0 && NC > 0);
        u32 Wt = W/TILE_WIDTH;
        u32 Ht = H/TILE_HEIGHT;
        float scaler = do_max ? 1.0f : 1.0f/H;
        uint32_t num_tensor_tiles = NC*H*W / (TILE_WIDTH*TILE_HEIGHT);

        uint32_t single_tile_bytes = 2 * 1024;
        uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = tt_metal::CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_bytes, dram_buffer_src0_addr);
        auto dst_dram_buffer = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_bytes/Ht, dram_buffer_dst_addr);
        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_buffer_tiles = 32;
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
        uint32_t num_output_buffer_tiles = 32;
        // this buffer is used in writer_unary.cpp BRISC kernel
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_buffer_tiles,
            num_output_buffer_tiles * single_tile_bytes,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto cb_temp_reduce_tile = tt_metal::CreateCircularBuffer(
            program, device, CB::c_in2 /*ouput_cb_index*/, core, 2 /* two tiles */,
            2 * single_tile_bytes, 400*1024, /* buf addr */
            tt::DataFormat::Float16_b
        );

        TT_ASSERT(num_tensor_tiles%Ht == 0);

        TT_ASSERT(multibank == true);
        auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ? "tt_metal/kernels/dataflow/reader_unary_transpose_wh_8bank.cpp"
                      : "tt_metal/kernels/dataflow/reader_unary_transpose_wh.cpp", // TODO(AP): not ported for reduce with scaler
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ? "tt_metal/kernels/dataflow/writer_unary_8bank.cpp" // no need to transpose the output since output Ht=1
                      : "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        vector<uint32_t> compute_kernel_args = {
            *reinterpret_cast<uint32_t*>(&scaler),
            uint(Ht),
            uint(Wt),
            uint(NC),
        };

        tt_metal::ComputeKernelArgs *compute_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto reduce_h_compute_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/reduce_h.cpp",
            core,
            compute_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        reduce_h_compute_kernel->add_define("REDUCE_OP", do_max ? "PoolType::MAX" : "PoolType::SUM");
        reduce_h_compute_kernel->add_define("REDUCE_DIM", "ReduceDim::REDUCE_COL");

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 10.0f, 0x1234, -4.5f);
        if (multibank)
            pass &= tt_metal::WriteToDeviceDRAMChannelsInterleavedTiles(device, src0_vec, src0_dram_buffer->address());
        else
            pass &= tt_metal::WriteToDeviceDRAMChannel(device, dram_src0_channel_id, src0_vec, src0_dram_buffer->address());

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {
                dram_buffer_src0_addr,
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                num_tensor_tiles, N, Ht, Wt, Ht*Wt,
                *reinterpret_cast<uint32_t*>(&scaler),
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
                num_tensor_tiles/Ht
            }
        );

        pass &= tt_metal::LaunchKernels(device, program);

        // The kernel will view the input as TILED32_4FACES
        vector<uint32_t> result_vec;
        if (multibank)
            tt_metal::ReadFromDeviceDRAMChannelsInterleavedTiles(
                device, dst_dram_buffer->address(), result_vec, dst_dram_buffer->size());
        else
            tt_metal::ReadFromDeviceDRAMChannel(
                device, dram_dst_channel_id, dst_dram_buffer->address(), result_vec, dst_dram_buffer->size());
        TT_ASSERT(result_vec.size() == NC*W*32/2); // we are expecting one tile in H, and half the elements since the vector packs 2 uint16_ts

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        int argfail = -1;
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.06f;
            const float atol = 1e-2f;
            float maxabs = fmaxf(fabsf(a), fabsf(b));
            float absdiff = fabsf(a - b);
            auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
            return result;
        };

        // recover a linear view of input vector for consumption by gold_ function
        auto u16_src0_vec = u16_from_u32_vector(src0_vec);
        vector<u16> src_linear = convert_layout<u16>(u16_src0_vec, shape, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
        vector<u16> gold_reduced = gold_reduce_h(src_linear, shape, scaler, do_max ? true : false ); // result is u16 untilized

        // Tilize from row major and convert to pairs (u32)
        vector<uint32_t> shapeR{shape[0], shape[1], TILE_HEIGHT, shape[3]};
        auto gold_4f_u32 = u32_from_u16_vector(convert_layout<u16>(gold_reduced, shapeR, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES));

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
    } // for do_max loop

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
