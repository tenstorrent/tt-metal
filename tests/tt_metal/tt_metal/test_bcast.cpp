#include <algorithm>
#include <functional>
#include <random>
#include <vector>
#include <map>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include "test_tiles.hpp"
#include "test_gold_impls.hpp"
#include "constants.hpp"

#include "llrt/tt_debug_print_server.hpp"

using namespace tt;
using namespace constants;

using u32 = std::uint32_t;
using u16 = std::uint16_t;
using std::vector;


namespace {
const char* get_reader_name(bool multibank, BcastDim::Enum bcast_dim) {
    TT_ASSERT(multibank && "Only multibank is supported correctly.");
    if (bcast_dim == BcastDim::H) {
        return multibank ?
            "tt_metal/kernels/dataflow/reader_bcast_h_8bank.cpp" :
            "tt_metal/kernels/dataflow/reader_bcast_h.cpp";
    } else if (bcast_dim == BcastDim::W) {
        return multibank ?
            "tt_metal/kernels/dataflow/reader_bcast_w_8bank.cpp" :
            "tt_metal/kernels/dataflow/reader_bcast_w.cpp";
    } if (bcast_dim == BcastDim::HW) {
        return multibank ?
            "tt_metal/kernels/dataflow/reader_bcast_hw_8bank.cpp" :
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp";
    }
    TT_ASSERT(false && "Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastDim::Enum bcast_dim) {
    switch (bcast_dim) {
        case BcastDim::H:  return "tt_metal/kernels/compute/bcast_h.cpp";
        case BcastDim::W:  return "tt_metal/kernels/compute/bcast_w.cpp";
        case BcastDim::HW: return "tt_metal/kernels/compute/bcast_hw.cpp";
        default:           TT_ASSERT(false && "Unexpected bcast_dim!");
    }
    return "";
}

}

//////////////////////////////////////////////////////////////////////////////////////////
// Tests reduce_h kernel in H dimension (NCHW->NC1W)
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;

    const char* bdim_to_log_string[] = { "", "BCAST_H", "BCAST_W", "", "BCAST_HW" };
    const char* op_id_to_op_define[] = {"add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast"};
    const char* op_id_to_llkop_define[] = {"ELWADD", "ELWSUB", "ELWMUL"};
    const char* bdim_to_llkdim_define[] = { "", "BroadcastType::ROW", "BroadcastType::COL", "", "BroadcastType::SCALAR" };
    const char* op_id_to_op_name[] = {"ADD", "SUB", "MUL"};
    bool multibank = true;

    auto bdims = BcastDim::all();
    auto ops = BcastOp::all();
    //ops = { BcastOp::MUL }; // TODO(AP): for debugging
    for (auto bcast_op: ops) {
    for (auto bcast_dim: bdims) {

    log_info(LogTest, "=============================================================");
    log_info(LogTest, "======= Running bcast test for bdim={}, op={}", bdim_to_log_string[bcast_dim], op_id_to_op_name[bcast_op]);
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

        // Also tests that the debug print server terminates cleanly with new tt_metal APIs
        // (it was previously crashing due to different termination sequence)
        tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        vector<uint32_t> shape = {2, 4, 2*TILE_HEIGHT, 3*TILE_WIDTH};
        u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0], N = shape[0], C = shape[1];
        u32 HW = H*W;
        TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
        TT_ASSERT(H > 0 && W > 0 && NC > 0);
        u32 Wt = W/TILE_WIDTH;
        u32 Ht = H/TILE_HEIGHT;
        uint32_t num_tensor_tiles = NC*H*W / (32*32);

        uint32_t single_tile_bytes = 2 * 1024;
        uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024; // needs to be at a different address for multi-bank
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        uint32_t page_size = single_tile_bytes;
        if (not multibank) {
            page_size = dram_buffer_bytes;
        }

        auto src0_dram_buffer = tt_metal::Buffer(device, dram_buffer_bytes, dram_buffer_src0_addr, page_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_bytes, dram_buffer_dst_addr, page_size, tt_metal::BufferType::DRAM);
        auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_buffer_tiles = 2;
        // this buffer is used in transpose_hc.cpp NCRISC kernel
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_buffer_tiles,
            num_buffer_tiles * single_tile_bytes,
            tt::DataFormat::Float16_b,
            src0_cb_addr
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
        auto cb_src1 = tt_metal::CreateCircularBuffer(
            program,
            src1_cb_index,
            core,
            num_buffer_tiles,
            num_buffer_tiles * single_tile_bytes,
            tt::DataFormat::Float16_b,
            src1_cb_addr
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_buffer_tiles = 2;
        // this buffer is used in writer_unary.cpp BRISC kernel
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_output_buffer_tiles,
            num_output_buffer_tiles * single_tile_bytes,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        vector<uint16_t> tiled_bcast_values;
        vector<uint16_t> ref_bcast_values;
        vector<uint32_t> ref_bcast_shape = {N,C,1,1};
        float bcast_1value = 10.0f;
        uint16_t bcast_1value16 = bfloat16(bcast_1value).to_uint16();
        unsigned num_bcast_tiles = 0;
        // build the constant tiles to be broadcast
        if (bcast_dim == BcastDim::HW) {
            num_bcast_tiles = NC;
            ref_bcast_values.resize(NC, 0);
            for (int j = 0; j < NC; j++)
                // add something not too large but different between tiles
                ref_bcast_values[j] = bfloat16(bcast_1value+(j%7)).to_uint16();
            // convert the reference broadcast tensor to tiled format
            tiled_bcast_values = convert_layout<u16>(
                ref_bcast_values, ref_bcast_shape, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
            TT_ASSERT(tiled_bcast_values[0] == bcast_1value16);
            // restore ref values and shape to 1
            ref_bcast_shape[3] = 1;
            ref_bcast_shape[4] = 1;
        } else if (bcast_dim == BcastDim::H) {
            // For bcast_h a.k.a. Dim::R we broadcast _over_ H, meaning we take a W vector and += it over each element in the H dimension
            // At least that's the behavior i've seen from a single tile bcast-H
            // So this is why here we create a W-sized vector
            // Same for the if branch for BCAST_W below
            TT_ASSERT(W%32 == 0);
            // pad values and shape with extra 32 values because the reader kernel expects it
            // generate broadcast values along the W axis with one extra tile (needed by the kernel I believe)
            // TODO(AP): need to figure out why the extra tile in broadcast inputs is expected by the kernel
            ref_bcast_values.resize(NC*W, 0);
            ref_bcast_shape[3] = W;
            for (int j = 0; j < NC*W; j++)
                // add something not too large but different between tiles
                ref_bcast_values[j] = bfloat16(bcast_1value+(j%7)).to_uint16();
            tiled_bcast_values = convert_layout<u16>(
                ref_bcast_values, ref_bcast_shape, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
            num_bcast_tiles = NC*Wt;
            // restore values and shape to W
        } else if (bcast_dim == BcastDim::W) {
            // see the comments above for BCAST_H
            ref_bcast_values.resize(NC*H, 0);
            ref_bcast_shape[2] = H;
            for (int j = 0; j < NC*H; j++)
                // add something not too large but different between tiles
                ref_bcast_values[j] = bfloat16(bcast_1value+(j%7)).to_uint16();
            tiled_bcast_values = convert_layout<u16>(
                ref_bcast_values, ref_bcast_shape, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
            num_bcast_tiles = NC*Ht;
        }

        auto bcast_tiled_u32 = u32_from_u16_vector(tiled_bcast_values);
        auto bcast_vals_nbytes = bcast_tiled_u32.size()*sizeof(bcast_tiled_u32[0]);
        uint32_t src1_page_size = single_tile_bytes;
        if (not multibank) {
            src1_page_size = bcast_vals_nbytes;
        }
        auto src1_dram_buffer = tt_metal::Buffer(device, bcast_vals_nbytes, dram_buffer_src1_addr, src1_page_size, tt_metal::BufferType::DRAM);
        auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
        tt_metal::WriteToBuffer(src1_dram_buffer, bcast_tiled_u32);

        bool src0_is_dram = true;
        bool src1_is_dram = true;
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

        const char* reader_name = get_reader_name(multibank, bcast_dim);
        auto binary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            reader_name,
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ? "tt_metal/kernels/dataflow/writer_unary_8bank.cpp"
                      : "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        uint32_t nc1 = 0;
        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel,
            core,
            {dram_buffer_src0_addr, // 0
            (std::uint32_t)dram_src0_noc_xy.x, // 1
            (std::uint32_t)dram_src0_noc_xy.y, // 2
            num_tensor_tiles, // 3
            dram_buffer_src1_addr, // 4
            (std::uint32_t)dram_src1_noc_xy.x, // 5
            (std::uint32_t)dram_src1_noc_xy.y, // 6
            num_bcast_tiles, NC*Ht*Wt, NC, Ht, Wt, nc1}); // 7 8 9 10 11 12

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tensor_tiles});

        vector<uint32_t> compute_kernel_args = {
            uint(NC),
            uint(Ht),
            uint(Wt)
        };

        std::map<string, string> compute_defines = {
            {"BCAST_DIM", bdim_to_llkdim_define[bcast_dim]},
            {"BCAST_OP", op_id_to_op_define[bcast_op]},
            {"BCAST_LLKOP", op_id_to_llkop_define[bcast_op]}
        };
        auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
            program,
            get_compute_name(bcast_dim),
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = compute_defines}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 10.0f, 0x1234);
        tt_metal::WriteToBuffer(src0_dram_buffer, src0_vec);
        tt_metal::WriteRuntimeArgsToDevice(device, program);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        pass &= tt_metal::LaunchKernels(device, program);

        // The kernel will view the input as TILED32_4FACES
        vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        int argfail = -1;
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.02f;
            const float atol = 1e-3f;
            float maxabs = fmaxf(fabsf(a), fabsf(b));
            float absdiff = fabsf(a - b);
            auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
            return result;
        };

        // recover a linear view of input vector for consumption by gold_ function
        auto u16_src0_vec = u16_from_u32_vector(src0_vec);
        vector<u16> src_linear = convert_layout<u16>(
            u16_src0_vec, shape, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
        vector<u16> gold_added = gold_bcast_op(
            src_linear, shape, ref_bcast_values, bcast_dim, bcast_op); // result is u16 untilized

        // Tilize from row major and convert to pairs (u32)
        vector<uint32_t> shapeR{shape[0], shape[1], shape[2], shape[3]};
        auto gold_4f_u32 = u32_from_u16_vector(
            convert_layout<u16>(
                gold_added, shapeR, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES));

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

    } // for bcast_op loop
    } // for bcast_mode loop

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
