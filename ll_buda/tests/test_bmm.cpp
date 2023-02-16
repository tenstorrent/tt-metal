#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"
#include "llrt/tt_debug_print_server.hpp"

using namespace tt;
using namespace tt::ll_buda;

namespace bmm_kernel {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    uint32_t batch; // batch
    uint32_t Mt; // number of tiles in M
    uint32_t Kt; // number of tiles in K
    uint32_t Nt; // number of tiles in N
};
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    log_info(LogTest, "====================================================================");
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair core = {0, 0};
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

        auto src0_dram_buffer = ll_buda::CreateDramBuffer(0, bytesA, dram_buffer_src0_addr);
        auto src1_dram_buffer = ll_buda::CreateDramBuffer(0, bytesB, dram_buffer_src1_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(0, bytesC, dram_buffer_dst_addr);

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 2;
        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
        auto cb_src1 = ll_buda::CreateCircularBuffer(
            program,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src1_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = 2;
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );
        
        auto reader = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_bmm_8bank.cpp",
            core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);
        
        auto writer = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_bmm_8bank.cpp",
            core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

        void *hlk_args = new bmm_kernel::hlk_args_t{ .batch = B, .Mt = Mt, .Kt = Kt, .Nt = Nt };
        ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(bmm_kernel::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/bmm.cpp",
            core,
            eltwise_binary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
        pass &= ll_buda::WriteToDeviceDRAMChannelsInterleavedTiles(device, src0_vec, src0_dram_buffer->address());
        pass &= ll_buda::WriteToDeviceDRAMChannelsInterleavedTiles(device, src1_vec, src1_dram_buffer->address());
        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);
        ll_buda::WriteRuntimeArgsToDevice(
            device, reader, core,
            {dram_buffer_src0_addr, dram_buffer_src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
        );
        ll_buda::WriteRuntimeArgsToDevice(
            device, writer, core,
            {dram_buffer_dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
        );
        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAMChannelsInterleavedTiles(
            device, dst_dram_buffer->address(), result_vec, dst_dram_buffer->size());

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
