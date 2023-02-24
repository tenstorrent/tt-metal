#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tensor/tensor.hpp"
#include "test_tiles.hpp"

using namespace tt;

namespace eltwise_binary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
}

namespace matmul {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    int block_tile_dim;
    int dst_tile_rows;
    int dst_tile_cols;
    int block_cnt;
    int in0_block_tile_cnt;
    int in1_block_tile_cnt;
    int out_block_tile_cnt;
};
}

ll_buda::Program *setup_program_one(ll_buda::Device *device, const tt_xy_pair &core, uint32_t single_tile_size) {
    ll_buda::Program *program = new ll_buda::Program();

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

    auto binary_reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_binary.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    void *hlk_args = new eltwise_binary::hlk_args_t{
        .per_core_block_cnt = 1,
        .per_core_block_size = 1
    };
    ll_buda::ComputeKernelArgs *eltwise_binary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(eltwise_binary::hlk_args_t));
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = ll_buda::CreateComputeKernel(
        program,
        "kernels/compute/eltwise_binary.cpp",
        core,
        eltwise_binary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    eltwise_binary_kernel->add_define("ELTWISE_OP", "add_tiles");

    return program;
}

ll_buda::Program *setup_program_two(ll_buda::Device *device, const tt_xy_pair &core, uint32_t single_tile_size) {
    ll_buda::Program *program = new ll_buda::Program();

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

    auto mm_reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_matmul_small_block.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    void *hlk_args = new matmul::hlk_args_t{
        .block_tile_dim = 1,
        .dst_tile_rows = 1,
        .dst_tile_cols = 1,
        .block_cnt = 1,
        .in0_block_tile_cnt = 1,
        .in1_block_tile_cnt = 1,
        .out_block_tile_cnt = 1
    };
    ll_buda::ComputeKernelArgs *mm_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(matmul::hlk_args_t));
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto mm_kernel = ll_buda::CreateComputeKernel(
        program,
        "kernels/compute/matmul.cpp",
        core,
        mm_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    return program;
}

void write_program_runtime_args_to_device(
    ll_buda::Device *device,
    ll_buda::Program *program,
    const tt_xy_pair &core,
    uint32_t num_tiles,
    ll_buda::DramBuffer *src0_dram_buffer,
    ll_buda::DramBuffer *src1_dram_buffer,
    ll_buda::DramBuffer *dst_dram_buffer) {

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    for (auto dm_kernel : program->data_movement_kernels()) {
        if (dm_kernel->name() == "reader_binary" or dm_kernel->name() == "reader_matmul_small_block") {
            ll_buda::WriteRuntimeArgsToDevice(
                device, dm_kernel, core,
                {src0_dram_buffer->address(),
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                src1_dram_buffer->address(),
                (std::uint32_t)dram_src1_noc_xy.x,
                (std::uint32_t)dram_src1_noc_xy.y,
                num_tiles});
        } else if (dm_kernel->name() == "writer_unary") {
            ll_buda::WriteRuntimeArgsToDevice(
                device, dm_kernel, core,
                {dst_dram_buffer->address(),
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tiles});
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
// 1. First program runs eltwise binary on logical core {0, 0}
// 2. Host read the results from eltwise binary
// 3. Second program runs matmul, using results from step 2 as input activation
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

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_xy_pair core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;

        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_src1_addr = 0;
        int dram_src1_channel_id = 1;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_size, dram_buffer_src0_addr);
        auto src1_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src1_channel_id, dram_buffer_size, dram_buffer_src1_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        ll_buda::Program *program1 = setup_program_one(device, core, single_tile_size);

        ll_buda::Program *program2 = setup_program_two(device, core, single_tile_size);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Applications
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program1, skip_hlkc);

        // Both programs use the same CB addresses but they can be compiled one after
        // the other because they use the same data formats
        pass &= ll_buda::CompileProgram(device, program2, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Program One
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, 32, 32};
        tt::Tensor<bfloat16> src0_tensor = tt::initialize_tensor<bfloat16>(shape, tt::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto src0_activations_tile_layout = convert_to_tile_layout(src0_tensor.get_values());
        auto src0_activations = pack_bfloat16_vec_into_uint32_vec(src0_activations_tile_layout);
        pass &= ll_buda::WriteToDeviceDRAM(device, src0_dram_buffer, src0_activations);

        tt::Tensor<bfloat16> src1_tensor = tt::initialize_tensor<bfloat16>(shape, tt::Initialize::ZEROS, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto src1_activations_tile_layout = convert_to_tile_layout(src1_tensor.get_values());
        auto src1_activations = pack_bfloat16_vec_into_uint32_vec(src1_activations_tile_layout);
        pass &= ll_buda::WriteToDeviceDRAM(device, src1_dram_buffer, src1_activations);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program1);

        write_program_runtime_args_to_device(device, program1, core, num_tiles, src0_dram_buffer, src1_dram_buffer, dst_dram_buffer);

        pass &= ll_buda::LaunchKernels(device, program1);

        std::vector<uint32_t> intermediate_result_vec;
        ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, intermediate_result_vec, dst_dram_buffer->size());

        ////////////////////////////////////////////////////////////////////////////
        //                      Validatie Intermediate Result
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src0_activations == intermediate_result_vec);  // src1 is ZEROS
        if (pass) {
            log_info(LogTest, "Eltwise binary ran successfully");
        } else {
            log_error(LogTest, "Eltwise binary did not run sucessfully!");
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Program Two
        ////////////////////////////////////////////////////////////////////////////
        // Write matmul weights to DRAM
        auto identity = create_identity_matrix(32, 32, 32); //bflaot16 32x32 identity
        auto weights_tile_layout = convert_to_tile_layout(identity);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        pass &= ll_buda::WriteToDeviceDRAM(device, src1_dram_buffer, weights);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program2);

        write_program_runtime_args_to_device(device, program2, core, num_tiles, src0_dram_buffer, src1_dram_buffer, dst_dram_buffer);

        pass &= ll_buda::LaunchKernels(device, program2);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_vec, dst_dram_buffer->size());

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (intermediate_result_vec == result_vec); // src1 is identity matrix

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
