#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor eltwise_unary_single_core(const Tensor &a, UnaryOpType::Enum op_type) {
    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume() / TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );
    // no need to create c_in2 buffer since we pass scaler=0 to reader

    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        num_tiles, // per_core_block_cnt
        1 // per_core_block_size
    };
    tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core, compute_kernel_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    eltwise_unary_op_utils::add_defines(eltwise_unary_kernel, op_type);


    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        uint32_t(dram_src0_noc_xy.x),
        uint32_t(dram_src0_noc_xy.y),
        num_tiles, 0,0,0,0,0 } // TODO(AP): [8] is scaler
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        uint32_t(dram_dst_noc_xy.x),
        uint32_t(dram_dst_noc_xy.y),
        num_tiles }
    );

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
