#include "tt_metal/op_library/eltwise_binary/eltwise_binary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

namespace eltwise_binary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
}

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor eltwise_binary(const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type) {
    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::InterleavedDramBuffer *src0_dram_buffer = a.buffer();
    tt_metal::InterleavedDramBuffer *src1_dram_buffer = b.buffer();

    TT_ASSERT(src0_dram_buffer->size() == src1_dram_buffer->size(), "Operand to eltwise binary need to be the same size!");

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;

    // InterleavedDramBuffer stores buffers across multiple dram banks but reader kernels only need the location of the first one
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates().at(0);
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates().at(0);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    // InterleavedDramBuffer stores buffers across multiple dram banks but writer kernel only needs the location of the first one
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates().at(0);

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
        DataFormat::Float16_b
    );

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
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
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
        DataFormat::Float16_b
    );

    tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        //"kernels/dataflow/reader_binary.cpp",
        "kernels/dataflow/reader_dual_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        //"kernels/dataflow/writer_unary.cpp",
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles), // per_core_block_cnt
        1, // per_core_block_size
    };
    tt_metal::ComputeKernelArgs *eltwise_binary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "kernels/compute/eltwise_binary.cpp",
        core,
        eltwise_binary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    string op_name;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    tt_metal::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        binary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        (std::uint32_t)dram_src0_noc_xy.x,
        (std::uint32_t)dram_src0_noc_xy.y,
        (std::uint32_t)num_tiles,
        src1_dram_buffer->address(),
        (std::uint32_t)dram_src1_noc_xy.x,
        (std::uint32_t)dram_src1_noc_xy.y,
        (std::uint32_t)num_tiles});

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles});

    tt_metal::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
