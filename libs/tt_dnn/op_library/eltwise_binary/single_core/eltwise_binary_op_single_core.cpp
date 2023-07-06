#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {
operation::ProgramWithCallbacks eltwise_binary_single_core(const Tensor &a, const Tensor &b, Tensor& output, BinaryOpType::Enum op_type) {

    Program program{};
    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.storage_type() == StorageType::DEVICE and b.storage_type() == StorageType::DEVICE, "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to eltwise binary need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();

    TT_ASSERT(src0_dram_buffer->size() == src1_dram_buffer->size(), "Operand to eltwise binary need to be the same size!");

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume() / TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        //"tt_metal/kernels/dataflow/reader_binary.cpp",
        "tt_metal/kernels/dataflow/reader_dual_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        //"tt_metal/kernels/dataflow/writer_unary.cpp",
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        num_tiles, // per_core_block_cnt
        1, // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    eltwise_binary_op_utils::add_defines(eltwise_binary_kernel, op_type);


    tt_metal::SetRuntimeArgs(
        binary_reader_kernel,
        core,
        {
            src0_dram_buffer->address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tiles,
            src1_dram_buffer->address(),
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            num_tiles
        }
    );

    tt_metal::SetRuntimeArgs(
        unary_writer_kernel,
        core,
        {
            dst_dram_buffer->address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles
        }
    );

    auto override_runtime_args_callback = [
            binary_reader_kernel,
            unary_writer_kernel
        ]
    (
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_noc_xy_a = src_dram_buffer_a->noc_coordinates();

        auto src_dram_buffer_b = input_buffers.at(1);
        auto src_dram_noc_xy_b = src_dram_buffer_b->noc_coordinates();

        auto dst_dram_buffer = output_buffers.at(0);
        auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(binary_reader_kernel, core);
            runtime_args[0] = src_dram_buffer_a->address();
            runtime_args[1] = uint32_t(src_dram_noc_xy_a.x);
            runtime_args[2] = uint32_t(src_dram_noc_xy_a.y);
            runtime_args[4] = src_dram_buffer_b->address();
            runtime_args[5] = uint32_t(src_dram_noc_xy_b.x);
            runtime_args[6] = uint32_t(src_dram_noc_xy_b.y);
            SetRuntimeArgs(binary_reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(unary_writer_kernel, core);
            runtime_args[0] = dst_dram_buffer->address();
            runtime_args[1] = uint32_t(dst_dram_noc_xy.x);
            runtime_args[2] = uint32_t(dst_dram_noc_xy.y);
            SetRuntimeArgs(unary_writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
