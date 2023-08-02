#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {
operation::ProgramWithCallbacks eltwise_binary_single_core(const Tensor &a, const Tensor &b, Tensor& output, BinaryOpType::Enum op_type) {

    Program program{};
    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    uint32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * src0_single_tile_size,
        src0_cb_data_format
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * src1_single_tile_size,
        src1_cb_data_format
    );

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * dst_single_tile_size,
        dst_cb_data_format
    );
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) src1_is_dram
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    KernelID binary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {
        num_tiles, // per_core_block_cnt
        1, // per_core_block_size
    };

    auto eltwise_binary_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = eltwise_binary_op_utils::get_defines(op_type)}
    );

    tt_metal::SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {
            src0_buffer->address(),
            src1_buffer->address(),
            num_tiles,
            0
        }
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            num_tiles,
            0
        }
    );

    auto override_runtime_args_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer_a = input_buffers.at(0);

        auto src_buffer_b = input_buffers.at(1);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, binary_reader_kernel_id, core);
            runtime_args[0] = src_buffer_a->address();
            runtime_args[1] = src_buffer_b->address();
            SetRuntimeArgs(program, binary_reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
