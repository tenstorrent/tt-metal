#include "tt_dnn/op_library/move/move_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks move_single_core(const Tensor &input, Tensor &output) {
    Program program{};

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = output.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = output.device();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t output_cb_index = 0; // same as input cb
    /* If we need dataformat conversion, use output buffer + compute kernel
    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * output_single_tile_size,
        output_cb_data_format
    );
    */

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    // NOTE: If both src and dst is DRAM, need to read forwards since DRAM is allocated bottom up.
    //       If src and dst is not the same, it doesn't matter which way we read.
    bool src_and_dst_is_dram = src_is_dram and dst_is_dram;

    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(input_cb_data_format), (uint32_t)src_is_dram};
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        src_and_dst_is_dram ? "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp" : "tt_metal/kernels/dataflow/reader_unary_backwards_interleaved_start_id.cpp",
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        static_cast<uint32_t>(output_cb_data_format),
        (uint32_t)dst_is_dram,
    };
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        src_and_dst_is_dram ? "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp" : "tt_metal/kernels/dataflow/writer_unary_backwards_interleaved_start_id.cpp",
        core,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    /* If we need dataformat conversion, use compute kernel
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    vector<uint32_t> compute_kernel_args = {
        num_tiles
    };
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    */

    SetRuntimeArgs(
        unary_reader_kernel,
        core,
        {
            src_buffer->address(),
            num_tiles,
            src_and_dst_is_dram ? 0 : num_tiles - 1
        }
    );

    SetRuntimeArgs(
        unary_writer_kernel,
        core,
        {
            dst_buffer->address(),
            num_tiles,
            src_and_dst_is_dram ? 0 : num_tiles - 1
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel, unary_writer_kernel](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(unary_reader_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            SetRuntimeArgs(unary_reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(unary_writer_kernel, core);
            runtime_args[0] = dst_dram_buffer->address();
            SetRuntimeArgs(unary_writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
