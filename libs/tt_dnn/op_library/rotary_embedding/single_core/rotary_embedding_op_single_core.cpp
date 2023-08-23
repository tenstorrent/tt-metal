#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_single_core(const Tensor &input, const Tensor &cos, const Tensor &sin, Tensor &output, std::optional<uint32_t> token_idx) {
    Program program{};

    CoreRangeSet core({CoreRange{.start={0, 0}, .end={0, 0}}});

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt_metal::detail::TileSize(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt_metal::detail::TileSize(sin_cb_data_format);

    tt::DataFormat scalar_cb_data_format = DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt_metal::detail::TileSize(scalar_cb_data_format);

    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;
    uint32_t num_rows = input.volume()  / input.shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.shape()[-1] * sizeof(bfloat16);

    tt_metal::Device *device = input.device();

    uint32_t input_cb_index = 0;
    uint32_t num_input_tiles = 2 * Wt;
    auto cb_input = tt_metal::CreateCircularBuffers(
        program,
        input_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t rotated_input_cb_index = 1;
    auto cb_rotated_input = tt_metal::CreateCircularBuffers(
        program,
        rotated_input_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t num_cos_sin_tiles = 2 * Wt;
    // TODO: Debug why this can't be double buffered
    if (token_idx.has_value()) {
        num_cos_sin_tiles = Wt;
    }
    uint32_t cos_cb_index = 2;
    auto cb_cos = tt_metal::CreateCircularBuffers(
        program,
        cos_cb_index,
        core,
        num_cos_sin_tiles,
        num_cos_sin_tiles * cos_single_tile_size,
        cos_cb_data_format
    );

    uint32_t sin_cb_index = 3;
    auto cb_sin = tt_metal::CreateCircularBuffers(
        program,
        sin_cb_index,
        core,
        num_cos_sin_tiles,
        num_cos_sin_tiles * sin_single_tile_size,
        sin_cb_data_format
    );

    // Used for bcast scalar
    uint32_t src_scalar_cb_index = 4;
    uint32_t num_scalar_tiles = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src_scalar_cb_index,
        core,
        num_scalar_tiles,
        num_scalar_tiles * scalar_single_tile_size,
        scalar_cb_data_format
    );

    uint32_t rotated_input_interm_cb_index = 24;
    auto cb_rotated_input_interm = tt_metal::CreateCircularBuffers(
        program,
        rotated_input_interm_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * input_single_tile_size,
        input_cb_data_format
    );

    uint32_t cos_interm_cb_index = 25;
    auto cb_cos_interm = tt_metal::CreateCircularBuffers(
        program,
        cos_interm_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * cos_single_tile_size,
        cos_cb_data_format
    );

    uint32_t sin_interm_cb_index = 26;
    auto cb_sin_interm = tt_metal::CreateCircularBuffers(
        program,
        sin_interm_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * sin_single_tile_size,
        sin_cb_data_format
    );


    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2 * Wt;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * output_single_tile_size,
        output_cb_data_format
    );

    uint32_t untilized_cos_interm_cb_index = 27;
    uint32_t untilized_cos_sync_cb_index = 28;
    uint32_t untilized_sin_interm_cb_index = 29;
    uint32_t untilized_sin_sync_cb_index = 30;
    uint32_t retilized_cos_cb_index = 5;
    uint32_t retilized_sin_cb_index = 6;
    std::map<string, string> kernel_defines;
    if (token_idx.has_value()) {
        auto cb_cos2 = tt_metal::CreateCircularBuffers(
            program,
            retilized_cos_cb_index,
            core,
            num_cos_sin_tiles,
            num_cos_sin_tiles * cos_single_tile_size,
            cos_cb_data_format
        );


        auto cb_sin2 = tt_metal::CreateCircularBuffers(
            program,
            retilized_sin_cb_index,
            core,
            num_cos_sin_tiles,
            num_cos_sin_tiles * sin_single_tile_size,
            sin_cb_data_format
        );
        auto cb_untilized_cos_interm = tt_metal::CreateCircularBuffers(
            program,
            {untilized_cos_interm_cb_index, untilized_cos_sync_cb_index},
            core,
            Wt,
            Wt * scalar_single_tile_size,
            scalar_cb_data_format
        );
        auto cb_untilized_sin_interm = tt_metal::CreateCircularBuffers(
            program,
            {untilized_sin_interm_cb_index, untilized_sin_sync_cb_index},
            core,
            Wt,
            Wt * scalar_single_tile_size,
            scalar_cb_data_format
        );
        kernel_defines["DECODE_MODE"] = "1";
    }


    const uint16_t bfloat16_scalar = bfloat16(-1.0f).to_uint16();

    auto src_buffer = input.buffer();
    auto cos_buffer = cos.buffer();
    auto sin_buffer = sin.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool cos_is_dram = cos_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool sin_is_dram = sin_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)cos_is_dram,
        (std::uint32_t)sin_is_dram,
        (std::uint32_t)bfloat16_scalar,
        (std::uint32_t)Ht,
        (std::uint32_t)Wt,
        (std::uint32_t)HtWt,
        (std::uint32_t)half_Wt,
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    if (token_idx.has_value()) {
        writer_compile_time_args.insert(writer_compile_time_args.end(), {
            untilized_cos_interm_cb_index,
            untilized_cos_sync_cb_index,
            untilized_sin_interm_cb_index,
            untilized_sin_sync_cb_index,
            Wt,
            Wbytes
        });
    }

    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args, .defines=kernel_defines});

    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args, .defines=kernel_defines});

	vector<uint32_t> compute_kernel_args = {
		(std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
		(std::uint32_t)num_rows,
        (std::uint32_t)Wt,
        (std::uint32_t)half_Wt
    };
    if (token_idx.has_value()) {
        compute_kernel_args.insert(compute_kernel_args.end(), {
            (std::uint32_t)untilized_cos_interm_cb_index,
            (std::uint32_t)untilized_cos_sync_cb_index,
            (std::uint32_t)untilized_sin_interm_cb_index,
            (std::uint32_t)untilized_sin_sync_cb_index,
            (std::uint32_t)retilized_cos_cb_index,
            (std::uint32_t)retilized_sin_cb_index
        });
    }


	auto rotary_embedding_kernel_group_1_id = tt_metal::CreateComputeKernel(
		program,
		"tt_metal/kernels/compute/rotary_embedding.cpp",
		core,
		tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines=kernel_defines}
	);

    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (token_idx.has_value()) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            src_buffer->address(),
            cos_buffer->address(),
            sin_buffer->address(),
            num_rows,
            0, 0,
            cos_sin_start_id
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            num_tiles,
            0,
            cos_sin_offset,
            Wt, Wbytes
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto cos_buffer = input_buffers.at(1);
        auto sin_buffer = input_buffers.at(2);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = cos_buffer->address();
            runtime_args[2] = sin_buffer->address();
            SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
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
