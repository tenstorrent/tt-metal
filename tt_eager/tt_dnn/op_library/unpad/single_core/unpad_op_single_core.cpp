// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


std::pair<std::vector<uint32_t>, std::vector<uint32_t> >   get_unpad_runtime_args_rm(const Tensor &input_tensor,
                                                                                Tensor& output_tensor,
                                                                                const Shape &output_tensor_start
                                                                                ){

    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();
    uint32_t num_dims = input_shape.rank();
    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * output_tensor.element_size();

    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims, 0);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for(int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t num_unpadded_sticks = output_tensor.volume() / output_shape[-1];

    uint32_t start_offset = get_rm_start_offset(input_tensor, output_tensor_start);
    vector<uint32_t> reader_kernel_args = {
        input_tensor.buffer()->address() + output_tensor_start[-1] * output_tensor.element_size(),
        padded_row_size_bytes,
        unpadded_row_size_bytes,
        num_dims,
        start_offset,
        num_unpadded_sticks
    };
    reader_kernel_args.insert(reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    reader_kernel_args.insert(reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());
    reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

    vector<uint32_t> writer_kernel_args = {
        output_tensor.buffer()->address(),
        unpadded_row_size_bytes,
        num_unpadded_sticks, 0,
    };

    return {reader_kernel_args, writer_kernel_args};

}

operation::ProgramWithCallbacks unpad_rm_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {

    const Shape output_shape = output.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t padded_row_size_bytes = a.get_legacy_shape()[-1] * a.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * a.element_size();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src_stick_size = padded_row_size_bytes;
    uint32_t dst_stick_size = unpadded_row_size_bytes;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_pages = 2;

    uint32_t cb_page_size = round_up(unpadded_row_size_bytes, TILE_WIDTH);

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_pages * cb_page_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    auto all_runtime_args = get_unpad_runtime_args_rm(a, output, output_tensor_start);
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(src_stick_size);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(src_stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args_vec = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) src_stick_size_is_power_of_two,
        (std::uint32_t) src_log2_stick_size
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(dst_stick_size);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(dst_stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args_vec = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) src_stick_size_is_power_of_two,
        (std::uint32_t) src_log2_stick_size,
        (std::uint32_t) dst_stick_size_is_power_of_two,
        (std::uint32_t) dst_log2_stick_size,
    };

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));


    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        all_runtime_args.first
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        all_runtime_args.second
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);

        CoreCoord core = {0, 0};
        const auto tensor_start = static_cast<const Unpad*>(operation)->output_tensor_start;
        auto all_runtime_args = get_unpad_runtime_args_rm(src_tensor, dst_tensor, tensor_start);

        {
            SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args.first);
        }

        {
            SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args.second);
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t> >   get_unpad_runtime_args_tile(const Tensor &input_tensor,
                                                                                Tensor& output_tensor,
                                                                                const Shape &output_tensor_start
                                                                                ){

    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();
    uint32_t num_dims = input_shape.rank();

    std::vector<uint32_t> num_unpadded_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_padded_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims, 0);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    for(int32_t i = 2; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t num_unpadded_tiles = output_tensor.volume() / TILE_HW;
    uint32_t start_offset = get_tiled_start_offset(input_tensor, output_tensor_start);

    vector<uint32_t> reader_kernel_args = {
        input_tensor.buffer()->address(),
        num_dims,
        start_offset,
        num_unpadded_tiles
    };
    reader_kernel_args.insert(reader_kernel_args.end(), num_unpadded_tiles_per_dim.begin(), num_unpadded_tiles_per_dim.end());
    reader_kernel_args.insert(reader_kernel_args.end(), num_padded_tiles_per_dim.begin(), num_padded_tiles_per_dim.end());
    reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

    vector<uint32_t> writer_kernel_args = {
        output_tensor.buffer()->address(),
        num_unpadded_tiles, 0,
    };

    return {reader_kernel_args, writer_kernel_args};


}

operation::ProgramWithCallbacks unpad_tile_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {


    const Shape output_shape = output.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);


    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_is_dram
    };
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));



    auto all_runtime_args = get_unpad_runtime_args_tile(a, output, output_tensor_start);

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        all_runtime_args.first
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        all_runtime_args.second
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        CoreCoord core = {0, 0};
        const auto tensor_start = static_cast<const Unpad*>(operation)->output_tensor_start;
        auto all_runtime_args = get_unpad_runtime_args_tile(src_tensor, dst_tensor, tensor_start);

        {
            SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args.first);
        }

        {
            SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args.second);
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

operation::ProgramWithCallbacks unpad_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {
    switch (a.get_layout()) {
        case Layout::ROW_MAJOR:
            return unpad_rm_single_core(a, output, output_tensor_start, output_tensor_end);
        case Layout::TILE:
            return unpad_tile_single_core(a, output, output_tensor_start, output_tensor_end);
        default:
            TT_ASSERT(false, "Unsupported Layout");
    }
    return {};
}

}  // namespace tt_metal

}  // namespace tt
