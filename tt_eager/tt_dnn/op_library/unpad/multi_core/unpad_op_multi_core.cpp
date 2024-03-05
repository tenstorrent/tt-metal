// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "optional"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > get_unpad_runtime_args_rm(
                                                                                const Tensor &input_tensor,
                                                                                Tensor& output_tensor,
                                                                                const Shape &output_tensor_start,
                                                                                uint32_t num_cores_total,
                                                                                uint32_t num_cores,
                                                                                uint32_t num_cores_y,
                                                                                CoreRangeSet core_group_1,
                                                                                CoreRangeSet core_group_2,
                                                                                uint32_t num_sticks_per_core_group_1,
                                                                                uint32_t num_sticks_per_core_group_2
                                                                                ){

    tt_metal::Device *device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

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

    vector<uint32_t> common_reader_kernel_args = {
        input_tensor.buffer()->address() + output_tensor_start[-1] * output_tensor.element_size(),
        padded_row_size_bytes,
        unpadded_row_size_bytes,
        num_dims,
        0,
        0
    };
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val (num_cores_total);

    uint32_t start_offset = get_rm_start_offset(input_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_total; i++){

        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            //no-op
            num_sticks_per_core = 0;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for(uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        //
        uint32_t addr_offset = 4; //input buffer addr, padded_row_size_bytes, unpadded_row_size_bytes, num_dims
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset] = num_sticks_per_core;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        vector<uint32_t> writer_kernel_args = {
            output_buffer->address(),
            unpadded_row_size_bytes,
            num_sticks_per_core,
            num_sticks_written,
            0
        };
        num_sticks_written+=num_sticks_per_core;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}



operation::ProgramWithCallbacks unpad_rm_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {

    const Shape output_shape = output.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    uint32_t num_unpadded_sticks = output.volume() / output.get_legacy_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});
    uint32_t num_cores_total = num_cores_x*num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

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
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);


    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args_vec = {
        (std::uint32_t) src0_is_dram
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;


    std::vector<uint32_t> writer_compile_time_args_vec = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        total_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/unpad/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        total_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    auto all_runtime_args = get_unpad_runtime_args_rm(a, output, output_tensor_start,
                                                    num_cores_total, num_cores, num_cores_y, core_group_1, core_group_2,
                                                    num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_total; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            all_runtime_args[i].first
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            all_runtime_args[i].second
        );
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = num_cores_x*num_cores_y;
        uint32_t num_unpadded_sticks = dst_tensor.volume() / dst_tensor.get_legacy_shape()[-1];
        auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

        const auto tensor_start = static_cast<const Unpad*>(operation)->output_tensor_start;
        auto all_runtime_args = get_unpad_runtime_args_rm(src_tensor, dst_tensor, tensor_start,
                                             num_cores_total, num_cores, num_cores_y, core_group_1, core_group_2,
                                             num_sticks_per_core_group_1, num_sticks_per_core_group_2);


        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
            }

            {
                SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}


// Each element of outer vector corresponds to a core
// Each core has a pair of std::vector<uint32_t>
// First of pair is reader args
// Second of pair is writer args
std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > get_unpad_runtime_args_tile(const Tensor &input_tensor,
                                                                                Tensor& output_tensor,
                                                                                const Shape &output_tensor_start,
                                                                                uint32_t num_cores_total,
                                                                                uint32_t num_cores,
                                                                                uint32_t num_cores_y,
                                                                                CoreRangeSet core_group_1,
                                                                                CoreRangeSet core_group_2,
                                                                                uint32_t num_tiles_per_core_group_1,
                                                                                uint32_t num_tiles_per_core_group_2
                                                                                ){
    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_padded_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

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


    vector<uint32_t> common_reader_kernel_args = {
        input_buffer->address(),
        num_dims,
        0, 0
    };
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), num_unpadded_tiles_per_dim.begin(), num_unpadded_tiles_per_dim.end());
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), num_padded_tiles_per_dim.begin(), num_padded_tiles_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val (num_cores_total);

    uint32_t start_offset = get_tiled_start_offset(input_tensor, output_tensor_start);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            //no-op
            num_tiles_per_core = 0;
        }

        id_per_dim[0] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for(uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        uint32_t addr_offset = 2; //input buffer addr, num_dims
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset] = num_tiles_per_core;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        vector<uint32_t> writer_kernel_args = {
            output_buffer->address(),
            num_tiles_per_core,
            num_tiles_written,
            0
        };
        num_tiles_written+=num_tiles_per_core;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks unpad_tile_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {

    const Shape output_shape = output.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    uint32_t num_unpadded_tiles = output.volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cores_total = num_cores_x*num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);




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
        total_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));



    auto all_runtime_args = get_unpad_runtime_args_tile(a, output, output_tensor_start, num_cores_total, num_cores, num_cores_y, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);



    for (uint32_t i = 0; i < num_cores_total; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            all_runtime_args[i].first
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            all_runtime_args[i].second
        );
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>> & ,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        uint32_t num_unpadded_tiles = dst_tensor.volume() / TILE_HW;

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto num_cores_total = num_cores_x*num_cores_y;

        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

        const auto tensor_start = static_cast<const Unpad*>(operation)->output_tensor_start;
        auto all_runtime_args = get_unpad_runtime_args_tile(src_tensor, dst_tensor, tensor_start, num_cores_total,  num_cores, num_cores_y, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);


        for (uint32_t i = 0; i < num_cores_total; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
            }
            {
                SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

operation::ProgramWithCallbacks unpad_multi_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {
    switch (a.get_layout()) {
        case Layout::ROW_MAJOR:
            return unpad_rm_multi_core(a, output, output_tensor_start, output_tensor_end);
        case Layout::TILE:
            return unpad_tile_multi_core(a, output, output_tensor_start, output_tensor_end);
        default:
            TT_ASSERT(false, "Unsupported Layout");
    }
    return {};
}

}  // namespace tt_metal

}  // namespace tt
