// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/concat/concat_op.hpp"

#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// start is inclusive, end is exclusive
struct PageRange {
    uint32_t start;
    uint32_t end;
};

struct CorePageRange {
    CoreCoord core;
    PageRange range;
};




operation::ProgramWithCallbacks s2s_rm_concat_multi_core(const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output) {
    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device *device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_output_rows = output.shape()[-1];
    uint32_t num_input_tensors = input_tensors.size();

    vector<CBHandle> cb_input(num_input_tensors);
    vector<uint32_t> input_num_units_per_shard_height(num_input_tensors);
    vector<uint32_t> input_num_units_per_shard_width(num_input_tensors);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    auto all_cores = input_tensors[0].shard_spec().value().grid;

    vector<uint32_t> cb_ids(num_input_tensors);
    uint32_t input_unit_size = input_tensors[0].shard_spec().value().shape[1] * input_tensors[0].element_size();
    //input CBs
    for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_units_per_shard_height[input_id] = shard_spec.shape[0];
        input_num_units_per_shard_width[input_id] = 1;
        auto num_input_units = input_num_units_per_shard_height[input_id] * input_num_units_per_shard_width[input_id];
        auto input_page_size = round_up_to_mul32(input_unit_size);
        tt_metal::CircularBufferConfig input_cb_config =
            tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_id, cb_data_format}})
                .set_page_size(input_id, input_page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_input[input_id] = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);
        cb_ids[input_id] = input_id;
    }

    //output CB
    uint32_t cb_dst_id = 16;
    auto num_output_units = input_num_units_per_shard_height[0] * input_num_units_per_shard_width[0] * num_input_tensors;
    auto output_page_size = round_up_to_mul32(input_unit_size);
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(num_output_units * output_page_size, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, output_page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);



    auto output_shard_spec = output.shard_spec().value();


    bool dst_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector <uint32_t> reader_compile_time_args = {num_input_tensors};
    std::vector <uint32_t> writer_compile_time_args = {num_input_tensors, std::uint32_t(dst_is_dram), cb_dst_id};

   tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/concat/kernels/dataflow/reader_s2i_width.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

   tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/concat/kernels/dataflow/writer_s2s_width.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
    auto input_cores = input_tensors[0].shard_spec().value().grid;
    uint32_t num_output_rows_per_core = div_up(num_output_rows, input_cores.num_cores());

    uint32_t core_id = 0;
    for(auto core: cores) {
        auto input_shard_spec = input_tensors[0].shard_spec().value();
        uint32_t curr_num_input_tensors;
        uint32_t curr_num_output_rows;
        if(input_cores.core_coord_in_core_ranges(core)) {
            curr_num_input_tensors = num_input_tensors;
            curr_num_output_rows = num_output_rows_per_core;
        }
        else {
            curr_num_input_tensors = 0;
            curr_num_output_rows = 0;
        }

        vector<uint32_t> reader_runtime_args = {};
        vector<uint32_t> writer_runtime_args = {output.buffer()->address(), core_id,
                            curr_num_output_rows, input_unit_size, num_input_tensors*input_shard_spec.shape[0], input_shard_spec.shape[0]};
        for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            reader_runtime_args.push_back(input_id);
            reader_runtime_args.push_back(input_shard_spec.shape[0]);
            writer_runtime_args.push_back(input_id);
        }
        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            reader_runtime_args
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            writer_runtime_args
        );
        core_id++;
    }


    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, all_cores, num_input_tensors](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors)
    {

        bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
        auto dst_buffer = output_tensors.at(0).buffer();
        auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
        auto input_cores = input_tensors[0].shard_spec().value().grid;
        uint32_t num_output_rows = output_tensors[0].shape()[-1];
        uint32_t num_output_rows_per_core = div_up(num_output_rows, input_cores.num_cores());
        for(auto core: cores) {
            uint32_t curr_num_input_tensors;
            uint32_t curr_num_output_rows;
            if(input_cores.core_coord_in_core_ranges(core)) {
                curr_num_input_tensors = num_input_tensors;
                curr_num_output_rows = num_output_rows_per_core;
            }
            else {
                curr_num_input_tensors = 0;
                curr_num_output_rows = 0;
            }

            vector<uint32_t> reader_runtime_args = {curr_num_input_tensors};
            vector<uint32_t> writer_runtime_args = {dst_buffer->address(), curr_num_input_tensors, curr_num_output_rows};
            for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
                UpdateDynamicCircularBufferAddress(program, input_id, *dst_buffer);
                auto input_shard_spec = input_tensors[input_id].shard_spec().value();
                reader_runtime_args.push_back(input_id);
                reader_runtime_args.push_back(input_shard_spec.shape[1]);
                writer_runtime_args.push_back(input_id);
            }
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                reader_runtime_args
            );

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                writer_runtime_args
            );
        }


    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}



operation::ProgramWithCallbacks s2i_rm_concat_multi_core(const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output) {
    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device *device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    //CoreRangeSet all_cores({CoreRange(CoreCoord(0,0), compute_with_storage_grid_size)});

    uint32_t num_output_rows = output.shape()[-1];
    uint32_t num_input_tensors = input_tensors.size();

    vector<CBHandle> cb_input(num_input_tensors);
    vector<uint32_t> input_num_units_per_shard_height(num_input_tensors);
    vector<uint32_t> input_num_units_per_shard_width(num_input_tensors);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    auto all_cores = input_tensors[0].shard_spec().value().grid;

    vector<uint32_t> cb_ids(num_input_tensors);
    uint32_t input_unit_size = input_tensors[0].shard_spec().value().shape[1] * input_tensors[0].element_size();
    //input CBs
    for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_units_per_shard_height[input_id] = shard_spec.shape[0];
        input_num_units_per_shard_width[input_id] = 1;
        auto num_input_units = input_num_units_per_shard_height[input_id] * input_num_units_per_shard_width[input_id];
        auto input_page_size = round_up_to_mul32(input_unit_size);
        tt_metal::CircularBufferConfig input_cb_config =
            tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_id, cb_data_format}})
                .set_page_size(input_id, input_page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_input[input_id] = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);
        cb_ids[input_id] = input_id;
    }


    bool dst_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector <uint32_t> reader_compile_time_args = {num_input_tensors};
    std::vector <uint32_t> writer_compile_time_args = {num_input_tensors, std::uint32_t(dst_is_dram)};

   tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/concat/kernels/dataflow/reader_s2i_width.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

   tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/concat/kernels/dataflow/writer_s2i_width.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
    auto input_cores = input_tensors[0].shard_spec().value().grid;
    uint32_t num_output_rows_per_core = div_up(num_output_rows, input_cores.num_cores());

    uint32_t core_id = 0;
    for(auto core: cores) {
        auto input_shard_spec = input_tensors[0].shard_spec().value();
        uint32_t curr_num_input_tensors;
        uint32_t curr_num_output_rows;
        if(input_cores.core_coord_in_core_ranges(core)) {
            curr_num_input_tensors = num_input_tensors;
            curr_num_output_rows = num_output_rows_per_core;
        }
        else {
            curr_num_input_tensors = 0;
            curr_num_output_rows = 0;
        }

        vector<uint32_t> reader_runtime_args = {};
        vector<uint32_t> writer_runtime_args = {output.buffer()->address(), core_id,
                            curr_num_output_rows, input_unit_size, num_input_tensors*input_shard_spec.shape[0], input_shard_spec.shape[0] };
        for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            reader_runtime_args.push_back(input_id);
            reader_runtime_args.push_back(input_shard_spec.shape[0]);
            writer_runtime_args.push_back(input_id);
        }
        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            reader_runtime_args
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            writer_runtime_args
        );
        core_id++;
    }


    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, all_cores, num_input_tensors](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors)
    {

        bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
        auto dst_buffer = output_tensors.at(0).buffer();
        auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
        auto input_cores = input_tensors[0].shard_spec().value().grid;
        uint32_t num_output_rows = output_tensors[0].shape()[-1];
        uint32_t num_output_rows_per_core = div_up(num_output_rows, input_cores.num_cores());
        for(auto core: cores) {
            uint32_t curr_num_input_tensors;
            uint32_t curr_num_output_rows;
            if(input_cores.core_coord_in_core_ranges(core)) {
                curr_num_input_tensors = num_input_tensors;
                curr_num_output_rows = num_output_rows_per_core;
            }
            else {
                curr_num_input_tensors = 0;
                curr_num_output_rows = 0;
            }

            vector<uint32_t> reader_runtime_args = {curr_num_input_tensors};
            vector<uint32_t> writer_runtime_args = {dst_buffer->address(), curr_num_input_tensors, curr_num_output_rows};
            for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
                UpdateDynamicCircularBufferAddress(program, input_id, *dst_buffer);
                auto input_shard_spec = input_tensors[input_id].shard_spec().value();
                reader_runtime_args.push_back(input_id);
                reader_runtime_args.push_back(input_shard_spec.shape[1]);
                writer_runtime_args.push_back(input_id);
            }
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                reader_runtime_args
            );

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                writer_runtime_args
            );
        }


    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}


operation::ProgramWithCallbacks sharded_concat_multi_core(const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output) {
    if(output.is_sharded()) {
        return s2s_rm_concat_multi_core(input_tensors, dim, output);
    }
    else {
        return s2i_rm_concat_multi_core(input_tensors, dim, output);
    }
}

operation::ProgramWithCallbacks concat_multi_core(const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output) {

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device *device = output.device();

    uint32_t num_output_tiles = output.volume() / TILE_HW;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_output_tiles);

    uint32_t num_input_tensors = input_tensors.size();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t num_dims = input_tensors[0].shape().rank();

    std::vector<uint32_t> src_addr(num_input_tensors);
    std::vector<bool> is_dram(num_input_tensors);
    std::vector<uint32_t> num_tiles_per_block(num_input_tensors);
    std::vector<uint32_t> tile_id_per_tensor(num_input_tensors);

    uint32_t num_accum_tiles = 1;
    uint32_t scale_factor = 1;
    if (dim == num_dims - 2) {
        scale_factor = TILE_HEIGHT;
    } else if (dim == num_dims - 1) {
        scale_factor = TILE_WIDTH;
    }
    for (uint32_t i = dim + 1; i < num_dims; i++) {
        num_accum_tiles *=  input_tensors[0].shape()[i];
        if (i == num_dims - 2) {
            num_accum_tiles /= TILE_HEIGHT;
        } else if (i == num_dims - 1) {
            num_accum_tiles /= TILE_WIDTH;
        }
    }

    uint32_t num_output_tiles_per_block = 0;

    for(uint32_t i = 0; i < num_input_tensors; i++) {
        auto buffer = input_tensors[i].buffer();
        src_addr[i] = buffer->address();
        is_dram[i] = buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        uint32_t dim_tiles = input_tensors[i].shape()[dim] / scale_factor;
        num_tiles_per_block[i] = num_accum_tiles * dim_tiles;
        num_output_tiles_per_block += num_accum_tiles * dim_tiles;
    }
    vector<uint32_t> common_reader_kernel_args = {
        num_input_tensors,
        0, 0, 0
    };
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), src_addr.begin(), src_addr.end());
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), is_dram.begin(), is_dram.end());
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), num_tiles_per_block.begin(), num_tiles_per_block.end());

    // Reader compile-time args
    // Data is 32 byte aligned
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_cb_index
    };
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/concat/kernels/dataflow/reader_concat_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        uint32_t block_id = num_tiles_written / num_output_tiles_per_block;
        uint32_t id_within_block = num_tiles_written % num_output_tiles_per_block;
        uint32_t curr_tensor = 0;
        uint32_t curr_tensor_id = 0;
        for (uint32_t j = 0; j < num_input_tensors; j++) {
            tile_id_per_tensor[j] = block_id * num_tiles_per_block[j];
            if (id_within_block == 0) {
            } else if (id_within_block >= num_tiles_per_block[j]) {
                tile_id_per_tensor[j] += num_tiles_per_block[j];
                id_within_block -= num_tiles_per_block[j];
                curr_tensor = j + 1;
            } else {
                tile_id_per_tensor[j] += id_within_block;
                curr_tensor = j;
                curr_tensor_id = id_within_block;
                id_within_block = 0;
            }
        }

        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[1] = num_tiles_per_core;
        reader_kernel_args[2] = curr_tensor;
        reader_kernel_args[3] = curr_tensor_id;
        reader_kernel_args.insert(reader_kernel_args.end(), tile_id_per_tensor.begin(), tile_id_per_tensor.end());

        vector<uint32_t> writer_kernel_args = {
            dst_buffer->address(),
            num_tiles_per_core,
            num_tiles_written
        };
        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            reader_kernel_args
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            writer_kernel_args
        );
        num_tiles_written+=num_tiles_per_core;
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_y
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        std::vector<uint32_t> src_addrs(input_buffers.size());
        for(uint32_t i = 0; i < input_buffers.size(); i++) {
            src_addrs[i] = input_buffers.at(0)->address();
        }

        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                std::copy(src_addrs.begin(), src_addrs.end(), runtime_args.begin() + 4);
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
