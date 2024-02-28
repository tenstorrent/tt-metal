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


std::vector<std::unordered_map <CoreCoord, std::vector<CorePageRange> > > get_core_page_ranges_width (
    Buffer * input_buffer,
    Buffer * output_buffer,
    uint32_t num_input_tensors
    )

{

    std::vector<std::unordered_map <CoreCoord, std::vector<CorePageRange> > >  ret_vec;
    const std::vector<uint32_t> & output_shard_to_host_mapping = output_buffer->get_dev_page_to_host_page_mapping();



    for(uint32_t input_tensor_id = 0; input_tensor_id < num_input_tensors; input_tensor_id++) {
        const std::vector<uint32_t> & input_page_to_local_page_mapping = input_buffer->get_host_page_to_local_shard_page_mapping() ;
        const std::vector<uint32_t> & host_page_to_input_page_mapping = input_buffer->get_host_page_to_dev_page_mapping();
        uint32_t num_pages = output_buffer->num_pages();

    // First get output_core to vector< pair<input_core, input_page> (num_pages_in_output)
        std::unordered_map <CoreCoord, std::vector<std::pair<CoreCoord, uint32_t> > > output_core_to_vector_input_core_page;

        for(uint32_t output_page_id = 0; output_page_id < num_pages; output_page_id++) {
            // because height sharded
            auto output_host_page = output_page_id;
            auto input_host_page = output_host_page/num_input_tensors;
            auto output_core = output_buffer->get_core_from_dev_page_id(output_page_id);

            //height_sharded
            auto input_page = input_host_page;
            auto local_input_page = input_page_to_local_page_mapping[input_host_page];
            auto input_core = input_buffer->get_core_from_dev_page_id(input_page);
            if (output_core_to_vector_input_core_page.find(output_core) == output_core_to_vector_input_core_page.end()) {
                output_core_to_vector_input_core_page[output_core] = {{input_core, local_input_page}};
            }
            else {
                output_core_to_vector_input_core_page[output_core].push_back({input_core, local_input_page});
            }
        }


    //now compress to output_core to vector<pair<input_core, input_page_range> (num_page_ranges_in_output)
        std::unordered_map <CoreCoord, std::vector<CorePageRange> > ret_map;
        auto output_cores = corerange_to_cores(output_buffer->shard_spec().grid());

        for(auto output_core: output_cores) {
            if( output_core_to_vector_input_core_page.find(output_core) != output_core_to_vector_input_core_page.end()) {
                auto vector_of_input_core_input_page = output_core_to_vector_input_core_page.at(output_core);
                ret_map.insert({output_core, std::vector<CorePageRange>()});
                for(uint32_t outer_input_core_page_id=0;
                    outer_input_core_page_id < vector_of_input_core_input_page.size();
                    outer_input_core_page_id++) {
                    //we know its on the same output core, get range where input page is
                    // consecutively increasing and on same input core
                    uint32_t curr_inner_page = vector_of_input_core_input_page[outer_input_core_page_id].second;
                    CoreCoord curr_input_core = vector_of_input_core_input_page[outer_input_core_page_id].first;
                    for(uint32_t inner_input_core_page_id=outer_input_core_page_id ;
                        inner_input_core_page_id < vector_of_input_core_input_page.size();
                        inner_input_core_page_id++){
                        auto core = vector_of_input_core_input_page[inner_input_core_page_id].first;
                        auto input_page = vector_of_input_core_input_page[inner_input_core_page_id].second;
                        if(core != curr_input_core || input_page != curr_inner_page + 1) {
                            outer_input_core_page_id = inner_input_core_page_id;
                            break;
                        }
                        curr_inner_page = vector_of_input_core_input_page[inner_input_core_page_id].second;
                    }
                    CorePageRange page_range;
                    page_range.core = curr_input_core;
                    page_range.range.start = vector_of_input_core_input_page[outer_input_core_page_id].second;
                    page_range.range.end = curr_inner_page + 1;
                    ret_map.at(output_core).push_back(page_range);

                }
            }
        }
        ret_vec.push_back(ret_map);
    }
    return ret_vec;
}


operation::ProgramWithCallbacks s2s_rm_concat_multi_core(const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output) {
    tt_metal::Program program = tt_metal::CreateProgram();
    uint32_t num_input_tensors = input_tensors.size();
    tt_metal::Device *device = output.device();

    //assuming all shard_specs for input tensors are the same
    auto output_to_input_mapping = get_core_page_ranges_width(input_tensors[0].buffer(), output.buffer(), num_input_tensors);

    auto output_shard_spec = output.shard_spec().value();
    auto all_cores = output_shard_spec.grid;

    uint32_t dst_cb_index = 16;
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_s2s_width.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig({dst_cb_index, num_input_tensors}));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig({dst_cb_index}));

    auto input_shard_spec = input_tensors[0].shard_spec().value();
    auto unit_size = input_shard_spec.shape[1] * input_tensors[0].element_size();
    auto page_size = input_tensors[0].shape()[-1] * input_tensors[0].element_size();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    auto output_shard_shape = output_shard_spec.shape;
    //using input_size as page_size, so need to multiply num pages by num_input tensors
    tt_metal::CircularBufferConfig cb_dst_config =
        tt_metal::CircularBufferConfig(output_shard_shape[0]*output_shard_shape[1]*output.element_size()*num_input_tensors , {{dst_cb_index, data_format}})
                .set_page_size(dst_cb_index, unit_size)
                .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    bool row_wise = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
    for(auto core: cores) {
        std::vector<uint32_t> runtime_args = {0};
        uint32_t num_output_pages = 0;
        for (uint32_t input_tensor_id = 0; input_tensor_id < num_input_tensors; input_tensor_id++) {
            auto page_range_vector = output_to_input_mapping[input_tensor_id].at(core);
            uint32_t num_ranges = page_range_vector.size();
            runtime_args.push_back(input_tensors[input_tensor_id].buffer()->address());
            runtime_args.push_back(num_ranges);
            for (uint32_t range_id = 0; range_id < num_ranges; range_id++) {
                auto physical_input_core = device->worker_core_from_logical_core(page_range_vector[range_id].core);
                runtime_args.push_back(physical_input_core.x);
                runtime_args.push_back(physical_input_core.y);
                runtime_args.push_back(page_range_vector[range_id].range.start * unit_size); //start addr_offset
                runtime_args.push_back((page_range_vector[range_id].range.end - page_range_vector[range_id].range.start)*unit_size); //size
                num_output_pages += page_range_vector[range_id].range.end - page_range_vector[range_id].range.start;
            }
        }
        runtime_args[0] = num_output_pages;
        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {num_output_pages});
    }

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, page_size, cb_dst0, num_input_tensors](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        auto output_core_to_page_range_pair = get_core_page_ranges_width(src_buffer, dst_buffer, num_input_tensors);

        auto input_shard_spec = input_tensors.at(0).shard_spec().value();
        auto output_shard_spec = output_tensors.at(0).shard_spec().value();
        auto all_cores = input_shard_spec.grid.merge(output_shard_spec.grid);
        auto cores = corerange_to_cores(all_cores);
        auto device = input_tensors.at(0).device();
        auto unit_size = input_shard_spec.shape[1] * input_tensors[0].element_size();
        auto page_size = input_tensors[0].shape()[-1] * input_tensors[0].element_size();

        for(auto core: cores) {
            std::vector<uint32_t> runtime_args = {0};
            uint32_t num_output_pages = 0;
            for (uint32_t input_tensor_id = 0; input_tensor_id < num_input_tensors; input_tensor_id++) {
                auto page_range_vector = output_core_to_page_range_pair[input_tensor_id].at(core);
                uint32_t num_ranges = page_range_vector.size();
                runtime_args.push_back(input_tensors[input_tensor_id].buffer()->address());
                runtime_args.push_back(num_ranges);
                for (uint32_t range_id = 0; range_id < num_ranges; range_id++) {
                    auto physical_input_core = device->worker_core_from_logical_core(page_range_vector[range_id].core);
                    runtime_args.push_back(physical_input_core.x);
                    runtime_args.push_back(physical_input_core.y);
                    runtime_args.push_back(page_range_vector[range_id].range.start * unit_size); //start addr_offset
                    runtime_args.push_back((page_range_vector[range_id].range.end - page_range_vector[range_id].range.start)*unit_size); //size
                    num_output_pages += page_range_vector[range_id].range.end - page_range_vector[range_id].range.start;
                }
            }
            runtime_args[0] = num_output_pages;
            tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {num_output_pages});
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
                            curr_num_output_rows, input_unit_size, num_input_tensors*input_shard_spec.shape[0], };
        for(uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            reader_runtime_args.push_back(input_id);
            reader_runtime_args.push_back(input_shard_spec.shape[0]);
            writer_runtime_args.push_back(input_id);
            writer_runtime_args.push_back(input_shard_spec.shape[0]);
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
