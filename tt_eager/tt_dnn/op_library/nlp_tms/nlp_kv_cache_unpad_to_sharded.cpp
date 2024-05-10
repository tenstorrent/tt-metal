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

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > get_unpad_runtime_args_tile_sharded(const Tensor &input_tensor,
                                                                                Tensor& output_tensor,
                                                                                const Shape &output_tensor_start,
                                                                                uint32_t num_cores_total,
                                                                                uint32_t num_cores,
                                                                                uint32_t num_cores_x,
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

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH; // 4
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH; // 4
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt; // 0
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT; // 1
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT; // 68
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt; // (68-1)*4 = 268

    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt; // 68 * 4 = 272

    for(int32_t i = 2; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    log_info("[xuncai] get_unpad_runtime_args_tile_sharded");
    log_info("[xuncai] input_shape: {}", input_shape);
    log_info("[xuncai] output_shape: {}", output_shape);
    log_info("[xuncai] num_unpadded_tiles_per_dim: {}", num_unpadded_tiles_per_dim);
    log_info("[xuncai] num_padded_tiles_per_dim: {}", num_padded_tiles_per_dim);
    log_info("[xuncai] accumulated_total_per_dim: {}", accumulated_total_per_dim);

    vector<uint32_t> common_reader_kernel_args = {
        input_buffer->address(),
        num_dims,
        0, 0
    };
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), num_unpadded_tiles_per_dim.begin(), num_unpadded_tiles_per_dim.end());
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), num_padded_tiles_per_dim.begin(), num_padded_tiles_per_dim.end());

    // log_info("[xuncai] common_reader_kernel_args: {}", common_reader_kernel_args);

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val (num_cores_total);

    uint32_t start_offset = get_tiled_start_offset(input_tensor, output_tensor_start);

    // log_info("[xuncai] start_offset: {}", start_offset);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; i++){
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            //no-op
            num_tiles_per_core = 0;
        }

        log_info("[xuncai] Iteration: {} ------------------------------------------", i);
        log_info("[xuncai] core: {}", core);
        log_info("[xuncai] num_tiles_per_core: {}", num_tiles_per_core);

        id_per_dim[0] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for(uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        log_info("[xuncai] num_tiles_written: {}", num_tiles_written);
        // log_info("[xuncai] unpadded_written: {}", unpadded_written);
        // log_info("[xuncai] start_id: {}", start_id);
        log_info("[xuncai] id_per_dim: {}", id_per_dim);

        // reader kernel args
        vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        uint32_t addr_offset = 2; //input buffer addr, num_dims
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset] = num_tiles_per_core;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        // auto shard_spec = output_tensor.shard_spec().value();
        // uint32_t num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        // uint32_t num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        // uint32_t num_units_per_row = input_tensor.get_legacy_shape()[-1] / TILE_WIDTH;
        // uint32_t num_units_offset = num_units_per_row;
        // uint32_t shard_height = num_units_per_shard_height;
        // uint32_t shard_width = num_units_per_shard_width;
        // uint32_t curr_num_units_per_shard = shard_height * shard_width;
        // vector<uint32_t> reader_kernel_args = {
        //     input_buffer->address(),
        //     shard_height,
        //     shard_width,
        //     num_units_offset,
        //     curr_num_units_per_shard,
        //     start_id,
        // };

        // log_info("[xuncai] reader_kernel_args: {}", reader_kernel_args);

        // writer kernel args
        // vector<uint32_t> writer_kernel_args = {
        //     output_buffer->address(),
        //     num_tiles_per_core,
        //     num_tiles_written,
        //     0
        // };

        vector<uint32_t> writer_kernel_args = {
            num_tiles_per_core,
        };

        num_tiles_written+=num_tiles_per_core;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks multi_core_nlp_kv_cache_unpad_to_sharded(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {

    const Shape output_shape = output.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    uint32_t num_unpadded_tiles = output.volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cores_total = num_cores_x*num_cores_y;
    //CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    // Print all variables
    // log_info("[xuncai] unpad_tile_multi_core_sharded");
    // log_info("[xuncai] output_shape: {}", output_shape);
    // log_info("[xuncai] num_unpadded_tiles: {}", num_unpadded_tiles);
    // log_info("[xuncai] core info (old)");
    // log_info("[xuncai] compute_with_storage_grid_size: {}", compute_with_storage_grid_size);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);
    // log_info("[xuncai] num_cores_x: {}", num_cores_x);
    // log_info("[xuncai] num_cores_y: {}", num_cores_y);
    // log_info("[xuncai] num_cores_total: {}", num_cores_total);
    // log_info("[xuncai] num_cores: {}", num_cores);
    // log_info("[xuncai] all_cores: {}", all_cores);
    // log_info("[xuncai] core_group_1: {}", core_group_1);
    // log_info("[xuncai] core_group_2: {}", core_group_2);
    // log_info("[xuncai] num_tiles_per_core_group_1: {}", num_tiles_per_core_group_1);
    // log_info("[xuncai] num_tiles_per_core_group_2: {}", num_tiles_per_core_group_2);

    // log_info("[xuncai] core info new");
    auto shard_spec = output.shard_spec().value();
    all_cores = shard_spec.grid;
    num_cores = all_cores.num_cores();
    num_cores_total = num_cores;
    auto first_core_range = *all_cores.ranges().begin();
    num_cores_x = first_core_range.grid_size().x;
    num_cores_y = first_core_range.grid_size().y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});
    core_group_1 = shard_spec.grid; // all cores should evenly devide the work if shard_spec is correct
    uint32_t num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
    num_tiles_per_core_group_1 = num_units_per_shard_height * num_units_per_shard_width;
    num_tiles_per_core_group_2 = 0;
    log_info("[xuncai] num_cores_x: {}", num_cores_x);
    log_info("[xuncai] num_cores_y: {}", num_cores_y);
    log_info("[xuncai] num_cores_total: {}", num_cores_total);
    log_info("[xuncai] num_cores: {}", num_cores);
    log_info("[xuncai] all_cores: {}", all_cores);
    log_info("[xuncai] core_group_1: {}", core_group_1);
    log_info("[xuncai] core_group_2: {}", core_group_2);
    log_info("[xuncai] num_tiles_per_core_group_1: {}", num_tiles_per_core_group_1);
    log_info("[xuncai] num_tiles_per_core_group_2: {}", num_tiles_per_core_group_2);
    log_info("[xuncai] total_cores: {}", total_cores);


    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    // uint32_t src0_cb_index = 0;
    // uint32_t num_input_tiles = 2;
    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_tiles_per_core_group_1;
    // tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
	// 	.set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
	 	.set_page_size(src0_cb_index, single_tile_size)
        .set_globally_allocated_address(*output.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);




    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    // Reader
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_is_dram

    };
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/nlp_tms/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp",
        total_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)src0_is_dram};

    // tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
    //     program,
    //     "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp",
    //     total_cores,
    //     tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer
    // std::vector<uint32_t> writer_compile_time_args = {
    //     // interleaved accessor args
    //     (std::uint32_t) src0_cb_index,
    //     (std::uint32_t) dst_is_dram
    // };
    // tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
    //     program,
    //     "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
    //     total_cores,
    //     tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) src0_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        total_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));



    auto all_runtime_args = get_unpad_runtime_args_tile_sharded(a, output, output_tensor_start, num_cores_total, num_cores, num_cores_x, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);



    for (uint32_t i = 0; i < num_cores_total; i++){
        CoreCoord core = {i % num_cores_x, i / num_cores_x};

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

        // log_info("[xuncai] override_runtime_args_callback");

        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        uint32_t num_unpadded_tiles = dst_tensor.volume() / TILE_HW;

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto num_cores_total = num_cores_x*num_cores_y;

        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

        // log_info("[xuncai] core info new");
        auto shard_spec = dst_tensor.shard_spec().value();
        all_cores = shard_spec.grid;
        num_cores = all_cores.num_cores();
        num_cores_total = num_cores;
        auto first_core_range = *all_cores.ranges().begin();
        num_cores_x = first_core_range.grid_size().x;
        num_cores_y = first_core_range.grid_size().y;
        CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});
        core_group_1 = shard_spec.grid; // all cores should evenly devide the work if shard_spec is correct
        uint32_t num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        uint32_t num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_tiles_per_core_group_1 = num_units_per_shard_height * num_units_per_shard_width;
        num_tiles_per_core_group_2 = 0;

        const auto tensor_start = static_cast<const Unpad*>(operation)->output_tensor_start;
        auto all_runtime_args = get_unpad_runtime_args_tile_sharded(src_tensor, dst_tensor, tensor_start, num_cores_total,  num_cores, num_cores_x, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2);


        for (uint32_t i = 0; i < num_cores_total; i++){
            CoreCoord core = {i % num_cores_x, i / num_cores_x};
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

}  // namespace tt_metal

}  // namespace tt
