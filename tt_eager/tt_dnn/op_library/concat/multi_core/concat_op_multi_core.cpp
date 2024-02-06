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
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args});

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

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
