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

operation::ProgramWithCallbacks concat_single_core(const std::vector<Tensor> &input_tensors, uint32_t dim, Tensor &output) {

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt_metal::Device *device = output.device();

    uint32_t num_output_tiles = output.volume() / TILE_HW;

    uint32_t num_input_tensors = input_tensors.size();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

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
        tile_id_per_tensor[i] = 0;
    }
    vector<uint32_t> reader_kernel_args = {
        num_input_tensors,
        num_output_tiles, 0, 0
    };
    reader_kernel_args.insert(reader_kernel_args.end(), src_addr.begin(), src_addr.end());
    reader_kernel_args.insert(reader_kernel_args.end(), is_dram.begin(), is_dram.end());
    reader_kernel_args.insert(reader_kernel_args.end(), num_tiles_per_block.begin(), num_tiles_per_block.end());
    reader_kernel_args.insert(reader_kernel_args.end(), tile_id_per_tensor.begin(), tile_id_per_tensor.end());

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
        core,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args});

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        num_output_tiles,
        0
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

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id
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

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            std::copy(src_addrs.begin(), src_addrs.end(), runtime_args.begin() + 4);
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
