// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/concat/concat_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks concat_single_core(
    const std::vector<Tensor> &input_tensors, const uint32_t dim, const Tensor &output) {
    tt_metal::Program program = tt_metal::CreateProgram();

    const CoreRange core({0, 0}, {0, 0});

    tt_metal::Device *device = output.device();

    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    const bool rm_layout = output.get_layout() == Layout::ROW_MAJOR;

    uint32_t num_output_pages;
    uint32_t single_page_size;
    if (rm_layout) {
        num_output_pages = output.volume() / output.get_legacy_shape()[-1];
        single_page_size = align(output.element_size() * output.get_legacy_shape()[-1], ADDRESS_ALIGNMENT);
    } else {
        num_output_pages = output.volume() / TILE_HW;
        single_page_size = tt_metal::detail::TileSize(cb_data_format);
    }

    uint32_t num_input_tensors = input_tensors.size();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_pages = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_pages * single_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t num_dims = output.get_legacy_shape().rank();

    std::vector<uint32_t> src_addr(num_input_tensors);
    std::vector<bool> is_dram(num_input_tensors);
    std::vector<uint32_t> num_pages_per_block(num_input_tensors);
    std::vector<uint32_t> page_id_per_tensor(num_input_tensors);
    // Only used for RM
    std::vector<uint32_t> page_size_per_tensor(num_input_tensors);

    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;

    // RM is special cased in the loop (dim_units = 1 for last dim else it's the dim size)
    if (!rm_layout) {
        if (dim == num_dims - 2) {
            scale_factor = TILE_HEIGHT;
        } else if (dim == num_dims - 1) {
            scale_factor = TILE_WIDTH;
        }
    }

    for (uint32_t i = dim + 1; i < num_dims; ++i) {
        num_accum_pages *= output.get_legacy_shape()[i];
    }
    if (rm_layout) {
        if (num_dims > 1 && dim < num_dims - 1) {
            num_accum_pages /= output.get_legacy_shape()[-1];
        }
    } else {
        if (dim < num_dims - 2) {
            num_accum_pages /= TILE_HW;
        } else if (dim == num_dims - 2) {
            num_accum_pages /= TILE_WIDTH;
        }
    }

    uint32_t num_output_pages_per_block = 0;

    if (rm_layout) {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            is_dram[i] = buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
            page_size_per_tensor[i] = buffer->page_size();
            if (dim == num_dims - 1) {
                num_pages_per_block[i] = num_accum_pages;
            } else {
                uint32_t dim_pages = input_tensors[i].get_legacy_shape()[dim];
                num_pages_per_block[i] = num_accum_pages * dim_pages;
                num_output_pages_per_block += num_accum_pages * dim_pages;
            }
        }
        if (dim == num_dims - 1) {
            num_output_pages_per_block = 1;
        }
    } else {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            is_dram[i] = buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
            uint32_t dim_pages = input_tensors[i].get_legacy_shape()[dim] / scale_factor;
            num_pages_per_block[i] = num_accum_pages * dim_pages;
            num_output_pages_per_block += num_accum_pages * dim_pages;
        }
    }

    vector<uint32_t> reader_kernel_args = {num_output_pages, 0, 0};
    reader_kernel_args.insert(reader_kernel_args.end(), src_addr.begin(), src_addr.end());
    reader_kernel_args.insert(reader_kernel_args.end(), is_dram.begin(), is_dram.end());
    reader_kernel_args.insert(reader_kernel_args.end(), num_pages_per_block.begin(), num_pages_per_block.end());
    if (rm_layout) {
        reader_kernel_args.insert(reader_kernel_args.end(), page_size_per_tensor.begin(), page_size_per_tensor.end());
    }
    reader_kernel_args.insert(reader_kernel_args.end(), page_id_per_tensor.begin(), page_id_per_tensor.end());

    // Reader compile-time args
    // Data is 32 byte aligned
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_cb_index,
                                                      (std::uint32_t)num_input_tensors};

    std::map<string, string> concat_defines;

    if (rm_layout && dim == num_dims - 1) {
        concat_defines["WIDTH_CONCAT"] = "1";
    }

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_cb_index,
                                                      (std::uint32_t)dst_is_dram};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        rm_layout
            ? "tt_eager/tt_dnn/op_library/concat/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp"
            : "tt_eager/tt_dnn/op_library/concat/kernels/dataflow/reader_concat_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, concat_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        rm_layout ? "tt_eager/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp"
                  : "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> writer_kernel_args;
    if (rm_layout) {
        writer_kernel_args = {dst_buffer->address(), output.buffer()->page_size(), num_output_pages, 0};
    } else {
        writer_kernel_args = {dst_buffer->address(), num_output_pages, 0};
    }
    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        std::vector<uint32_t> src_addrs(input_buffers.size());
        for (uint32_t i = 0; i < input_buffers.size(); ++i) {
            src_addrs[i] = input_buffers[i]->address();
        }

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            std::copy(src_addrs.begin(), src_addrs.end(), runtime_args.begin() + 3);
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
