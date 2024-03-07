// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/fold/fold_op.hpp"
#include "tt_dnn/op_library/math.hpp"

namespace tt::tt_metal {
operation::ProgramWithCallbacks fold_single_core(
    const Tensor &input, const Tensor &output, uint8_t stride_h, uint8_t stride_w) {
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    uint32_t pixel_size = input.get_legacy_shape()[-1] * input.element_size();
    uint32_t num_pixels = input.volume() / input.get_legacy_shape()[-1];

    // chunk consists of channel values of stride_w neighboring pixels along the W dimension
    uint32_t width = input.get_legacy_shape()[2];
    uint32_t chunk_size = stride_w * pixel_size;
    uint32_t row_size = width * pixel_size;
    uint32_t dst_pixel_size = stride_h * chunk_size;
    uint32_t dst_row_size = stride_h * row_size;
    uint32_t num_dst_rows = num_pixels / (width * stride_h);
    uint32_t cb_pages_per_dst_row = stride_h * width;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = output.device();

    Buffer *src_buffer = input.buffer();
    Buffer *dst_buffer = output.buffer();
    bool src_is_dram = (src_buffer->buffer_type() == tt_metal::BufferType::DRAM);
    bool dst_is_dram = (dst_buffer->buffer_type() == tt_metal::BufferType::DRAM);

    // Setup CB.
    uint32_t cb_src0_index = CB::c_in0;
    uint32_t aligned_pixel_size = round_up_to_mul32(pixel_size);
    tt_metal::CircularBufferConfig cb_src0_config(
        2 * cb_pages_per_dst_row * aligned_pixel_size, {{cb_src0_index, cb_data_format}});
    cb_src0_config.set_page_size(cb_src0_index, aligned_pixel_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Since we only rejigger data around, we use a single CB in both reader and writer kernels.
    uint32_t cb_dst0_index = cb_src0_index;

    // We also create a scratch space CB for a single output page.
    uint32_t aligned_dst_pixel_size = round_up_to_mul32(dst_pixel_size);

    tt_metal::InterleavedBufferConfig l1_config{
        .device = device,
        .size = aligned_dst_pixel_size,
        .page_size = aligned_dst_pixel_size,
        .buffer_type = tt_metal::BufferType::L1};
    std::shared_ptr<Buffer> scratch_buffer = CreateBuffer(l1_config);

    // Setup kernels
    uint32_t src_unit_size_is_power_of_two = is_power_of_two_at_least_32(aligned_pixel_size);
    uint32_t src_log2_unit_size = src_unit_size_is_power_of_two ? (std::uint32_t)log2(aligned_pixel_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        cb_src0_index,
        src_is_dram,
        src_unit_size_is_power_of_two,
        src_log2_unit_size,
    };

    uint32_t dst_unit_size_is_power_of_two = is_power_of_two_at_least_32(aligned_dst_pixel_size);
    uint32_t dst_log2_unit_size = dst_unit_size_is_power_of_two ? (std::uint32_t)log2(aligned_dst_pixel_size) : 0;

    std::vector<uint32_t> writer_compile_time_args = {
        cb_dst0_index,
        dst_is_dram,
        dst_unit_size_is_power_of_two,
        dst_log2_unit_size,
    };

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp",
        core,
        ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/fold/kernels/dataflow/writer_unary_stick_layout_concatenate_rows_interleaved.cpp",
        core,
        WriterDataMovementConfig(writer_compile_time_args));

    SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), pixel_size, num_pixels, 0});

    // Writer run-time args
    std::vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        dst_pixel_size,
        scratch_buffer->address(),
        pixel_size,
        aligned_pixel_size,
        stride_w * aligned_pixel_size,
        width * aligned_pixel_size,
        stride_h,
        stride_w,
        num_dst_rows,
        width / stride_w,
        cb_pages_per_dst_row,
    };

    SetRuntimeArgs(program, writer_kernel_id, core, writer_kernel_args);

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        Buffer *src_buffer = input_buffers.at(0);
        Buffer *dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};
        GetRuntimeArgs(program, reader_kernel_id, core)[0] = src_buffer->address();
        GetRuntimeArgs(program, writer_kernel_id, core)[0] = dst_buffer->address();
    };

    return {std::move(program), override_runtime_args_callback};
}
}  // namespace tt::tt_metal
