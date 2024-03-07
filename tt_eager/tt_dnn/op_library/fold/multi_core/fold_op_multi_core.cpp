// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/fold/fold_op.hpp"
#include "tt_dnn/op_library/math.hpp"

namespace tt::tt_metal {
operation::ProgramWithCallbacks fold_multi_core(
    const Tensor& input, const Tensor& output, uint8_t stride_h, uint8_t stride_w) {
    Program program = CreateProgram();
    Device* device = output.device();

    auto all_cores = input.shard_spec()->grid;
    auto shard_shape = input.shard_spec()->shape;

    DataFormat cb_data_format = datatype_to_dataformat_converter(input.get_dtype());

    uint32_t pixel_size = shard_shape[1] * input.element_size();
    uint32_t num_pixels = shard_shape[0];
    uint32_t num_dst_pixels = num_pixels / (stride_h * stride_w);

    // chunk consists of channel values of stride_w neighboring pixels along the W dimension
    uint32_t width = input.get_legacy_shape()[2];
    uint32_t chunk_size = stride_w * pixel_size;
    uint32_t row_size = width * pixel_size;
    uint32_t dst_pixel_size = stride_h * chunk_size;
    uint32_t dst_row_size = stride_h * row_size;
    uint32_t num_dst_rows = num_pixels / (width * stride_h);
    uint32_t pixels_per_dst_row = stride_h * width;

    // input CB
    uint32_t cb_src0_index = CB::c_in0;
    uint32_t aligned_pixel_size = round_up_to_mul32(pixel_size);
    auto src_cb_config = CircularBufferConfig(num_pixels * aligned_pixel_size, {{cb_src0_index, cb_data_format}})
                             .set_page_size(cb_src0_index, aligned_pixel_size)
                             .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output CB
    uint32_t cb_dst0_index = CB::c_out0;
    uint32_t aligned_dst_pixel_size = round_up_to_mul32(dst_pixel_size);
    auto dst_cb_config =
        CircularBufferConfig(num_dst_pixels * aligned_dst_pixel_size, {{cb_dst0_index, cb_data_format}})
            .set_page_size(cb_dst0_index, aligned_dst_pixel_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = CreateCircularBuffer(program, all_cores, dst_cb_config);

    // Setup kernel
    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/fold/kernels/dataflow/writer_cb2s_row_major.cpp",
        all_cores,
        WriterDataMovementConfig({cb_src0_index, cb_dst0_index}));

    // Writer run-time args
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        all_cores,
        {
            pixel_size,
            aligned_pixel_size,
            aligned_dst_pixel_size,
            num_pixels,
            num_dst_pixels,
            stride_w * aligned_pixel_size,
            width * aligned_pixel_size,
            stride_h,
            stride_w,
            num_dst_rows,
            width / stride_w,
            pixels_per_dst_row * aligned_pixel_size,
        });

    auto override_runtime_args_callback = [writer_kernel_id, stride_h, stride_w, cb_src0, cb_dst0](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto shard_shape = input_tensors[0].shard_spec()->shape;
        auto all_cores = input_tensors[0].shard_spec()->grid;

        DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensors[0].get_dtype());

        uint32_t pixel_size = shard_shape[1] * input_tensors[0].element_size();
        uint32_t num_pixels = shard_shape[0];
        uint32_t num_dst_pixels = num_pixels / (stride_h * stride_w);

        uint32_t width = input_tensors[0].get_legacy_shape()[2];
        uint32_t chunk_size = stride_w * pixel_size;
        uint32_t row_size = width * pixel_size;
        uint32_t dst_pixel_size = stride_h * chunk_size;
        uint32_t dst_row_size = stride_h * row_size;
        uint32_t num_dst_rows = num_pixels / (width * stride_h);
        uint32_t cb_pages_per_dst_row = stride_h * width;

        uint32_t aligned_pixel_size = round_up_to_mul32(pixel_size);
        uint32_t aligned_dst_pixel_size = round_up_to_mul32(dst_pixel_size);

        auto src_buffer = input_tensors[0].buffer();
        auto dst_buffer = output_tensors[0].buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_dst0, *dst_buffer);

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            all_cores,
            {
                pixel_size,
                aligned_pixel_size,
                aligned_dst_pixel_size,
                num_pixels,
                num_dst_pixels,
                stride_w * aligned_pixel_size,
                width * aligned_pixel_size,
                stride_h,
                stride_w,
                num_dst_rows,
                width / stride_w,
                cb_pages_per_dst_row,
            });
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}
}  // namespace tt::tt_metal
