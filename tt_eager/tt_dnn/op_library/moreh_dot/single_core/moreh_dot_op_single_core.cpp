// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_dot/moreh_dot_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_dot_single_core(const Tensor &a, const Tensor &b, Tensor &output) {
    Program program{};
    CoreRange core = {.start = {0, 0}, .end = {0, 0}};

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    uint32_t num_tiles = a.volume() / TILE_HW;
    float scaler = 1.0f;
    const auto &a_shape_wo_padding = a.shape().without_padding();
    uint32_t pad_h = a_shape_wo_padding[2] % TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (TILE_WIDTH) : (pad_w);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(dst_single_tile_size, {{CB::c_in2, dst_cb_data_format}})
            .set_page_size(CB::c_in2, dst_single_tile_size);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, core, cb_scaler_config);

    uint32_t interm0_cb_index = 24;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(dst_single_tile_size, {{interm0_cb_index, dst_cb_data_format}})
            .set_page_size(interm0_cb_index, dst_single_tile_size);
    auto cb_interm0 = tt_metal::CreateCircularBuffer(program, core, interm0_cb_config);

    uint32_t interm1_cb_index = 25;
    tt_metal::CircularBufferConfig interm1_cb_config =
        tt_metal::CircularBufferConfig(dst_single_tile_size, {{interm1_cb_index, dst_cb_data_format}})
            .set_page_size(interm1_cb_index, dst_single_tile_size);
    auto cb_interm1 = tt_metal::CreateCircularBuffer(program, core, interm1_cb_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_is_dram, (std::uint32_t)src1_is_dram, *reinterpret_cast<uint32_t *>(&scaler)};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    KernelID binary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_dot/single_core/kernels/reader_binary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_dot/single_core/kernels/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    // defines["ELTWISE_OP"] = "mul_tiles";
    // defines["ELTWISE_OP_CODE"] = "2";

    auto dot_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_dot/single_core/kernels/dot.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = defines});

    tt_metal::SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {src0_buffer->address(), src1_buffer->address(), num_tiles, 0, mask_h, mask_w});

    tt_metal::SetRuntimeArgs(program, dot_kernel, core, {num_tiles, 1});

    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {output.buffer()->address(), 1, 0});

    auto override_runtime_arguments_callback = [binary_reader_kernel_id, unary_writer_kernel_id, dot_kernel](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        uint32_t num_tiles = input_tensors.at(0).volume() / TILE_HW;

        {
            auto runtime_args = GetRuntimeArgs(program, binary_reader_kernel_id, core);
            runtime_args[0] = src_buffer_a->address();
            runtime_args[1] = src_buffer_b->address();
            runtime_args[2] = num_tiles;
            SetRuntimeArgs(program, binary_reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, dot_kernel, core);
            runtime_args[0] = num_tiles;
            SetRuntimeArgs(program, dot_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = 1;
            SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
        }
    };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
