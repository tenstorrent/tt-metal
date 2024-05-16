// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_llama_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_llama_single_core(
    const Tensor &input,
    const Tensor &cos,
    const Tensor &sin,
    const Tensor &trans_mat,
    Tensor &output,
    DeviceComputeKernelConfig compute_kernel_config
) {
    Program program{};

    CoreRangeSet core({CoreRange({0, 0}, {0, 0})});

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.get_dtype());
    uint32_t cos_single_tile_size = tt_metal::detail::TileSize(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.get_dtype());
    uint32_t sin_single_tile_size = tt_metal::detail::TileSize(sin_cb_data_format);

    tt::DataFormat trans_mat_cb_data_format = tt_metal::datatype_to_dataformat_converter(trans_mat.get_dtype());
    uint32_t trans_mat_single_tile_size = tt_metal::detail::TileSize(trans_mat_cb_data_format);

    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;
    uint32_t num_rows = input.volume() / input.get_legacy_shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.get_legacy_shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.get_legacy_shape()[-1] * sizeof(bfloat16);

    tt_metal::Device *device = input.device();

    MathFidelity math_fidelity;
    bool fp32_dest_acc_en;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::WORMHOLE_B0, "kernel config is not for wormhole_b0");
            math_fidelity = compute_kernel_config.math_fidelity;
            fp32_dest_acc_en = input_cb_data_format == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    uint32_t input_cb_index = CB::c_in0;
    uint32_t num_input_tiles = 2 * Wt;
    tt_metal::CircularBufferConfig cb_input_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size);
    auto cb_input = tt_metal::CreateCircularBuffer(program, core, cb_input_config);

    uint32_t num_cos_sin_tiles = 2 * Wt;

    uint32_t cos_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig cb_cos_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size);
    auto cb_cos = tt_metal::CreateCircularBuffer(program, core, cb_cos_config);

    uint32_t sin_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig cb_sin_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size);
    auto cb_sin = tt_metal::CreateCircularBuffer(program, core, cb_sin_config);

    uint32_t trans_mat_cb_index = CB::c_in3;
    uint32_t num_trans_mat_tiles = 2;
    tt_metal::CircularBufferConfig cb_trans_mat_config =
        tt_metal::CircularBufferConfig(num_input_tiles * trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
            .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size);
    auto cb_trans_mat = tt_metal::CreateCircularBuffer(program, core, cb_trans_mat_config);

    uint32_t num_interm_tiles = 1;
    uint32_t rotated_input_interm_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig cb_rotated_input_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size);
    auto cb_rotated_input_interm = tt_metal::CreateCircularBuffer(program, core, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * cos_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    auto cb_cos_interm = tt_metal::CreateCircularBuffer(program, core, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = CB::c_intermed2;
    tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * sin_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    auto cb_sin_interm = tt_metal::CreateCircularBuffer(program, core, cb_sin_interm_config);

    uint32_t output_cb_index = CB::c_out0;  // output operands start at index 16
    uint32_t num_output_tiles = 2 * Wt;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    std::map<string, string> kernel_defines;

    auto src_buffer = input.buffer();
    auto cos_buffer = cos.buffer();
    auto sin_buffer = sin.buffer();
    auto trans_mat_buffer = trans_mat.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool cos_is_dram = cos_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool sin_is_dram = sin_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool trans_mat_is_dram = trans_mat_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)trans_mat_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)cos_is_dram,
        (std::uint32_t)sin_is_dram,
        (std::uint32_t)trans_mat_is_dram
        (std::uint32_t)Ht,
        (std::uint32_t)Wt,
        (std::uint32_t)HtWt,
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/rotary_embedding/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/rotary_embedding/kernels/dataflow/writer_rotary_embedding_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    vector<uint32_t> compute_kernel_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)trans_mat_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)num_rows,
        (std::uint32_t)Wt
        };

    auto rotary_embedding_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/rotary_embedding/kernels/compute/rotary_embedding.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity=math_fidelity, .fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_kernel_args, .defines = kernel_defines});

    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    // if (token_idx.has_value()) {
    //     cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
    //     cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    // }

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src_buffer->address(), cos_buffer->address(), sin_buffer->address(), num_rows, 0, 0, cos_sin_start_id});

    SetRuntimeArgs(
        program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0, cos_sin_offset, Wt, Wbytes});

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, Wbytes, Wt](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        const auto token_idx = static_cast<const RotaryEmbedding *>(operation)->token_idx;

        auto src_buffer = input_tensors.at(0).buffer();
        auto cos_buffer = input_tensors.at(1).buffer();
        auto sin_buffer = input_tensors.at(2).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        uint32_t cos_sin_offset = 0;
        uint32_t cos_sin_start_id = 0;
        if (token_idx.has_value()) {
            cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
            cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
        }

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = cos_buffer->address();
            runtime_args[2] = sin_buffer->address();
            runtime_args[6] = cos_sin_start_id;
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[3] = cos_sin_offset;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
