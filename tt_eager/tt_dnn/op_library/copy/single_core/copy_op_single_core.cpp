// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "common/bfloat8.hpp"
#include "common/bfloat4.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks copy_single_core(const Tensor &input, const Tensor &output, bool backwards) {
    Program program{};

    CoreRange core({0, 0}, {0, 0});

    bool tilized = output.get_layout() == Layout::TILE;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_unit_size = tilized ? tt_metal::detail::TileSize(input_cb_data_format) : input.get_legacy_shape()[-1] * input.element_size();
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_unit_size = tilized ? tt_metal::detail::TileSize(output_cb_data_format) : output.get_legacy_shape()[-1] * output.element_size();
    bool convert_dtype = input_cb_data_format != output_cb_data_format;

    uint32_t num_units = tilized ? output.volume() / TILE_HW : output.volume() / output.get_legacy_shape()[-1];

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = output.device();

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_units = 2;
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_units * aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
		.set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index; // same as input cb

    if (convert_dtype) {
        output_cb_index = 16; // output operands start at index 16
        uint32_t num_output_units = 2;
        uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
        tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_units * aligned_output_unit_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, aligned_output_unit_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, output_cb_config);
    }

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    // NOTE: If both src and dst is DRAM, need to read forwards since DRAM is allocated bottom up.
    //       If src and dst is not the same, it doesn't matter which way we read.
    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        reader_compile_time_args = {(uint32_t)src_is_dram};
        writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram
        };
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_unit_size);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(input_unit_size) : 0;
        reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) src_is_dram,
            (std::uint32_t) src_stick_size_is_power_of_two,
            (std::uint32_t) src_log2_stick_size
        };
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_unit_size);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(output_unit_size) : 0;
        writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram,
            (std::uint32_t) dst_stick_size_is_power_of_two,
            (std::uint32_t) dst_log2_stick_size
        };
    }
    std::map<string, string> kernel_defines;
    if (backwards) {
        kernel_defines["BACKWARDS"] = "1";
    }
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        tilized ? "tt_eager/tt_dnn/kernels/dataflow/reader_unary_interleaved_start_id.cpp" : "tt_eager/tt_dnn/kernels/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        tilized ? "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp" : "tt_eager/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    if (convert_dtype) {
        vector<uint32_t> compute_kernel_args = {
            num_units
        };
        auto eltwise_unary_kernel = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args=compute_kernel_args}
        );
    }

   if (tilized) {
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_units,
                backwards ? 0 : num_units - 1
            }
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_units,
                backwards ? 0 : num_units - 1
            }
        );
    } else {
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                input_unit_size,
                num_units,
                backwards ? 0 : num_units - 1
            }
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                output_unit_size,
                num_units,
                backwards ? 0 : num_units - 1
            }
        );
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
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
