// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/topk/topk_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

namespace tt {
namespace tt_metal{

operation::ProgramWithCallbacks single_core_topk_interleaved(const Tensor &input_tensor, const uint32_t k, Tensor &value_tensor, Tensor &index_tensor) {
    Program program{};

    CoreRange core({0, 0}, {0, 0});
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat value_cb_data_format = tt_metal::datatype_to_dataformat_converter(value_tensor.get_dtype());
    tt::DataFormat index_cb_data_format = tt_metal::datatype_to_dataformat_converter(index_tensor.get_dtype());

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto input_buffer = input_tensor.buffer();
    auto values_buffer = value_tensor.buffer();
    auto index_buffer = index_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool values_is_dram = values_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool index_is_dram = index_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    uint32_t num_input_tiles = input_tensor.volume()/TILE_HW;
    uint32_t num_value_tiles = value_tensor.volume()/TILE_HW;

    auto input_shape = input_tensor.get_legacy_shape();
    uint32_t Ht = (input_shape[0]*input_shape[1]*input_shape[2])/TILE_HEIGHT;
    uint32_t Wt = input_shape[3]/TILE_WIDTH;
    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four tiles of space
    // TODO: In theory if we have enough memory we could allocate 2*Wt tiles to reduce stalls
    uint32_t input_cb_index = CB::c_in0;
    tt_metal::CircularBufferConfig input_cb_config = tt_metal::CircularBufferConfig(
        cb_in_units  * value_tile_size, {{input_cb_index, input_cb_data_format}})
		.set_page_size(input_cb_index, input_tile_size);
    auto cb_input_tensor = tt_metal::CreateCircularBuffer(program, core, input_cb_config);

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four tiles of space
    // This CB carries the indices that are created in the reader kernel
    uint32_t index_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig index_input_intermed0_config = tt_metal::CircularBufferConfig(
        cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
		.set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor = tt_metal::CreateCircularBuffer(program, core, index_input_intermed0_config);

    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig input_transposed_cb_config = tt_metal::CircularBufferConfig(
         Wt * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
		.set_page_size(input_transposed_cb_index, input_tile_size);
    auto cb_input_transposed_tiles = tt_metal::CreateCircularBuffer(program, core, input_transposed_cb_config);

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig index_transposed_cb_config = tt_metal::CircularBufferConfig(
         Wt * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
		.set_page_size(index_transposed_cb_index, index_tile_size);
    auto cb_index_transposed_tiles = tt_metal::CreateCircularBuffer(program, core, index_transposed_cb_config);

    // Output topk values
    uint32_t values_cb_index = CB::c_out0;
    tt_metal::CircularBufferConfig values_cb_config = tt_metal::CircularBufferConfig(
        num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
        .set_page_size(values_cb_index, value_tile_size);
    auto cb_values_tensor = tt_metal::CreateCircularBuffer(program, core, values_cb_config);


    // Output topk indices
    uint32_t output_ind_cb_index = CB::c_out1;
    tt_metal::CircularBufferConfig output_ind_cb_config = tt_metal::CircularBufferConfig(
        num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
        .set_page_size(output_ind_cb_index, index_tile_size);
    auto cb_output_ind_tensor = tt_metal::CreateCircularBuffer(program, core, output_ind_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
                                                        input_cb_index,
                                                        index_cb_index,
                                                        (uint32_t)input_is_dram,
                                                        Ht,
                                                        Wt};
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));


    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            input_buffer->address(),
        }
    );

    std::vector<uint32_t> writer_compile_time_args = {
                                                        values_cb_index,
                                                        output_ind_cb_index,
                                                        (std::uint32_t) values_is_dram,
                                                        (std::uint32_t) index_is_dram,
                                                        Ht,
                                                        k};
    tt_metal::KernelHandle binary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    SetRuntimeArgs(
        program,
        binary_writer_kernel_id,
        core,
        {
            values_buffer->address(),
            index_buffer->address(),

        }
    );

    std::vector<uint32_t> compute_args = {
                                        input_cb_index,
                                        index_cb_index,
                                        input_transposed_cb_index,
                                        index_transposed_cb_index,
                                        values_cb_index,
                                        output_ind_cb_index,
                                        Ht,
                                        Wt,
                                        k,
                                        (std::uint32_t) std::log2(k),
                                        (std::uint32_t) std::log2(Wt),
                                        };
    tt_metal::KernelHandle topk_compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );


    auto override_runtime_args_callback = [unary_reader_kernel_id, binary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto input_buffer = input_buffers.at(0);
        auto values_buffer = output_buffers.at(0);
        auto index_buffer = output_buffers.at(1);

        CoreCoord core = {0, 0};

        {
            auto &reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto &writer_runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, core);
            writer_runtime_args[0] = values_buffer->address();
            writer_runtime_args[1] = index_buffer->address();
        }

    };

    return {std::move(program), override_runtime_args_callback};
}

}
}
