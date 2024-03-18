// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/move/move_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// Sharded buffers are mapped to CBs. Move from top of src CB to dst CB
operation::ProgramWithCallbacks move_multi_core_sharded(const Tensor &input, Tensor &output) {
    tt_metal::Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
    auto shard_spec = input.shard_spec().value();
    auto shard_shape = shard_spec.shape;
    auto shard_grid = shard_spec.grid;
    auto input_shape = input.get_legacy_shape();
    auto input_dtype = input.get_dtype();
    auto input_layout = input.get_layout();
    TT_FATAL(input_layout == output.get_layout() && input_dtype == output.get_dtype() && shard_shape == output.shard_spec().value().shape && input_shape == output.get_legacy_shape());
    const uint32_t src_cb_sharded = CB::c_in0;
    const uint32_t dst_cb_sharded = CB::c_in1;
    uint32_t tile_size_bytes = tile_size(cb_data_format);
    uint32_t shard_shape_num_tiles = div_up(shard_shape[0] * shard_shape[1], TILE_HEIGHT * TILE_WIDTH);
    uint32_t total_size_bytes = 0;
    uint32_t page_size_bytes = 0;
    if ((shard_shape[0] * shard_shape[1]) % (TILE_HEIGHT * TILE_WIDTH) == 0) {
        uint32_t tile_size_bytes = tile_size(cb_data_format);
        total_size_bytes = shard_shape_num_tiles * tile_size_bytes;
        page_size_bytes = tile_size_bytes;
    } else {
        uint32_t datum_size_bytes =  datum_size(cb_data_format);
        total_size_bytes = shard_shape[0] * shard_shape[1] * datum_size_bytes;
        page_size_bytes = shard_shape[1] *  datum_size_bytes;
    }
    CircularBufferConfig src_cb_sharded_config = CircularBufferConfig(total_size_bytes, {{src_cb_sharded, cb_data_format}})
		    .set_page_size(src_cb_sharded, page_size_bytes);
    src_cb_sharded_config.set_globally_allocated_address(*input.buffer());
    auto src_sharded_cb = tt_metal::CreateCircularBuffer(program, shard_grid, src_cb_sharded_config);

    CircularBufferConfig dst_cb_sharded_config = CircularBufferConfig(total_size_bytes, {{dst_cb_sharded, cb_data_format}})
		    .set_page_size(dst_cb_sharded, page_size_bytes);
    dst_cb_sharded_config.set_globally_allocated_address(*output.buffer());
    auto dst_sharded_cb = tt_metal::CreateCircularBuffer(program, shard_grid, dst_cb_sharded_config);

    auto input_buffer_address = input.buffer()->address();
    auto output_buffer_address = output.buffer()->address();
    TT_FATAL(output_buffer_address > input_buffer_address, "Expected output buffer to be allocated at a higher address than input buffer");
    uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
    TT_FATAL(move_chunk_size_bytes % ADDRESS_ALIGNMENT == 0, "Expected chunk size bytes to move to be {} byte aligned.", ADDRESS_ALIGNMENT);
    uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
    uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;

    std::vector<uint32_t> reader_compile_time_args = {src_cb_sharded, dst_cb_sharded};
    KernelHandle kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/move/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp",
        shard_grid,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::NOC_1,
            .compile_args = reader_compile_time_args}
    );
    std::vector<uint32_t> runtime_args = {total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
    SetRuntimeArgs(program, kernel_id, shard_grid, runtime_args);

    auto override_runtime_args_callback = [shard_grid=shard_grid, kernel_id=kernel_id, src_sharded_cb=src_sharded_cb, dst_sharded_cb=dst_sharded_cb, total_size_bytes=total_size_bytes](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        UpdateDynamicCircularBufferAddress(program, src_sharded_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, dst_sharded_cb, *dst_buffer);
        auto input_buffer_address = src_buffer->address();
        auto output_buffer_address = dst_buffer->address();
        uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
        uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
        uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;
        std::vector<uint32_t> runtime_args = {total_size_bytes, num_chunks, move_chunk_size_bytes, remainder_chunk_size_bytes};
        SetRuntimeArgs(program, kernel_id, shard_grid, runtime_args);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
