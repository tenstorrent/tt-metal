// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "tt_dnn/op_library/cb_utils.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt::tt_metal {

operation::ProgramWithCallbacks tilize_multi_core_interleaved(const Tensor& a, Tensor& output) {
    tt_metal::Program program = tt_metal::CreateProgram();

    DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = detail::TileSize(output_cb_data_format);

    int32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = a.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t nblocks = std::ceil((float)ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.get_legacy_shape()[-1] * a.element_size();

    Device* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        split_blocks_for_tilize(grid_size, nblocks);

    create_cb(CB::c_in0, program, all_cores, input_single_tile_size, ntiles_per_block, input_cb_data_format);

    auto [output_cb_index, _] =
        create_cb(CB::c_out0, program, all_cores, output_single_tile_size, ntiles_per_block, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(block_size_nbytes);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (uint32_t)std::log2(block_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {src0_is_dram, stick_size_is_power_of_two, log2_stick_size};
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/tilize/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_ct_args = {output_cb_index, out_is_dram};
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    vector<uint32_t> compute_args = {nblocks_per_core, ntiles_per_block};
    vector<uint32_t> compute_args_cliff = {nblocks_per_core_cliff, ntiles_per_block};

    if (core_range.ranges().size() > 0) {
        auto tilize_kernel_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/tilize.cpp",
            core_range,
            ComputeConfig{.compile_args = compute_args});
    }
    if (core_range_cliff.size() > 0) {
        auto tilize_cliff_kernel_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{.compile_args = compute_args_cliff});
    }

    // 1D distribution of blocks across cores
    bool has_cliff = core_range_cliff.size() > 0;

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t ncores_x = grid_size.x;
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];

        // reader runtime args
        vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core * TILE_HEIGHT,
            block_size_nbytes,
            ntiles_per_block,
            block_size_nbytes,
            1,  // full blocks in row
            0,  // num leftover tiles
            0,  // leftover width in row
            row_start_id};

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core,  // ntiles per core
            tile_start_id                         // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (has_cliff) {
        // the last core is a cliff core with nblocks_per_core_cliff blocks
        const CoreCoord& core = cores.back();

        // reader runtime args
        vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core_cliff * TILE_HEIGHT,
            block_size_nbytes,
            ntiles_per_block,
            block_size_nbytes,
            1,  // full blocks in row
            0,  // num leftover tiles
            0,  // leftover width in row
            row_start_id};

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core_cliff,  // ntiles per core
            tile_start_id                               // start id
        };

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    auto override_runtime_args_callback =
        [reader_kernel_id = unary_reader_kernel_id, writer_kernel_id = unary_writer_kernel_id, cores = cores](
            const Program& program,
            const std::vector<Buffer*>& input_buffers,
            const std::vector<Buffer*>& output_buffers) {
            auto src_buffer = input_buffers.at(0);
            auto dst_buffer = output_buffers.at(0);

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                }
                {
                    auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = dst_buffer->address();
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks tilize_multi_core_sharded(const Tensor& input, Tensor& output) {
    tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t num_tiles = input.volume() / TILE_HW;

    tt_metal::Device* device = input.device();

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    auto all_cores = shard_spec.grid;
    uint32_t num_cores_x = device->compute_with_storage_grid_size().x;
    uint32_t num_cores = all_cores.num_cores();

    auto [src0_cb_index, cb_src0] = create_cb(
        CB::c_in0,
        program,
        all_cores,
        input_single_tile_size,
        num_tiles_per_shard,
        input_cb_data_format,
        input.buffer());

    auto [output_cb_index, cb_output] = create_cb(
        CB::c_out0,
        program,
        all_cores,
        output_single_tile_size,
        num_tiles_per_shard,
        output_cb_data_format,
        output.buffer());

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {uint32_t(num_tiles_per_shard / num_tiles_per_row), uint32_t(num_tiles_per_row)};

    auto untilize_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/compute/tilize.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = compute_args});

    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, {num_tiles_per_shard});

    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, {num_tiles_per_shard});

    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_output](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();

            auto dst_buffer = output_tensors.at(0).buffer();

            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);

            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks tilize_multi_core(const Tensor& a, Tensor& output) {
    if (a.memory_config().is_sharded()) {
        return tilize_multi_core_sharded(a, output);
    } else {
        return tilize_multi_core_interleaved(a, output);
    }
}

operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_interleaved(
    const Tensor& a, Tensor& output, const float pad_value) {
    tt_metal::Program program = tt_metal::CreateProgram();

    DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = detail::TileSize(output_cb_data_format);

    const Shape& input_shape = a.get_legacy_shape();
    const Shape& output_shape = output.get_legacy_shape();

    Device* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t num_blocks = output.volume() / output_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = output.get_legacy_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        split_blocks_for_tilize(grid_size, num_blocks);

    bool has_cliff = core_range_cliff.size() > 0;

    uint32_t unpadded_row_size_bytes = input_shape[-1] * a.element_size();  // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output_shape[-1] * a.element_size();   // Assuming bfloat16 dataformat

    auto [src0_cb_index, cb_src0] =
        create_cb(CB::c_in0, program, all_cores, input_single_tile_size, num_tiles_per_row, input_cb_data_format);

    auto [output_cb_index, cb_output] =
        create_cb(CB::c_out0, program, all_cores, output_single_tile_size, num_tiles_per_row, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/tilize/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig({src0_is_dram, stick_size_is_power_of_two, log2_stick_size}));

    /** writer
     */
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig({output_cb_index, out_is_dram}));

    /** compute
     */
    if (core_range.size() > 0) {
        auto tilize_kernel_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/tilize.cpp",
            core_range,
            ComputeConfig{.compile_args = {nblocks_per_core, num_tiles_per_row}});
    }
    if (has_cliff) {
        auto tilize_cliff_kernel_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{.compile_args = {nblocks_per_core_cliff, num_tiles_per_row}});
    }

    /* RUNTIME ARGS */

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    // 1D distribution of blocks across cores
    auto core_assignments = distribute_work(
        output_shape.without_padding(),
        output_shape.padding(),
        ncores,
        nblocks_per_core,
        has_cliff,
        nblocks_per_core_cliff);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    uint32_t ncores_x = grid_size.x;

    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // reader runtime args
        vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            unpadded_row_size_bytes,
            padded_row_size_bytes,
            packed_pad_value,
            row_start_id,
            static_cast<unsigned int>(assignment.size()),
        };

        uint32_t nblocks_per_core = 0;

        for (const auto& el : assignment) {
            nblocks_per_core += el.block_count();
            row_start_id += el.data_row_count();
            reader_rt_args.push_back(el.n_data);
            reader_rt_args.push_back(el.n_mixed);
            reader_rt_args.push_back(el.n_pads);
            reader_rt_args.push_back(el.times);
        }

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core;

        // writer runtime args
        vector<uint32_t> writer_rt_args = {dst_buffer->address(), num_tiles_per_core, tile_start_id};

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += num_tiles_per_core;
    }

    auto override_runtime_args_callback =
        [reader_kernel_id = unary_reader_kernel_id, writer_kernel_id = unary_writer_kernel_id, cores = cores](
            const Program& program,
            const std::vector<Buffer*>& input_buffers,
            const std::vector<Buffer*>& output_buffers) {
            auto src_buffer = input_buffers.at(0);
            auto dst_buffer = output_buffers.at(0);

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = src_buffer->address();
                }
                {
                    auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                    runtime_args[0] = dst_buffer->address();
                }
            }
        };

    return {std::move(program), override_runtime_args_callback};
}

// This purely supports input width shard -> output width shard for now
operation::ProgramWithCallbacks tilize_with_val_padding_multi_core_sharded(
    const Tensor& a, Tensor& output, const float pad_value) {
    tt_metal::Program program = tt_metal::CreateProgram();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();

    auto input_shard_spec = a.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();

    auto all_cores = output_shard_spec.grid;

    uint32_t num_batches = output.volume() / (output.get_legacy_shape()[-2] * output.get_legacy_shape()[-1]);

    uint32_t num_input_rows = input_shard_spec.shape[0];
    uint32_t input_shard_width_bytes = input_shard_spec.shape[1] * a.element_size();
    uint32_t ntiles_per_core = output_shard_spec.shape[0] * output_shard_spec.shape[1] / TILE_HW;
    uint32_t ntiles_per_batch = ntiles_per_core / num_batches;
    uint32_t ntiles_per_block = output_shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = output_shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_padded_rows = output.get_legacy_shape()[-2] - a.get_legacy_shape()[-2];

    auto [src0_cb_index, cb_src0] = create_cb(
        CB::c_in1,
        program,
        all_cores,
        input_shard_width_bytes,
        num_input_rows,
        input_cb_data_format,
        src_sharded ? a.buffer() : nullptr);

    auto [src1_cb_index, cb_src1] =
        create_cb(CB::c_in0, program, all_cores, input_single_tile_size, ntiles_per_batch * 2, input_cb_data_format);

    auto [src2_cb_index, cb_src2] =
        create_cb(CB::c_in2, program, all_cores, input_shard_width_bytes, 1, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        CB::c_out0,
        program,
        all_cores,
        output_single_tile_size,
        ntiles_per_core,
        output_cb_data_format,
        out_sharded ? output.buffer() : nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelHandle unary_reader_kernel_id;
    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
    };

    unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/tilize/kernels/dataflow/reader_unary_pad_height_width_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    KernelHandle unary_writer_kernel_id;
    bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    vector<uint32_t> writer_ct_args = {
        output_cb_index,
    };
    unary_writer_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    auto tilize_kernel_id = CreateKernel(
        program, "tt_eager/tt_dnn/kernels/compute/tilize.cpp", all_cores, ComputeConfig{.compile_args = compute_args});

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_rt_args = {
        num_input_rows,
        input_shard_width_bytes,
        (num_input_rows / num_batches) * input_shard_width_bytes,
        ntiles_per_batch,
        num_padded_rows,
        num_batches,
        packed_pad_value};
    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, reader_rt_args);

    vector<uint32_t> writer_rt_args = {ntiles_per_core};
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, writer_rt_args);

    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_output = cb_output](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks tilize_with_val_padding_multi_core(
    const Tensor& a, Tensor& output, const float pad_value) {
    if (a.memory_config().is_sharded()) {
        return tilize_with_val_padding_multi_core_sharded(a, output, pad_value);
    } else {
        return tilize_with_val_padding_multi_core_interleaved(a, output, pad_value);
    }
}

}  // namespace tt::tt_metal
