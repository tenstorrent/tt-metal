// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "write_tile_block_program_factory.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "tt-metalium/host_api.hpp"
#include "ttnn/types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/write_tile_block/device/kernels/dataflow/reader_write_tile_block.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/write_tile_block/device/kernels/dataflow/writer_write_tile_block.cpp";

// reader runtime args: (source_addr, num_tiles, start_tile)
constexpr uint32_t kSourceBufferIdx = 0;
// writer runtime args: (destination_addr, num_tiles, start_tile, tile_row_offset, tile_col_offset, Wt_src, Wt_dst)
constexpr uint32_t kDestinationBufferIdx = 0;
constexpr uint32_t kTileRowOffsetIdx = 3;
constexpr uint32_t kTileColOffsetIdx = 4;

constexpr auto kCbIndex = tt::CBIndex::c_0;
constexpr uint32_t kBlockSize = 4U;

}  // namespace

namespace ttml::metal::ops::write_tile_block::device {

WriteTileBlockProgramFactory::cached_program_t WriteTileBlockProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t&) {
    const auto& source = tensor_args.source;
    const auto& destination = tensor_args.destination;
    auto* device = source.device();

    tt::tt_metal::Program program{};

    const tt::DataFormat data_format = datatype_to_dataformat_converter(source.dtype());
    TT_FATAL(data_format == tt::DataFormat::Float16_b, "WriteTileBlock: source data format must be Float16_b");
    const uint32_t tile_size_bytes = tt::tile_size(data_format);

    const auto src_shape = source.padded_shape();
    const uint32_t Wt_src = src_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Wt_dst = destination.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t num_tiles = src_shape.volume() / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
    // Source tile i (row i / Wt_src, column i % Wt_src) lands at destination tile
    // (tile_row_offset + i / Wt_src) * Wt_dst + tile_col_offset + i % Wt_src: one contiguous range when the widths
    // match, a strided column block otherwise (the writer maps every tile).

    auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_group_1, tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_tiles);

    [[maybe_unused]] auto cb =
        create_circular_buffer(program, all_cores, kCbIndex, data_format, tile_size_bytes, 2U * kBlockSize);

    auto* source_buffer = source.buffer();
    auto* destination_buffer = destination.buffer();
    TT_FATAL(
        source_buffer->buffer_type() == ttnn::BufferType::DRAM &&
            destination_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "WriteTileBlock: source and destination must be in DRAM (got {} and {})",
        enchantum::to_string(source_buffer->buffer_type()),
        enchantum::to_string(destination_buffer->buffer_type()));

    std::map<std::string, std::string> defines;
    tt::tt_metal::KernelHandle reader_kernel{};
    tt::tt_metal::KernelHandle writer_kernel{};
    {
        std::vector<uint32_t> compile_time_args{kBlockSize};
        tt::tt_metal::TensorAccessorArgs(source_buffer).append_to(compile_time_args);
        reader_kernel = create_reader_kernel(program, all_cores, compile_time_args, defines, kReaderKernelPath);
    }
    {
        std::vector<uint32_t> compile_time_args{kBlockSize};
        tt::tt_metal::TensorAccessorArgs(destination_buffer).append_to(compile_time_args);
        writer_kernel = create_writer_kernel(program, all_cores, compile_time_args, defines, kWriterKernelPath);
    }

    for_each_core_with_work(
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        tiles_per_core_group_1,
        tiles_per_core_group_2,
        [&](const CoreWork& work) {
            SetRuntimeArgs(program, reader_kernel, work.core, {source_buffer->address(), work.num_units, work.start});
            SetRuntimeArgs(
                program,
                writer_kernel,
                work.core,
                {destination_buffer->address(),
                 work.num_units,
                 work.start,
                 operation_attributes.tile_row_offset,
                 operation_attributes.tile_col_offset,
                 Wt_src,
                 Wt_dst});
        });

    return cached_program_t{std::move(program), {reader_kernel, writer_kernel, num_cores, num_cores_y}};
}

void WriteTileBlockProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t&) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    auto* source_buffer = tensor_args.source.buffer();
    auto* destination_buffer = tensor_args.destination.buffer();

    auto& reader_runtime_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for_each_core(shared.num_cores, shared.num_cores_y, [&](const tt::tt_metal::CoreCoord& core) {
        reader_runtime_args[core.x][core.y][kSourceBufferIdx] = source_buffer->address();
        auto& writer_args = writer_runtime_args[core.x][core.y];
        writer_args[kDestinationBufferIdx] = destination_buffer->address();
        writer_args[kTileRowOffsetIdx] = operation_attributes.tile_row_offset;
        writer_args[kTileColOffsetIdx] = operation_attributes.tile_col_offset;
    });
}

}  // namespace ttml::metal::ops::write_tile_block::device
