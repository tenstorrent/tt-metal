// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "tt_dnn/op_library/cb_utils.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

namespace untilize_helpers {
uint32_t get_num_cores(CoreCoord grid_size, uint32_t nblocks) {
    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    int32_t ncores = ncores_x * ncores_y;
    if (nblocks <= ncores) {
        ncores = nblocks;
    } else {
        uint32_t nblocks_per_core = ceil((float)nblocks / ncores);
        ncores = ceil((float)nblocks / nblocks_per_core);
    }
    return ncores;
}
}  // namespace untilize_helpers

operation::ProgramWithCallbacks untilize_multi_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt_metal::Program program = tt_metal::CreateProgram();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();

    uint32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = a.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t nblocks = ceil((float)ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.get_legacy_shape()[-1] * output.element_size();

    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        split_blocks_for_tilize(grid_size, nblocks);
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);

    bool row_major = true;
    bool src_block_sharded = false;
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    std::vector<CoreCoord> cores_with_rtargs;

    if (src_sharded) {
        auto shard_spec = a.shard_spec().value();
        src_block_sharded = a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
        ncores_y = device->compute_with_storage_grid_size().y;
        all_cores = shard_spec.grid;
        uint32_t num_cores = all_cores.num_cores();
        ncores = num_cores;
        core_range = all_cores;
        core_range_cliff = CoreRangeSet({});
        ntiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
        nblocks_per_core = shard_spec.shape[0] / TILE_HEIGHT;
        nblocks_per_core_cliff = 0;

        num_rows_block = shard_spec.shape[0];
        block_row_size = shard_spec.shape[1] * output.element_size();  // in0_block_w * TILE_WIDTH * dtype_nbytes
        output_row_size = output.get_legacy_shape()[-1] * output.element_size();  // output row size bytes
        last_block_row_size_unpadded = block_row_size - (round_up(output.get_legacy_shape()[-1], shard_spec.shape[1]) -
                                                         output.get_legacy_shape()[-1]) *
                                                            output.element_size();
        uint32_t num_output_rows = output.volume() / output.get_legacy_shape()[-1];
        num_output_rows_unpadded = num_rows_block - (round_up(num_output_rows, shard_spec.shape[0]) - num_output_rows);
        end_core = (*shard_spec.grid.ranges().begin()).end;
    }

    uint32_t num_input_tiles = src_sharded ? ntiles_per_block * nblocks_per_core : ntiles_per_block * 2;
    auto [src0_cb_index, cb_src0] = create_cb(
        CB::c_in0,
        program,
        all_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        src_sharded ? a.buffer() : nullptr);

    uint32_t num_output_tiles = out_sharded ? ntiles_per_block * nblocks_per_core : ntiles_per_block * 2;
    auto [output_cb_index, cb_output] = create_cb(
        CB::c_out0,
        program,
        all_cores,
        output_single_tile_size,
        num_output_tiles,
        output_cb_data_format,
        out_sharded ? output.buffer() : nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelHandle unary_reader_kernel_id;

    if (src_sharded) {
        std::vector<uint32_t> reader_ct_args = {(std::uint32_t)src0_cb_index};

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_ct_args));
    } else {
        bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        vector<uint32_t> reader_ct_args = {(uint32_t)src0_is_dram};

        unary_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            all_cores,
            ReaderDataMovementConfig(reader_ct_args));
    }

    /** writer
     */
    KernelHandle unary_writer_kernel_id;
    if (out_sharded) {
        std::vector<uint32_t> writer_ct_args = {(std::uint32_t)output_cb_index};
        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_ct_args));
    } else {
        bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        if (src_block_sharded) {
            vector<uint32_t> writer_ct_args = {
                (uint32_t)out_is_dram, (uint32_t)(input_cb_data_format == tt::DataFormat::Float32)};
            unary_writer_kernel_id = CreateKernel(
                program,
                "tt_eager/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp",
                all_cores,
                WriterDataMovementConfig(writer_ct_args));
        } else {
            bool stick_size_is_power_of_two = is_power_of_two_at_least_32(block_size_nbytes);
            uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(block_size_nbytes) : 0;
            vector<uint32_t> writer_ct_args = {
                (uint32_t)out_is_dram,
                (uint32_t)stick_size_is_power_of_two,
                (uint32_t)log2_stick_size,
            };

            unary_writer_kernel_id = CreateKernel(
                program,
                "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/"
                "writer_unary_stick_layout_split_rows_interleaved.cpp",
                all_cores,
                WriterDataMovementConfig(writer_ct_args));
        }
    }

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };
    vector<uint32_t> compute_args_cliff = {
        (uint32_t)nblocks_per_core_cliff,
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    std::string compute_kernel("tt_eager/tt_dnn/op_library/untilize/kernels/compute/pack_untilize.cpp");
    if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(LogOp, "Using slow untilize.");
        compute_kernel = std::string("tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp");
    } else {
        log_debug(LogOp, "Using fast pack untilize.");
    }

    if (core_range.ranges().size() > 0) {
        auto untilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});
    }
    if (core_range_cliff.ranges().size() > 0) {
        auto untilize_cliff_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range_cliff,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args_cliff});
    }

    // 1D distribution of blocks across all cores
    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
    for (uint32_t i = 0; i < cores.size(); i++) {
        CoreCoord core = cores[i];
        if (!full_cores.core_coord_in_core_ranges(core)) {
            continue;
        }
        // reader runtime args
        vector<uint32_t> reader_rt_args;

        if (src_sharded) {
            reader_rt_args = {
                ntiles_per_block * nblocks_per_core  // ntiles
            };
        } else {
            reader_rt_args = {
                src0_buffer->address(),               // src_addr
                ntiles_per_block * nblocks_per_core,  // ntiles
                tile_start_id                         // start_id
            };
        }
        // log_debug("reader[{}]: {},{} = {} ({})", src0_buffer->address(), core.x, core.y, tile_start_id,
        // ntiles_per_block * nblocks_per_core);

        // writer runtime args
        vector<uint32_t> writer_rt_args;
        if (out_sharded) {
            writer_rt_args = {
                ntiles_per_block * nblocks_per_core  // ntiles
            };
        } else {
            if (src_block_sharded) {
                uint32_t block_start_row_offset;
                uint32_t block_start_row_id_offset;
                uint32_t row_size_unpadded = block_row_size;
                uint32_t num_rows_unpadded = num_rows_block;
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }

                writer_rt_args = {
                    dst_buffer->address(),  // dst_addr
                    num_rows_block,
                    block_row_size,
                    1,
                    1,
                    1,
                    output_row_size,
                    row_size_unpadded,
                    num_rows_unpadded,
                    block_start_row_id_offset,
                    block_start_row_offset};
            } else {
                writer_rt_args = {
                    dst_buffer->address(),           // dst_addr
                    nblocks_per_core * TILE_HEIGHT,  // nblocks per core
                    block_size_nbytes,               // block_size_nbytes
                    ntiles_per_block,                // ntiles_per_block
                    block_size_nbytes,               // block_size_nbytes
                    1,                               // full blocks in a row
                    0,
                    0,
                    row_start_id};
            }
        }
        // log_debug("writer[{}]: {},{} = {} {}", dst_buffer->address(), core.x, core.y, block_size_nbytes,
        // row_start_id);

        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);

        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x}
                                   : CoreCoord{ncores_full / ncores_y, ncores_full % ncores_y};
        // reader runtime args
        vector<uint32_t> reader_rt_args;

        if (src_sharded) {
            reader_rt_args = {
                ntiles_per_block * nblocks_per_core_cliff  // ntiles
            };
        } else {
            reader_rt_args = {
                src0_buffer->address(),                               // src_addr
                (uint32_t)ntiles_per_block * nblocks_per_core_cliff,  // ntiles
                tile_start_id                                         // start_id
            };
        }
        // log_debug("reader: {},{} = {} ({})", core.x, core.y, tile_start_id, ntiles_per_block *
        // nblocks_per_core_cliff);

        // writer runtime args
        vector<uint32_t> writer_rt_args;
        if (out_sharded) {
            writer_rt_args = {
                ntiles_per_block * nblocks_per_core_cliff  // ntiles
            };
        } else {
            if (src_block_sharded) {
                uint32_t block_start_row_offset;
                uint32_t block_start_row_id_offset;
                uint32_t row_size_unpadded = block_row_size;
                uint32_t num_rows_unpadded = num_rows_block;
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                writer_rt_args = {
                    dst_buffer->address(),  // dst_addr
                    num_rows_block,
                    block_row_size,
                    1,
                    1,
                    1,
                    output_row_size,
                    row_size_unpadded,
                    num_rows_unpadded,
                    block_start_row_id_offset,
                    block_start_row_offset};
            } else {
                writer_rt_args = {
                    dst_buffer->address(),                 // dst_addr
                    nblocks_per_core_cliff * TILE_HEIGHT,  // nsticks
                    block_size_nbytes,                     // stick_size_nbytes
                    ntiles_per_block,                      // ntiles_per_block
                    block_size_nbytes,                     // block_width_nbytes
                    1,                                     // full blocks in a row
                    0,                                     // UNUSED
                    0,                                     // UNUSED
                    row_start_id};
            }
        }
        // log_debug("writer: {},{} = {} {}", core.x, core.y, block_size_nbytes, row_start_id);

        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);

        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
    }
    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_output = cb_output,
                                                cores_with_rtargs](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool src_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        if (src_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        } else {
            auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer->address();
            }
        }

        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
        } else {
            auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const CoreCoord& core : cores_with_rtargs) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks untilize_with_unpadding_multi_core_interleaved(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt_metal::Program program = tt_metal::CreateProgram();

    DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = detail::TileSize(output_cb_data_format);

    const Shape& input_shape = a.get_legacy_shape();
    const Shape& output_shape = output.get_legacy_shape();

    Device* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();

    uint32_t num_blocks = a.volume() / input_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.get_legacy_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        split_blocks_for_tilize(grid_size, num_blocks);

    bool has_cliff = core_range_cliff.size() > 0;

    uint32_t padded_row_size_bytes = input_shape[-1] * a.element_size();     // Assuming bfloat16 dataformat
    uint32_t unpadded_row_size_bytes = output_shape[-1] * a.element_size();  // Assuming bfloat16 dataformat

    create_cb(CB::c_in0, program, all_cores, input_single_tile_size, num_tiles_per_row, input_cb_data_format);
    create_cb(CB::c_out0, program, all_cores, output_single_tile_size, num_tiles_per_row, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    uint32_t src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig({src0_is_dram}));

    /** writer
     */
    uint32_t out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    uint32_t stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multicore.cpp",
        all_cores,
        WriterDataMovementConfig(
            {out_is_dram,
             stick_size_is_power_of_two,
             log2_stick_size,
             input_cb_data_format == tt::DataFormat::Float32}));

    /** compute
     */
    std::string compute_kernel("tt_eager/tt_dnn/op_library/untilize/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_row > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        compute_kernel = "tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp";
    }

    if (core_range.size() > 0) {
        auto tilize_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range,
            ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {nblocks_per_core, num_tiles_per_row}});
    }
    if (has_cliff) {
        auto tilize_cliff_kernel_id = CreateKernel(
            program,
            compute_kernel,
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {nblocks_per_core_cliff, num_tiles_per_row}});
    }

    auto input_w = input_shape.rank() >= 4 ? input_shape[-4] : 1;
    auto input_z = input_shape.rank() >= 3 ? input_shape[-3] : 1;
    auto input_y = input_shape.rank() >= 2 ? input_shape[-2] : 1;
    auto input_x = input_shape[-1];

    auto output_w = output_shape.rank() >= 4 ? output_shape[-4] : 1;
    auto output_z = output_shape.rank() >= 3 ? output_shape[-3] : 1;
    auto output_y = output_shape.rank() >= 2 ? output_shape[-2] : 1;
    auto output_x = output_shape[-1];

    Padding padding(
        {{0, input_w - output_w}, {0, input_z - output_z}, {0, input_y - output_y}, {0, input_x - output_x}},
        Padding::PadValue::Any);
    auto core_assignments =
        distribute_work(output_shape, padding, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    uint32_t ncores_x = grid_size.x;

    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            unpadded_row_size_bytes,
            padded_row_size_bytes,
            row_start_id,
            static_cast<unsigned int>(assignment.size()),
        };

        uint32_t nblocks_per_core = 0;

        for (const auto& el : assignment) {
            nblocks_per_core += el.block_count();
            row_start_id += el.data_row_count();
            writer_rt_args.push_back(el.n_data);
            writer_rt_args.push_back(el.n_mixed);
            writer_rt_args.push_back(el.n_pads);
            writer_rt_args.push_back(el.times);
        }

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core;

        // reader runtime args
        vector<uint32_t> reader_rt_args = {src0_buffer->address(), num_tiles_per_core, tile_start_id};

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

// This purely supports input block shard -> output interleaved for now
operation::ProgramWithCallbacks untilize_with_unpadding_multi_core_sharded(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    tt_metal::Program program = tt_metal::CreateProgram();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    Device* device = a.device();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    uint32_t last_idx = 0;
    auto shard_spec = a.shard_spec().value();

    // I am not sure it is correct to ever use the shard_spec here.
    auto out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto grid = *shard_spec.grid.ranges().begin();
    uint32_t ncores_x = grid.end.x + 1;
    uint32_t ncores_y = grid.end.y + 1;
    auto all_cores = shard_spec.grid;
    uint32_t ncores = all_cores.num_cores();
    uint32_t ntiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t batch = a.volume() / (a.get_legacy_shape()[-2] * a.get_legacy_shape()[-1]);
    uint32_t ntiles_per_batch = ntiles_per_block * nblocks_per_core / batch;

    num_rows_block = out_shard_spec.shape[0];
    block_row_size = out_shard_spec.shape[1] * output.element_size();         // in0_block_w * TILE_WIDTH * dtype_nbytes
    output_row_size = output.get_legacy_shape()[-1] * output.element_size();  // output row size bytes
    last_block_row_size_unpadded = block_row_size - (round_up(output.get_legacy_shape()[-1], out_shard_spec.shape[1]) -
                                                     output.get_legacy_shape()[-1]) *
                                                        output.element_size();
    uint32_t num_output_rows = output.volume() / output.get_legacy_shape()[-1];
    num_output_rows_unpadded = num_rows_block - (round_up(num_output_rows, out_shard_spec.shape[0]) - num_output_rows);
    if (a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        last_idx = div_up(output.get_legacy_shape()[-1], out_shard_spec.shape[1]) - 1;
    } else if (a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        last_idx = div_up(num_output_rows, out_shard_spec.shape[0]) - 1;
    } else {
        end_core = {
            div_up(output.get_legacy_shape()[-1], out_shard_spec.shape[1]) - 1,
            div_up(num_output_rows, out_shard_spec.shape[0]) - 1};
    }
    if (!row_major) {
        std::swap(end_core.x, end_core.y);
    }

    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    auto [src0_cb_index, cb_src0] = create_cb(
        CB::c_in0,
        program,
        all_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        src_sharded ? a.buffer() : nullptr);

    uint32_t num_output_tiles = out_sharded ? ntiles_per_batch * 2 : ntiles_per_block * 2;
    auto [output_cb_index, cb_output] =
        create_cb(CB::c_out0, program, all_cores, output_single_tile_size, num_output_tiles, output_cb_data_format);

    auto [sharded_output_cb_index, cb_sharded_output] = out_sharded ? create_cb(
                                                                          CB::c_out1,
                                                                          program,
                                                                          all_cores,
                                                                          block_row_size,
                                                                          num_output_rows_unpadded,
                                                                          output_cb_data_format,
                                                                          output.buffer())
                                                                    : std::make_tuple(CB::c_out1, CBHandle{});

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelHandle unary_reader_kernel_id;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t)src0_cb_index};

    unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    KernelHandle unary_writer_kernel_id;
    if (out_sharded) {
        vector<uint32_t> writer_ct_args = {(uint32_t)output_cb_index, (uint32_t)sharded_output_cb_index};
        unary_writer_kernel_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_unpad_batch_rows_sharded.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct_args));
    } else {
        bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
        vector<uint32_t> writer_ct_args = {
            (uint32_t)out_is_dram, (uint32_t)(input_cb_data_format == tt::DataFormat::Float32)};
        unary_writer_kernel_id = CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct_args));
    }

    /** compute
     */
    vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    std::string compute_kernel("tt_eager/tt_dnn/op_library/untilize/kernels/compute/pack_untilize.cpp");
    if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(LogOp, "Using slow untilize.");
        compute_kernel = std::string("tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp");
    } else {
        log_debug(LogOp, "Using fast pack untilize.");
    }

    auto untilize_kernel_id = CreateKernel(
        program,
        compute_kernel,
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args});

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        ntiles_per_block * nblocks_per_core  // ntiles
    };
    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, reader_rt_args);
    std::vector<CoreCoord> cores;

    if (out_sharded) {
        vector<uint32_t> writer_rt_args = {
            num_output_rows_unpadded,
            ntiles_per_batch,
            out_shard_spec.shape[0] / batch,
            shard_spec.shape[1] * output.element_size(),
            block_row_size,
            batch};
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, writer_rt_args);
    } else {
        uint32_t tile_start_id = 0;
        uint32_t row_start_id = 0;
        cores = grid_to_cores(ncores, ncores_x, ncores_y, row_major);
        for (uint32_t i = 0; i < cores.size(); ++i) {
            CoreCoord& core = cores[i];

            // writer runtime args
            vector<uint32_t> writer_rt_args;
            uint32_t block_start_row_offset;
            uint32_t block_start_row_id_offset;
            uint32_t row_size_unpadded = block_row_size;
            uint32_t num_rows_unpadded = num_rows_block;
            if (a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                block_start_row_offset = i * block_row_size;
                block_start_row_id_offset = 0;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    num_rows_unpadded = num_output_rows_unpadded;
                    if (i == last_idx) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                }
            } else if (a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                block_start_row_offset = 0;
                block_start_row_id_offset = i * num_rows_block;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    if (i == last_idx) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                    row_size_unpadded = last_block_row_size_unpadded;
                }
            } else {
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                if (core.x > end_core.x || core.y > end_core.y) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                }
            }

            writer_rt_args = {
                dst_buffer->address(),  // dst_addr
                num_rows_block,
                block_row_size,
                1,
                1,
                1,
                output_row_size,
                row_size_unpadded,
                num_rows_unpadded,
                block_start_row_id_offset,
                block_start_row_offset};

            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        }
    }

    auto override_runtime_arguments_callback = [reader_kernel_id = unary_reader_kernel_id,
                                                writer_kernel_id = unary_writer_kernel_id,
                                                cb_src0 = cb_src0,
                                                cb_sharded_output = cb_sharded_output,
                                                cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        bool src_sharded = input_tensors.at(0).memory_config().is_sharded();
        bool out_sharded = output_tensors.at(0).memory_config().is_sharded();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);

        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_sharded_output, *dst_buffer);
        } else {
            auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
            for (const CoreCoord& core : cores) {
                auto& runtime_args = runtime_args_by_core[core.x][core.y];
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {
    if (a.memory_config().is_sharded()) {
        return untilize_with_unpadding_multi_core_sharded(a, output, use_pack_untilize, fp32_dest_acc_en);
    } else {
        return untilize_with_unpadding_multi_core_interleaved(a, output, use_pack_untilize, fp32_dest_acc_en);
    }
}

}  // namespace tt_metal

}  // namespace tt
