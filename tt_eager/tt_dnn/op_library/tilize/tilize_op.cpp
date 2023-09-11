// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/math.hpp"
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"

#include "tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


inline std::tuple<int32_t, int32_t, int32_t, int32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t>
    split_blocks_across_cores(CoreCoord grid_size, uint32_t nblocks) {

    int32_t ncores_x = grid_size.x;
    int32_t ncores_y = grid_size.y;
    int32_t ncores = ncores_x * ncores_y;
    uint32_t nblocks_per_core = nblocks;
    uint32_t nblocks_per_core_cliff = 0;
    int32_t ncores_x_cliff = 0;
    std::set<CoreRange> all_cores;
    std::set<CoreRange> core_range, core_range_cliff;
    if (nblocks <= ncores) {
        nblocks_per_core = 1;
        ncores = nblocks;
        ncores_y = ceil((float) ncores / ncores_x);
        ncores_x_cliff = ncores - (ncores_x * (ncores_y - 1));
        if (ncores_x_cliff == ncores_x) {
            // no cliff, all is perfectly divisible
            ncores_x_cliff = 0;
            core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
            all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
        } else if (ncores_x_cliff == 1) {
            // just one cliff core in the last row
            nblocks_per_core_cliff = 1;
            if (ncores_y > 1) {
                core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
            }
            core_range_cliff.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
            all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
        } else if (ncores_x_cliff > 1) {
            // both normal and cliff cores in the last row
            nblocks_per_core_cliff = 1;
            if (ncores_y > 1) {
                core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
            }
            core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
            core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
            all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
        } else {
            TT_ASSERT(false, "Something went really wrong in splitting blocks across cores {} {}!!", ncores_x, ncores_x_cliff);
        }
    } else {
        nblocks_per_core = ceil((float) nblocks / ncores);
        ncores = ceil((float) nblocks / nblocks_per_core);
        nblocks_per_core_cliff = nblocks - nblocks_per_core * (ncores - 1);
        ncores_y = ceil((float) ncores / ncores_x);
        ncores_x_cliff = ncores - ncores_x * (ncores_y - 1);
        if (nblocks_per_core_cliff == nblocks_per_core) {
            // no special cliff at block level for per core
            if (ncores_x_cliff == ncores_x) {
                // no x_cliff row => all cores are equal
                ncores_x_cliff = 0;
                core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
            } else if (ncores_x_cliff == 1) {
                // just 1 core as cliff in the last core row
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range_cliff.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
            } else if (ncores_x_cliff < ncores_x) {
                // last core row has last core as cliff, rest are normal
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
                core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
            } else {
                TT_ASSERT("Something went really wrong in calculating the core ranges {} {}", ncores_x, ncores_x_cliff);
            }
        } else if (nblocks_per_core_cliff < nblocks_per_core) {
            // last core has unequal blocks
            if (ncores_x_cliff == ncores_x) {
                // ncores x is same throughout
                ncores_x_cliff = 0;
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
                core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
            } else if (ncores_x_cliff == 1) {
                // last core row only has 1 core, as cliff
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range_cliff.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(0, ncores_y - 1)});
            } else if (ncores_x_cliff < ncores_x) {
                if (ncores_y > 1) {
                    core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                    all_cores.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
                }
                core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 2, ncores_y - 1)});
                core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x_cliff - 1, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
                all_cores.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_x_cliff - 1, ncores_y - 1)});
            } else {
                TT_ASSERT(false, "Something went very wrong in calculating core ranges (case 2)");
            }
        } else {
            TT_ASSERT(false, "Somehting went really wrong in splitting blocks across cores (case else)");
        }
    }
    return std::make_tuple(ncores, ncores_x, ncores_x_cliff, ncores_y, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff);
}

operation::ProgramWithCallbacks tilize_multi_core(const Tensor &a, Tensor& output) {
    tt_metal::Program program = tt_metal::Program();

    TT_ASSERT(a.dtype() == DataType::BFLOAT16, "Only BFLOAT16 data type supported for tilize.");
    TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "Input is not in RM. This case is not yet supported");

    DataFormat cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    int32_t ntiles = a.volume() / TILE_HW;
    uint32_t ntiles_per_block = a.shape()[3] / TILE_WIDTH;
    uint32_t nblocks = ceil((float) ntiles / ntiles_per_block);
    uint32_t block_size_nbytes = a.shape()[3] * a.element_size();

    {
        log_debug(LogOp, "ntiles: {}", ntiles);
        log_debug(LogOp, "ntiles_per_block: {}", ntiles_per_block);
        log_debug(LogOp, "nblocks: {}", nblocks);
        log_debug(LogOp, "block_size_nbytes: {}", block_size_nbytes);
    }

    Device *device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, ncores_x, ncores_x_cliff, ncores_y, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] = split_blocks_across_cores(grid_size, nblocks);

    {
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "ncores_x_cliff: {}", ncores_x_cliff);
        log_debug(LogOp, "ncores_y: {}", ncores_y);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_per_core_cliff: {}", nblocks_per_core_cliff);
    }

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = ntiles_per_block;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t output_cb_index = CB::c_out0;
    uint32_t num_output_tiles = ntiles_per_block;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t block_width_nbytes = ntiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = 1;
    uint32_t num_leftover_tiles = 0;
    uint32_t leftover_width_in_row = 0;

    /** reader
     */
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(block_size_nbytes);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t) std::log2(block_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };
    KernelID unary_reader_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    /** writer
     */
    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) out_is_dram
    };
    KernelID unary_writer_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    /** compute
     */
    vector<uint32_t> compute_args = {
        nblocks_per_core,
        ntiles_per_block
    };
    vector<uint32_t> compute_args_cliff = {
        nblocks_per_core_cliff,
        ntiles_per_block
    };

    if (core_range.ranges().size() > 0) {
        auto tilize_kernel_id = CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/tilize.cpp",
            core_range,
            ComputeConfig{
                .compile_args = compute_args});
    }
    if (core_range_cliff.ranges().size() > 0) {
        auto tilize_cliff_kernel_id = CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/tilize.cpp",
            core_range_cliff,
            ComputeConfig{
                .compile_args = compute_args_cliff});

    }

    // 1D distribution of blocks across cores
    uint32_t ncores_full = ncores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff on the last core
        ncores_full -= 1;
    }
    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;
    for (uint32_t i = 0; i < ncores_full; ++ i) {
        CoreCoord core = {i % ncores_x, i / ncores_x};

        // reader runtime args
        vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core * TILE_HEIGHT,
            block_size_nbytes,
            ntiles_per_block,
            block_size_nbytes,
            1,              // full blocks in row
            0,              // num leftover tiles
            0,              // leftover width in row
            row_start_id
        };
        // log_debug("reader: {},{} = {} {}", core.x, core.y, block_size_nbytes, row_start_id);

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core,    // ntiles per core
            tile_start_id                           // start id
        };
        // log_debug("writer: {},{} = {} ({})", core.x, core.y, tile_start_id, ntiles_per_block * nblocks_per_core_cliff);

        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            reader_rt_args
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            writer_rt_args
        );

        tile_start_id += ntiles_per_block * nblocks_per_core;
        row_start_id += TILE_HEIGHT * nblocks_per_core;
    }
    if (ncores_full < ncores) {
        // the last core is a cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = {ncores_full % ncores_x, ncores_full / ncores_x};

        // reader runtime args
        vector<uint32_t> reader_rt_args = {
            src0_buffer->address(),
            nblocks_per_core_cliff * TILE_HEIGHT,
            block_size_nbytes,
            ntiles_per_block,
            block_size_nbytes,
            1,              // full blocks in row
            0,              // num leftover tiles
            0,              // leftover width in row
            row_start_id
        };
        // log_debug("reader: {},{} = {} {}", core.x, core.y, block_size_nbytes, row_start_id);

        // writer runtime args
        vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            ntiles_per_block * nblocks_per_core_cliff,    // ntiles per core
            tile_start_id                           // start id
        };
        // log_debug("writer: {},{} = {} ({})", core.x, core.y, tile_start_id, ntiles_per_block * nblocks_per_core_cliff);

        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            reader_rt_args
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            writer_rt_args
        );
    }

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id,
        ncores=ncores,
        ncores_x=ncores_x
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < ncores; ++ i) {
            CoreCoord core = {i % ncores_x, i / ncores_x};
            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }
            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks tilize_single_core(const Tensor &a, Tensor& output) {

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = output.shape();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_sticks = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] * a.shape()[2] : a.shape()[0] * a.shape()[2] * a.shape()[3];
    uint32_t stick_size = stick_s * a.element_size(); // Assuming bfloat16 dataformat

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - L1_UNRESERVED_BASE;
    uint32_t max_tiles = max_l1_size / (2 * single_tile_size); // 2 CBs
    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, src0_cb_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
        0                       // row_start_id
    };

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) out_is_dram
    };

    // Tilized reader
    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    // Tilized writer
    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block), // per_core_block_cnt
        uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
    };

    auto tilize_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {dst_buffer->address(),
        (uint32_t) num_tiles,
        (uint32_t) 0}
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void Tilize::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to tilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);

    uint32_t stick_s =  input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.shape()[3] : input_tensor_a.shape()[1];
    uint32_t num_sticks = input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.volume() / input_tensor_a.shape()[3] : input_tensor_a.volume() / input_tensor_a.shape()[1];
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);

    uint32_t stick_size = stick_s * input_tensor_a.element_size(); // Assuming bfloat16 dataformat

    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");
}

std::vector<Shape> Tilize::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto output_shape = input_tensor_a.shape();
    return {output_shape};
}

std::vector<Tensor> Tilize::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Tilize::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (use_multicore) {
        return tilize_multi_core(input_tensor_a, output_tensor);
    } else {
        return tilize_single_core(input_tensor_a, output_tensor);
    }
}

tt::stl::reflection::Attributes Tilize::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor tilize(const Tensor &input_tensor_a, const MemoryConfig& output_mem_config, bool use_multicore) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return input_tensor_a;
    }
    return operation::run_without_autoformat(Tilize{output_mem_config, use_multicore}, {input_tensor_a}).at(0);
}

operation::ProgramWithCallbacks tilize_with_val_padding_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value) {


    auto output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    int32_t num_tiles = output.volume() / TILE_HW;

    auto true_input_shape = a.shape();
    auto true_output_shape = output.shape();

    uint32_t unpadded_row_size_datum = true_input_shape[3];
    uint32_t padded_row_size_datum = true_output_shape[3];

    uint32_t num_rows_padded = true_output_shape[2];
    uint32_t num_cols_padded = true_output_shape[3] - unpadded_row_size_datum;


    uint32_t num_2d_faces = true_output_shape[0] * true_output_shape[1];

    uint32_t unpadded_row_size_bytes = unpadded_row_size_datum * a.element_size(); // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = padded_row_size_datum * a.element_size(); // Assuming bfloat16 dataformat

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = true_output_shape[3] / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - L1_UNRESERVED_BASE;
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * a.element_size();
    uint32_t num_blocks_w_output = padded_row_size_bytes / block_row_size;
    uint32_t num_blocks_w_input = unpadded_row_size_bytes / block_row_size;

    // Leftover size if input is not divisible by block size
    uint32_t block_row_leftover_size = unpadded_row_size_bytes - num_blocks_w_input * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_output - num_blocks_w_input - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (true_output_shape[2] - true_input_shape[2]) / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_Z_diff_blocks = (true_output_shape[1] - true_input_shape[1]) * true_output_shape[2] / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_W_diff_blocks = (true_output_shape[0] - true_input_shape[0]) * true_output_shape[1] * true_output_shape[2] / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t num_leftover_Y = true_input_shape[2] - true_input_shape[2] / TILE_HEIGHT * TILE_HEIGHT;


    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    assert(num_input_tiles > 0);
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, src0_cb_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
	auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        true_input_shape[0],
        padded_W_diff_blocks,
        true_input_shape[1],
        padded_Z_diff_blocks,
        true_input_shape[2],
        padded_Y_diff_blocks,
        num_leftover_Y,
        true_input_shape[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        packed_pad_value,
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size
    };

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) out_is_dram
    };

    // Tilized reader
    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_pad_dims_split_rows.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    // Tilized writer
    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {
        uint32_t(num_tiles / num_tiles_per_block),
        uint32_t(num_tiles_per_block)
    };

    auto tilize_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {dst_buffer->address(),
        (uint32_t) num_tiles, 0}
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void TilizeWithValPadding::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);

    TT_ASSERT(input_tensor_a.shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0]);
    TT_ASSERT(input_tensor_a.shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1]);
    TT_ASSERT(input_tensor_a.shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2]);
    TT_ASSERT(input_tensor_a.shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3]);
    TT_ASSERT((this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 && this->input_tensor_start[3] == 0), "On device padding only supports padding at end of dims");

    uint32_t num_rows = this->output_tensor_shape[2];
    uint32_t inner_dim = this->output_tensor_shape[3];
    TT_ASSERT(num_rows % TILE_HEIGHT == 0, "Output shape must be tilizable");
    TT_ASSERT(inner_dim % TILE_WIDTH == 0, "Output shape must be tilizable");
}
std::vector<Shape> TilizeWithValPadding::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto input_shape = input_tensors.at(0).shape();
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape.rank(); index++) {
        auto front = this->input_tensor_start[index];
        auto back = this->output_tensor_shape[index] - (this->input_tensor_start[index] + input_shape[index]);
        dimensions_pads.push_back(Padding::PadDimension{.front=front, .back=back});
    }
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {Shape(this->output_tensor_shape, padding)};
}
std::vector<Tensor> TilizeWithValPadding::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.dtype(), Layout::TILE, this->output_mem_config);
}

// TODO: If pad is called on a tile and output is not tile, we could untilize then pad, and output is RM
// Currently calling pad on a tile requires the output pad shape to be tile
operation::ProgramWithCallbacks TilizeWithValPadding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return tilize_with_val_padding_single_core(input_tensor_a, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
}

tt::stl::reflection::Attributes TilizeWithValPadding::attributes() const {
    return {
        {"output_tensor_shape", this->output_tensor_shape},
        {"input_tensor_start", this->input_tensor_start},
        {"pad_value", this->pad_value},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor tilize_with_val_padding(const Tensor &input_tensor_a, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value, const MemoryConfig& output_mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    if (input_tensor_a.layout() == Layout::TILE) {
        if (output_tensor_shape == input_tensor_a.shape()) {
            log_warning("Perf warning: tilize with padding called on already tilized tensor of target shape.");
            return input_tensor_a;
        } else {
            TT_ASSERT(false, "Cannot tilize and pad tensor that is already tilized");
        }
    }
    return operation::run_without_autoformat(TilizeWithValPadding{output_tensor_shape, input_tensor_start, pad_value, output_mem_config}, {input_tensor_a}).at(0);

}

Tensor tilize_with_zero_padding(const Tensor &input_tensor_a, const MemoryConfig& output_mem_config) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return input_tensor_a;
    }
    auto shape = input_tensor_a.shape();


    shape[2] = round_up(shape[2], TILE_HEIGHT);
    shape[3] = round_up(shape[3], TILE_WIDTH);
    return tilize_with_val_padding(input_tensor_a, shape, {0, 0, 0, 0}, 0, output_mem_config);
}

}  // namespace tt_metal

}  // namespace tt
