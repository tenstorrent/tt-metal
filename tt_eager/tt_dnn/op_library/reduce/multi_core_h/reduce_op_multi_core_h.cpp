// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using u32 = std::uint32_t;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks reduce_multi_core_h(const Tensor &a, Tensor& output, ReduceOpMath reduce_op, ReduceOpDim reduce_dim, float scaler) {

    TT_ASSERT(reduce_dim == ReduceOpDim::H);
    const auto shape = a.shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1]*shape[0];

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    tt_metal::Program program = tt_metal::Program();

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format = DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    uint32_t num_tiles = a.volume()/TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cols = NC * Wt;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_cols);

    string compute_kernel_name = reduce_op_utils::dim_to_kernel_name(reduce_dim, reduce_op);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffers(program, all_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_scaler_config = tt_metal::CircularBufferConfig(num_input_tiles * scaler_single_tile_size, {{CB::c_in2, scaler_cb_data_format}})
		.set_page_size(CB::c_in2, scaler_single_tile_size);
    auto cb_scaler = tt_metal::CreateCircularBuffers(program, all_cores, cb_scaler_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffers(program, all_cores, cb_output_config);

    tt_metal::Buffer *src0_buffer = a.buffer();
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        Ht,
        Wt,
        HtWt,
        *reinterpret_cast<uint32_t*>(&scaler) // scaler
    };

    tt_metal::Buffer *dst_buffer = output.buffer();
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };
    std::map<string, string> reader_defines;
    reader_defines["REDUCE_SCALER"] = "1";
    tt_metal::KernelID reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args, .defines = reader_defines});

    tt_metal::KernelID writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    std::map<string, string> reduce_defines = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    vector<uint32_t> compute_kernel_args_group_1 = {
        Ht, // Ht
        num_cols_per_core_group_1, // Wt
        1, // NC
    };

    auto reduce_compute_kernel_group_1_id = tt_metal::CreateComputeKernel(
        program,
        compute_kernel_name,
        core_group_1,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_1, .defines = reduce_defines}
    );

    if(!core_group_2.ranges().empty()){
        vector<uint32_t> compute_kernel_args_group_2 = {
            Ht, // Ht
            num_cols_per_core_group_2, // Wt
            1, // NC
        };

        auto reduce_compute_kernel_group_2_id = tt_metal::CreateComputeKernel(
            program,
            compute_kernel_name,
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_2, .defines = reduce_defines}
        );
    }

    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_cols_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core,
            {
                a.buffer()->address(),
                num_cols_read / Wt * HtWt + num_cols_read % Wt,
                num_cols_read % Wt,
                num_cols_per_core
            }
        );

        tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core,
            {
                output.buffer()->address(),
                num_cols_per_core, // number of tiles to write
                num_cols_read // output tile start index
            }
        );
        num_cols_read += num_cols_per_core;
    }

    auto override_runtime_args_callback = [
            reader_kernel_id,
            writer_kernel_id,
            num_cores,
            num_cores_y
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);
        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
