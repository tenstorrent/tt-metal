// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks eltwise_binary_multi_core(const Tensor &a, const Tensor &b, Tensor& output, BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations) {

    Program program{};

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    uint32_t num_tiles = a.volume() / TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
		.set_page_size(src1_cb_index, src1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);


    std::map<string, string> eltwise_defines = eltwise_binary_op_utils::get_defines(op_type, fused_activations);

    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN0_0") != eltwise_defines.end()) {
        tt_metal::CircularBufferConfig cb_interm_config = tt_metal::CircularBufferConfig(1 * src0_single_tile_size, {{CB::c_intermed0, src0_cb_data_format}})
		    .set_page_size(CB::c_intermed0, src0_single_tile_size);
        auto cb_interm = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm_config);
    }
    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN1_0") != eltwise_defines.end()) {
        tt_metal::CircularBufferConfig cb_interm2_config = tt_metal::CircularBufferConfig(1 * src1_single_tile_size, {{CB::c_intermed1, src1_cb_data_format}})
		    .set_page_size(CB::c_intermed1, src1_single_tile_size);
        auto cb_interm2 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm2_config);
    }

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;

    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
        .set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) src1_is_dram
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    KernelID binary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {
        1 // per_core_block_size
    };

    auto eltwise_binary_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = eltwise_defines}
    );

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_x * num_cores_y; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            SetRuntimeArgs(program, binary_reader_kernel_id, core, {0, 0, 0, 0});
            SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {0});
            SetRuntimeArgs(program, unary_writer_kernel_id, core, {0, 0, 0});
            continue;
        }
        tt_metal::SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {
                src0_buffer->address(),
                src1_buffer->address(),
                num_tiles_per_core,
                num_tiles_read
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            eltwise_binary_kernel_id,
            core,
            {
                num_tiles_per_core, // per_core_block_cnt
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_tiles_per_core,
                num_tiles_read
            }
        );
        num_tiles_read+=num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
            eltwise_binary_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        uint32_t num_tiles = input_tensors.at(0).volume() / TILE_HW;

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tiles);

        for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_x * num_cores_y; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_tiles_per_core = 0;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                SetRuntimeArgs(program, binary_reader_kernel_id, core, {0, 0, 0, 0});
                SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {0});
                SetRuntimeArgs(program, unary_writer_kernel_id, core, {0, 0, 0});
                continue;
            }
            SetRuntimeArgs(program, binary_reader_kernel_id, core, {src_buffer_a->address(), src_buffer_b->address(), num_tiles_per_core, num_tiles_read});
            SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {num_tiles_per_core});
            SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_read});
            num_tiles_read += num_tiles_per_core;
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
