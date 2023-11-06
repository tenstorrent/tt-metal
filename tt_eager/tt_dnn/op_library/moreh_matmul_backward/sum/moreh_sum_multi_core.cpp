// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_sum_multi_core(const Tensor &src, const Tensor &dst) {
    Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(dst.dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    Buffer *src_buffer = src.buffer();
    Buffer *dst_buffer = dst.buffer();

    // src shape
    const auto &src_shape = src.shape();
    uint32_t src_B1 = src_shape[0];
    uint32_t src_B2 = src_shape[1];
    uint32_t Ht = src_shape[2] / TILE_HEIGHT;
    uint32_t Wt = src_shape[3] / TILE_WIDTH;
    uint32_t HtWt = Ht * Wt;
    uint32_t src_B2HtWt = src_B2 * HtWt;

    // dst shape
    const auto &dst_shape = dst.shape();
    uint32_t dst_B1 = dst_shape[0];
    uint32_t dst_B2 = dst_shape[1];

    uint32_t num_read_tiles_per_core = 1;
    if (src_B1 != dst_B1)
        num_read_tiles_per_core *= src_B1;
    if (src_B2 != dst_B2)
        num_read_tiles_per_core *= src_B2;

    uint32_t read_tile_offset = (src_B1 != dst_B1 && src_B2 == dst_B2) ? (src_B2HtWt) : (HtWt);
    bool b1_batched = (src_B1 == dst_B1 && src_B1 != 1 && src_B2 != dst_B2) ? (true) : (false);
    uint32_t num_dst_tiles = dst.volume() / TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    Device *device = dst.device();
    CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord = {.x = core_grid.x_, .y = num_cores_y};

    const uint32_t in0_t = 2;   // src
    const uint32_t in1_t = 1;   // zero
    const uint32_t im0_t = 1;   // temp
    const uint32_t out0_t = 2;  // dst
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, num_dst_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},        // src
            {CB::c_in1, in1_t},        // zero
            {CB::c_intermed0, im0_t},  // temp
            {CB::c_out0, out0_t},      // dst
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(src.memory_config().buffer_type == BufferType::DRAM)};
    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(dst.memory_config().buffer_type == BufferType::DRAM)};

    const auto reader_kernel_file = "tt_eager/tt_dnn/op_library/moreh_matmul_backward/kernels/reader_moreh_sum.cpp";

    const auto writer_kernel_file = "tt_eager/tt_dnn/op_library/moreh_matmul_backward/kernels/writer_moreh_sum.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<string, string> compute_defines;
    const auto compute_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_matmul_backward/kernels/moreh_sum_multi_core.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1}, compute_defines);

    std::optional<KernelID> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src_buffer->address(),
             num_read_tiles_per_core,
             num_tiles_per_core,
             read_tile_offset,
             tile_offset,
             static_cast<uint32_t>(b1_batched),
             HtWt,
             src_B2HtWt});

        SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, tile_offset});

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_1_id, core, {num_read_tiles_per_core, num_tiles_per_core});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_read_tiles_per_core, num_tiles_per_core});
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        Buffer *src_buffer = input_tensors.at(0).buffer();
        Buffer *dst_buffer = input_tensors.at(1).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
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

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace primary
}  // namespace operations
}  // namespace tt
