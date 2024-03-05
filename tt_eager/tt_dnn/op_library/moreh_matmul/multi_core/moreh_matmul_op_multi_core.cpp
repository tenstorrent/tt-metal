// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace operations {

namespace primary {

std::tuple<bool, bool> get_bcast_batch(const Shape &input0_shape, const Shape &input1_shape) {
    return {(input0_shape[1] < input1_shape[1]), (input0_shape[1] > input1_shape[1])};
}

operation::ProgramWithCallbacks moreh_matmul_multi_core(
    const Tensor &a,
    const Tensor &b,
    const Tensor &output,
    bool transpose_a,
    bool transpose_b,
    uint32_t a_start_tile_id,
    uint32_t b_start_tile_id,
    uint32_t output_start_tile_id) {
    tt_metal::Program program{};
    const auto &ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    tt_metal::Device *device = a.device();
    Shape cshape = output.get_legacy_shape();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t M = (transpose_a) ? (ashape[3]) : (ashape[2]);
    uint32_t N = (transpose_b) ? (bshape[2]) : (bshape[3]);
    uint32_t K = (transpose_a) ? (ashape[2]) : (ashape[3]);
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t B1 = cshape[0];
    uint32_t B2 = cshape[1];
    uint32_t a_B1 = ashape[0];
    uint32_t a_B2 = ashape[1];
    uint32_t b_B1 = bshape[0];
    uint32_t b_B2 = bshape[1];
    uint32_t a_B2MtKt = a_B2 * MtKt;
    uint32_t b_B2KtNt = b_B2 * KtNt;
    uint32_t B2MtNt = B2 * MtNt;

    const auto &a_shape_wo_padding = a.get_legacy_shape().without_padding();
    const auto &b_shape_wo_padding = b.get_legacy_shape().without_padding();

    uint32_t a_pad_h = a_shape_wo_padding[2] % TILE_HEIGHT;
    uint32_t a_pad_w = a_shape_wo_padding[3] % TILE_WIDTH;
    uint32_t a_mask_h = (a_pad_h == 0) ? (TILE_HEIGHT) : (a_pad_h);
    uint32_t a_mask_w = (a_pad_w == 0) ? (TILE_WIDTH) : (a_pad_w);

    uint32_t b_pad_h = b_shape_wo_padding[2] % TILE_HEIGHT;
    uint32_t b_pad_w = b_shape_wo_padding[3] % TILE_WIDTH;
    uint32_t b_mask_h = (b_pad_h == 0) ? (TILE_HEIGHT) : (b_pad_h);
    uint32_t b_mask_w = (b_pad_w == 0) ? (TILE_WIDTH) : (b_pad_w);

    auto [a_bcast_batch, b_bcast_batch] = get_bcast_batch(ashape, bshape);
    log_debug(LogTest, "B1 {} B2 {} Mt {} Nt {} Kt {}", B1, B2, Mt, Nt, Kt);
    log_debug(LogTest, "a_bcast_batch {} b_bcast_batch {}", a_bcast_batch, b_bcast_batch);
    log_debug(LogTest, "transpose_a {} transpose_b {}", transpose_a, transpose_b);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // 1 x B2 x Mt x Nt
    auto num_output_tiles_total = cshape[1] * cshape[2] * cshape[3] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    log_debug("num_cores {}", num_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 2;
    const uint32_t in1_t = 2;
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;
    const uint32_t im2_t = 1;
    const uint32_t out0_t = 2;

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},
            {CB::c_in1, in1_t},
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t},
            {CB::c_intermed2, im2_t},
            {CB::c_out0, out0_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(is_dram(src0_buffer)), static_cast<uint32_t>(is_dram(src1_buffer))};

    const std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)CB::c_out0, (std::uint32_t)is_dram(dst_buffer)};

    const auto reader_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/kernels/reader_moreh_matmul.cpp";
    const auto writer_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/kernels/writer_moreh_matmul.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<string, string> compute_defines;
    const auto compute_kernel_file = "tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/kernels/moreh_matmul.cpp";
    const std::vector<uint32_t> compute_args_group_1 = {
        1,                                  // B
        1,                                  // Mt
        Kt,                                 // Kt
        num_output_tiles_per_core_group_1,  // Nt
        uint32_t(transpose_a),
        uint32_t(transpose_b)};
    const auto compute_kernel_1_id = CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_output_tiles_per_core_group_1, compute_args_group_1}, compute_defines);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2 = {
            1,                                  // B
            1,                                  // Mt
            Kt,                                 // Kt
            num_output_tiles_per_core_group_2,  // Nt
            uint32_t(transpose_a),
            uint32_t(transpose_b)};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_output_tiles_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src0_buffer->address(),
             src1_buffer->address(),
             Mt,
             Kt,
             Nt,
             MtKt,
             KtNt,
             uint32_t(a_bcast_batch),
             uint32_t(b_bcast_batch),
             num_tiles_written,
             num_output_tiles_per_core,
             MtNt,
             uint32_t(transpose_a),
             uint32_t(transpose_b),
             a_start_tile_id,
             b_start_tile_id,
             a_mask_h,
             a_mask_w,
             b_mask_h,
             b_mask_w});
        tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {dst_buffer->address(), num_output_tiles_per_core, num_tiles_written + output_start_tile_id});
        num_tiles_written += num_output_tiles_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id,
                                           writer_kernel_id,
                                           num_cores,
                                           num_cores_y,
                                           a_start_tile_id,
                                           b_start_tile_id,
                                           output_start_tile_id](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer_a->address();
                runtime_args[1] = src_dram_buffer_b->address();
                runtime_args[14] = a_start_tile_id;
                runtime_args[15] = b_start_tile_id;
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
                runtime_args[2] = output_start_tile_id;
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
