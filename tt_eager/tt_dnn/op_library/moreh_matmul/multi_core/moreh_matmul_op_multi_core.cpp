// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace operations {

namespace primary {

std::tuple<bool, bool> get_bcast_batch(const Shape &input0_shape, const Shape &input1_shape) {
    bool in0_bcast_batch = false;
    bool in1_bcast_batch = false;

    // TODO: revise check code
    if (input0_shape[1] > input1_shape[1]) {
        in1_bcast_batch = true;
    } else if (input0_shape[1] < input1_shape[1]) {
        in0_bcast_batch = true;
    } else {
        in0_bcast_batch = false;
        in1_bcast_batch = false;
    }
    return {in0_bcast_batch, in1_bcast_batch};
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
    TT_ASSERT(!(transpose_a == true && transpose_b == true));

    tt_metal::Program program{};
    const auto &ashape = a.shape(), bshape = b.shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    Shape cshape = output.shape();  // C=A*B, N1MK*11KN->N1MN

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
            split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    log_debug("num_cores {}", num_cores);

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = A*B
    // MN = MK*KN
    uint32_t M = (transpose_a) ? (ashape[3]) : (ashape[2]);
    uint32_t N = (transpose_b) ? (bshape[2]) : (bshape[3]);
    uint32_t K = (transpose_a) ? (ashape[2]) : (ashape[3]);
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t B1 = cshape[0];  // B1
    uint32_t B2 = cshape[1];  // B2
    uint32_t a_B1 = ashape[0];
    uint32_t a_B2 = ashape[1];
    uint32_t b_B1 = bshape[0];
    uint32_t b_B2 = bshape[1];
    uint32_t a_B2MtKt = a_B2 * MtKt;
    uint32_t b_B2KtNt = b_B2 * KtNt;
    uint32_t B2MtNt = B2 * MtNt;

    const auto &a_shape_wo_padding = a.shape().without_padding();
    const auto &b_shape_wo_padding = b.shape().without_padding();

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

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t interm0_cb_index = 24;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(in1_single_tile_size, {{interm0_cb_index, in1_data_format}})
            .set_page_size(interm0_cb_index, in1_single_tile_size);
    auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_cores, interm0_cb_config);

    uint32_t interm1_cb_index = 25;
    tt_metal::CircularBufferConfig interm1_cb_config =
        tt_metal::CircularBufferConfig(in1_single_tile_size, {{interm1_cb_index, in1_data_format}})
            .set_page_size(interm1_cb_index, in1_single_tile_size);
    auto cb_interm1 = tt_metal::CreateCircularBuffer(program, all_cores, interm1_cb_config);

    uint32_t interm2_cb_index = 26;
    tt_metal::CircularBufferConfig interm2_cb_config =
        tt_metal::CircularBufferConfig(in1_single_tile_size, {{interm2_cb_index, in1_data_format}})
            .set_page_size(interm2_cb_index, in1_single_tile_size);
    auto cb_interm2 = tt_metal::CreateCircularBuffer(program, all_cores, interm2_cb_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    auto reader_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/kernels/reader_matmul_8bank_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/kernels/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args_group_1 = {
        1,                                  // B
        1,                                  // Mt
        Kt,                                 // Kt
        num_output_tiles_per_core_group_1,  // Nt
        uint32_t(transpose_a),
        uint32_t(transpose_b)};  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1
                                 // large loop, so only set Nt for simplicity

    auto eltwise_binary_kernel_group_1_id = tt_metal::CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/kernels/matmul.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_1});

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_args_group_2 = {
            1,                                  // B
            1,                                  // Mt
            Kt,                                 // Kt
            num_output_tiles_per_core_group_2,  // Nt
            uint32_t(transpose_a),
            uint32_t(transpose_b)};  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1
                                     // large loop, so only set Nt for simplicity

        auto eltwise_binary_kernel_group_2_id = tt_metal::CreateComputeKernel(
            program,
            "tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/kernels/matmul.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_2});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {src0_addr,
             src1_addr,
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
            program, writer_id, core, {dst_addr, num_output_tiles_per_core, num_tiles_written + output_start_tile_id});
        num_tiles_written += num_output_tiles_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_id = reader_id,
                                           writer_kernel_id = writer_id,
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
