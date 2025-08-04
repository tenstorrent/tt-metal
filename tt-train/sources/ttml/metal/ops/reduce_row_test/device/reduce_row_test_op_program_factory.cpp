// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_row_test_op_program_factory.hpp"

#include <bit>
#include <cstdint>

#include "metal/ops/common/program_utils.hpp"
#include "reduce_row_test_op_device_operation_types.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/reduce_row_test/device/kernels/dataflow/"
    "reduce_row_test_op_writer_kernel.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/reduce_row_test/device/kernels/dataflow/"
    "reduce_row_test_op_reader_kernel.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/reduce_row_test/device/kernels/compute/reduce_row_test_op_compute_kernel.cpp";

// reader runtime args
constexpr uint32_t kInputBufferIdx = 0;
// writer runtime args
constexpr uint32_t kOutputBufferIdx = 0;

constexpr auto kFirstInputCbIndex = tt::CBIndex::c_0;
constexpr auto kReductionScalerCbIndex = tt::CBIndex::c_1;
constexpr auto kMatMulCbIndex = tt::CBIndex::c_2;
constexpr auto kBeforeReductionCbIndex = tt::CBIndex::c_3;
constexpr auto kOutputCbIndex = tt::CBIndex::c_4;

constexpr uint32_t kNumScalerTiles = 1U;

const std::string kUseMatmul = "USE_MATMUL";
}  // namespace

namespace ttml::metal::ops::reduce_row_test_op::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct ReduceRowTestOperationKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const ReduceRowTestOperationKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* output_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_rows_per_core_group_1,
    uint32_t num_rows_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many rows this core will process
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        // Reader kernel: (input_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(program, kernels.reader, core, {input_buffer->address(), num_rows_per_core, num_rows_written});

        // Writer kernel: (dst_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(program, kernels.writer, core, {output_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

ReduceRowTestProgramFactory::cached_program_t ReduceRowTestProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;

    auto* device = input.device();

    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();
    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    auto data_format = input_data_format;

    auto cb_input = create_circular_buffer(
        program, all_cores, kFirstInputCbIndex, data_format, bfloat16_single_tile_size_bytes, Wt);

    auto cb_reduction_scaler = create_circular_buffer(
        program, all_cores, kReductionScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    auto cb_matmul_reduce = create_circular_buffer(
        program, all_cores, kMatMulCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    auto cb_before_reduction = create_circular_buffer(
        program, all_cores, kBeforeReductionCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    auto cb_output =
        create_circular_buffer(program, all_cores, kOutputCbIndex, data_format, bfloat16_single_tile_size_bytes, Wt);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        magic_enum::enum_name(input_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        magic_enum::enum_name(output_buffer->buffer_type()));

    // configure defines
    std::map<std::string, std::string> defines;

    // setup defines for reduce
    // Compute kernel does not compile without these defines
    // LLK reduction uses define values as default template parameters
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    if (args.use_matmul) {
        defines[kUseMatmul] = "1";
    }

    ReduceRowTestOperationKernels kernels;
    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */ {Wt},
        defines,
        kReaderKernelPath);
    kernels.writer =
        create_writer_kernel(program, all_cores, /* writer_compile_args */ {Wt}, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for reduce_row_test_op
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        Wt                          // num_inner / TILE_W
    };

    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            Wt                          // num_inner / TILE_W
        };

        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, true);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        output_buffer,
        num_cores,
        num_cores_y,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        core_group_1,
        core_group_2);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {/* reduce_row_test_op_reader_kernel_id  = */ kernels.reader,
         /* reduce_row_test_op_writer_kernel_id  = */ kernels.writer,
         /* reduce_row_test_op_kernel_group_1_id = */ kernels.compute_group_1,
         /* reduce_row_test_op_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void ReduceRowTestProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& reduce_row_test_op_reader_kernel = shared_vars.reduce_row_test_op_reader_kernel_id;
    auto& reduce_row_test_op_writer_kernel = shared_vars.reduce_row_test_op_writer_kernel_id;
    auto& reduce_row_test_op_group_1_kernel = shared_vars.reduce_row_test_op_kernel_group_1_id;
    auto& reduce_row_test_op_group_2_kernel = shared_vars.reduce_row_test_op_kernel_group_2_id;
    auto& core_group_1 = shared_vars.core_group_1;
    auto& core_group_2 = shared_vars.core_group_2;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto& input = tensor_args.input;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, reduce_row_test_op_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, reduce_row_test_op_writer_kernel);
    auto& group_1_runtime_args = GetRuntimeArgs(program, reduce_row_test_op_group_1_kernel);
    // we need to initialize it with something, but if group 2 is  empty it will be used in the loop
    auto& group_2_runtime_args = core_group_2.ranges().empty()
                                     ? group_1_runtime_args
                                     : GetRuntimeArgs(program, reduce_row_test_op_group_2_kernel);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
        }
        // Update destination buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::reduce_row_test_op::device
