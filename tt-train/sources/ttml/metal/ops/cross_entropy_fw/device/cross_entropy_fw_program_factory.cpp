// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_fw_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <tt-metalium/buffer.hpp>

#include "cross_entropy_fw_device_operation_types.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/cross_entropy_fw/device/kernels/dataflow/"
    "reader_cross_entropy_fw_interleaved_start_id.cpp";

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/cross_entropy_fw/device/kernels/dataflow/"
    "writer_cross_entropy_fw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/cross_entropy_fw/device/kernels/compute/cross_entropy_fw_kernel.cpp";

// reader runtime args
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kTargetBufferIdx = 1U;
// writer runtime args
constexpr uint32_t kOutputBufferIdx = 0;

constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kTargetCbIndex = tt::CBIndex::c_1;
constexpr auto kMaskCbIndex = tt::CBIndex::c_2;
constexpr auto kMaxMaskCbIndex = tt::CBIndex::c_3;
constexpr auto kScalerCbIndex = tt::CBIndex::c_4;  // used to reduction
constexpr auto kInputTileCbIndex = tt::CBIndex::c_5;
constexpr auto kTargetLogitsCbIndex = tt::CBIndex::c_6;
constexpr auto kMaxValueBeforeReductionCbIndex = tt::CBIndex::c_7;
constexpr auto kMaxValueAfterReductionCbIndex = tt::CBIndex::c_8;
constexpr auto kExpSumBeforeReductionCbIndex = tt::CBIndex::c_9;
constexpr auto KExpSumAfterReductionCbIndex = tt::CBIndex::c_10;
constexpr auto kOutputCbIndex = tt::CBIndex::c_11;

constexpr uint32_t kNumTargetIndexesTiles = 1U;
constexpr uint32_t kNumMaskTiles = 1U;
constexpr uint32_t kNumMaxValueTiles = 1U;
constexpr uint32_t kNumInputTilesByIndx = 1U;
constexpr uint32_t kMaxValueBeforeReductionTiles = 2U;
constexpr uint32_t kNumTargetLogitsTiles = 2U;
constexpr uint32_t kNumMaxValueAfterReductionTiles = 2U;
constexpr uint32_t kNumExpSumBeforeReductionTiles = 2U;
constexpr uint32_t kNumExpSumAfterReductionTiles = 2U;
constexpr uint32_t kNumScalerTiles = 1U;  // used it to reduction
constexpr uint32_t kNumOutputTiles = 1U;

constexpr uint32_t kPageElementsNumber = 32U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kEverythingFitsInL1DefineKey = "EVERYTHING_FITS_IN_L1";

uint32_t get_block_size(uint32_t num_inner) {
    const uint32_t max_block_size = 4U;  // 4 is the maximum block size for enabled fp32 dest acc
    for (uint32_t block_size = max_block_size; block_size > 1; block_size--) {
        if (num_inner % block_size == 0) {  // if num_inner is divisible by block_size - choose this block_size
            return block_size;
        }
    }
    return 1U;
}

}  // namespace

namespace ttml::metal::ops::cross_entropy_fw::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct CrossEntropyForwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

/**
 *   Create and configure a circular buffer, returning both the configuration and the handle.
 */
tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
            .set_page_size(cb_index, single_tile_size);

    auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
    return cb_handle;
}

/**
 *   Create a reader kernel with the given compile-time arguments.
 */
tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));
}

/**
 *   Create a writer kernel with the given compile-time arguments.
 */
tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program, kernel_path, core_ranges, tt::tt_metal::WriterDataMovementConfig(compile_time_args, defines));
}

/**
 * Create a compute kernel with the given compile-time arguments.
 */
tt::tt_metal::KernelHandle create_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::map<std::string, std::string>& defines,
    const std::string& kernel_path) {
    return tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compile_time_args,
            .defines = defines});
}

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const CrossEntropyForwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* target_buffer,
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

        // Reader kernel: (input_addr, gamma_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {input_buffer->address(), target_buffer->address(), num_rows_per_core, num_rows_written});

        // Writer kernel: (dst_addr, dst_rms_addr number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.writer,
            core,
            {output_buffer->address(), /*temporary*/ num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

CrossEntropyForwardProgramFactory::cached_program_t CrossEntropyForwardProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& target = tensor_args.target;

    auto* device = input.device();

    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.get_dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float32);

    auto padded_tensor_shape = input.get_padded_shape();
    auto padded_tensor_volume = input.volume();

    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");

    //
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;  // <- number of tiles in inner dimension
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // get size of target indexes inner dimension
    uint32_t target_indexes_inner_dim_size = target.get_logical_shape()[-1] * target.element_size();
    // read target indexes by pages(32 indexes in page)
    uint32_t uint32_read_page_size = tt::datum_size(tt::DataFormat::UInt32) * kPageElementsNumber;

    // get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // get the number of inner dimension
    uint32_t num_inner = input.get_logical_shape()[-1];  // (N, 1, C, H)

    // mask_w - this mask used to avoid calculation of extra data(data which will be added to create full tile 32x32)??
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;  // width index of first trash value in tile

    // compile arguments
    uint32_t block_size = get_block_size(Wt);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    uint32_t twice_block_size = 2U * block_size;

    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    const uint64_t mask_scaler_maxmask_memory =
        (kNumMaskTiles + kNumScalerTiles + kNumMaskTiles) * bfloat16_single_tile_size_bytes;
    const uint64_t output_memory = kNumOutputTiles * bfloat16_single_tile_size_bytes;
    const uint64_t max_value_memory =
        (kNumMaxValueAfterReductionTiles + kMaxValueBeforeReductionTiles) * bfloat16_single_tile_size_bytes;
    const uint64_t exp_sum_memory =
        (kNumExpSumBeforeReductionTiles + kNumExpSumAfterReductionTiles) * float32_single_tile_size_bytes;
    const uint64_t input_memory =
        Wt * bfloat16_single_tile_size_bytes + 2U * bfloat16_single_tile_size_bytes + uint32_read_page_size;

    // Total L1 memory required
    const uint64_t required_L1_in_bytes =
        input_memory + mask_scaler_maxmask_memory + max_value_memory + exp_sum_memory + output_memory;
    // Is everything fits in L1
    const bool everything_fits_in_l1 = required_L1_in_bytes <= available_L1_in_bytes;

    const uint32_t num_input_tiles = (everything_fits_in_l1) ? Wt : twice_block_size;

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b
    auto precise_data_format = tt::DataFormat::Float32;
    auto target_indexes_data_format = tt::DataFormat::UInt32;

    auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, num_input_tiles);

    auto cb_target = create_circular_buffer(
        program, all_cores, kTargetCbIndex, target_indexes_data_format, uint32_read_page_size, kNumTargetIndexesTiles);

    auto cb_mask = create_circular_buffer(
        program, all_cores, kMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);

    auto cb_max_mask = create_circular_buffer(
        program, all_cores, kMaxMaskCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskTiles);

    auto cb_scaler = create_circular_buffer(
        program, all_cores, kScalerCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumScalerTiles);

    auto cb_input_tile_by_idx = create_circular_buffer(
        program, all_cores, kInputTileCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumInputTilesByIndx);

    auto cb_target_inputs = create_circular_buffer(
        program, all_cores, kTargetLogitsCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumTargetLogitsTiles);

    auto cb_max_value_before_reduction = create_circular_buffer(
        program,
        all_cores,
        kMaxValueBeforeReductionCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kMaxValueBeforeReductionTiles);

    auto cb_max_value_after_reduction = create_circular_buffer(
        program,
        all_cores,
        kMaxValueAfterReductionCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kNumMaxValueAfterReductionTiles);

    auto cb_exp_sum_before_reduction = create_circular_buffer(
        program,
        all_cores,
        kExpSumBeforeReductionCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kNumExpSumBeforeReductionTiles);

    auto cb_exp_sum_after_refuction = create_circular_buffer(
        program,
        all_cores,
        KExpSumAfterReductionCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kNumExpSumAfterReductionTiles);

    auto cb_output = create_circular_buffer(
        program,
        all_cores,
        kOutputCbIndex,
        data_format,
        bfloat16_single_tile_size_bytes,
        kNumOutputTiles);  // create twice block_size variable it looks like I need 1 tile here

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------

    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        magic_enum::enum_name(input_buffer->buffer_type()));

    auto* target_buffer = target.buffer();
    TT_FATAL(
        target_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Target buffer must be in DRAM. Target buffer of type {}",
        magic_enum::enum_name(target_buffer->buffer_type()));

    auto* output_buffer = output.buffer();
    TT_FATAL(
        output_buffer->buffer_type() == ttnn::BufferType::DRAM,
        "Output buffer must be in DRAM. Output buffer of type {}",
        magic_enum::enum_name(output_buffer->buffer_type()));

    // configure defines
    std::map<std::string, std::string> defines;
    // define whether I need mask or not
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }

    // define whether everything fits in L1 or not
    if (everything_fits_in_l1) {
        defines[kEverythingFitsInL1DefineKey] = "1";
    }

    // setup defines for reduce
    // Compute kernel does not compile without these defines
    // LLK reduction uses define values as default template parameters
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    CrossEntropyForwardKernels kernels;
    kernels.reader = create_reader_kernel(
        program,
        all_cores,
        /* reader_compile_args */
        {block_size, Wt, mask_w, target_indexes_inner_dim_size, Ht, uint32_read_page_size},
        defines,
        kReaderKernelPath);

    kernels.writer = create_writer_kernel(
        program, all_cores, /* writer_compile_args */ {block_size, Wt}, defines, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for cross_entropy_fw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt                          // num_inner / TILE_W
    };

    kernels.compute_group_1 =
        create_compute_kernel(program, core_group_1, compute_group_1_args, defines, kComputeKernelPath);

    // Group 2 (if present) compile-time arguments
    std::vector<uint32_t> compute_group_2_args = {
        num_rows_per_core_group_2,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt                          // num_inner / TILE_W
    };

    kernels.compute_group_2 =
        create_compute_kernel(program, core_group_2, compute_group_2_args, defines, kComputeKernelPath);

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------

    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        target_buffer,
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
        {/* rmsnorm_fw_reader_kernel_id  = */ kernels.reader,
         /* rmsnorm_fw_writer_kernel_id  = */ kernels.writer,
         /* rmsnorm_fw_kernel_group_1_id = */ kernels.compute_group_1,
         /* rmsnorm_fw_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void CrossEntropyForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& cross_entropy_fw_reader_kernel_id = shared_variables.reader_kernel_id;
    auto& cross_entropy_fw_writer_kernel_id = shared_variables.writer_kernel_id;
    auto& cross_entropy_fw_kernel_group_1_id = shared_variables.compute_kernel_group_1_id;
    auto& cross_entropy_fw_kernel_group_2_id = shared_variables.compute_kernel_group_2_id;
    auto& core_group_1 = shared_variables.core_group_1;
    auto& core_group_2 = shared_variables.core_group_2;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* input_buffer = tensor_args.input.buffer();
    auto* target_buffer = tensor_args.target.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, cross_entropy_fw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, cross_entropy_fw_writer_kernel_id);
    auto& group_1_runtime_args = GetRuntimeArgs(program, cross_entropy_fw_kernel_group_1_id);
    // we need to initialize it with something, but if group 2 is  empty it will be used in the loop
    auto& group_2_runtime_args = core_group_2.ranges().empty()
                                     ? group_1_runtime_args
                                     : GetRuntimeArgs(program, cross_entropy_fw_kernel_group_2_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
            runtime_args[kTargetBufferIdx] = target_buffer->address();
        }

        // Update output buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kOutputBufferIdx] = output_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::cross_entropy_fw::device
