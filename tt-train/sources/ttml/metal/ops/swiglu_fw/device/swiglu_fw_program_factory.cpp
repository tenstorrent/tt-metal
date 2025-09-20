// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_fw_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <iostream>
#include <tt-metalium/assert.hpp>

#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/writer_swiglu_fw_interleaved_start_id.cpp";

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/reader_swiglu_fw_interleaved_start_id.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/compute/swiglu_fw_kernel.cpp";

// Reader buffer indices
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kW1BufferIdx = 1U;
constexpr uint32_t kW2BufferIdx = 2U;
constexpr uint32_t kW3BufferIdx = 3U;

// Writer buffer indices
constexpr uint32_t kSwigluBufferIdx = 0;

// CBs with input data
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kW1CbIndex = tt::CBIndex::c_1;
constexpr auto kW2CbIndex = tt::CBIndex::c_2;
constexpr auto kW3CbIndex = tt::CBIndex::c_3;
// CBs with masks
constexpr auto kMaskWCbIndex = tt::CBIndex::c_4;
constexpr auto kMaskHWCbIndex = tt::CBIndex::c_5;
// CBs with intermediate computations
constexpr auto kXW1CbIndex = tt::CBIndex::c_6;       // keeps track of (X @ W1) [r, k]
constexpr auto kXW3CbIndex = tt::CBIndex::c_7;       // keeps track of (X @ W3) [r, k]
constexpr auto kMCbIndex = tt::CBIndex::c_8;         // keeps track of M[r, k]
constexpr auto kYPartialCbIndex = tt::CBIndex::c_9;  // keeps track of partial Y[r, c]
// CB with output data
constexpr auto kYCbIndex = tt::CBIndex::c_10;  // keeps track of final Y[r, c]

constexpr uint32_t kNumMaskWTiles = 1U;
constexpr uint32_t kNumMaskHWTiles = 1U;
constexpr uint32_t kNumXW1Tiles = 2U;  // Likely 1U is sufficient, but using 2U for double buffering
constexpr uint32_t kNumXW3Tiles = 2U;

const std::string kMaskWDefineKey = "DO_MASK_W";
const std::string kMaskHWDefineKey = "DO_MASK_HW";

}  // namespace

namespace ttml::metal::ops::swiglu_fw::device {

struct SwiGLUForwardKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

// this possibly could be moved to a common utils file
void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const SwiGLUForwardKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* w1,
    const tt::tt_metal::Buffer* w2,
    const tt::tt_metal::Buffer* w3,
    const tt::tt_metal::Buffer* swiglu_buffer,
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
        // Reader kernel: (input_addr, w1_addr, w2_addr, w3_addr, num_rows, offset)
        SetRuntimeArgs(
            program,
            kernels.reader,
            core,
            {input_buffer->address(),
             w1->address(),
             w2->address(),
             w3->address(),
             num_rows_per_core,
             num_rows_written});

        // Writer kernel: (swiglu_addr, num_rows, offset)
        SetRuntimeArgs(program, kernels.writer, core, {swiglu_buffer->address(), num_rows_per_core, num_rows_written});

        num_rows_written += num_rows_per_core;
    }
}

// TODO: It would be good to setup a memory check, for the big matrices that might not fit into L1

SwiGLUForwardProgramFactory::cached_program_t SwiGLUForwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    const auto& w1 = tensor_args.w1;
    const auto& w2 = tensor_args.w2;
    const auto& w3 = tensor_args.w3;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t bfloat16_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t float32_single_tile_size_bytes = tt::tt_metal::detail::TileSize(tt::DataFormat::Float32);

    auto padded_tensor_shape = input.padded_shape();
    auto padded_tensor_volume = input.physical_volume();
    TT_FATAL(
        padded_tensor_volume % tt::constants::TILE_HW == 0, "Padded input tensor volume must be divisible by TILE_HW");
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // Get the num_inner and hidden_num_inner
    uint32_t num_inner = input.logical_shape()[-1];
    uint32_t hidden_num_inner = w1.logical_shape()[-1];

    // Get the hidden dimension // 32
    uint32_t hidden_Wt = hidden_num_inner / tt::constants::TILE_WIDTH;

    // These parameters are used to determine if we need to mask tiles along input/hidden dimension, i.e. if the
    // operation applied over input/hidden dimension might produce incorrect results due to some random data in the end
    // of the last tile.
    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;
    uint32_t mask_hw = hidden_num_inner % tt::constants::TILE_WIDTH;

    // TODO(maciek): Consider to add masking. Now we assume that the N and C % 32 == 0.
    TT_FATAL(mask_w == 0, "Input inner dimension must be multiple of TILE_WIDTH");
    TT_FATAL(mask_hw == 0, "Hidden inner dimension must be multiple of TILE_WIDTH");

    // Get number of free cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Compile arguments
    // We enforce to use block_size of 4. If num_inner % 4 != 0 or hidden_num_inner % 4 != 0, we will take care of it in
    // the kernels.
    const uint32_t block_size = 4U;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;

    // TODO(maciek): Optimize memory access to minimize DRAM reads and redundant computation.
    // Key optimization ideas:
    // 1. Cache M[r,:] in L1 memory if possible—this avoids recomputing M[r,:] for each matmul with W2.
    // 2. Cache X[r,:] in L1 memory if possible—this avoids repeatedly streaming X[r,:] from DRAM for every M[r,:]
    // computation. Leveraging L1 caching for these tiles could significantly improve performance by reducing DRAM
    // bandwidth pressure. Currently, all data is streamed from DRAM in blocks, and M[r,:] is recomputed each time it is
    // needed.

    auto data_format = input_data_format;  // tt::DataFormat::Float16_b
    auto precise_data_format = tt::DataFormat::Float32;

    // NOTE(maciek): It is likely that there is no way we can benefit from fp32 data type. The tricky things are:
    // - input CBs must be bf16
    // - output CBs must be bf16
    // - matmul_tiles supports only one data format for both input and output CBs (otherwise produces some NaNs/infs)
    // - matmul_tiles and accumulation is done on FPU, which supports only bfloat16
    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_w1 = create_circular_buffer(
        program, all_cores, kW1CbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_w2 = create_circular_buffer(
        program, all_cores, kW2CbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_w3 = create_circular_buffer(
        program, all_cores, kW3CbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_mask_w = create_circular_buffer(
        program, all_cores, kMaskWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskWTiles);
    [[maybe_unused]] auto cb_mask_hw = create_circular_buffer(
        program, all_cores, kMaskHWCbIndex, data_format, bfloat16_single_tile_size_bytes, kNumMaskHWTiles);
    [[maybe_unused]] auto cb_x_w1 = create_circular_buffer(
        program, all_cores, kXW1CbIndex, data_format, bfloat16_single_tile_size_bytes, kNumXW1Tiles);
    [[maybe_unused]] auto cb_x_w3 = create_circular_buffer(
        program, all_cores, kXW3CbIndex, data_format, bfloat16_single_tile_size_bytes, kNumXW3Tiles);
    [[maybe_unused]] auto cb_m = create_circular_buffer(
        program, all_cores, kMCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_y_partial = create_circular_buffer(
        program, all_cores, kYPartialCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_y = create_circular_buffer(
        program, all_cores, kYCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    TT_FATAL(
        input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "Input buffer must be in DRAM. Input buffer of type {}",
        enchantum::to_string(input_buffer->buffer_type()));

    auto* w1_buffer = w1.buffer();
    TT_FATAL(
        w1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "W1 buffer must be in DRAM. w1 buffer of type {}",
        enchantum::to_string(w1_buffer->buffer_type()));

    auto* w2_buffer = w2.buffer();
    TT_FATAL(
        w2_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "W2 buffer must be in DRAM. w2 buffer of type {}",
        enchantum::to_string(w2_buffer->buffer_type()));

    auto* w3_buffer = w3.buffer();
    TT_FATAL(
        w3_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "W3 buffer must be in DRAM. w3 buffer of type {}",
        enchantum::to_string(w3_buffer->buffer_type()));

    auto* swiglu_buffer = output.buffer();
    TT_FATAL(
        swiglu_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "SwiGLU buffer must be in DRAM. SwiGLU buffer of type {}",
        enchantum::to_string(swiglu_buffer->buffer_type()));

    std::map<std::string, std::string> defines;
    if (mask_w != 0) {
        defines[kMaskWDefineKey] = "1";
    }
    if (mask_hw != 0) {
        defines[kMaskHWDefineKey] = "1";
    }

    SwiGLUForwardKernels kernels;
    // Q: consider what kind of compiled time args should be passed
    kernels.reader =
        create_reader_kernel(program, all_cores, {block_size, Wt, hidden_Wt, mask_w, mask_hw}, {}, kReaderKernelPath);

    kernels.writer = create_writer_kernel(program, all_cores, {block_size, Wt}, {}, kWriterKernelPath);

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for swiglu_fw
    // -------------------------------------------------------------------------

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1,  // per_core_block_cnt
        block_size,                 // per_core_block_size
        Wt,                         // num_inner / TILE_W
        hidden_Wt,                  // hidden_num_inner / TILE_W
        mask_w,                     // mask_w
        mask_hw                     // mask_hw
    };

    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2,  // per_core_block_cnt
            block_size,                 // per_core_block_size
            Wt,                         // num_inner / TILE_W
            hidden_Wt,                  // hidden_num_inner / TILE_W
            mask_w,                     // mask_w
            mask_hw                     // mask_hw

        };

        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }
    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
        kernels,
        input_buffer,
        w1_buffer,
        w2_buffer,
        w3_buffer,
        swiglu_buffer,
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
        {/* swiglu_fw_reader_kernel_id  = */ kernels.reader,
         /* swiglu_fw_writer_kernel_id  = */ kernels.writer,
         /* swiglu_fw_kernel_group_1_id = */ kernels.compute_group_1,
         /* swiglu_fw_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void SwiGLUForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto& swiglu_fw_reader_kernel_id = shared_variables.swiglu_fw_reader_kernel_id;
    auto& swiglu_fw_writer_kernel_id = shared_variables.swiglu_fw_writer_kernel_id;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_y = shared_variables.num_cores_y;

    auto* input_buffer = tensor_args.input.buffer();
    auto* w1_buffer = tensor_args.w1.buffer();
    auto* w2_buffer = tensor_args.w2.buffer();
    auto* w3_buffer = tensor_args.w3.buffer();

    auto* swiglu_buffer = output.buffer();

    // Only address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, swiglu_fw_reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, swiglu_fw_writer_kernel_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update input buffers for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
            runtime_args[kW1BufferIdx] = w1_buffer->address();
            runtime_args[kW2BufferIdx] = w2_buffer->address();
            runtime_args[kW3BufferIdx] = w3_buffer->address();
        }

        // Update output buffers for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kSwigluBufferIdx] = swiglu_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::swiglu_fw::device
