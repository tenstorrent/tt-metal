// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TRUE FLASH SwiGLU PROGRAM FACTORY
//
// This factory creates programs using the "True Flash" optimization that
// avoids materializing the full M row in L1. Instead, M tiles are computed
// on-demand for each k_block.
//
// Memory savings vs original:
//   - XW1_partial: block_size tiles (vs full hidden_Wt)
//   - XW3_partial: block_size tiles (vs full hidden_Wt)
//   - M_partial: block_size tiles (vs full hidden_Wt)
//   - Y_partial: full Wt tiles (need to accumulate all output columns)
//
// For NanoLlama3: 560 KB → 280 KB (50% reduction)
// ============================================================================

#include "swiglu_fw_true_flash_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/writer_swiglu_fw_true_flash.cpp";

constexpr auto kReaderSenderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/reader_swiglu_fw_true_flash_sender.cpp";

constexpr auto kReaderReceiverKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/dataflow/reader_swiglu_fw_true_flash_receiver.cpp";

constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_fw/device/kernels/compute/swiglu_fw_true_flash_kernel.cpp";

// Buffer indices
constexpr uint32_t kInputBufferIdx = 0;
constexpr uint32_t kW1BufferIdx = 1U;
constexpr uint32_t kW2BufferIdx = 2U;
constexpr uint32_t kW3BufferIdx = 3U;
constexpr uint32_t kSwigluBufferIdx = 0;

// CB indices
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kW1CbIndex = tt::CBIndex::c_1;
constexpr auto kW2CbIndex = tt::CBIndex::c_2;
constexpr auto kW3CbIndex = tt::CBIndex::c_3;
constexpr auto kXW1PartialCbIndex = tt::CBIndex::c_4;
constexpr auto kXW3PartialCbIndex = tt::CBIndex::c_5;
constexpr auto kXW1CbIndex = tt::CBIndex::c_6;
constexpr auto kXW3CbIndex = tt::CBIndex::c_7;
constexpr auto kMCbIndex = tt::CBIndex::c_8;
constexpr auto kYPartialCbIndex = tt::CBIndex::c_9;
constexpr auto kYCbIndex = tt::CBIndex::c_10;

}  // namespace

namespace ttml::metal::ops::swiglu_fw::device {

struct TrueFlashKernels {
    tt::tt_metal::KernelHandle reader_sender;
    tt::tt_metal::KernelHandle reader_receiver;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

void assign_true_flash_runtime_args(
    tt::tt_metal::Program& program,
    const TrueFlashKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    const tt::tt_metal::Buffer* w1,
    const tt::tt_metal::Buffer* w2,
    const tt::tt_metal::Buffer* w3,
    const tt::tt_metal::Buffer* swiglu_buffer,
    const uint32_t num_cores,
    const uint32_t num_cores_x,
    const uint32_t num_cores_y,
    const uint32_t num_rows_per_core_group_1,
    const uint32_t num_rows_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    ttnn::IDevice* device,
    const uint32_t mcast_sender_semaphore_id,
    const uint32_t mcast_receiver_semaphore_id,
    const bool use_multicast,
    const uint32_t max_rows_across_all_cores) {
    // Compute multicast bounding box
    uint32_t mcast_start_physical_x = 0;
    uint32_t mcast_start_physical_y = 0;
    uint32_t mcast_end_physical_x = 0;
    uint32_t mcast_end_physical_y = 0;
    uint32_t num_receivers_excluding_sender = 0;

    if (use_multicast && num_cores > 1) {
        uint32_t max_x = 0;
        uint32_t max_y = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            tt::tt_metal::CoreCoord core = {i % num_cores_x, i / num_cores_x};
            max_x = std::max(max_x, static_cast<uint32_t>(core.x));
            max_y = std::max(max_y, static_cast<uint32_t>(core.y));
        }

        tt::tt_metal::CoreCoord mcast_start_core = {0, 0};
        tt::tt_metal::CoreCoord mcast_end_core = {max_x, max_y};
        auto mcast_start_physical = device->worker_core_from_logical_core(mcast_start_core);
        auto mcast_end_physical = device->worker_core_from_logical_core(mcast_end_core);

        mcast_start_physical_x = mcast_start_physical.x;
        mcast_start_physical_y = mcast_start_physical.y;
        mcast_end_physical_x = mcast_end_physical.x;
        mcast_end_physical_y = mcast_end_physical.y;
        num_receivers_excluding_sender = num_cores - 1;
    }

    auto sender_physical = device->worker_core_from_logical_core({0, 0});

    for (uint32_t i = 0, num_rows_written = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i % num_cores_x, i / num_cores_x};

        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        uint32_t max_rows_for_sync = use_multicast ? max_rows_across_all_cores : num_rows_per_core;
        bool is_sender = (core.x == 0 && core.y == 0);

        if (is_sender) {
            SetRuntimeArgs(
                program,
                kernels.reader_sender,
                core,
                {input_buffer->address(),
                 w1->address(),
                 w2->address(),
                 w3->address(),
                 num_rows_per_core,
                 max_rows_for_sync,
                 num_rows_written,
                 mcast_start_physical_x,
                 mcast_start_physical_y,
                 mcast_end_physical_x,
                 mcast_end_physical_y,
                 num_receivers_excluding_sender,
                 mcast_sender_semaphore_id,
                 mcast_receiver_semaphore_id});
        } else if (use_multicast) {
            SetRuntimeArgs(
                program,
                kernels.reader_receiver,
                core,
                {input_buffer->address(),
                 num_rows_per_core,
                 max_rows_for_sync,
                 num_rows_written,
                 static_cast<uint32_t>(sender_physical.x),
                 static_cast<uint32_t>(sender_physical.y),
                 mcast_sender_semaphore_id,
                 mcast_receiver_semaphore_id});
        } else {
            SetRuntimeArgs(
                program,
                kernels.reader_sender,
                core,
                {input_buffer->address(),
                 w1->address(),
                 w2->address(),
                 w3->address(),
                 num_rows_per_core,
                 max_rows_for_sync,
                 num_rows_written,
                 0,
                 0,
                 0,
                 0,  // unused mcast coords
                 0,  // no receivers
                 mcast_sender_semaphore_id,
                 mcast_receiver_semaphore_id});
        }

        SetRuntimeArgs(program, kernels.writer, core, {swiglu_buffer->address(), num_rows_per_core, num_rows_written});
        num_rows_written += num_rows_per_core;
    }
}

bool true_flash_fits_in_l1(
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t bfloat16_single_tile_size_bytes,
    ttnn::IDevice* device) {
    const uint32_t twice_block_size = 2U * block_size;
    const uint32_t available_L1_in_bytes =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // True Flash CB sizes:
    // Input: 2 × block_size (double-buffered)
    // W1/W2/W3: 2 × block_size² each (double-buffered batched)
    // XW1_partial, XW3_partial: block_size each (only k_block tiles!)
    // XW1, XW3: block_size each (only k_block tiles!)
    // M: block_size (only k_block tiles!)
    // Y_partial: Wt tiles (full output row for accumulation)
    // Y: Wt tiles (full output row)

    const uint64_t input_memory = twice_block_size * bfloat16_single_tile_size_bytes;
    const uint64_t w1_memory = (2U * block_size * block_size) * bfloat16_single_tile_size_bytes;
    const uint64_t w2_memory = (2U * block_size * block_size) * bfloat16_single_tile_size_bytes;
    const uint64_t w3_memory = (2U * block_size * block_size) * bfloat16_single_tile_size_bytes;
    const uint64_t xw1_partial_memory = block_size * bfloat16_single_tile_size_bytes;
    const uint64_t xw3_partial_memory = block_size * bfloat16_single_tile_size_bytes;
    const uint64_t xw1_memory = block_size * bfloat16_single_tile_size_bytes;
    const uint64_t xw3_memory = block_size * bfloat16_single_tile_size_bytes;
    const uint64_t m_memory = block_size * bfloat16_single_tile_size_bytes;
    const uint64_t y_partial_memory = Wt * bfloat16_single_tile_size_bytes;  // Full row!
    const uint64_t y_memory = Wt * bfloat16_single_tile_size_bytes;          // Full row!

    const uint64_t total_memory = input_memory + w1_memory + w2_memory + w3_memory + xw1_partial_memory +
                                  xw3_partial_memory + xw1_memory + xw3_memory + m_memory + y_partial_memory + y_memory;

    return total_memory <= available_L1_in_bytes;
}

SwiGLUTrueFlashProgramFactory::cached_program_t SwiGLUTrueFlashProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& w1 = tensor_args.w1;
    const auto& w2 = tensor_args.w2;
    const auto& w3 = tensor_args.w3;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    auto padded_tensor_shape = input.padded_shape();
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    uint32_t num_inner = input.logical_shape()[-1];
    uint32_t hidden_num_inner = w1.logical_shape()[-1];
    uint32_t hidden_Wt = hidden_num_inner / tt::constants::TILE_WIDTH;

    // Validate weight shapes
    const auto& w1_shape = w1.logical_shape();
    const auto& w2_shape = w2.logical_shape();
    const auto& w3_shape = w3.logical_shape();

    TT_FATAL(w1_shape[-2] == num_inner, "W1 shape mismatch");
    TT_FATAL(w3_shape[-2] == num_inner && w3_shape[-1] == hidden_num_inner, "W3 shape mismatch");
    TT_FATAL(w2_shape[-2] == hidden_num_inner && w2_shape[-1] == num_inner, "W2 shape mismatch");

    uint32_t mask_w = num_inner % tt::constants::TILE_WIDTH;
    uint32_t mask_hw = hidden_num_inner % tt::constants::TILE_WIDTH;
    TT_FATAL(mask_w == 0, "Input inner dimension must be multiple of TILE_WIDTH");
    TT_FATAL(mask_hw == 0, "Hidden inner dimension must be multiple of TILE_WIDTH");

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const uint32_t block_size = 4U;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows_to_process, /*row_wise=*/true);

    // Validate True Flash fits in L1
    TT_FATAL(
        true_flash_fits_in_l1(Wt, block_size, bfloat16_single_tile_size_bytes, device),
        "True Flash SwiGLU: Y row too large for L1! Wt={} tiles. "
        "Consider using Tensor Parallelism (TP) to reduce embed_dim per device.",
        Wt);

    bool use_multicast = (num_cores > 1);
    uint32_t max_rows_across_all_cores = num_rows_per_core_group_1;

    // -------------------------------------------------------------------------
    // Create circular buffers - TRUE FLASH sizing
    // -------------------------------------------------------------------------
    const uint32_t twice_block_size = 2U * block_size;
    const uint32_t w_cb_tiles = 2U * block_size * block_size;  // Batched mcast

    // TRUE FLASH: Intermediate CBs are only block_size tiles (not full hidden_Wt!)
    const uint32_t intermediate_tiles = block_size;
    // TRUE FLASH: Y_partial and Y need full Wt tiles for accumulation
    const uint32_t y_tiles = Wt;

    auto data_format = input_data_format;

    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, all_cores, kInputCbIndex, data_format, bfloat16_single_tile_size_bytes, twice_block_size);
    [[maybe_unused]] auto cb_w1 = create_circular_buffer(
        program, all_cores, kW1CbIndex, data_format, bfloat16_single_tile_size_bytes, w_cb_tiles);
    [[maybe_unused]] auto cb_w2 = create_circular_buffer(
        program, all_cores, kW2CbIndex, data_format, bfloat16_single_tile_size_bytes, w_cb_tiles);
    [[maybe_unused]] auto cb_w3 = create_circular_buffer(
        program, all_cores, kW3CbIndex, data_format, bfloat16_single_tile_size_bytes, w_cb_tiles);
    // TRUE FLASH: Small intermediate CBs
    [[maybe_unused]] auto cb_xw1_partial = create_circular_buffer(
        program, all_cores, kXW1PartialCbIndex, data_format, bfloat16_single_tile_size_bytes, intermediate_tiles);
    [[maybe_unused]] auto cb_xw3_partial = create_circular_buffer(
        program, all_cores, kXW3PartialCbIndex, data_format, bfloat16_single_tile_size_bytes, intermediate_tiles);
    [[maybe_unused]] auto cb_xw1 = create_circular_buffer(
        program, all_cores, kXW1CbIndex, data_format, bfloat16_single_tile_size_bytes, intermediate_tiles);
    [[maybe_unused]] auto cb_xw3 = create_circular_buffer(
        program, all_cores, kXW3CbIndex, data_format, bfloat16_single_tile_size_bytes, intermediate_tiles);
    [[maybe_unused]] auto cb_m = create_circular_buffer(
        program, all_cores, kMCbIndex, data_format, bfloat16_single_tile_size_bytes, intermediate_tiles);
    // TRUE FLASH: Full Y row CBs
    [[maybe_unused]] auto cb_y_partial = create_circular_buffer(
        program, all_cores, kYPartialCbIndex, data_format, bfloat16_single_tile_size_bytes, y_tiles);
    [[maybe_unused]] auto cb_y =
        create_circular_buffer(program, all_cores, kYCbIndex, data_format, bfloat16_single_tile_size_bytes, y_tiles);

    // -------------------------------------------------------------------------
    // Create kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    auto* w1_buffer = w1.buffer();
    auto* w2_buffer = w2.buffer();
    auto* w3_buffer = w3.buffer();
    auto* swiglu_buffer = output.buffer();

    std::map<std::string, std::string> defines;
    defines["TRUE_FLASH"] = "1";

    // Split cores into sender and receivers
    std::vector<tt::tt_metal::CoreRange> sender_ranges;
    std::vector<tt::tt_metal::CoreRange> receiver_ranges;

    for (const auto& core_range : all_cores.ranges()) {
        for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
            for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                tt::tt_metal::CoreCoord core = {x, y};
                if (use_multicast) {
                    if (core.x == 0 && core.y == 0) {
                        sender_ranges.push_back(tt::tt_metal::CoreRange(core, core));
                    } else {
                        receiver_ranges.push_back(tt::tt_metal::CoreRange(core, core));
                    }
                } else {
                    sender_ranges.push_back(tt::tt_metal::CoreRange(core, core));
                }
            }
        }
    }

    tt::tt_metal::CoreRangeSet sender_core_set(sender_ranges);
    tt::tt_metal::CoreRangeSet receiver_core_set;
    if (!receiver_ranges.empty()) {
        receiver_core_set = tt::tt_metal::CoreRangeSet(receiver_ranges);
    }

    uint32_t mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    uint32_t mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    TrueFlashKernels kernels;

    // Sender kernel
    std::vector<uint32_t> sender_compile_time_args{block_size, Wt, hidden_Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w1_buffer).append_to(sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w2_buffer).append_to(sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(w3_buffer).append_to(sender_compile_time_args);
    kernels.reader_sender =
        create_reader_kernel(program, sender_core_set, sender_compile_time_args, defines, kReaderSenderKernelPath);

    // Receiver kernel
    if (use_multicast && !receiver_ranges.empty()) {
        std::vector<uint32_t> receiver_compile_time_args{block_size, Wt, hidden_Wt};
        tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(receiver_compile_time_args);
        kernels.reader_receiver = create_reader_kernel(
            program, receiver_core_set, receiver_compile_time_args, defines, kReaderReceiverKernelPath);
    }

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(swiglu_buffer).append_to(writer_compile_time_args);
    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_time_args, defines, kWriterKernelPath);

    // Compute kernels
    std::vector<uint32_t> compute_group_1_args = {
        num_rows_per_core_group_1, max_rows_across_all_cores, block_size, Wt, hidden_Wt};
    kernels.compute_group_1 = create_compute_kernel(
        program, core_group_1, compute_group_1_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_rows_per_core_group_2, max_rows_across_all_cores, block_size, Wt, hidden_Wt};
        kernels.compute_group_2 = create_compute_kernel(
            program, core_group_2, compute_group_2_args, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
    }

    // Assign runtime args
    assign_true_flash_runtime_args(
        program,
        kernels,
        input_buffer,
        w1_buffer,
        w2_buffer,
        w3_buffer,
        swiglu_buffer,
        num_cores,
        num_cores_x,
        num_cores_y,
        num_rows_per_core_group_1,
        num_rows_per_core_group_2,
        core_group_1,
        core_group_2,
        device,
        mcast_sender_semaphore_id,
        mcast_receiver_semaphore_id,
        use_multicast,
        max_rows_across_all_cores);

    return cached_program_t{
        std::move(program),
        {kernels.reader_sender,
         kernels.reader_receiver,
         kernels.writer,
         kernels.compute_group_1,
         kernels.compute_group_2,
         core_group_1,
         core_group_2,
         num_cores,
         num_cores_x,
         num_cores_y,
         use_multicast}};
}

void SwiGLUTrueFlashProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    uint32_t num_cores = shared_variables.num_cores;
    uint32_t num_cores_x = shared_variables.num_cores_x;
    bool use_multicast = shared_variables.use_multicast;

    auto* input_buffer = tensor_args.input.buffer();
    auto* w1_buffer = tensor_args.w1.buffer();
    auto* w2_buffer = tensor_args.w2.buffer();
    auto* w3_buffer = tensor_args.w3.buffer();
    auto* swiglu_buffer = output.buffer();

    auto& sender_runtime_args = GetRuntimeArgs(program, shared_variables.reader_sender_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, shared_variables.writer_kernel_id);

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i % num_cores_x, i / num_cores_x};
        bool is_sender = (core.x == 0 && core.y == 0);

        if (use_multicast && !is_sender) {
            auto& receiver_runtime_args = GetRuntimeArgs(program, shared_variables.reader_receiver_kernel_id);
            auto& runtime_args = receiver_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
        } else {
            auto& runtime_args = sender_runtime_args[core.x][core.y];
            runtime_args[kInputBufferIdx] = input_buffer->address();
            runtime_args[kW1BufferIdx] = w1_buffer->address();
            runtime_args[kW2BufferIdx] = w2_buffer->address();
            runtime_args[kW3BufferIdx] = w3_buffer->address();
        }

        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kSwigluBufferIdx] = swiglu_buffer->address();
        }
    }
}

}  // namespace ttml::metal::ops::swiglu_fw::device
