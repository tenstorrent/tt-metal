// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_gate_up_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kIn0SenderKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_gate_up/device/kernels/dataflow/reader_in0_sender.cpp";
constexpr auto kIn0ReceiverKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_gate_up/device/kernels/dataflow/reader_in0_receiver.cpp";
constexpr auto kIn1SenderWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_gate_up/device/kernels/dataflow/reader_in1_sender_writer.cpp";
constexpr auto kIn1ReceiverWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_gate_up/device/kernels/dataflow/reader_in1_receiver_writer.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/swiglu_gate_up/device/kernels/compute/swiglu_gate_up_compute.cpp";

constexpr auto kCbIn0Index = tt::CBIndex::c_0;     // X tiles (double-buffered)
constexpr auto kCbW1Index = tt::CBIndex::c_1;      // W1 batch (double-buffered)
constexpr auto kCbW3Index = tt::CBIndex::c_2;      // W3 batch (double-buffered)
constexpr auto kCbXW1AccIndex = tt::CBIndex::c_3;  // XW1 accumulation (L1 acc target)
constexpr auto kCbXW3AccIndex = tt::CBIndex::c_4;  // XW3 accumulation (L1 acc target)
constexpr auto kCbMOutIndex = tt::CBIndex::c_5;    // M output tiles

constexpr uint32_t kBlockSize = 4U;
constexpr uint32_t kTilesPerBatch = kBlockSize * kBlockSize;

// Runtime arg indices for IN0 sender
constexpr uint32_t kIn0SenderXAddrIdx = 0U;

// Runtime arg indices for IN1 sender+writer
constexpr uint32_t kIn1SenderW1AddrIdx = 0U;
constexpr uint32_t kIn1SenderW3AddrIdx = 1U;
constexpr uint32_t kIn1SenderMAddrIdx = 2U;

// Runtime arg indices for IN1 receiver+writer
constexpr uint32_t kIn1ReceiverMAddrIdx = 0U;

uint32_t round_up(uint32_t val, uint32_t multiple) {
    return ((val + multiple - 1U) / multiple) * multiple;
}

}  // namespace

namespace ttml::metal::ops::swiglu_gate_up::device {

SwiGLUGateUpProgramFactory::cached_program_t SwiGLUGateUpProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& w1 = tensor_args.w1;
    const auto& w3 = tensor_args.w3;

    auto* device = input.device();
    tt::tt_metal::Program program{};

    tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(data_format);

    // -------------------------------------------------------------------------
    // 1) Compute dimensions
    // -------------------------------------------------------------------------
    auto padded_shape = input.padded_shape();
    TT_FATAL(padded_shape.rank() == 4U, "Input tensor must be 4D");
    uint32_t Wt = padded_shape[-1] / tt::constants::TILE_WIDTH;  // K dimension in tiles
    uint32_t Ht = padded_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_shape[0] * padded_shape[1];
    uint32_t total_M_tiles = NC * Ht;

    uint32_t hidden_dim = w1.logical_shape()[-1];
    uint32_t hidden_Wt = hidden_dim / tt::constants::TILE_WIDTH;  // N dimension in tiles

    // -------------------------------------------------------------------------
    // 2) Determine 2D core grid: R rows × C columns
    // -------------------------------------------------------------------------
    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_r = std::min(static_cast<uint32_t>(grid_size.y), total_M_tiles);
    uint32_t num_cores_c = std::min(static_cast<uint32_t>(grid_size.x), hidden_Wt);

    uint32_t per_core_M = (total_M_tiles + num_cores_r - 1U) / num_cores_r;
    uint32_t per_core_N = (hidden_Wt + num_cores_c - 1U) / num_cores_c;
    uint32_t per_core_N_rounded = round_up(per_core_N, kBlockSize);
    uint32_t num_n_blocks = per_core_N_rounded / kBlockSize;

    // -------------------------------------------------------------------------
    // 3) Build core ranges
    // -------------------------------------------------------------------------
    auto all_cores = tt::tt_metal::CoreRange({0, 0}, {num_cores_c - 1U, num_cores_r - 1U});
    tt::tt_metal::CoreRangeSet all_cores_set({all_cores});

    // Left column: IN0 senders (x=0)
    auto left_column = tt::tt_metal::CoreRange({0, 0}, {0, num_cores_r - 1U});
    tt::tt_metal::CoreRangeSet left_column_set({left_column});

    // Non-left columns: IN0 receivers (x>0)
    tt::tt_metal::CoreRangeSet non_left_column_set;
    if (num_cores_c > 1U) {
        auto non_left = tt::tt_metal::CoreRange({1, 0}, {num_cores_c - 1U, num_cores_r - 1U});
        non_left_column_set = tt::tt_metal::CoreRangeSet({non_left});
    }

    // Top row: IN1 senders (y=0)
    auto top_row = tt::tt_metal::CoreRange({0, 0}, {num_cores_c - 1U, 0});
    tt::tt_metal::CoreRangeSet top_row_set({top_row});

    // Non-top rows: IN1 receivers (y>0)
    tt::tt_metal::CoreRangeSet non_top_row_set;
    if (num_cores_r > 1U) {
        auto non_top = tt::tt_metal::CoreRange({0, 1}, {num_cores_c - 1U, num_cores_r - 1U});
        non_top_row_set = tt::tt_metal::CoreRangeSet({non_top});
    }

    // -------------------------------------------------------------------------
    // 4) Allocate circular buffers
    // -------------------------------------------------------------------------
    uint32_t x_cb_tiles = 2U * kBlockSize;      // double-buffered
    uint32_t w_cb_tiles = 2U * kTilesPerBatch;  // double-buffered
    uint32_t acc_cb_tiles = per_core_N_rounded;
    uint32_t m_out_cb_tiles = per_core_N_rounded;

    [[maybe_unused]] auto cb_in0 =
        create_circular_buffer(program, all_cores_set, kCbIn0Index, data_format, single_tile_size, x_cb_tiles);
    [[maybe_unused]] auto cb_w1 =
        create_circular_buffer(program, all_cores_set, kCbW1Index, data_format, single_tile_size, w_cb_tiles);
    [[maybe_unused]] auto cb_w3 =
        create_circular_buffer(program, all_cores_set, kCbW3Index, data_format, single_tile_size, w_cb_tiles);
    [[maybe_unused]] auto cb_xw1_acc =
        create_circular_buffer(program, all_cores_set, kCbXW1AccIndex, data_format, single_tile_size, acc_cb_tiles);
    [[maybe_unused]] auto cb_xw3_acc =
        create_circular_buffer(program, all_cores_set, kCbXW3AccIndex, data_format, single_tile_size, acc_cb_tiles);
    [[maybe_unused]] auto cb_m_out =
        create_circular_buffer(program, all_cores_set, kCbMOutIndex, data_format, single_tile_size, m_out_cb_tiles);

    // -------------------------------------------------------------------------
    // 5) Create semaphores (4 for 2D multicast)
    // -------------------------------------------------------------------------
    auto in0_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_set, 0U);
    auto in0_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_set, 0U);
    auto in1_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_set, 0U);
    auto in1_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_set, 0U);

    // -------------------------------------------------------------------------
    // 6) NOC selection (match matmul 2D mcast pattern)
    // -------------------------------------------------------------------------
    tt::tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    // -------------------------------------------------------------------------
    // 7) Get buffer pointers
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    auto* w1_buffer = w1.buffer();
    auto* w3_buffer = w3.buffer();
    auto* m_buffer = output.buffer();

    // -------------------------------------------------------------------------
    // 8) Create dataflow kernels
    // -------------------------------------------------------------------------

    // --- RISCV_1: IN0 sender (left column) ---
    std::vector<uint32_t> in0_sender_ct_args = {
        kBlockSize,
        Wt,
        per_core_M,
        num_n_blocks,
        in0_mcast_sender_semaphore_id,
        in0_mcast_receiver_semaphore_id,
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(in0_sender_ct_args);
    auto in0_sender_kernel_id = tt::tt_metal::CreateKernel(
        program,
        kIn0SenderKernelPath,
        left_column_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_ct_args});

    // --- RISCV_1: IN0 receiver (non-left columns) ---
    tt::tt_metal::KernelHandle in0_receiver_kernel_id = 0;
    if (num_cores_c > 1U) {
        std::vector<uint32_t> in0_receiver_ct_args = {
            kBlockSize,
            Wt,
            per_core_M,
            num_n_blocks,
            in0_mcast_sender_semaphore_id,
            in0_mcast_receiver_semaphore_id,
        };
        in0_receiver_kernel_id = tt::tt_metal::CreateKernel(
            program,
            kIn0ReceiverKernelPath,
            non_left_column_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_receiver_ct_args});
    }

    // --- RISCV_0: IN1 sender + M writer (top row) ---
    std::vector<uint32_t> in1_sender_writer_ct_args = {
        kBlockSize,
        Wt,
        hidden_Wt,
        per_core_M,
        per_core_N,
        per_core_N_rounded,
        num_n_blocks,
        in1_mcast_sender_semaphore_id,
        in1_mcast_receiver_semaphore_id,
    };
    tt::tt_metal::TensorAccessorArgs(w1_buffer).append_to(in1_sender_writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(w3_buffer).append_to(in1_sender_writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(m_buffer).append_to(in1_sender_writer_ct_args);
    auto in1_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        kIn1SenderWriterKernelPath,
        top_row_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_ct_args});

    // --- RISCV_0: IN1 receiver + M writer (non-top rows) ---
    tt::tt_metal::KernelHandle in1_receiver_writer_kernel_id = 0;
    if (num_cores_r > 1U) {
        std::vector<uint32_t> in1_receiver_writer_ct_args = {
            kBlockSize,
            Wt,
            hidden_Wt,
            per_core_M,
            per_core_N,
            per_core_N_rounded,
            num_n_blocks,
            in1_mcast_sender_semaphore_id,
            in1_mcast_receiver_semaphore_id,
        };
        tt::tt_metal::TensorAccessorArgs(m_buffer).append_to(in1_receiver_writer_ct_args);
        in1_receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            kIn1ReceiverWriterKernelPath,
            non_top_row_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
                .compile_args = in1_receiver_writer_ct_args});
    }

    // -------------------------------------------------------------------------
    // 9) Create compute kernel
    // -------------------------------------------------------------------------
    std::vector<uint32_t> compute_ct_args = {
        per_core_M,
        per_core_N,
        per_core_N_rounded,
        kBlockSize,
        Wt,
        num_n_blocks,
    };
    auto compute_kernel_id = create_compute_kernel(
        program, all_cores_set, compute_ct_args, {}, kComputeKernelPath, /*fp32_dest_acc_en=*/true);

    // -------------------------------------------------------------------------
    // 10) Assign runtime args per core
    // -------------------------------------------------------------------------
    for (uint32_t r = 0U; r < num_cores_r; ++r) {
        for (uint32_t c = 0U; c < num_cores_c; ++c) {
            tt::tt_metal::CoreCoord core = {c, r};

            uint32_t m_start = r * per_core_M;
            uint32_t n_start = c * per_core_N;
            uint32_t actual_m_tiles = std::min(per_core_M, total_M_tiles - m_start);
            uint32_t actual_n_tiles = std::min(per_core_N, hidden_Wt - n_start);

            bool is_in0_sender = (c == 0U);
            bool is_in1_sender = (r == 0U);

            // --- RISCV_1: IN0 sender or receiver ---
            if (is_in0_sender) {
                // Row multicast range: IN0 runs on in0_noc (RISCV_1).
                // On Wormhole NOC_1, coordinates are inverted relative to
                // worker_core_from_logical_core (NOC_0 space).  Pass the
                // end-of-row as "start" and start-of-row as "end" so that
                // after NOC_1 inversion the rectangle is valid (start ≤ end).
                auto row_lo_phys = device->worker_core_from_logical_core({0, r});
                auto row_hi_phys = device->worker_core_from_logical_core({num_cores_c - 1U, r});
                uint32_t num_in0_receivers = num_cores_c - 1U;

                SetRuntimeArgs(
                    program,
                    in0_sender_kernel_id,
                    core,
                    {input_buffer->address(),
                     m_start,
                     actual_m_tiles,
                     static_cast<uint32_t>(row_hi_phys.x),
                     static_cast<uint32_t>(row_hi_phys.y),
                     static_cast<uint32_t>(row_lo_phys.x),
                     static_cast<uint32_t>(row_lo_phys.y),
                     num_in0_receivers});
            } else {
                // IN0 receiver: needs sender's physical coords for semaphore
                auto sender_phys = device->worker_core_from_logical_core({0, r});
                SetRuntimeArgs(
                    program,
                    in0_receiver_kernel_id,
                    core,
                    {static_cast<uint32_t>(sender_phys.x), static_cast<uint32_t>(sender_phys.y)});
            }

            // --- RISCV_0: IN1 sender+writer or receiver+writer ---
            if (is_in1_sender) {
                // Compute column multicast range
                auto col_start_phys = device->worker_core_from_logical_core({c, 0});
                auto col_end_phys = device->worker_core_from_logical_core({c, num_cores_r - 1U});
                uint32_t num_in1_receivers = num_cores_r - 1U;

                SetRuntimeArgs(
                    program,
                    in1_sender_writer_kernel_id,
                    core,
                    {w1_buffer->address(),
                     w3_buffer->address(),
                     m_buffer->address(),
                     m_start,
                     n_start,
                     actual_m_tiles,
                     actual_n_tiles,
                     static_cast<uint32_t>(col_start_phys.x),
                     static_cast<uint32_t>(col_start_phys.y),
                     static_cast<uint32_t>(col_end_phys.x),
                     static_cast<uint32_t>(col_end_phys.y),
                     num_in1_receivers});
            } else {
                // IN1 receiver+writer: needs sender's physical coords
                auto sender_phys = device->worker_core_from_logical_core({c, 0});
                SetRuntimeArgs(
                    program,
                    in1_receiver_writer_kernel_id,
                    core,
                    {m_buffer->address(),
                     m_start,
                     n_start,
                     actual_m_tiles,
                     actual_n_tiles,
                     static_cast<uint32_t>(sender_phys.x),
                     static_cast<uint32_t>(sender_phys.y)});
            }
        }
    }

    // -------------------------------------------------------------------------
    // 11) Return cached program
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {in0_sender_kernel_id,
         in0_receiver_kernel_id,
         in1_sender_writer_kernel_id,
         in1_receiver_writer_kernel_id,
         compute_kernel_id,
         num_cores_r,
         num_cores_c,
         per_core_M,
         per_core_N}};
}

void SwiGLUGateUpProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;

    auto* input_buffer = tensor_args.input.buffer();
    auto* w1_buffer = tensor_args.w1.buffer();
    auto* w3_buffer = tensor_args.w3.buffer();
    auto* m_buffer = output.buffer();

    auto& in0_sender_args = GetRuntimeArgs(program, sv.in0_sender_kernel_id);
    auto& in1_sender_writer_args = GetRuntimeArgs(program, sv.in1_sender_writer_kernel_id);
    auto& in1_receiver_writer_args = GetRuntimeArgs(program, sv.in1_receiver_writer_kernel_id);

    for (uint32_t r = 0U; r < sv.num_cores_r; ++r) {
        for (uint32_t c = 0U; c < sv.num_cores_c; ++c) {
            tt::tt_metal::CoreCoord core = {c, r};
            bool is_in0_sender = (c == 0U);
            bool is_in1_sender = (r == 0U);

            if (is_in0_sender) {
                auto& rt_args = in0_sender_args[core.x][core.y];
                rt_args[kIn0SenderXAddrIdx] = input_buffer->address();
            }

            if (is_in1_sender) {
                auto& rt_args = in1_sender_writer_args[core.x][core.y];
                rt_args[kIn1SenderW1AddrIdx] = w1_buffer->address();
                rt_args[kIn1SenderW3AddrIdx] = w3_buffer->address();
                rt_args[kIn1SenderMAddrIdx] = m_buffer->address();
            } else {
                auto& rt_args = in1_receiver_writer_args[core.x][core.y];
                rt_args[kIn1ReceiverMAddrIdx] = m_buffer->address();
            }
        }
    }
}

}  // namespace ttml::metal::ops::swiglu_gate_up::device
