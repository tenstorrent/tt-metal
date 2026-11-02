// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "skinny_contract_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include "skinny_contract_device_operation.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/skinny_contract/device/kernels/dataflow/reader_skinny_contract.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/skinny_contract/device/kernels/dataflow/writer_skinny_contract.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/skinny_contract/device/kernels/compute/skinny_contract_kernel.cpp";

constexpr uint32_t kSmallCb = 0;    // [Kt] bf16: small^T, resident (multicast from the sender core)
constexpr uint32_t kBigCb = 1;      // [2*prefetch*cols] bf16: K rows of a chunk's `cols` adjacent column tiles
constexpr uint32_t kOutCb = 2;      // [2*cols] bf16
constexpr uint32_t kMaxCols = 4U;   // fp32 DEST holds four accumulators
constexpr uint32_t kPrefetch = 8U;  // K rows read per barrier (DRAM latency hiding; 32 = one batch per chunk is 4 %
                                    // slower, O/iter40_contract_after.log)

constexpr uint32_t kReaderSmallIdx = 0U;
constexpr uint32_t kReaderBigIdx = 1U;
constexpr uint32_t kWriterOutIdx = 0U;

}  // namespace

namespace ttml::metal::ops::skinny_contract::device {

// Every core of the grid takes a contiguous range of output column tiles (Nt / cores, rounded: 4-5 of the 512 of
// the gate_up dB; iteration 40, was items of 4 columns handed out round-robin, 2 items on 18 cores and 1 on the
// rest) and works it in chunks of up to kMaxCols columns: out tile (0, n) = sum_t small_t[0, t] @ big[t, n] over
// the Kt tiles of T, the `cols` accumulators in fp32 DEST, scaled, transposed in DEST when transpose_out, packed
// bf16. small^T is read from DRAM by the sender core (0, 0) and multicast into every core's resident CB. The reader
// streams one K row of `cols` contiguous pages per push.
SkinnyContractProgramFactory::cached_program_t SkinnyContractProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, ttnn::Tensor& output) {
    const auto& small_t = tensor_args.small_t;
    const auto& big = tensor_args.big;
    auto* device = big.device();
    tt::tt_metal::Program program{};

    const uint32_t Kt =
        (args.big_transposed ? big.padded_shape()[3] : big.padded_shape()[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t Nt =
        (args.big_transposed ? big.padded_shape()[2] : big.padded_shape()[3]) / tt::constants::TILE_WIDTH;
    const auto grid = device->compute_with_storage_grid_size();
    // iteration 47: the op may be confined to the first grid_rows core rows (a sub-device split with a collective)
    const uint32_t rows = args.grid_rows == 0U ? grid.y : std::min<uint32_t>(args.grid_rows, grid.y);
    const uint32_t grid_cores = grid.x * rows;
    const tt::tt_metal::CoreCoord sender_core{0, 0};
    const tt::tt_metal::CoreRange grid_rect(sender_core, {grid.x - 1U, rows - 1U});
    const tt::tt_metal::CoreRangeSet all_cores(grid_rect);
    // the receivers: the grid rectangle without the sender
    std::vector<tt::tt_metal::CoreRange> receiver_ranges;
    if (grid.x > 1U) {
        receiver_ranges.emplace_back(tt::tt_metal::CoreCoord{1, 0}, tt::tt_metal::CoreCoord{grid.x - 1U, 0});
    }
    if (rows > 1U) {
        receiver_ranges.emplace_back(tt::tt_metal::CoreCoord{0, 1}, tt::tt_metal::CoreCoord{grid.x - 1U, rows - 1U});
    }
    const tt::tt_metal::CoreRangeSet receiver_cores(receiver_ranges);
    const uint32_t cols = kMaxCols;
    const uint32_t work_cores = std::min(Nt, grid_cores);

    const auto bf16 = tt::DataFormat::Float16_b;
    const uint32_t tile = tt::tile_size(bf16);
    create_circular_buffer(program, all_cores, kSmallCb, bf16, tile, Kt);
    const uint32_t prefetch = std::min(kPrefetch, Kt);
    create_circular_buffer(program, all_cores, kBigCb, bf16, tile, 2U * prefetch * cols);
    create_circular_buffer(program, all_cores, kOutCb, bf16, tile, 2U * cols);

    std::map<std::string, std::string> defines;
    if (args.transpose_out) {
        defines["TRANSPOSE_OUT"] = "1";
    }
    if (args.big_transposed) {
        defines["BIG_TRANSPOSED"] = "1";  // big tile (t, n) at page n * Kt + t
    }
    const uint32_t scale_bits = std::bit_cast<uint32_t>(args.scale);

    const auto sender_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const auto receiver_sem = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const auto start_noc = device->worker_core_from_logical_core(sender_core);
    const auto end_noc = device->worker_core_from_logical_core({grid.x - 1U, rows - 1U});
    std::vector<uint32_t> reader_ct{
        Kt,
        Nt,
        cols,
        prefetch,
        sender_sem,
        receiver_sem,
        static_cast<uint32_t>(start_noc.x),
        static_cast<uint32_t>(start_noc.y),
        static_cast<uint32_t>(end_noc.x),
        static_cast<uint32_t>(end_noc.y),
        grid_cores - 1U};
    tt::tt_metal::TensorAccessorArgs(small_t.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(big.buffer()).append_to(reader_ct);
    auto sender_defines = defines;
    sender_defines["MCAST_SENDER"] = "1";
    auto reader_sender = create_reader_kernel(
        program,
        tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(sender_core, sender_core)),
        reader_ct,
        sender_defines,
        kReaderKernelPath);
    auto reader_receiver = create_reader_kernel(program, receiver_cores, reader_ct, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct{Kt, Nt, cols};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    auto writer = create_writer_kernel(program, all_cores, writer_ct, defines, kWriterKernelPath);
    auto compute = create_compute_kernel(
        program, all_cores, {Kt, Nt, cols, scale_bits, prefetch}, defines, kComputeKernelPath, true);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(grid_cores);
    for (uint32_t i = 0; i < grid_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / rows, i % rows};
        const uint32_t n_begin = i < work_cores ? (i * Nt) / work_cores : Nt;
        const uint32_t n_end = i < work_cores ? ((i + 1U) * Nt) / work_cores : Nt;
        const uint32_t n_count = n_end - n_begin;
        SetRuntimeArgs(program, compute, core, {n_begin, n_count});
        SetRuntimeArgs(
            program,
            core == sender_core ? reader_sender : reader_receiver,
            core,
            {small_t.buffer()->address(), big.buffer()->address(), n_begin, n_count});
        SetRuntimeArgs(program, writer, core, {output.buffer()->address(), n_begin, n_count});
        cores.push_back(core);
    }
    return cached_program_t{std::move(program), {reader_sender, reader_receiver, writer, std::move(cores)}};
}

void SkinnyContractProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    ttnn::Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& sender_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& receiver_args = GetRuntimeArgs(program, shared.reader_receiver_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = (core.x == 0 && core.y == 0) ? sender_args[core.x][core.y] : receiver_args[core.x][core.y];
        r[kReaderSmallIdx] = tensor_args.small_t.buffer()->address();
        r[kReaderBigIdx] = tensor_args.big.buffer()->address();
        writer_args[core.x][core.y][kWriterOutIdx] = output.buffer()->address();
    }
}

}  // namespace ttml::metal::ops::skinny_contract::device
