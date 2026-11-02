// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "skinny_contract_program_factory.hpp"

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

constexpr uint32_t kSmallCb = 0;    // [Kt] bf16: small^T, resident
constexpr uint32_t kBigCb = 1;      // [2*cols] bf16: one K row of the item's `cols` adjacent column tiles
constexpr uint32_t kOutCb = 2;      // [cols] bf16
constexpr uint32_t kMaxCols = 4U;   // fp32 DEST holds four accumulators
constexpr uint32_t kPrefetch = 8U;  // K rows read per barrier (DRAM latency hiding)

constexpr uint32_t kReaderSmallIdx = 0U;
constexpr uint32_t kReaderBigIdx = 1U;
constexpr uint32_t kWriterOutIdx = 0U;

}  // namespace

namespace ttml::metal::ops::skinny_contract::device {

// Items are groups of `cols` adjacent output column tiles (one item per core when Nt <= 4 x cores): out tile
// (0, n) = sum_t small_t[0, t] @ big[t, n] over the Kt tiles of T, the `cols` accumulators in fp32 DEST, scaled,
// transposed in DEST when transpose_out, packed bf16. The reader streams one K row of `cols` contiguous pages
// per push.
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
    const uint32_t num_cores_total = grid.x * grid.y;
    const uint32_t cols = std::min(kMaxCols, std::max<uint32_t>(1U, (Nt + num_cores_total - 1U) / num_cores_total));
    const uint32_t items = (Nt + cols - 1U) / cols;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_1, work_2] =
        tt::tt_metal::split_work_to_cores(grid, items);

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

    std::vector<uint32_t> reader_ct{Kt, Nt, cols, prefetch};
    tt::tt_metal::TensorAccessorArgs(small_t.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(big.buffer()).append_to(reader_ct);
    auto reader = create_reader_kernel(program, all_cores, reader_ct, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct{Kt, Nt, cols};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    auto writer = create_writer_kernel(program, all_cores, writer_ct, defines, kWriterKernelPath);

    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        return create_compute_kernel(
            program, cores, {work_count, Kt, Nt, cols, scale_bits, prefetch}, defines, kComputeKernelPath, true);
    };
    const auto compute_1 = make_compute(core_group_1, work_1);
    const auto compute_2 = core_group_2.ranges().empty() ? compute_1 : make_compute(core_group_2, work_2);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t i = 0, work_start = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / grid.y, i % grid.y};
        const bool in_group_1 = core_group_1.contains(core);
        const uint32_t work_count = in_group_1 ? work_1 : work_2;
        SetRuntimeArgs(program, in_group_1 ? compute_1 : compute_2, core, {work_start});
        SetRuntimeArgs(
            program, reader, core, {small_t.buffer()->address(), big.buffer()->address(), work_start, work_count});
        SetRuntimeArgs(program, writer, core, {output.buffer()->address(), work_start, work_count});
        cores.push_back(core);
        work_start += work_count;
    }
    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void SkinnyContractProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    ttnn::Tensor& output) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id);
    for (const auto& core : shared.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderSmallIdx] = tensor_args.small_t.buffer()->address();
        r[kReaderBigIdx] = tensor_args.big.buffer()->address();
        writer_args[core.x][core.y][kWriterOutIdx] = output.buffer()->address();
    }
}

}  // namespace ttml::metal::ops::skinny_contract::device
