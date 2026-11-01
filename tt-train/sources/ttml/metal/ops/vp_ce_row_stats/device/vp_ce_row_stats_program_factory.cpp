// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vp_ce_row_stats_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/vp_ce_row_stats/device/kernels/dataflow/reader_vp_ce_row_stats.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/vp_ce_row_stats/device/kernels/dataflow/writer_vp_ce_row_stats.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/vp_ce_row_stats/device/kernels/compute/vp_ce_row_stats_kernel.cpp";

// CB layout (shared with the kernels):
constexpr uint32_t kInCb = 0;       // bf16 logits tiles, `block` at a time
constexpr uint32_t kMaxCb = 1;      // the item's row max tile (mode 1)
constexpr uint32_t kScalerCb = 2;   // bf16 1.0 tile (the reduce scaler, mode 0)
constexpr uint32_t kOnesCb = 3;     // bf16 ones-column tile (the matmul row reduce, mode 1)
constexpr uint32_t kTmpCb = 5;      // fp32 the reduced max tile before the column broadcast (mode 0)
constexpr uint32_t kOutCb = 6;      // fp32 output tiles
constexpr uint32_t kBlock = 4U;     // mode 0 (the reduce holds one DEST register)
constexpr uint32_t kSumBlock = 2U;  // mode 1 (two exp tiles, the accumulator and the mask in the fp32 DEST)

constexpr uint32_t kReaderLogitsIdx = 0U;
constexpr uint32_t kReaderMaxIdx = 1U;
constexpr uint32_t kWriterOutIdx = 0U;

}  // namespace

namespace ttml::metal::ops::vp_ce_row_stats::device {

uint32_t num_slices(uint32_t rows, uint32_t Wt, uint32_t num_cores) {
    uint32_t slices = std::min<uint32_t>(Wt, std::max<uint32_t>(1U, (2U * num_cores + rows - 1U) / rows));
    uint32_t St = (Wt + slices - 1U) / slices;
    if (St % kBlock != 0U) {  // a multiple of both block sizes
        St += kBlock - (St % kBlock);
    }
    return (Wt + St - 1U) / St;
}

VpCeRowStatsProgramFactory::cached_program_t VpCeRowStatsProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& logits = tensor_args.logits;
    auto* device = logits.device();
    tt::tt_metal::Program program{};

    const auto padded = logits.padded_shape();
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded[2] / tt::constants::TILE_HEIGHT;
    const uint32_t rows = padded[0] * padded[1] * Ht;
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t slices = num_slices(rows, Wt, grid.x * grid.y);
    const uint32_t St =
        (Wt + slices - 1U) / slices +
        ((((Wt + slices - 1U) / slices) % kBlock) != 0U ? kBlock - (((Wt + slices - 1U) / slices) % kBlock) : 0U);
    TT_FATAL((Wt + St - 1U) / St == slices, "VpCeRowStats: slice arithmetic mismatch");
    const uint32_t total_items = rows * slices;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_1, work_2] =
        tt::tt_metal::split_work_to_cores(grid, total_items);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const bool sum_mode = args.mode == 1U;
    const uint32_t block = sum_mode ? kSumBlock : kBlock;
    const auto max_format = sum_mode ? datatype_to_dataformat_converter(tensor_args.max_col->dtype()) : f32;
    create_circular_buffer(program, all_cores, kInCb, bf16, tt::tile_size(bf16), 2U * block);
    create_circular_buffer(program, all_cores, kMaxCb, max_format, tt::tile_size(max_format), 2U);
    create_circular_buffer(program, all_cores, kScalerCb, bf16, tt::tile_size(bf16), 1U);
    create_circular_buffer(program, all_cores, kOnesCb, bf16, tt::tile_size(bf16), 1U);
    create_circular_buffer(program, all_cores, kTmpCb, f32, tt::tile_size(f32), 1U);
    create_circular_buffer(program, all_cores, kOutCb, f32, tt::tile_size(f32), 2U);

    std::map<std::string, std::string> defines;
    defines[sum_mode ? "SUM_EXP" : "ROW_MAX"] = "1";

    std::vector<uint32_t> reader_ct{Wt, slices, St, block};
    tt::tt_metal::TensorAccessorArgs(logits.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs((sum_mode ? *tensor_args.max_col : logits).buffer()).append_to(reader_ct);
    auto reader = create_reader_kernel(program, all_cores, reader_ct, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct{Wt, slices, St, block};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    auto writer = create_writer_kernel(program, all_cores, writer_ct, defines, kWriterKernelPath);

    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        return create_compute_kernel(
            program,
            cores,
            {work_count, Wt, slices, St, block},
            defines,
            kComputeKernelPath,
            /*fp32_dest_acc_en=*/true);
    };
    const auto compute_1 = make_compute(core_group_1, work_1);
    const auto compute_2 = core_group_2.ranges().empty() ? compute_1 : make_compute(core_group_2, work_2);

    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(num_cores);
    const uint32_t max_addr = sum_mode ? tensor_args.max_col->buffer()->address() : logits.buffer()->address();
    for (uint32_t i = 0, work_start = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core = {i / grid.y, i % grid.y};
        const bool in_group_1 = core_group_1.contains(core);
        const uint32_t work_count = in_group_1 ? work_1 : work_2;
        SetRuntimeArgs(program, in_group_1 ? compute_1 : compute_2, core, {work_start});
        SetRuntimeArgs(program, reader, core, {logits.buffer()->address(), max_addr, work_start, work_count});
        SetRuntimeArgs(program, writer, core, {output.buffer()->address(), work_start, work_count});
        cores.push_back(core);
        work_start += work_count;
    }
    return cached_program_t{std::move(program), {reader, writer, std::move(cores)}};
}

void VpCeRowStatsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& sv = cached_program.shared_variables;
    auto& reader_args = GetRuntimeArgs(program, sv.reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, sv.writer_kernel_id);
    const uint32_t max_addr =
        args.mode == 1U ? tensor_args.max_col->buffer()->address() : tensor_args.logits.buffer()->address();
    for (const auto& core : sv.cores) {
        auto& r = reader_args[core.x][core.y];
        r[kReaderLogitsIdx] = tensor_args.logits.buffer()->address();
        r[kReaderMaxIdx] = max_addr;
        writer_args[core.x][core.y][kWriterOutIdx] = output.buffer()->address();
    }
}

}  // namespace ttml::metal::ops::vp_ce_row_stats::device
