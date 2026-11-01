"""Iteration 32: `vp_ce_row_stats`, one pass over the local logits shard for the vocab-parallel cross-entropy forward:
mode 0 = the row max (per (row, slice) partial, broadcast over the tile's columns), mode 1 = the row sum of
exp(x - max) (per (row, slice) partial in column 0, the other columns 0); the caller reduces the [B, 1, S, slices * 32]
fp32 partials with ttnn::max / ttnn::sum. Replaces ttnn::max + (sub, exp into a 128 MB fp32 tensor) + ttnn::sum."""
import os
import sys

ROOT = "tt-train/sources/ttml"
OP = f"{ROOT}/metal/ops/vp_ce_row_stats"
os.makedirs(f"{OP}/device/kernels/compute", exist_ok=True)
os.makedirs(f"{OP}/device/kernels/dataflow", exist_ok=True)

HDR = "// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.\n//\n// SPDX-License-Identifier: Apache-2.0\n\n"

files = {}
files[f"{OP}/vp_ce_row_stats.hpp"] = (
    HDR
    + """#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Row statistics of a [B, 1, S, V] bf16 TILE shard for the vocab-parallel cross-entropy forward (iteration 32 of the
// Mistral loop). mode 0: per (row, K slice) partial row max, the value broadcast over the tile's 32 columns; mode 1:
// per (row, K slice) partial sum of exp(x - max_col) in column 0 (the other columns 0), `max_col` [B, 1, S, 1] TILE.
// Output [B, 1, S, slices * 32] fp32 TILE; reduce it over the last dim with ttnn::max (mode 0) or ttnn::sum (mode 1).
ttnn::Tensor vp_ce_row_stats(
    const ttnn::Tensor& logits, uint32_t mode, const std::optional<ttnn::Tensor>& max_col = std::nullopt);

}  // namespace ttml::metal
"""
)
files[f"{OP}/vp_ce_row_stats.cpp"] = (
    HDR
    + """#include "vp_ce_row_stats.hpp"

#include "device/vp_ce_row_stats_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor vp_ce_row_stats(const ttnn::Tensor& logits, uint32_t mode, const std::optional<ttnn::Tensor>& max_col) {
    return ttnn::prim::ttml_vp_ce_row_stats(logits, mode, max_col);
}

}  // namespace ttml::metal
"""
)
files[f"{OP}/device/vp_ce_row_stats_device_operation_types.hpp"] = (
    HDR
    + """#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::vp_ce_row_stats::device {

struct Params {
    uint32_t mode{0U};  // 0: row max, 1: row sum of exp(x - max_col)
};

struct Inputs {
    const ttnn::Tensor& logits;              // [B, 1, S, V] TILE bf16
    std::optional<ttnn::Tensor> max_col{};  // [B, 1, S, 1] TILE bf16 / fp32 (mode 1)
};

using operation_attributes_t = Params;
using tensor_args_t = Inputs;
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::vp_ce_row_stats::device
"""
)
files[f"{OP}/device/vp_ce_row_stats_program_factory.hpp"] = (
    HDR
    + """#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "vp_ce_row_stats_device_operation_types.hpp"

namespace ttml::metal::ops::vp_ce_row_stats::device {

struct VpCeRowStatsProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

// The slice count of a [rows, Wt] shard on `num_cores` cores (shared with the output shape).
uint32_t num_slices(uint32_t rows, uint32_t Wt, uint32_t num_cores);

}  // namespace ttml::metal::ops::vp_ce_row_stats::device
"""
)
files[f"{OP}/device/vp_ce_row_stats_device_operation.hpp"] = (
    HDR
    + """#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "vp_ce_row_stats_device_operation_types.hpp"
#include "vp_ce_row_stats_program_factory.hpp"

namespace ttml::metal::ops::vp_ce_row_stats::device {

struct VpCeRowStatsDeviceOperation {
    using operation_attributes_t = device::operation_attributes_t;
    using tensor_args_t = device::tensor_args_t;
    using spec_return_value_t = device::spec_return_value_t;
    using tensor_return_value_t = device::tensor_return_value_t;
    using program_factory_t = std::variant<VpCeRowStatsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::vp_ce_row_stats::device

namespace ttnn::prim {
ttnn::Tensor ttml_vp_ce_row_stats(
    const ttnn::Tensor& logits, uint32_t mode, const std::optional<ttnn::Tensor>& max_col);
}  // namespace ttnn::prim
"""
)
files[f"{OP}/device/vp_ce_row_stats_device_operation.cpp"] = (
    HDR
    + """#include "vp_ce_row_stats_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::vp_ce_row_stats::device {

VpCeRowStatsDeviceOperation::program_factory_t VpCeRowStatsDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return VpCeRowStatsProgramFactory{};
}

void VpCeRowStatsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& logits = tensor_args.logits;
    check_device_tensor(logits, "VpCeRowStats", "logits", {.dtypes = {tt::tt_metal::DataType::BFLOAT16}});
    const auto& ls = logits.logical_shape();
    TT_FATAL(ls.rank() == 4U && ls[1] == 1U, "VpCeRowStats: logits must be [B, 1, S, V], got {}", ls);
    TT_FATAL(args.mode <= 1U, "VpCeRowStats: mode must be 0 (max) or 1 (sum of exp), got {}", args.mode);
    if (args.mode == 1U) {
        TT_FATAL(tensor_args.max_col.has_value(), "VpCeRowStats: mode 1 needs max_col");
        check_device_tensor(
            *tensor_args.max_col,
            "VpCeRowStats",
            "max_col",
            {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32}});
        const auto& ms = tensor_args.max_col->logical_shape();
        TT_FATAL(
            ms.rank() == 4U && ms[0] == ls[0] && ms[1] == 1U && ms[2] == ls[2] && ms[3] == 1U,
            "VpCeRowStats: max_col must be [B, 1, S, 1] for logits {}, got {}",
            ls,
            ms);
    }
}

void VpCeRowStatsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

spec_return_value_t VpCeRowStatsDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& logits = tensor_args.logits;
    const auto padded = logits.padded_shape();
    const uint32_t Wt = padded[3] / tt::constants::TILE_WIDTH;
    const uint32_t rows = padded[0] * padded[1] * padded[2] / tt::constants::TILE_HEIGHT;
    const auto grid = logits.device()->compute_with_storage_grid_size();
    const uint32_t slices = num_slices(rows, Wt, grid.x * grid.y);
    const auto& ls = logits.logical_shape();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape{ls[0], ls[1], ls[2], slices * tt::constants::TILE_WIDTH},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), logits.memory_config()));
}

tensor_return_value_t VpCeRowStatsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.logits.device());
}

ttsl::hash::hash_t VpCeRowStatsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<VpCeRowStatsDeviceOperation>(
        args.mode,
        tensor_args.logits.logical_shape(),
        tensor_args.max_col.has_value() ? static_cast<uint32_t>(tensor_args.max_col->dtype()) : 0U);
}

}  // namespace ttml::metal::ops::vp_ce_row_stats::device

namespace ttnn::prim {

ttnn::Tensor ttml_vp_ce_row_stats(
    const ttnn::Tensor& logits, uint32_t mode, const std::optional<ttnn::Tensor>& max_col) {
    using OperationType = ttml::metal::ops::vp_ce_row_stats::device::VpCeRowStatsDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{.mode = mode};
    auto tensor_args = OperationType::tensor_args_t{.logits = logits, .max_col = max_col};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
"""
)
files[f"{OP}/device/vp_ce_row_stats_program_factory.cpp"] = (
    HDR
    + """#include "vp_ce_row_stats_program_factory.hpp"

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
constexpr uint32_t kInCb = 0;      // bf16 logits tiles, `block` at a time
constexpr uint32_t kMaxCb = 1;     // the item's row max tile (mode 1)
constexpr uint32_t kScalerCb = 2;  // bf16 1.0 tile (the reduce scaler, mode 0)
constexpr uint32_t kOnesCb = 3;    // bf16 ones-column tile (the matmul row reduce, mode 1)
constexpr uint32_t kExpCb = 4;     // fp32 exp tiles of a block (mode 1)
constexpr uint32_t kTmpCb = 5;     // fp32 the reduced max tile before the column broadcast (mode 0)
constexpr uint32_t kOutCb = 6;     // fp32 output tiles
constexpr uint32_t kBlock = 4U;

constexpr uint32_t kReaderLogitsIdx = 0U;
constexpr uint32_t kReaderMaxIdx = 1U;
constexpr uint32_t kWriterOutIdx = 0U;

}  // namespace

namespace ttml::metal::ops::vp_ce_row_stats::device {

uint32_t num_slices(uint32_t rows, uint32_t Wt, uint32_t num_cores) {
    uint32_t slices = std::min<uint32_t>(Wt, std::max<uint32_t>(1U, (2U * num_cores + rows - 1U) / rows));
    uint32_t St = (Wt + slices - 1U) / slices;
    if (St % kBlock != 0U) {
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
    const uint32_t St = (Wt + slices - 1U) / slices + ((((Wt + slices - 1U) / slices) % kBlock) != 0U
                                                           ? kBlock - (((Wt + slices - 1U) / slices) % kBlock)
                                                           : 0U);
    TT_FATAL((Wt + St - 1U) / St == slices, "VpCeRowStats: slice arithmetic mismatch");
    const uint32_t total_items = rows * slices;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_1, work_2] =
        tt::tt_metal::split_work_to_cores(grid, total_items);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto f32 = tt::DataFormat::Float32;
    const bool sum_mode = args.mode == 1U;
    const auto max_format = sum_mode ? datatype_to_dataformat_converter(tensor_args.max_col->dtype()) : f32;
    create_circular_buffer(program, all_cores, kInCb, bf16, tt::tile_size(bf16), 2U * kBlock);
    create_circular_buffer(program, all_cores, kMaxCb, max_format, tt::tile_size(max_format), 2U);
    create_circular_buffer(program, all_cores, kScalerCb, bf16, tt::tile_size(bf16), 1U);
    create_circular_buffer(program, all_cores, kOnesCb, bf16, tt::tile_size(bf16), 1U);
    create_circular_buffer(program, all_cores, kExpCb, f32, tt::tile_size(f32), kBlock);
    create_circular_buffer(program, all_cores, kTmpCb, f32, tt::tile_size(f32), 1U);
    create_circular_buffer(program, all_cores, kOutCb, f32, tt::tile_size(f32), 2U);

    std::map<std::string, std::string> defines;
    defines[sum_mode ? "SUM_EXP" : "ROW_MAX"] = "1";

    std::vector<uint32_t> reader_ct{Wt, slices, St, kBlock};
    tt::tt_metal::TensorAccessorArgs(logits.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs((sum_mode ? *tensor_args.max_col : logits).buffer()).append_to(reader_ct);
    auto reader = create_reader_kernel(program, all_cores, reader_ct, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_ct{Wt, slices, St, kBlock};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_ct);
    auto writer = create_writer_kernel(program, all_cores, writer_ct, defines, kWriterKernelPath);

    auto make_compute = [&](const tt::tt_metal::CoreRangeSet& cores, uint32_t work_count) {
        return create_compute_kernel(
            program, cores, {work_count, Wt, slices, St, kBlock}, defines, kComputeKernelPath, /*fp32_dest_acc_en=*/true);
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
"""
)
files[f"{OP}/device/kernels/dataflow/reader_vp_ce_row_stats.cpp"] = (
    HDR
    + """#include "api/dataflow/dataflow_api.h"

// Per item (row r, slice s): the row's max tile (SUM_EXP) and the slice's logits tiles in blocks.
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_max = 1;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t slices = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr auto in_args = TensorAccessorArgs<4>();
constexpr auto max_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t max_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t in_tile = get_tile_size(cb_in);
    const auto in_acc = TensorAccessor(in_args, in_addr, in_tile);
#ifdef SUM_EXP
    const uint32_t max_tile = get_tile_size(cb_max);
    const auto max_acc = TensorAccessor(max_args, max_addr, max_tile);
#endif

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / slices;
        const uint32_t s = work - r * slices;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Wt) ? St : (Wt - k0);
#ifdef SUM_EXP
        cb_reserve_back(cb_max, 1);
        noc_async_read_page(r, max_acc, get_write_ptr(cb_max));
        noc_async_read_barrier();
        cb_push_back(cb_max, 1);
#endif
        for (uint32_t k = 0; k < nk; k += block) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_reserve_back(cb_in, block);
            uint32_t l1 = get_write_ptr(cb_in);
            for (uint32_t j = 0; j < n; ++j) {
                noc_async_read_page(r * Wt + k0 + k + j, in_acc, l1);
                l1 += in_tile;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, block);
        }
    }
}
"""
)
files[f"{OP}/device/kernels/dataflow/writer_vp_ce_row_stats.cpp"] = (
    HDR
    + """#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Generates the reduce scaler (1.0) and the ones-column tile once, then writes each item's output tile to page
// r * slices + s of the [rows, slices] tile grid.
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_ones = 3;
constexpr uint32_t cb_out = 6;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t slices = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr auto out_args = TensorAccessorArgs<4>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t out_tile = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tile);

    constexpr uint16_t one_bf16 = 0x3F80U;
    generate_tile_with_bfloat16_value(cb_scaler, one_bf16);  // the reduce scaler
    generate_matmul_row_reduce_tile(cb_ones);                // ones in column 0: the matmul row sum

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(work, out_acc, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
"""
)
files[f"{OP}/device/kernels/compute/vp_ce_row_stats_kernel.cpp"] = (
    HDR
    + """#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/common/sdpa_compute_utils_common.hpp"

// Per item (row r, slice s of St tiles):
//   ROW_MAX: the row max over the slice's tiles (FPU reduce, one DEST register held over the blocks), packed to a
//            scratch tile, then column 0 broadcast over the tile's columns (so the caller's max over the partial
//            columns is exact) into the output tile.
//   SUM_EXP: per block exp(x - max) on the SFPU at fp32 into the exp CB, the block's row sums by a matmul with the
//            ones-column tile (column 0 = the sums, the other columns 0) L1-accumulated onto the output tile.
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_max = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_ones = 3;
constexpr uint32_t cb_exp = 4;
constexpr uint32_t cb_tmp = 5;
constexpr uint32_t cb_out = 6;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t slices = get_compile_time_arg_val(2);
constexpr uint32_t St = get_compile_time_arg_val(3);
constexpr uint32_t block = get_compile_time_arg_val(4);

void kernel_main() {
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_ones, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t s = work - (work / slices) * slices;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Wt) ? St : (Wt - k0);
        cb_reserve_back(cb_out, 1);
#ifdef ROW_MAX
        reconfig_data_format(cb_in, cb_scaler);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_tmp);
        tile_regs_acquire();
        for (uint32_t k = 0; k < nk; k += block) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_wait_front(cb_in, block);
            for (uint32_t j = 0; j < n; ++j) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, j, 0, /* dst */ 0);
            }
            cb_pop_front(cb_in, block);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_tmp, 1);
        pack_reconfig_data_format(cb_tmp);
        pack_tile(0, cb_tmp);
        tile_regs_release();
        cb_push_back(cb_tmp, 1);
        // Column 0 broadcast over the columns.
        cb_wait_front(cb_tmp, 1);
        tile_regs_acquire();
        reconfig_data_format_srcb(cb_tmp);
        UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
            false, false, cb_tmp)));
        MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, BroadcastType::COL>(
            cb_tmp)));
        unary_bcast<BroadcastType::COL>(cb_tmp, /* tile_idx */ 0, /* dst */ 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_pop_front(cb_tmp, 1);
#else
        cb_wait_front(cb_max, 1);
        for (uint32_t k = 0, b = 0; k < nk; k += block, ++b) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_wait_front(cb_in, block);
            // exp(x - max) at fp32 into the exp CB.
            tile_regs_acquire();
            reconfig_data_format(cb_in, cb_max);
            sub_bcast_cols_init(cb_in, cb_max);
            for (uint32_t j = 0; j < n; ++j) {
                sub_tiles_bcast_cols(cb_in, cb_max, j, 0, j);
            }
            sdpa_exp_tile_init();
            for (uint32_t j = 0; j < n; ++j) {
                sdpa_exp_tile(j);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_exp, block);
            pack_reconfig_data_format(cb_exp);
            pack_reconfig_l1_acc(false);
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_exp);
            }
            tile_regs_release();
            cb_push_back(cb_exp, block);
            cb_pop_front(cb_in, block);
            // The block's row sums: matmul with the ones column (in1 bf16 through SrcA, in0 fp32 through SrcB),
            // L1-accumulated onto the output tile from the second block on.
            cb_wait_front(cb_exp, block);
            tile_regs_acquire();
            reconfig_data_format(/* srca */ cb_ones, /* srcb */ cb_exp);
            matmul_init(cb_exp, cb_ones, /* transpose */ 0);
            for (uint32_t j = 0; j < n; ++j) {
                matmul_tiles(cb_exp, cb_ones, j, 0, /* dst */ 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_reconfig_l1_acc(b > 0U);
            pack_tile</* out_of_order */ true>(0, cb_out, /* offset */ 0);
            pack_reconfig_l1_acc(false);
            tile_regs_release();
            cb_pop_front(cb_exp, block);
        }
        cb_pop_front(cb_max, 1);
#endif
        cb_push_back(cb_out, 1);
    }
}
"""
)
for p, c in files.items():
    open(p, "w").write(c)


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:80]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


sub(
    f"{ROOT}/CMakeLists.txt",
    "    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/vp_cross_entropy_bw/device/vp_cross_entropy_bw_program_factory.cpp\n",
    "    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/vp_cross_entropy_bw/device/vp_cross_entropy_bw_program_factory.cpp\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/vp_ce_row_stats/vp_ce_row_stats.cpp\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/vp_ce_row_stats/device/vp_ce_row_stats_device_operation.cpp\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/metal/ops/vp_ce_row_stats/device/vp_ce_row_stats_program_factory.cpp\n",
)
sub(
    f"{ROOT}/metal/operations.hpp",
    '#include "ops/vp_cross_entropy_bw/vp_cross_entropy_bw.hpp"\n',
    '#include "ops/vp_ce_row_stats/vp_ce_row_stats.hpp"\n#include "ops/vp_cross_entropy_bw/vp_cross_entropy_bw.hpp"\n',
)
sub(
    f"{ROOT}/nanobind/nb_ops.cpp",
    '#include "metal/ops/vp_cross_entropy_bw/vp_cross_entropy_bw.hpp"\n',
    '#include "metal/ops/vp_ce_row_stats/vp_ce_row_stats.hpp"\n#include "metal/ops/vp_cross_entropy_bw/vp_cross_entropy_bw.hpp"\n',
)
sub(
    f"{ROOT}/nanobind/nb_ops.cpp",
    """        py_metal.def(
            "swiglu_packed_bw",
            &ttml::metal::swiglu_packed_bw,""",
    """        py_metal.def(
            "vp_ce_row_stats",
            &ttml::metal::vp_ce_row_stats,
            nb::arg("logits"),
            nb::arg("mode"),
            nb::arg("max_col") = nb::none(),
            "Row statistics of a [B, 1, S, V] bf16 shard: mode 0 the per-slice row max (broadcast over the tile "
            "columns), mode 1 the per-slice row sum of exp(x - max_col); [B, 1, S, slices * 32] fp32, reduce it "
            "with max / sum over the last dim.");
        py_metal.def(
            "swiglu_packed_bw",
            &ttml::metal::swiglu_packed_bw,""",
)
# losses.cpp: the forward stats through the new op
L = f"{ROOT}/ops/distributed/losses.cpp"
sub(
    L,
    '#include "metal/ops/vp_cross_entropy_bw/device/vp_cross_entropy_bw_device_operation.hpp"\n',
    '#include "metal/ops/vp_ce_row_stats/vp_ce_row_stats.hpp"\n#include "metal/ops/vp_cross_entropy_bw/device/vp_cross_entropy_bw_device_operation.hpp"\n',
)
sub(
    L,
    """    // Step 1: local max [B,1,S,1] per device (BF16 — no precision loss)
    auto local_max = ttnn::max(logits->get_value(), 3, /* keepdim */ true);
""",
    """    // Step 1: local max [B,1,S,1] per device: one pass over the shard (iteration 32: `vp_ce_row_stats`, per-slice
    // partials broadcast over the tile columns, max over them). fp32 holding bf16 values exactly.
    auto local_max = ttnn::max(ttml::metal::vp_ce_row_stats(logits->get_value(), 0U), 3, /* keepdim */ true);
""",
)
sub(
    L,
    """    // Step 3: fused (logits - global_max).exp() into FP32 — single binary_ng kernel.
    auto local_exp = fused_subtract_exp_fp32(logits->get_value(), global_max);
    auto local_sum = ttnn::sum(local_exp, 3, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    local_exp.deallocate();
""",
    """    // Step 3: the row sums of exp(logits - global_max) in one pass over the shard (iteration 32: `vp_ce_row_stats`
    // mode 1, fp32 exp on the SFPU, per-slice partial sums in column 0, summed); the 128 MB fp32 exp tensor of the
    // sub-exp + sum path is gone.
    auto local_sum = ttnn::sum(
        ttml::metal::vp_ce_row_stats(logits->get_value(), 1U, global_max),
        3,
        /* keepdim */ true,
        std::nullopt,
        core::ComputeKernelConfig::precise());
""",
)
sub(
    L,
    """    // log_normalizer = global_max + log(global_sum)  [B,1,S,1]
    // global_max is BF16; log(global_sum) is FP32 — output_dtype promotes the add.
""",
    """    // log_normalizer = global_max + log(global_sum)  [B,1,S,1]
    // global_max is FP32 (bf16 values); log(global_sum) is FP32.
""",
)
print("iteration 32 files written")
