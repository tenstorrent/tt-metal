// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vp_ce_row_stats_device_operation.hpp"

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
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            logits.memory_config()));
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
