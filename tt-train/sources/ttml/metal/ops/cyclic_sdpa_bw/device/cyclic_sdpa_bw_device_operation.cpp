// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "core/tt_tensor_utils.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::cyclic_sdpa_bw::device {

namespace {

constexpr uint32_t kTile = 32U;

void check_on_device(const ttnn::Tensor& tensor, const std::string& name) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "cyclic_sdpa_bw requires {} on device; storage type is {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "cyclic_sdpa_bw: {} has no buffer allocated", name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "cyclic_sdpa_bw requires {} in tile layout; layout is {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "cyclic_sdpa_bw requires {} interleaved; memory layout is {}",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
}

}  // namespace

void CyclicSDPABackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& query = tensor_args.query;
    check_on_device(query, "query");
    check_on_device(tensor_args.key, "key");
    check_on_device(tensor_args.value, "value");
    check_on_device(tensor_args.grad_output, "grad_output");
    check_on_device(tensor_args.log_sum_exp, "log_sum_exp");
    check_on_device(tensor_args.row_scalar, "row_scalar");

    for (const auto* t : {&tensor_args.query, &tensor_args.key, &tensor_args.value, &tensor_args.grad_output}) {
        TT_FATAL(
            t->dtype() == tt::tt_metal::DataType::BFLOAT16,
            "cyclic_sdpa_bw takes bfloat16 operands: the matmul source registers do not accept "
            "Float32. Got {}",
            enchantum::to_string(t->dtype()));
    }
    for (const auto* t : {&tensor_args.log_sum_exp, &tensor_args.row_scalar}) {
        TT_FATAL(
            t->dtype() == tt::tt_metal::DataType::FLOAT32,
            "cyclic_sdpa_bw takes Float32 statistics, which carry running sums. Got {}",
            enchantum::to_string(t->dtype()));
    }

    const auto shape = query.logical_shape();
    TT_FATAL(shape.rank() == 4U, "cyclic_sdpa_bw takes rank-4 tensors (batch, head, sequence, head dim)");
    const uint32_t N = static_cast<uint32_t>(shape[2]);
    const uint32_t d = static_cast<uint32_t>(shape[3]);
    TT_FATAL(d % kTile == 0U, "cyclic_sdpa_bw needs a head dimension that is a multiple of {}; got {}", kTile, d);
    for (const auto* t : {&tensor_args.key, &tensor_args.value, &tensor_args.grad_output}) {
        TT_FATAL(
            t->logical_shape() == shape,
            "cyclic_sdpa_bw takes query, key, value and grad_output of one shape; {} differs",
            enchantum::to_string(t->dtype()));
    }

    // The layout planner carries the rest of the constraints -- that the
    // sequence length divides into whole cores, that a rectangle of that area
    // embeds the parity snake, and that the slices fit the grid -- and reports
    // each with the arithmetic that produced it.
    (void)plan_layout(
        query.device()->compute_with_storage_grid_size(),
        N,
        args.rows_per_block_tiles,
        static_cast<uint32_t>(shape[0]) * static_cast<uint32_t>(shape[1]));
}

CyclicSDPABackwardDeviceOperation::spec_return_value_t CyclicSDPABackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto make_spec = [&](const std::optional<ttnn::Tensor>& preallocated) {
        if (preallocated.has_value()) {
            return preallocated->tensor_spec();
        }
        return tt::tt_metal::TensorSpec(
            tensor_args.query.logical_shape(),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{}));
    };
    return {
        make_spec(tensor_args.preallocated_grad_query),
        make_spec(tensor_args.preallocated_grad_key),
        make_spec(tensor_args.preallocated_grad_value)};
}

CyclicSDPABackwardDeviceOperation::tensor_return_value_t CyclicSDPABackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.query.device();
    // Zeroed, not merely allocated. Every gradient here is accumulated into
    // and *read back*: dQ is seeded from memory at each streak start, and the
    // column gradients are re-read when a core revisits a column. An
    // uninitialised buffer therefore does not produce noise, it produces a
    // sum with whatever was there -- which, on a device that has just run the
    // same shapes, is the right answer added to itself. That is exactly the
    // 2x this op returned before the zeroing was added.
    const auto take = [&](const std::optional<ttnn::Tensor>& preallocated, size_t i) {
        if (preallocated.has_value()) {
            return preallocated.value();
        }
        return ttml::core::zeros(
            ttnn::Shape(specs[i].logical_shape()), device, specs[i].data_type());
    };
    auto grad_query = take(tensor_args.preallocated_grad_query, 0U);
    auto grad_key = take(tensor_args.preallocated_grad_key, 1U);
    auto grad_value = take(tensor_args.preallocated_grad_value, 2U);
    return {grad_query, grad_key, grad_value};
}

ttsl::hash::hash_t CyclicSDPABackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<CyclicSDPABackwardDeviceOperation>(
        args, tensor_args.query.dtype(), tensor_args.query.logical_shape());
}

}  // namespace ttml::metal::ops::cyclic_sdpa_bw::device

namespace ttnn::prim {

ttml::metal::ops::cyclic_sdpa_bw::device::CyclicSDPABackwardDeviceOperation::tensor_return_value_t
ttml_cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t rows_per_block_tiles,
    bool use_barrier,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value) {
    using OperationType = ttml::metal::ops::cyclic_sdpa_bw::device::CyclicSDPABackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .rows_per_block_tiles = rows_per_block_tiles, .use_barrier = use_barrier};
    auto tensor_args = OperationType::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .grad_output = grad_output,
        .log_sum_exp = log_sum_exp,
        .row_scalar = row_scalar,
        .preallocated_grad_query = preallocated_grad_query,
        .preallocated_grad_key = preallocated_grad_key,
        .preallocated_grad_value = preallocated_grad_value,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
