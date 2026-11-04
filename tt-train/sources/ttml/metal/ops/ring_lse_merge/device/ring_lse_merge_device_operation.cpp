// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_lse_merge_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ring_lse_merge_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::ring_lse_merge::device {

namespace {

constexpr std::string_view kOpName = "RingLseMerge";

// The reader and writer bake each tensor's buffer type into their TensorAccessor compile-time args, so every tensor
// must be interleaved DRAM: a cached program built for one buffer type would otherwise address another one wrongly.
DeviceTensorRequirements dram_tile(tt::tt_metal::DataType dtype) {
    return {.dtypes = {dtype}, .buffer_type = tt::tt_metal::BufferType::DRAM};
}

}  // namespace

void RingLseMergeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& accum = tensor_args.output_accum;
    const auto& running_lse = tensor_args.running_lse;
    const auto& step_output = tensor_args.step_output;
    const auto& step_intermediates = tensor_args.step_intermediates;

    check_device_tensor(accum, kOpName, "output_accum", dram_tile(tt::tt_metal::DataType::FLOAT32));
    check_device_tensor(running_lse, kOpName, "running_lse", dram_tile(tt::tt_metal::DataType::FLOAT32));
    check_device_tensor(step_output, kOpName, "step_output", dram_tile(tt::tt_metal::DataType::BFLOAT16));
    check_device_tensor(step_intermediates, kOpName, "step_intermediates", dram_tile(tt::tt_metal::DataType::FLOAT32));

    check_same_device(running_lse, accum, kOpName, "running_lse", "output_accum");
    check_same_device(step_output, accum, kOpName, "step_output", "output_accum");
    check_same_device(step_intermediates, accum, kOpName, "step_intermediates", "output_accum");

    TT_FATAL(
        accum.logical_shape().rank() == 4U,
        "{}: output_accum must be 4D (B, H, S, D), got {}",
        kOpName,
        accum.logical_shape());
    check_tile_padding(accum, kOpName, "output_accum");
    check_shape(step_output, kOpName, "step_output", accum.logical_shape());

    // Both logsumexp tensors use the SDPA intermediates layout: one FP32 tile column per (b, h, tile row).
    const auto [batch, heads, seq_len, dim] = accum.logical_shape().to_array_4D();
    const ttnn::Shape lse_shape{batch, heads, seq_len, tt::constants::TILE_WIDTH};
    check_shape(running_lse, kOpName, "running_lse", lse_shape);
    check_shape(step_intermediates, kOpName, "step_intermediates", lse_shape);

    TT_FATAL(
        running_lse.buffer() != step_intermediates.buffer(),
        "{}: running_lse must not alias step_intermediates",
        kOpName);
}

spec_return_value_t RingLseMergeDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.output_accum.tensor_spec(), tensor_args.running_lse.tensor_spec()};
}

tensor_return_value_t RingLseMergeDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.output_accum, tensor_args.running_lse};
}

std::vector<tt::tt_metal::TensorTopology> RingLseMergeDeviceOperation::compute_output_topologies(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return {tensor_args.output_accum.tensor_topology(), tensor_args.running_lse.tensor_topology()};
}

ttsl::hash::hash_t RingLseMergeDeviceOperation::compute_program_hash(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // Everything that reaches a compile-time arg or the work split: dtypes (CB formats), padded shapes (rows, Wt,
    // per-core row counts) and memory configs (the buffer type and memory layout baked into the TensorAccessor args).
    // Buffer addresses are runtime args, patched by override_runtime_arguments on a cache hit.
    const auto key = [](const ttnn::Tensor& t) {
        return std::make_tuple(t.dtype(), t.padded_shape(), t.memory_config());
    };
    return tt::tt_metal::operation::hash_operation<RingLseMergeDeviceOperation>(
        key(tensor_args.output_accum),
        key(tensor_args.running_lse),
        key(tensor_args.step_output),
        key(tensor_args.step_intermediates));
}

}  // namespace ttml::metal::ops::ring_lse_merge::device

namespace ttnn::prim {

ttml::metal::ops::ring_lse_merge::device::RingLseMergeDeviceOperation::tensor_return_value_t ttml_ring_lse_merge(
    const ttnn::Tensor& output_accum,
    const ttnn::Tensor& running_lse,
    const ttnn::Tensor& step_output,
    const ttnn::Tensor& step_intermediates) {
    using OperationType = ttml::metal::ops::ring_lse_merge::device::RingLseMergeDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{
        .output_accum = output_accum,
        .running_lse = running_lse,
        .step_output = step_output,
        .step_intermediates = step_intermediates,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
