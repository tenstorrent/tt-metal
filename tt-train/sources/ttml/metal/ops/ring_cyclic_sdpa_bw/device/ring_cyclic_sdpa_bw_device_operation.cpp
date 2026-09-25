// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_cyclic_sdpa_bw_device_operation.hpp"

#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/full/full.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_bw {

RingCyclicSDPABackwardDeviceOperation::program_factory_t
RingCyclicSDPABackwardDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RingCyclicSDPABackwardProgramFactory{};
}

namespace {
// The cyclic kernels read one Float32 tile per 32 query rows and take the
// statistic from its column 0, so a (B, H, S, 1) tensor serves as it is --
// the tiled layout is the same -- and so does the (B, H, S, 32) padded one
// the two-pass kernels want.
void validate_statistic_tensor(const ttnn::Tensor& statistic, const ttnn::Tensor& query) {
    TT_FATAL(
        statistic.dtype() == ttnn::DataType::FLOAT32, "Statistics must be FLOAT32, got {}", statistic.dtype());
    TT_FATAL(statistic.layout() == ttnn::Layout::TILE, "Statistics must be TILE layout, got {}", statistic.layout());
    const auto [batch, heads, seq_len, dim] = query.logical_shape().to_array_4D();
    const auto [sb, sh, ss, sw] = statistic.logical_shape().to_array_4D();
    TT_FATAL(
        sb == batch && sh == heads && ss == seq_len && (sw == 1U || sw == tt::constants::TILE_WIDTH),
        "Statistic shape {} must be ({}, {}, {}, 1 or 32): one value per query row in column 0 of a tile",
        statistic.logical_shape(),
        batch,
        heads,
        seq_len);
}
}  // namespace

void RingCyclicSDPABackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    ops::validate_ring_attributes(args, tensor_args.query);
    ops::validate_ring_qkv(tensor_args.query, tensor_args.key, tensor_args.value);
    ops::validate_output_like_tensor(tensor_args.grad_output, "grad_output", tensor_args.query, tensor_args.value);
    validate_statistic_tensor(tensor_args.log_sum_exp, tensor_args.query);
    validate_statistic_tensor(tensor_args.row_scalar, tensor_args.query);
}

void RingCyclicSDPABackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

RingCyclicSDPABackwardDeviceOperation::spec_return_value_t
RingCyclicSDPABackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // dQ has the query's shape, dK and dV the key's: with grouped-query
    // attention that is fewer heads.
    const auto make_spec = [&](const std::optional<ttnn::Tensor>& preallocated, const ttnn::Tensor& like) {
        if (preallocated.has_value()) {
            return preallocated->tensor_spec();
        }
        return tt::tt_metal::TensorSpec(
            like.logical_shape(),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig{}));
    };
    return {
        make_spec(tensor_args.preallocated_grad_query, tensor_args.query),
        make_spec(tensor_args.preallocated_grad_key, tensor_args.key),
        make_spec(tensor_args.preallocated_grad_value, tensor_args.value)};
}

RingCyclicSDPABackwardDeviceOperation::tensor_return_value_t
RingCyclicSDPABackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.query.device();
    // Zeroed, for the same reason the single-chip op zeroes: every gradient
    // here is accumulated into and read back. In the ring that matters twice
    // over, since the caller passes the running accumulators back in at every
    // step, and a chip that the causal schedule skips must leave them alone.
    const auto take = [&](const std::optional<ttnn::Tensor>& preallocated, size_t i) {
        if (preallocated.has_value()) {
            return preallocated.value();
        }
        // Filled on the device: core::zeros builds the tensor on the host
        // and writes it over PCIe, which at 7040 rows was 0.8 ms of a 1.8 ms
        // call (the ring driver's device_zeros_like, for the same reason).
        const auto& shape = specs[i].logical_shape();
        return ttnn::moreh_full(
            ttsl::SmallVector<uint32_t>(shape.cbegin(), shape.cend()), 0.0F, device, specs[i].data_type(),
            ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
    };
    return {
        take(tensor_args.preallocated_grad_query, 0U),
        take(tensor_args.preallocated_grad_key, 1U),
        take(tensor_args.preallocated_grad_value, 2U)};
}

ttsl::hash::hash_t RingCyclicSDPABackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // The step is in the hash because it decides, per chip, whether there is a
    // program at all and which schedule it runs.
    return tt::tt_metal::operation::hash_operation<RingCyclicSDPABackwardDeviceOperation>(
        args, tensor_args.query.dtype(), tensor_args.query.logical_shape(), tensor_args.key.logical_shape());
}

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_bw

namespace ttnn::prim {

ttml::metal::ops::ring_cyclic_sdpa_bw::RingCyclicSDPABackwardDeviceOperation::tensor_return_value_t
ttml_ring_cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_cyclic_sdpa_bw::RingDirection ring_direction,
    uint32_t rows_per_block_tiles,
    bool use_barrier,
    bool accumulate_into_outputs,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value,
    ttml::metal::ops::RingLayout layout,
    uint32_t zigzag_pair,
    bool grad_query_in_tile_transposed,
    bool grad_query_out_tile_transposed) {
    using OperationType = ttml::metal::ops::ring_cyclic_sdpa_bw::RingCyclicSDPABackwardDeviceOperation;

    auto attrs = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .mask_type = mask_type,
        .ring_direction = ring_direction,
        .rows_per_block_tiles = rows_per_block_tiles,
        .use_barrier = use_barrier,
        .accumulate_into_outputs = accumulate_into_outputs,
        .layout = layout,
        .zigzag_pair = zigzag_pair,
        .grad_query_in_tile_transposed = grad_query_in_tile_transposed,
        .grad_query_out_tile_transposed = grad_query_out_tile_transposed};
    auto tensors = OperationType::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .grad_output = grad_output,
        .log_sum_exp = log_sum_exp,
        .row_scalar = row_scalar,
        .preallocated_grad_query = preallocated_grad_query,
        .preallocated_grad_key = preallocated_grad_key,
        .preallocated_grad_value = preallocated_grad_value};

    return ttnn::device_operation::launch<OperationType>(attrs, tensors);
}

}  // namespace ttnn::prim
