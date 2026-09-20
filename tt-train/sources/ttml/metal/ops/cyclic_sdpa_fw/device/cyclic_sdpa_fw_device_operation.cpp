// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_fw_device_operation.hpp"

#include <algorithm>

#include <enchantum/enchantum.hpp>

#include "core/tt_tensor_utils.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::cyclic_sdpa_fw::device {

namespace {

constexpr uint32_t kTile = 32U;

void check_on_device(const ttnn::Tensor& tensor, const std::string& name) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "cyclic_sdpa_fw requires {} on device; storage type is {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "cyclic_sdpa_fw: {} has no buffer allocated", name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "cyclic_sdpa_fw requires {} in tile layout; layout is {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "cyclic_sdpa_fw requires {} interleaved; memory layout is {}",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
}

}  // namespace

void CyclicSDPAForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& query = tensor_args.query;
    check_on_device(query, "query");
    check_on_device(tensor_args.key, "key");
    check_on_device(tensor_args.value, "value");
    for (const auto* t : {&tensor_args.query, &tensor_args.key, &tensor_args.value}) {
        TT_FATAL(
            t->dtype() == tt::tt_metal::DataType::BFLOAT16,
            "cyclic_sdpa_fw takes bfloat16 operands: the matmul source registers do not accept Float32. Got {}",
            enchantum::to_string(t->dtype()));
    }
    TT_FATAL(
        args.mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "cyclic_sdpa_fw has no mask-tensor path: use Causal for the triangle or None for a full block");

    const auto shape = query.logical_shape();
    TT_FATAL(shape.rank() == 4U, "cyclic_sdpa_fw takes rank-4 tensors (batch, head, sequence, head dim)");
    const uint32_t N = static_cast<uint32_t>(shape[2]);
    const uint32_t d = static_cast<uint32_t>(shape[3]);
    TT_FATAL(d % kTile == 0U, "cyclic_sdpa_fw needs a head dimension that is a multiple of {}; got {}", kTile, d);
    const auto key_shape = tensor_args.key.logical_shape();
    TT_FATAL(
        tensor_args.value.logical_shape() == key_shape,
        "cyclic_sdpa_fw takes key and value of one shape; got key {} and value {}",
        key_shape,
        tensor_args.value.logical_shape());
    TT_FATAL(
        key_shape.rank() == 4U && key_shape[0] == shape[0] && key_shape[2] == shape[2] && key_shape[3] == shape[3],
        "cyclic_sdpa_fw takes key and value of the query's batch, sequence length and head dimension; query is "
        "{}, key is {}",
        shape,
        key_shape);
    const uint32_t q_heads = static_cast<uint32_t>(shape[1]);
    const uint32_t kv_heads = static_cast<uint32_t>(key_shape[1]);
    TT_FATAL(
        kv_heads >= 1U && q_heads % kv_heads == 0U,
        "cyclic_sdpa_fw: the {} query heads must be a multiple of the {} key/value heads",
        q_heads,
        kv_heads);
    if (tensor_args.preallocated_output.has_value()) {
        TT_FATAL(
            tensor_args.preallocated_output->logical_shape() == shape &&
                tensor_args.preallocated_output->dtype() == tt::tt_metal::DataType::BFLOAT16,
            "cyclic_sdpa_fw: the preallocated output must have the query's shape {} in bfloat16; got {} {}",
            shape,
            tensor_args.preallocated_output->logical_shape(),
            enchantum::to_string(tensor_args.preallocated_output->dtype()));
    }
    if (tensor_args.preallocated_intermediates.has_value()) {
        const ttnn::Shape want{shape[0], shape[1], shape[2], kTile};
        TT_FATAL(
            tensor_args.preallocated_intermediates->logical_shape() == want &&
                tensor_args.preallocated_intermediates->dtype() == tt::tt_metal::DataType::FLOAT32,
            "cyclic_sdpa_fw: the preallocated intermediates must be {} Float32; got {} {}",
            want,
            tensor_args.preallocated_intermediates->logical_shape(),
            enchantum::to_string(tensor_args.preallocated_intermediates->dtype()));
    }

    const uint32_t chunks = args.sequence_chunks;
    TT_FATAL(chunks >= 1U, "cyclic_sdpa_fw: sequence_chunks must be at least 1, got {}", chunks);
    TT_FATAL(N % chunks == 0U, "cyclic_sdpa_fw: the sequence of {} rows must split into {} equal chunks", N, chunks);
    TT_FATAL(
        args.row_chunks.size() == args.col_chunks.size(),
        "cyclic_sdpa_fw: row_chunks and col_chunks must have the same length; got {} and {}",
        args.row_chunks.size(),
        args.col_chunks.size());
    for (size_t p = 0; p < args.row_chunks.size(); ++p) {
        TT_FATAL(
            args.row_chunks[p] < chunks && args.col_chunks[p] < chunks,
            "cyclic_sdpa_fw: sub-problem {} names chunks ({}, {}) of {}",
            p,
            args.row_chunks[p],
            args.col_chunks[p],
            chunks);
        for (size_t q = p + 1; q < args.row_chunks.size(); ++q) {
            // Two sub-problems writing the same rows would each finish them
            // separately; the caller merges such partials itself (as a ring
            // step does) from separate launches.
            TT_FATAL(
                args.row_chunks[p] != args.row_chunks[q],
                "cyclic_sdpa_fw: sub-problems {} and {} share row chunk {}; each row chunk's output can be "
                "finished by one sub-problem per launch",
                p,
                q,
                args.row_chunks[p]);
        }
    }
    const auto pairs = static_cast<uint32_t>(std::max<size_t>(1, args.row_chunks.size()));
    (void)plan_layout(
        query.device()->compute_with_storage_grid_size(),
        N / chunks,
        args.rows_per_block_tiles,
        static_cast<uint32_t>(shape[0]) * q_heads * pairs,
        args.max_groups);
}

CyclicSDPAForwardDeviceOperation::spec_return_value_t CyclicSDPAForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto shape = tensor_args.query.logical_shape();
    const auto spec = [](const ttnn::Shape& s, tt::tt_metal::DataType dt) {
        return tt::tt_metal::TensorSpec(
            s,
            tt::tt_metal::TensorLayout(
                dt, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));
    };
    return {
        tensor_args.preallocated_output.has_value() ? tensor_args.preallocated_output->tensor_spec()
                                                    : spec(shape, tt::tt_metal::DataType::BFLOAT16),
        tensor_args.preallocated_intermediates.has_value()
            ? tensor_args.preallocated_intermediates->tensor_spec()
            : spec(ttnn::Shape{shape[0], shape[1], shape[2], kTile}, tt::tt_metal::DataType::FLOAT32),
        spec(shape, tt::tt_metal::DataType::FLOAT32),
        spec(ttnn::Shape{shape[0], shape[1], shape[2], 2U * kTile}, tt::tt_metal::DataType::FLOAT32)};
}

CyclicSDPAForwardDeviceOperation::tensor_return_value_t CyclicSDPAForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.query.device();
    // Nothing here is read before it is written: the state accumulator is
    // read only at a later streak start, after that row's spill, and the
    // outputs are written once per row at its last visit. Allocation only.
    const auto take = [&](const std::optional<ttnn::Tensor>& preallocated, size_t i) {
        if (preallocated.has_value()) {
            return preallocated.value();
        }
        return ttnn::create_device_tensor(specs[i], device);
    };
    auto output = take(tensor_args.preallocated_output, 0U);
    auto intermediates = take(tensor_args.preallocated_intermediates, 1U);
    auto accumulator = ttnn::create_device_tensor(specs[2], device);
    auto statistics = ttnn::create_device_tensor(specs[3], device);
    return {output, intermediates, accumulator, statistics};
}

ttsl::hash::hash_t CyclicSDPAForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<CyclicSDPAForwardDeviceOperation>(
        args, tensor_args.query.dtype(), tensor_args.query.logical_shape(), tensor_args.key.logical_shape());
}

}  // namespace ttml::metal::ops::cyclic_sdpa_fw::device

namespace ttnn::prim {

ttml::metal::ops::cyclic_sdpa_fw::device::CyclicSDPAForwardDeviceOperation::tensor_return_value_t
ttml_cyclic_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t rows_per_block_tiles,
    ttml::metal::AttentionMaskType mask_type,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates,
    uint32_t max_groups,
    uint32_t sequence_chunks,
    const std::vector<uint32_t>& row_chunks,
    const std::vector<uint32_t>& col_chunks) {
    using OperationType = ttml::metal::ops::cyclic_sdpa_fw::device::CyclicSDPAForwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .rows_per_block_tiles = rows_per_block_tiles,
        .mask_type = mask_type,
        .max_groups = max_groups,
        .sequence_chunks = sequence_chunks,
        .row_chunks = row_chunks,
        .col_chunks = col_chunks};
    auto tensor_args = OperationType::tensor_args_t{
        .query = query,
        .key = key,
        .value = value,
        .preallocated_output = preallocated_output,
        .preallocated_intermediates = preallocated_intermediates,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
