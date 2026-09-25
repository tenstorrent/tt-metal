// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cyclic_sdpa_bw_device_operation.hpp"

#include <algorithm>

#include <enchantum/enchantum.hpp>

#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/full/full.hpp"
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

    if (args.accumulate_into_outputs) {
        TT_FATAL(
            tensor_args.preallocated_grad_query.has_value() && tensor_args.preallocated_grad_key.has_value() &&
                tensor_args.preallocated_grad_value.has_value(),
            "cyclic_sdpa_bw: accumulate_into_outputs needs all three gradients preallocated, since they are "
            "what is accumulated into");
    }
    TT_FATAL(
        args.mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "cyclic_sdpa_bw has no mask-tensor path: use Causal for the triangle or None for a full block");

    const auto shape = query.logical_shape();
    TT_FATAL(shape.rank() == 4U, "cyclic_sdpa_bw takes rank-4 tensors (batch, head, sequence, head dim)");
    const uint32_t N = static_cast<uint32_t>(shape[2]);
    const uint32_t d = static_cast<uint32_t>(shape[3]);
    TT_FATAL(d % kTile == 0U, "cyclic_sdpa_bw needs a head dimension that is a multiple of {}; got {}", kTile, d);
    TT_FATAL(
        tensor_args.grad_output.logical_shape() == shape,
        "cyclic_sdpa_bw takes grad_output of the query's shape {}; got {}",
        shape,
        tensor_args.grad_output.logical_shape());
    // Grouped-query attention: key and value may carry fewer heads than the
    // query, a divisor of its head count; the heads of a group share one key
    // head and add into one dK and one dV. Everything else about the shape
    // must agree.
    const auto key_shape = tensor_args.key.logical_shape();
    TT_FATAL(
        tensor_args.value.logical_shape() == key_shape,
        "cyclic_sdpa_bw takes key and value of one shape; got key {} and value {}",
        key_shape,
        tensor_args.value.logical_shape());
    TT_FATAL(
        key_shape.rank() == 4U && key_shape[0] == shape[0] && key_shape[2] == shape[2] && key_shape[3] == shape[3],
        "cyclic_sdpa_bw takes key and value of the query's batch, sequence length and head dimension; query is "
        "{}, key is {}",
        shape,
        key_shape);
    const uint32_t q_heads = static_cast<uint32_t>(shape[1]);
    const uint32_t kv_heads = static_cast<uint32_t>(key_shape[1]);
    TT_FATAL(
        kv_heads >= 1U && q_heads % kv_heads == 0U,
        "cyclic_sdpa_bw: the {} query heads must be a multiple of the {} key/value heads (grouped-query "
        "attention shares one key head among a whole number of query heads)",
        q_heads,
        kv_heads);
    if (tensor_args.preallocated_grad_query.has_value()) {
        TT_FATAL(
            tensor_args.preallocated_grad_query->logical_shape() == shape,
            "cyclic_sdpa_bw: preallocated grad_query {} must have the query's shape {}",
            tensor_args.preallocated_grad_query->logical_shape(),
            shape);
    }
    for (const auto* t : {&tensor_args.preallocated_grad_key, &tensor_args.preallocated_grad_value}) {
        if (t->has_value()) {
            TT_FATAL(
                (*t)->logical_shape() == key_shape,
                "cyclic_sdpa_bw: preallocated grad_key and grad_value {} must have the key's shape {}",
                (*t)->logical_shape(),
                key_shape);
        }
    }

    const uint32_t chunks = args.sequence_chunks;
    TT_FATAL(chunks >= 1U, "cyclic_sdpa_bw: sequence_chunks must be at least 1, got {}", chunks);
    TT_FATAL(
        N % chunks == 0U,
        "cyclic_sdpa_bw: the sequence of {} rows must split into {} equal chunks",
        N,
        chunks);
    TT_FATAL(
        args.row_chunks.size() == args.col_chunks.size(),
        "cyclic_sdpa_bw: row_chunks and col_chunks name the sub-problems pairwise and must have the same "
        "length; got {} and {}",
        args.row_chunks.size(),
        args.col_chunks.size());
    for (size_t p = 0; p < args.row_chunks.size(); ++p) {
        TT_FATAL(
            args.row_chunks[p] < chunks && args.col_chunks[p] < chunks,
            "cyclic_sdpa_bw: sub-problem {} names chunks ({}, {}) of {}",
            p,
            args.row_chunks[p],
            args.col_chunks[p],
            chunks);
    }
    // Two sub-problems of one launch must not share a chunk on either side.
    // They run as independent slices, side by side on different groups or
    // one after the other on the same one, and nothing orders one slice's
    // final dQ spills or dK, dV write-backs before another slice's reads of
    // the same rows: the two would race, or the second would start from the
    // first's result where the caller expected a sum. The caller issues such
    // pairs as separate launches instead.
    for (size_t p = 0; p < args.row_chunks.size(); ++p) {
        for (size_t q = p + 1; q < args.row_chunks.size(); ++q) {
            TT_FATAL(
                args.row_chunks[p] != args.row_chunks[q] && args.col_chunks[p] != args.col_chunks[q],
                "cyclic_sdpa_bw: sub-problems {} ({}, {}) and {} ({}, {}) share a chunk; slices of one launch "
                "run independently and would race on it. Issue them as separate launches.",
                p,
                args.row_chunks[p],
                args.col_chunks[p],
                q,
                args.row_chunks[q],
                args.col_chunks[q]);
        }
    }
    const auto pairs = static_cast<uint32_t>(std::max<size_t>(1, args.row_chunks.size()));

    // The layout planner carries the rest of the constraints -- that the
    // chunk length divides into whole cores, that a rectangle of that area
    // embeds the parity snake, and that the slices fit the grid -- and reports
    // each with the arithmetic that produced it.
    (void)plan_layout(
        query.device()->compute_with_storage_grid_size(),
        N / chunks,
        args.rows_per_block_tiles,
        static_cast<uint32_t>(shape[0]) * static_cast<uint32_t>(shape[1]) * pairs,
        args.max_groups);
}

CyclicSDPABackwardDeviceOperation::spec_return_value_t CyclicSDPABackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
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
        // Filled on the device: core::zeros builds the tensor on the host
        // and writes it over PCIe, which at 7040 rows was 0.8 ms of a 1.8 ms
        // call (the ring driver's device_zeros_like, for the same reason).
        const auto& shape = specs[i].logical_shape();
        return ttnn::moreh_full(
            ttsl::SmallVector<uint32_t>(shape.cbegin(), shape.cend()), 0.0F, device, specs[i].data_type(),
            ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
    };
    auto grad_query = take(tensor_args.preallocated_grad_query, 0U);
    auto grad_key = take(tensor_args.preallocated_grad_key, 1U);
    auto grad_value = take(tensor_args.preallocated_grad_value, 2U);
    return {grad_query, grad_key, grad_value};
}

ttsl::hash::hash_t CyclicSDPABackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // args carries rows_per_block_tiles, the mask type and the barrier flag,
    // all of which change the compiled kernels, so they must be in the hash.
    // The key's shape too: its head count sets the grouped-query decode in
    // the runtime arguments and the seeded column path in the kernels.
    return tt::tt_metal::operation::hash_operation<CyclicSDPABackwardDeviceOperation>(
        args, tensor_args.query.dtype(), tensor_args.query.logical_shape(), tensor_args.key.logical_shape());
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
    ttml::metal::AttentionMaskType mask_type,
    bool accumulate_into_outputs,
    const std::optional<ttnn::Tensor>& preallocated_grad_query,
    const std::optional<ttnn::Tensor>& preallocated_grad_key,
    const std::optional<ttnn::Tensor>& preallocated_grad_value,
    uint32_t max_groups,
    uint32_t sequence_chunks,
    const std::vector<uint32_t>& row_chunks,
    const std::vector<uint32_t>& col_chunks,
    bool grad_query_in_tile_transposed,
    bool grad_query_out_tile_transposed) {
    using OperationType = ttml::metal::ops::cyclic_sdpa_bw::device::CyclicSDPABackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .rows_per_block_tiles = rows_per_block_tiles,
        .mask_type = mask_type,
        .use_barrier = use_barrier,
        .accumulate_into_outputs = accumulate_into_outputs,
        .max_groups = max_groups,
        .sequence_chunks = sequence_chunks,
        .row_chunks = row_chunks,
        .col_chunks = col_chunks,
        .grad_query_in_tile_transposed = grad_query_in_tile_transposed,
        .grad_query_out_tile_transposed = grad_query_out_tile_transposed};
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
