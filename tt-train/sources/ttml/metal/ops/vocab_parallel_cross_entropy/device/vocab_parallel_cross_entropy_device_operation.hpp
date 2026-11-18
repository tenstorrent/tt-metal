// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "metal/ttnn_all_includes.hpp"
#include "vocab_parallel_cross_entropy_device_operation_types.hpp"
#include "vocab_parallel_cross_entropy_program_factory.hpp"

namespace ttml::metal::ops::vocab_parallel_cross_entropy::device {

struct VocabParallelCrossEntropyDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::vocab_parallel_cross_entropy::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::vocab_parallel_cross_entropy::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::vocab_parallel_cross_entropy::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::vocab_parallel_cross_entropy::device::tensor_return_value_t;
    using program_factory_t = std::variant<VocabParallelCrossEntropyProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::vocab_parallel_cross_entropy::device

namespace ttnn::prim {

// Per shard and row, one FP32 tile row of (m, global first-maximum column, tail without it, target logit or 0).
ttnn::Tensor ttml_vocab_parallel_ce_stats(
    const ttnn::Tensor& logits, const ttnn::Tensor& target, std::optional<uint32_t> cluster_axis);

// {loss, row state} from the all-gathered shard records.
std::vector<ttnn::Tensor> ttml_vocab_parallel_ce_combine(const ttnn::Tensor& gathered_stats);

// The input gradient of every shard, scaled by `scaler` and, when given, the upstream gradient.
ttnn::Tensor ttml_vocab_parallel_ce_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& target,
    const ttnn::Tensor& state,
    const std::optional<ttnn::Tensor>& upstream,
    float scaler,
    std::optional<uint32_t> cluster_axis);

}  // namespace ttnn::prim
