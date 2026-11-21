// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vocab_parallel_cross_entropy.hpp"

#include "device/vocab_parallel_cross_entropy_device_operation.hpp"
#include "metal/ops/log_softmax/device/row_softmax_program.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttml::metal {

ttnn::Tensor vocab_parallel_ce_stats(
    const ttnn::Tensor& logits, const ttnn::Tensor& target, std::optional<uint32_t> cluster_axis) {
    return ttnn::prim::ttml_vocab_parallel_ce_stats(logits, target, cluster_axis);
}

std::array<ttnn::Tensor, 2> vocab_parallel_ce_combine(const ttnn::Tensor& gathered_stats) {
    auto result = ttnn::prim::ttml_vocab_parallel_ce_combine(gathered_stats);
    return {result[0], result[1]};
}

ttnn::Tensor vocab_parallel_ce_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& target,
    const ttnn::Tensor& state,
    const ttnn::Tensor& upstream,
    float scaler,
    std::optional<uint32_t> cluster_axis) {
    if (ops::row_softmax::can_fuse_upstream(logits, upstream)) {
        return ttnn::prim::ttml_vocab_parallel_ce_bw(logits, target, state, upstream, scaler, cluster_axis);
    }
    return ttnn::multiply(
        ttnn::prim::ttml_vocab_parallel_ce_bw(logits, target, state, std::nullopt, scaler, cluster_axis), upstream);
}

}  // namespace ttml::metal
