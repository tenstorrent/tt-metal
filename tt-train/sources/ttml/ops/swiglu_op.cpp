// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_op.hpp"

#include <cstdlib>
#include <tt_stl/assert.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <variant>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/distributed/distributed.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ops/dropout_op.hpp"
#include "ops/linear_op.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

namespace {
// Zero-copy flatten of all dims except the last into a single leading dim: [B,N,S,D] -> [B*N*S, D]
ttnn::Tensor flatten_leading(const ttnn::Tensor& t) {
    const auto vol = t.logical_volume() / static_cast<uint64_t>(t.logical_shape()[-1]);
    return t.reshape(ttnn::Shape({static_cast<uint32_t>(vol), t.logical_shape()[-1]}));
}

bool force_swiglu_composite() {
    // Benchmark-only override for model-level A/B without changing production call sites.
    return std::getenv("TTML_SWIGLU_FORCE_COMPOSITE") != nullptr;
}

// Whether `w2` (stored [out, in]) is sharded along its input features across a mesh axis of size > 1, as a
// row-parallel weight is. The matmul with it then leaves each device a partial sum still to be added up.
bool is_row_parallel(const ttnn::Tensor& w2) {
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    const auto placements = core::distributed::placements_per_axis(w2);
    const auto mesh_shape = w2.device()->shape();
    const int rank = static_cast<int>(w2.logical_shape().rank());
    for (size_t axis = 0; axis < placements.size(); ++axis) {
        const auto* shard = std::get_if<Shard>(&placements[axis]);
        if (mesh_shape[axis] > 1 && shard != nullptr && (shard->dim < 0 ? shard->dim + rank : shard->dim) == rank - 1) {
            return true;
        }
    }
    return false;
}

}  // namespace

autograd::TensorPtr swiglu_composite(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    float dropout_prob) {
    // Baseline-only reference used by the isolated benchmark A/B table.
    // Keep model/runtime paths on fused swiglu().
    const auto swished = ops::silu(ops::linear_op(tensor, w1, nullptr));
    const auto gate = ops::linear_op(tensor, w3, nullptr);
    const auto x = ops::linear_op(ops::mul(swished, gate), w2, nullptr);
    return ops::dropout(x, dropout_prob);
}

autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3,
    float dropout_prob) {
    if (force_swiglu_composite()) {
        return swiglu_composite(tensor, w1, w2, w3, dropout_prob);
    }
    // Composite forward: weights are [out, in] (LinearLayer convention)
    // Save linear1 and gate for backward (2 tensors vs autograd's 4+).
    // Fuse silu into multiply: silu(linear1) * gate in one kernel, no separate silu alloc.
    const auto x_shape = tensor->get_value().logical_shape();
    if (x_shape.rank() != 4) {
        throw std::runtime_error("swiglu only supports rank-4 input tensors.");
    }
    const auto w1_shape = w1->get_value().logical_shape();
    const auto w2_shape = w2->get_value().logical_shape();
    const auto w3_shape = w3->get_value().logical_shape();
    if (w1_shape.rank() < 2 || w2_shape.rank() < 2 || w3_shape.rank() < 2) {
        throw std::runtime_error(
            "swiglu expects weights with at least 2 dims; trailing dims must be w1,w3 [H,D], w2 [D,H].");
    }

    const auto d = x_shape[-1];
    const auto h = w1_shape[-2];
    if (w1_shape[-1] != d) {
        throw std::runtime_error("swiglu expects w1 trailing dims [H,D] with D == input[-1].");
    }
    if (w3_shape[-2] != h || w3_shape[-1] != d) {
        throw std::runtime_error("swiglu expects w3 trailing dims [H,D] matching w1.");
    }
    if (w2_shape[-2] != d || w2_shape[-1] != h) {
        throw std::runtime_error("swiglu expects w2 trailing dims [D,H] matching input D and hidden H.");
    }

    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary silu_act{ttnn::operations::unary::UnaryOpType::SILU};
    const ttsl::Span<const EltwiseUnary> no_acts;
    const ttsl::Span<const EltwiseUnary> silu_lhs(&silu_act, 1);

    auto saved_linear1 = ttnn_fixed::matmul(tensor->get_value(), w1->get_value(), false, true);
    auto saved_gate = ttnn_fixed::matmul(tensor->get_value(), w3->get_value(), false, true);
    auto saved_gated =
        ttnn::multiply(saved_linear1, saved_gate, std::nullopt, std::nullopt, std::nullopt, no_acts, silu_lhs);
    auto swiglu_result = ttnn_fixed::matmul(saved_gated, w2->get_value(), false, true);
    auto saved_linear1_for_bw = std::move(saved_linear1);
    auto saved_gate_for_bw = std::move(saved_gate);
    auto saved_gated_for_bw = std::move(saved_gated);
    // NOTE: gated = silu(linear1) * gate is an elemwise op and relatively light to recompute.
    // In our nanollama3/tinyllama/llama1b experiments, recomputing it increased step time,
    // while enabling larger DRAM-peak reduction. Keep eager save+reuse in this PR and track
    // recompute tradeoff as a benchmark reference.

    uint32_t dropout_seed = 0;
    float dropout_scaler = 1.0F;
    bool use_per_device_seed = false;
    if (dropout_prob > 0.0F) {
        // A partial sum's label cannot say it is one, so its seeding would be read wrong, and masking each partial
        // differently is not a dropout of their sum anyway.
        TT_FATAL(
            !is_row_parallel(w2->get_value()),
            "swiglu cannot apply dropout when w2 is sharded along its input features: each device then holds a partial "
            "sum still to be added up across devices, so apply dropout after that sum");
        use_per_device_seed = dropout_seed_per_device(swiglu_result);
        dropout_seed = static_cast<uint32_t>(autograd::ctx().get_generator()());
        dropout_scaler = 1.0F / (1.0F - dropout_prob);
        ttnn::experimental::dropout(
            swiglu_result,
            dropout_prob,
            dropout_scaler,
            dropout_seed,
            use_per_device_seed,
            std::nullopt,
            swiglu_result);
    }
    auto out = autograd::create_tensor(swiglu_result);

    autograd::GradFunction grad = [tensor,
                                   w1,
                                   w2,
                                   w3,
                                   out,
                                   saved_linear1_for_bw = std::move(saved_linear1_for_bw),
                                   saved_gate_for_bw = std::move(saved_gate_for_bw),
                                   saved_gated_for_bw = std::move(saved_gated_for_bw),
                                   dropout_prob,
                                   use_per_device_seed,
                                   dropout_seed,
                                   dropout_scaler]() mutable {
        auto dL_dout = out->get_grad();

        if (dropout_prob > 0.0F) {
            dL_dout =
                ttnn::experimental::dropout(dL_dout, dropout_prob, dropout_scaler, dropout_seed, use_per_device_seed);
        }

        ttnn::Tensor linear1 = std::move(saved_linear1_for_bw);
        ttnn::Tensor gate = std::move(saved_gate_for_bw);
        ttnn::Tensor gated = std::move(saved_gated_for_bw);

        // W2 grad: use saved gated directly — no recompute
        {
            auto dL_dW2 = ttnn_fixed::matmul(flatten_leading(dL_dout), flatten_leading(gated), true, false)
                              .reshape(w2->get_value().logical_shape());
            core::distributed::label_grad_like_parameter_shards(w2->get_value(), dL_dW2);
            w2->add_grad(dL_dW2);
        }
        gated.deallocate();
        // dL/d(prod) = dL_dout @ w2 (no transpose — w2 is [D, H])
        auto dL_dprod = ttnn_fixed::matmul(dL_dout, w2->get_value());
        dL_dout.deallocate();

        // Fused elemwise BW kernel: reads (linear1, gate, dL_dprod) once, produces (dL_dlinear1, dL_dgate)
        auto [dL_dlinear1, dL_dgate] = ttml::metal::swiglu_elemwise_bw(linear1, gate, dL_dprod, linear1);
        gate.deallocate();
        dL_dprod.deallocate();

        // Input grads: dL @ w (no transpose — w1,w3 are [H, D])
        auto dL_dtensor = ttnn_fixed::matmul(dL_dlinear1, w1->get_value());
        auto dL_dtensor_from_w3 = ttnn_fixed::matmul(dL_dgate, w3->get_value());
        ttnn::add_(dL_dtensor, dL_dtensor_from_w3);
        dL_dtensor_from_w3.deallocate();
        tensor->add_grad(dL_dtensor);
        dL_dtensor.deallocate();

        // W1 & W3 grads
        auto flat_x = flatten_leading(tensor->get_value());
        {
            auto dL_dW1 = ttnn_fixed::matmul(flatten_leading(dL_dlinear1), flat_x, true, false)
                              .reshape(w1->get_value().logical_shape());
            core::distributed::label_grad_like_parameter_shards(w1->get_value(), dL_dW1);
            w1->add_grad(dL_dW1);
        }
        dL_dlinear1.deallocate();

        {
            auto dL_dW3 = ttnn_fixed::matmul(flatten_leading(dL_dgate), flat_x, true, false)
                              .reshape(w3->get_value().logical_shape());
            core::distributed::label_grad_like_parameter_shards(w3->get_value(), dL_dW3);
            w3->add_grad(dL_dW3);
        }
        dL_dgate.deallocate();
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor, w1, w2, w3));

    return out;
}

}  // namespace ttml::ops
