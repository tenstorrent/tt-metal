// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/log_softmax/log_softmax.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward.hpp"
#include "ttnn/operations/moreh/moreh_softmax/moreh_softmax.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::relu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        auto res = ttnn::relu_bw(out->get_grad(), tensor->get_value(), mem_config);
        tensor->add_grad(res[0]);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

    return out;
}

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor, GeluVariant variant) {
    // Unlike ttnn, fast_lut variant is not supported (no fast-lut backward kernel)
    if (variant == GeluVariant::FAST_LUT) {
        throw std::invalid_argument("gelu: GeluVariant::FAST_LUT is not supported for training");
    }

    auto out = autograd::create_tensor();
    out->set_value(ttnn::gelu(tensor->get_value(), variant));
    autograd::GradFunction grad = [tensor, out, variant]() {
        auto dL_dt = ttnn::gelu_bw(out->get_grad(), tensor->get_value(), variant);
        tensor->add_grad(dL_dt[0].value());
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw) {
    auto out = autograd::create_tensor(ttnn::silu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out, use_composite_bw]() {
        auto res = use_composite_bw ? ttnn::silu_bw(out->get_grad(), tensor->get_value())
                                    : std::vector<std::optional<ttnn::Tensor>>(
                                          {ttml::metal::silu_bw(tensor->get_value(), out->get_grad())});
        assert(res.size() == 1U && "Silu backward should return only one gradient");
        tensor->add_grad(res.front().value());
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

    return out;
}

namespace {

// The composite's forward in FP32, before output rounding: the log-probabilities and the one-hot mask of the single
// selected (first) maximum, which the backward needs.
struct LogSoftmaxContext {
    ttnn::Tensor log_probabilities;
    ttnn::Tensor max_mask;
};

LogSoftmaxContext log_softmax_with_context(const ttnn::Tensor& t, int dim) {
    const auto rank = static_cast<int>(t.logical_shape().rank());
    TT_FATAL(dim >= -rank && dim < rank, "log_softmax: dimension {} out of range for rank {}", dim, rank);
    dim = dim < 0 ? dim + rank : dim;
    TT_FATAL(t.logical_shape()[dim] > 0, "log_softmax: reduction dimension must be nonempty");
    TT_FATAL(
        t.dtype() == ttnn::DataType::BFLOAT16 || t.dtype() == ttnn::DataType::FLOAT32,
        "log_softmax: expected BFLOAT16 or FLOAT32 input");

    auto input = ttnn::typecast(t, ttnn::DataType::FLOAT32);
    auto t_max = ttnn::max(input, dim, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto shifted = ttnn::subtract(input, t_max);

    // Select exactly one maximum, including for tied logits. FP32 indices are exact
    // up to 2^24; BF16 indices would alias adjacent classes beyond 256.
    TT_FATAL(t.logical_shape()[dim] <= (1U << 24), "log_softmax: reduction dimension exceeds exact FP32 index range");
    ttnn::Tensor max_indices;
    if (dim == rank - 1 && t.logical_volume() != 0) {
        // TILE W argmax uses a single scalar reader. Reuse the ROW_MAJOR
        // multicore path instead. keepdim=false batches the output rows and
        // enables its second data-movement processor where supported.
        auto row_major = ttnn::to_layout(input, ttnn::Layout::ROW_MAJOR);
        // The multicore op stages one max/index pair per row per worker in L1.
        // Bound each group to 256 rows, including for long sequence dimensions.
        const auto row_count = t.logical_volume() / t.logical_shape()[dim];
        auto group_rows = static_cast<uint32_t>(std::min<uint64_t>(row_count, 256));
        while (row_count % group_rows != 0) {
            --group_rows;
        }
        TT_FATAL(
            row_count / group_rows <= std::numeric_limits<uint32_t>::max(),
            "log_softmax: too many row groups for multicore argmax");
        auto grouped = ttnn::reshape(
            row_major,
            ttnn::Shape({static_cast<uint32_t>(row_count / group_rows), 1, group_rows, t.logical_shape()[dim]}));
        auto squeezed_indices = ttnn::argmax(grouped, /* dim */ 3, /* keepdim */ false);
        std::vector<uint32_t> indices_shape(t.logical_shape().cbegin(), t.logical_shape().cend());
        indices_shape[dim] = 1;
        max_indices = ttnn::reshape(squeezed_indices, ttnn::Shape(indices_shape));
    } else {
        max_indices = ttnn::argmax(input, dim, /* keepdim */ true);
    }
    auto indices = ttnn::typecast(ttnn::to_layout(max_indices, ttnn::Layout::TILE), ttnn::DataType::FLOAT32);
    auto positions = ttnn::arange(t.logical_shape()[dim], ttnn::DataType::FLOAT32, *t.device(), t.memory_config());
    std::vector<uint32_t> index_shape(rank, 1);
    index_shape[dim] = t.logical_shape()[dim];
    positions = ttnn::to_layout(ttnn::reshape(positions, ttnn::Shape(index_shape)), ttnn::Layout::TILE);
    auto max_mask = ttnn::eq(indices, positions);

    // Eq. (5), https://arxiv.org/abs/2610.31642: exclude the selected e^0 = 1
    // before summing. log1p preserves the tail even when 1 + tail rounds to 1.
    auto others = ttnn::where(max_mask, 0.F, ttnn::exp(shifted));
    auto log_normalizer = ttnn::log1p(ttnn_fixed::sum_ttnn(others, dim, /* keep_dim */ true));
    return {ttnn::subtract(shifted, log_normalizer), max_mask};
}

// The composite's TTNN ops need interleaved tensors (argmax rejects sharded ones): a sharded tensor moves to
// interleaved DRAM.
ttnn::Tensor to_interleaved(const ttnn::Tensor& tensor) {
    return tensor.is_sharded() ? ttnn::to_memory_config(tensor, ttnn::DRAM_MEMORY_CONFIG) : tensor;
}

// The corrected composite, built from TTNN ops: log_softmax's fallback for inputs the fused op does not support
// (see metal::log_softmax_supported). It computes in interleaved memory; the output and the gradient take the input's
// placement.
autograd::TensorPtr log_softmax_composite(const autograd::TensorPtr& tensor, int dim) {
    const auto& input = tensor->get_value(autograd::PreferredPrecision::NATIVE);
    auto context = log_softmax_with_context(to_interleaved(input), dim);
    const auto dtype = input.dtype();
    const auto placement = input.memory_config();
    const auto to_input_placement = [placement](const ttnn::Tensor& t) {
        return t.memory_config() == placement ? t : ttnn::to_memory_config(t, placement);
    };
    auto out = autograd::create_tensor(to_input_placement(ttnn::typecast(context.log_probabilities, dtype)));
    const auto rank = static_cast<int>(input.logical_shape().rank());
    dim = dim < 0 ? dim + rank : dim;
    autograd::GradFunction grad = [tensor, out, dim, dtype, context, to_input_placement]() {
        auto upstream = ttnn::typecast(to_interleaved(out->get_grad()), ttnn::DataType::FLOAT32);
        auto softmax = ttnn::exp(context.log_probabilities);
        auto sum_grad = ttnn_fixed::sum_ttnn(upstream, dim, /* keep_dim */ true);
        auto others = ttnn::where(context.max_mask, 0.F, ttnn::subtract(upstream, ttnn::multiply(softmax, sum_grad)));
        // Eq. (7): reconstruct the maximum's gradient from the other entries,
        // rather than subtracting nearly equal values at the dominant class.
        auto max_grad = ttnn::neg(ttnn_fixed::sum_ttnn(others, dim, /* keep_dim */ true));
        auto input_grad = ttnn::where(context.max_mask, max_grad, others);
        tensor->add_grad(to_input_placement(ttnn::typecast(input_grad, dtype)));
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace

autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim) {
    const auto& native = tensor->get_value(autograd::PreferredPrecision::NATIVE);
    const int rank = static_cast<int>(native.logical_shape().rank());
    TT_FATAL(dim >= -rank && dim < rank, "log_softmax dimension {} out of range for rank {}", dim, rank);
    dim = dim < 0 ? dim + rank : dim;
    if (!ttml::metal::log_softmax_supported(native)) {
        return log_softmax_composite(tensor, dim);
    }
    ttsl::SmallVector<int64_t> permutation{0, 1, 2, 3};
    std::swap(permutation[dim], permutation[3]);
    auto input = dim == 3 ? native : ttnn::permute(native, permutation);
    auto values = ttml::metal::log_softmax_fw(input);
    auto output = dim == 3 ? values[0] : ttnn::permute(values[0], permutation);
    auto out = autograd::create_tensor(output);
    auto state = values[1];
    autograd::GradFunction grad = [tensor, out, input, state, permutation, dim]() {
        auto upstream = out->get_grad();
        if (dim != 3)
            upstream = ttnn::permute(upstream, permutation);
        auto result = ttml::metal::log_softmax_bw(input, upstream, state);
        if (dim != 3)
            result = ttnn::permute(result, permutation);
        tensor->add_grad(result);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim) {
    auto log_softmax = ttnn::moreh_softmax(
        tensor->get_value(),
        /* axis */ dim,
        /* output */ std::nullopt,
        ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp::LOGSOFTMAX,
        ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::softmax());
    auto out = autograd::create_tensor(log_softmax);

    autograd::GradFunction grad = [tensor, out, dim]() {
        auto grad = ttnn::moreh_softmax_backward(
            out->get_value(),
            out->get_grad(),
            /* axis */ dim,
            /* output */ std::nullopt,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::LOGSOFTMAX,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            /* output_mem_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());
        tensor->add_grad(grad);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr mean(const autograd::TensorPtr& tensor) {
    auto shape = ttnn::Shape({1, 1, 1, 1});
    auto out =
        autograd::create_tensor(core::empty(shape, &autograd::ctx().get_device(), tensor->get_value().memory_config()));
    ttnn::moreh_mean(
        tensor->get_value(),
        std::nullopt,
        true,
        std::nullopt,
        out->get_value(),
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    autograd::GradFunction grad = [tensor, out]() {
        auto resulting_shape = tensor->get_value().logical_shape();
        auto res = ttnn::moreh_mean_backward(
            out->get_grad(),
            std::nullopt,
            false,
            resulting_shape,
            std::nullopt,
            std::nullopt,
            core::ComputeKernelConfig::precise());
        tensor->add_grad(res);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim) {
    if (new_batch_dim == 1 || tensor->get_value().logical_shape()[0] == new_batch_dim) {
        return tensor;
    }
    auto out = ttml::autograd::create_tensor();
    auto repeats = ttnn::Shape({new_batch_dim, 1, 1, 1});
    // currently assuming tensor came with shape: {1,X,Y,Z} and we want to get {B,X,Y,Z}
    out->set_value(ttnn::repeat(tensor->get_value(), repeats));

    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn_fixed::sum_over_batch(out->get_grad());
        tensor->add_grad(res);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr exp(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::exp(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn::exp_bw(out->get_grad(), tensor->get_value());
        tensor->add_grad(res[0].value());
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr clip(const autograd::TensorPtr& tensor, float lo, float hi) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::clip(tensor->get_value(), lo, hi));
    autograd::GradFunction grad = [tensor, out, lo, hi]() {
        auto res = ttnn::clip_bw(out->get_grad(), tensor->get_value(), lo, hi);
        tensor->add_grad(res[0]);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops
