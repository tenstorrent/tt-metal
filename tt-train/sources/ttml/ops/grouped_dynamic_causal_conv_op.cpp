// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_dynamic_causal_conv_op.hpp"

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <tt-metalium/constants.hpp>
#include <ttnn/operations/core/to_layout/to_layout_op.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/pad/pad.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

using tt::constants::TILE_WIDTH;

struct Geometry {
    uint32_t batch = 0;
    uint32_t seq = 0;
    uint32_t channels = 0;
    uint32_t kernel_size = 0;
    uint32_t groups = 0;
    uint32_t block = 0;
    uint32_t group = 0;

    [[nodiscard]] uint32_t rows() const {
        return batch * seq;
    }
    [[nodiscard]] uint32_t blocks() const {
        return rows() / block;
    }
};

void require_tile_device_tensor(const ttnn::Tensor& tensor, const char* name) {
    if (!ttnn::is_device_tensor(tensor) || !tensor.is_allocated()) {
        throw std::invalid_argument(
            fmt::format("grouped_dynamic_causal_conv: {} must be an allocated device tensor", name));
    }
    if (tensor.layout() != ttnn::Layout::TILE) {
        throw std::invalid_argument(fmt::format("grouped_dynamic_causal_conv: {} must be TILE layout", name));
    }
    if (tensor.logical_shape().rank() != 4U) {
        throw std::invalid_argument(
            fmt::format("grouped_dynamic_causal_conv: {} must be rank 4, got {}", name, tensor.logical_shape()));
    }
}

Geometry validate(
    const ttnn::Tensor& input,
    const ttnn::Tensor& dynamic_kernel,
    const ttnn::Tensor& base_kernel,
    uint32_t block_size,
    uint32_t group_size) {
    require_tile_device_tensor(input, "input");
    require_tile_device_tensor(dynamic_kernel, "dynamic_kernel");
    require_tile_device_tensor(base_kernel, "base_kernel");
    const auto& x = input.logical_shape();
    const auto& d = dynamic_kernel.logical_shape();
    const auto& b = base_kernel.logical_shape();
    Geometry g{x[0], x[2], x[3], b[2], 0U, block_size, group_size};
    if (x[1] != 1U) {
        throw std::invalid_argument(fmt::format("grouped_dynamic_causal_conv: input must be [B, 1, S, C], got {}", x));
    }
    if (block_size == 0U || g.seq % block_size != 0U) {
        throw std::invalid_argument(fmt::format(
            "grouped_dynamic_causal_conv: sequence length {} must be a nonzero multiple of block_size {}",
            g.seq,
            block_size));
    }
    if (group_size == 0U || g.channels % group_size != 0U) {
        throw std::invalid_argument(fmt::format(
            "grouped_dynamic_causal_conv: channels {} must be a multiple of group_size {}", g.channels, group_size));
    }
    g.groups = g.channels / group_size;
    if (g.channels % TILE_WIDTH != 0U || g.groups % TILE_WIDTH != 0U) {
        throw std::invalid_argument(fmt::format(
            "grouped_dynamic_causal_conv: channels {} and groups {} must be multiples of {}",
            g.channels,
            g.groups,
            TILE_WIDTH));
    }
    if (b[0] != 1U || b[1] != 1U || b[3] != g.channels || g.kernel_size == 0U || g.kernel_size > block_size) {
        throw std::invalid_argument(fmt::format(
            "grouped_dynamic_causal_conv: base_kernel must be [1, 1, K, {}] with 0 < K <= block_size {}, got {}",
            g.channels,
            block_size,
            b));
    }
    if (d[0] != g.batch || d[1] != 1U || d[2] != g.seq || d[3] != g.kernel_size * g.groups) {
        throw std::invalid_argument(fmt::format(
            "grouped_dynamic_causal_conv: dynamic_kernel must be [{}, 1, {}, {}], got {}",
            g.batch,
            g.seq,
            g.kernel_size * g.groups,
            d));
    }
    if (input.dtype() != dynamic_kernel.dtype() || input.dtype() != base_kernel.dtype()) {
        throw std::invalid_argument(
            "grouped_dynamic_causal_conv: input, dynamic_kernel and base_kernel must share a dtype");
    }
    return g;
}

ttnn::Tensor slice4(const ttnn::Tensor& t, const std::array<uint32_t, 4>& begin, const std::array<uint32_t, 4>& end) {
    const ttsl::SmallVector<uint32_t> begins(begin.begin(), begin.end());
    const ttsl::SmallVector<uint32_t> ends(end.begin(), end.end());
    const ttsl::SmallVector<uint32_t> steps(4U, 1U);
    return ttnn::slice(t, begins, ends, steps);
}

// [B, 1, S, W] -> [1, 1, B * S, W]: blocks are row runs, whatever the batch split.
ttnn::Tensor flat(const ttnn::Tensor& t, const Geometry& g) {
    const auto& s = t.logical_shape();
    return g.batch == 1U ? t : ttnn::reshape(t, ttnn::Shape({1U, 1U, g.rows(), s[3]}));
}

ttnn::Tensor unflat(const ttnn::Tensor& t, const Geometry& g) {
    const auto& s = t.logical_shape();
    return g.batch == 1U ? t : ttnn::reshape(t, ttnn::Shape({g.batch, 1U, g.seq, s[3]}));
}

// [1, 1, N, C] TILE -> [N / block, 1, block, C] ROW_MAJOR, one block per batch entry, so a row shift inside a block
// is a slice and a pad on dim 2.
ttnn::Tensor to_blocks(const ttnn::Tensor& t, const Geometry& g) {
    return ttnn::reshape(
        ttnn::to_layout(t, ttnn::Layout::ROW_MAJOR), ttnn::Shape({g.blocks(), 1U, g.block, g.channels}));
}

ttnn::Tensor from_blocks(const ttnn::Tensor& blocks, const Geometry& g) {
    return ttnn::to_layout(ttnn::reshape(blocks, ttnn::Shape({1U, 1U, g.rows(), g.channels})), ttnn::Layout::TILE);
}

// out[t] = in[t - offset] inside each block, zero for the block's first `offset` rows.
ttnn::Tensor shift_down(const ttnn::Tensor& blocks, uint32_t offset, const Geometry& g) {
    if (offset == 0U) {
        return blocks;
    }
    auto kept = slice4(blocks, {0U, 0U, 0U, 0U}, {g.blocks(), 1U, g.block - offset, g.channels});
    ttsl::SmallVector<std::array<uint32_t, 2>> padding{{0U, 0U}, {0U, 0U}, {offset, 0U}, {0U, 0U}};
    return ttnn::pad(kept, padding, 0.0F, /*use_multicore=*/true);
}

// The adjoint of shift_down: out[t] = in[t + offset] inside each block, zero for the block's last `offset` rows.
ttnn::Tensor shift_up(const ttnn::Tensor& blocks, uint32_t offset, const Geometry& g) {
    if (offset == 0U) {
        return blocks;
    }
    auto kept = slice4(blocks, {0U, 0U, offset, 0U}, {g.blocks(), 1U, g.block, g.channels});
    ttsl::SmallVector<std::array<uint32_t, 2>> padding{{0U, 0U}, {0U, 0U}, {0U, offset}, {0U, 0U}};
    return ttnn::pad(kept, padding, 0.0F, /*use_multicore=*/true);
}

// [1, 1, G, C] with expand[g, c] = (c / group_size == g): dynamic_kernel_o @ expand broadcasts each group's kernel
// value over its channels, and p @ expand^T sums each group's channels.
ttnn::Tensor group_expansion(const Geometry& g) {
    xt::xarray<float> expand = xt::zeros<float>({size_t{1}, size_t{1}, size_t{g.groups}, size_t{g.channels}});
    for (uint32_t c = 0; c < g.channels; ++c) {
        expand(0, 0, c / g.group, c) = 1.0F;
    }
    return core::from_xtensor(expand, &autograd::ctx().get_device());
}

// [1, 1, 1, C] row o of base_kernel, sliced in ROW_MAJOR since the row is not tile-aligned.
ttnn::Tensor base_kernel_row(const ttnn::Tensor& base_kernel, uint32_t offset, const Geometry& g) {
    auto row = slice4(
        ttnn::to_layout(base_kernel, ttnn::Layout::ROW_MAJOR), {0U, 0U, offset, 0U}, {1U, 1U, offset + 1U, g.channels});
    return ttnn::to_layout(row, ttnn::Layout::TILE);
}

// [1, 1, N, C] kernel of offset o: base_kernel[o] broadcast over rows plus each row's dynamic kernel, per group.
ttnn::Tensor kernel_at(
    const ttnn::Tensor& dynamic_kernel_flat,
    const ttnn::Tensor& base_kernel,
    const ttnn::Tensor& expand,
    uint32_t offset,
    const Geometry& g) {
    auto dynamic_kernel_o =
        slice4(dynamic_kernel_flat, {0U, 0U, 0U, offset * g.groups}, {1U, 1U, g.rows(), (offset + 1U) * g.groups});
    return ttnn::add(ttnn_fixed::matmul(dynamic_kernel_o, expand), base_kernel_row(base_kernel, offset, g));
}

}  // namespace

autograd::TensorPtr grouped_dynamic_causal_conv(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& dynamic_kernel,
    const autograd::TensorPtr& base_kernel,
    uint32_t block_size,
    uint32_t group_size) {
    const Geometry g =
        validate(input->get_value(), dynamic_kernel->get_value(), base_kernel->get_value(), block_size, group_size);
    const auto expand = group_expansion(g);

    const auto x_blocks = to_blocks(flat(input->get_value(), g), g);
    const auto dynamic_kernel_flat = flat(dynamic_kernel->get_value(), g);
    ttnn::Tensor out;
    for (uint32_t o = 0; o < g.kernel_size; ++o) {
        auto term = ttnn::multiply(
            kernel_at(dynamic_kernel_flat, base_kernel->get_value(), expand, o, g),
            from_blocks(shift_down(x_blocks, o, g), g));
        out = o == 0U ? term : ttnn::add(out, term);
    }

    auto result = autograd::create_tensor(unflat(out, g));
    // Recomputes the shifted inputs and the kernel values rather than saving K full-size copies of each.
    autograd::GradFunction grad = [input, dynamic_kernel, base_kernel, result, g, expand]() {
        const auto grad_out = flat(result->get_grad(), g);
        const auto x_blocks = to_blocks(flat(input->get_value(), g), g);
        const auto dynamic_kernel_flat = flat(dynamic_kernel->get_value(), g);

        std::vector<ttnn::Tensor> grad_dynamic_kernel;
        std::vector<ttnn::Tensor> grad_base_kernel;
        ttnn::Tensor grad_input;
        const bool want_input = input->get_requires_grad();
        const bool want_dynamic = dynamic_kernel->get_requires_grad();
        const bool want_base = base_kernel->get_requires_grad();
        for (uint32_t o = 0; o < g.kernel_size; ++o) {
            if (want_dynamic || want_base) {
                // d out / d kernel_o = shifted input, so its products with grad_out give both kernels' gradients.
                const auto product = ttnn::multiply(grad_out, from_blocks(shift_down(x_blocks, o, g), g));
                if (want_dynamic) {
                    grad_dynamic_kernel.push_back(
                        ttnn_fixed::matmul(product, expand, /*transpose_a=*/false, /*transpose_b=*/true));
                }
                if (want_base) {
                    // One-row results stack in ROW_MAJOR: TILE concat on dim 2 would need tile-aligned rows.
                    grad_base_kernel.push_back(
                        ttnn::to_layout(ttnn_fixed::sum_over_leading_dims(product), ttnn::Layout::ROW_MAJOR));
                }
            }
            if (want_input) {
                const auto kernel_times_grad =
                    ttnn::multiply(kernel_at(dynamic_kernel_flat, base_kernel->get_value(), expand, o, g), grad_out);
                auto term = from_blocks(shift_up(to_blocks(kernel_times_grad, g), o, g), g);
                grad_input = o == 0U ? term : ttnn::add(grad_input, term);
            }
        }

        if (want_input) {
            input->add_grad(unflat(grad_input, g));
        }
        if (want_dynamic) {
            dynamic_kernel->add_grad(unflat(
                g.kernel_size == 1U ? grad_dynamic_kernel.front() : ttnn::concat(grad_dynamic_kernel, /*dim=*/3), g));
        }
        if (want_base) {
            base_kernel->add_grad(ttnn::to_layout(
                g.kernel_size == 1U ? grad_base_kernel.front() : ttnn::concat(grad_base_kernel, /*dim=*/2),
                ttnn::Layout::TILE));
        }
    };
    result->set_node(autograd::add_backward_node(std::move(grad), result, input, dynamic_kernel, base_kernel));
    return result;
}

}  // namespace ttml::ops
