// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_op.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <ttnn/operations/core/to_layout/to_layout_op.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/pad/pad.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/experimental/conv3d/conv3d.hpp>
#include <ttnn/operations/experimental/conv3d/prepare_conv3d_weights.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <vector>

#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

// Activation channel counts must be divisible by the kernel's C_in block (16 or 32), so C_in and C_out are padded
// to TILE_WIDTH; C_out because dY becomes the transposed convolution's activation in the backward pass.
constexpr uint32_t kChannelAlignment = tt::constants::TILE_WIDTH;

struct Conv3dGeometry {
    uint32_t N = 0;
    Conv3dDims in_size{};   // D, H, W
    Conv3dDims out_size{};  // D_out, H_out, W_out
    uint32_t C_in = 0;      // total input channels (all groups)
    uint32_t C_out = 0;
    uint32_t groups = 1;
    Conv3dDims kernel{};
    Conv3dDims stride{};
    Conv3dDims padding{};
    Conv3dDims dilation{};

    [[nodiscard]] uint32_t kernel_volume() const {
        return kernel[0] * kernel[1] * kernel[2];
    }
    [[nodiscard]] uint32_t num_output_positions() const {
        return N * out_size[0] * out_size[1] * out_size[2];
    }
    [[nodiscard]] uint32_t effective_kernel(size_t i) const {
        return dilation[i] * (kernel[i] - 1U) + 1U;
    }
    [[nodiscard]] uint32_t C_in_per_group() const {
        return C_in / groups;
    }
    [[nodiscard]] uint32_t C_out_per_group() const {
        return C_out / groups;
    }
    [[nodiscard]] uint32_t c_in_block() const {
        return ttnn::operations::experimental::conv3d::default_c_in_block(kernel_volume());
    }
    [[nodiscard]] struct GroupGeometry per_group() const;
};

// Per-group channel counts; a distinct type so per-group helpers cannot be handed the whole-op geometry.
struct GroupGeometry : Conv3dGeometry {
    [[nodiscard]] uint32_t c_in_padded() const {
        return tt::round_up(C_in, kChannelAlignment);
    }
    [[nodiscard]] uint32_t c_out_padded() const {
        return tt::round_up(C_out, kChannelAlignment);
    }
};

GroupGeometry Conv3dGeometry::per_group() const {
    GroupGeometry g;
    static_cast<Conv3dGeometry&>(g) = *this;
    g.C_in = C_in_per_group();
    g.C_out = C_out_per_group();
    g.groups = 1U;
    return g;
}

template <typename MakeGroup>
std::vector<ttnn::Tensor> map_groups(uint32_t groups, MakeGroup&& make_group) {
    std::vector<ttnn::Tensor> parts;
    parts.reserve(groups);
    for (uint32_t g = 0; g < groups; ++g) {
        parts.push_back(make_group(g));
    }
    return parts;
}

ttnn::Tensor to_row_major(const ttnn::Tensor& tensor) {
    if (tensor.layout() == ttnn::Layout::ROW_MAJOR) {
        return tensor;
    }
    return ttnn::to_layout(tensor, ttnn::Layout::ROW_MAJOR);
}

ttnn::Tensor to_tile(const ttnn::Tensor& tensor) {
    if (tensor.layout() == ttnn::Layout::TILE) {
        return tensor;
    }
    return ttnn::to_layout(tensor, ttnn::Layout::TILE);
}

ttnn::Tensor to_layout_of(const ttnn::Tensor& tensor, const ttnn::Tensor& reference) {
    return reference.layout() == ttnn::Layout::TILE ? to_tile(tensor) : to_row_major(tensor);
}

ttnn::Tensor pad_after(const ttnn::Tensor& tensor, size_t dim, uint32_t amount) {
    if (amount == 0U) {
        return tensor;
    }
    ttsl::SmallVector<std::array<uint32_t, 2>> padding(tensor.logical_shape().rank(), {0U, 0U});
    padding[dim] = {0U, amount};
    return ttnn::pad(tensor, padding, 0.0F, /*use_multicore=*/true);
}

// The kernel must be launched with the C_in_block the weight was blocked with; a mismatch is silently wrong.
struct PreparedWeight {
    ttnn::Tensor tensor;
    uint32_t c_in_block = 0;
};

// ttnn's preparation pads C_in itself and stays on device for a device tensor with groups == 1; C_out must already
// be the launch count.
PreparedWeight prepare_weight(const ttnn::Tensor& raw_layout_weight) {
    const auto& shape = raw_layout_weight.logical_shape();
    const uint32_t c_in_block =
        ttnn::operations::experimental::conv3d::default_c_in_block(shape[2] * shape[3] * shape[4]);
    auto prepared = ttnn::operations::experimental::conv3d::prepare_conv3d_weights(
        raw_layout_weight, /*groups=*/1U, c_in_block, kChannelAlignment, raw_layout_weight.device());
    // The kernel accepts only the rank-2 prepared form from device.
    if (prepared.logical_shape().rank() != 2U) {
        throw std::logic_error(fmt::format(
            "conv3d: ttnn weight preparation returned rank {} for weight shape {}, expected rank 2",
            prepared.logical_shape().rank(),
            shape));
    }
    return {to_tile(prepared), c_in_block};
}

// The pad kernel cannot address a leading axis of a rank-5 tensor, hence the rank-4 view.
ttnn::Tensor pad_output_channels(const ttnn::Tensor& weight, uint32_t c_out_padded) {
    const auto shape = weight.logical_shape();
    if (shape[0] == c_out_padded) {
        return weight;
    }
    auto view = ttnn::reshape(weight, ttnn::Shape({shape[0], shape[1], shape[2] * shape[3], shape[4]}));
    view = pad_after(view, /*dim=*/0, c_out_padded - shape[0]);
    return ttnn::reshape(view, ttnn::Shape({c_out_padded, shape[1], shape[2], shape[3], shape[4]}));
}

ttnn::Tensor slice_dim(const ttnn::Tensor& tensor, size_t dim, uint32_t begin, uint32_t count) {
    const auto& shape = tensor.logical_shape();
    if (begin == 0U && count == shape[dim]) {
        return tensor;
    }
    ttsl::SmallVector<uint32_t> begins(shape.rank(), 0U);
    ttsl::SmallVector<uint32_t> ends(shape.cbegin(), shape.cend());
    ttsl::SmallVector<uint32_t> steps(shape.rank(), 1U);
    begins[dim] = begin;
    ends[dim] = begin + count;
    return ttnn::slice(tensor, begins, ends, steps);
}

ttnn::Tensor slice_channels(const ttnn::Tensor& tensor, uint32_t begin, uint32_t count) {
    return slice_dim(tensor, /*dim=*/4, begin, count);
}

ttnn::Tensor slice_weight_group(const ttnn::Tensor& weight_row_major, const Conv3dGeometry& geometry, uint32_t g) {
    return slice_dim(weight_row_major, /*dim=*/0, g * geometry.C_out_per_group(), geometry.C_out_per_group());
}

ttnn::Tensor concat_or_single(std::vector<ttnn::Tensor>& parts, int dim) {
    return parts.size() == 1U ? parts.front() : ttnn::concat(parts, dim);
}

PreparedWeight prepare_forward_weight(const ttnn::Tensor& raw_weight, uint32_t c_out_padded) {
    return prepare_weight(pad_output_channels(to_row_major(raw_weight), c_out_padded));
}

// Weight of the transposed convolution for dX: channel roles swap and the kernel is flipped on every spatial axis.
PreparedWeight prepare_transposed_weight(const ttnn::Tensor& raw_weight, uint32_t c_in_padded, uint32_t c_out_padded) {
    const auto& shape = raw_weight.logical_shape();
    const uint32_t C_out = shape[0];
    const uint32_t C_in = shape[1];
    const uint32_t kD = shape[2];
    const uint32_t kH = shape[3];
    const uint32_t kW = shape[4];
    const uint32_t kernel_volume = kD * kH * kW;

    auto weight = to_row_major(raw_weight);
    weight = ttnn::permute(weight, ttsl::SmallVector<int64_t>{2, 3, 4, 0, 1});  // [kD, kH, kW, C_out, C_in]
    weight = ttnn::reshape(weight, ttnn::Shape({kernel_volume, C_out, C_in}));

    if (kernel_volume > 1U) {
        std::vector<ttnn::Tensor> reversed_rows;
        reversed_rows.reserve(kernel_volume);
        for (uint32_t row = kernel_volume; row-- > 0U;) {
            reversed_rows.push_back(ttnn::slice(
                weight,
                ttsl::SmallVector<uint32_t>{row, 0U, 0U},
                ttsl::SmallVector<uint32_t>{row + 1U, C_out, C_in},
                ttsl::SmallVector<uint32_t>{1U, 1U, 1U}));
        }
        weight = ttnn::concat(reversed_rows, /*dim=*/0);
    }

    weight = pad_after(weight, /*dim=*/1, c_out_padded - C_out);
    weight = ttnn::reshape(weight, ttnn::Shape({kD, kH, kW, c_out_padded, C_in}));
    weight = ttnn::permute(weight, ttsl::SmallVector<int64_t>{4, 3, 0, 1, 2});
    return prepare_weight(pad_output_channels(weight, c_in_padded));
}

// Rank-4 view so the pad kernel addresses the padded axis directly.
ttnn::Tensor pad_spatial_dim_both_sides(const ttnn::Tensor& tensor, size_t dim, uint32_t amount) {
    if (amount == 0U) {
        return tensor;
    }
    const auto shape = tensor.logical_shape();
    uint32_t outer = 1U;
    for (size_t i = 0; i < dim; ++i) {
        outer *= shape[i];
    }
    uint32_t rows_after = 1U;
    for (size_t i = dim + 1; i + 1 < shape.rank(); ++i) {
        rows_after *= shape[i];
    }
    const uint32_t channels = shape[-1];
    const uint32_t length = shape[dim];

    auto view = ttnn::reshape(tensor, ttnn::Shape({outer, length, rows_after, channels}));
    view = ttnn::pad(
        view,
        ttsl::SmallVector<std::array<uint32_t, 2>>{{0U, 0U}, {amount, amount}, {0U, 0U}, {0U, 0U}},
        0.0F,
        /*use_multicore=*/true);

    ttsl::SmallVector<uint32_t> out_shape(shape.cbegin(), shape.cend());
    out_shape[dim] = length + 2U * amount;
    return ttnn::reshape(view, ttnn::Shape(out_shape));
}

ttnn::Tensor zero_upsample_spatial_dim(
    const ttnn::Tensor& tensor, size_t dim, uint32_t stride, uint32_t target_length) {
    const auto shape = tensor.logical_shape();
    const uint32_t length = shape[dim];
    if (stride == 1U && length == target_length) {
        return tensor;
    }

    uint32_t outer = 1U;
    for (size_t i = 0; i < dim; ++i) {
        outer *= shape[i];
    }
    uint32_t rows_after = 1U;
    for (size_t i = dim + 1; i + 1 < shape.rank(); ++i) {
        rows_after *= shape[i];
    }
    const uint32_t channels = shape[-1];

    auto view = tensor;
    if (stride > 1U) {
        // Padding a size-1 axis to `stride` and merging it back puts dY[l] at l * stride with zeros between.
        view = ttnn::reshape(view, ttnn::Shape({outer, length, 1U, rows_after, channels}));
        view = ttnn::pad(
            view,
            ttsl::SmallVector<std::array<uint32_t, 2>>{{0U, 0U}, {0U, 0U}, {0U, stride - 1U}, {0U, 0U}, {0U, 0U}},
            0.0F,
            /*use_multicore=*/true);
    }
    const uint32_t upsampled_length = length * stride;
    view = ttnn::reshape(view, ttnn::Shape({outer, upsampled_length, rows_after, channels}));

    if (upsampled_length > target_length) {
        view = ttnn::slice(
            view,
            ttsl::SmallVector<uint32_t>{0U, 0U, 0U, 0U},
            ttsl::SmallVector<uint32_t>{outer, target_length, rows_after, channels},
            ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
    } else if (upsampled_length < target_length) {
        view = pad_after(view, /*dim=*/1, target_length - upsampled_length);
    }

    ttsl::SmallVector<uint32_t> out_shape(shape.cbegin(), shape.cend());
    out_shape[dim] = target_length;
    return ttnn::reshape(view, ttnn::Shape(out_shape));
}

ttnn::Tensor run_ttnn_conv3d(
    const ttnn::Tensor& input_row_major,
    const PreparedWeight& prepared_weight,
    const std::optional<ttnn::Tensor>& bias_tile,
    uint32_t output_channels,
    const Conv3dDims& kernel,
    const Conv3dDims& stride,
    const Conv3dDims& padding,
    const Conv3dDims& dilation) {
    const ttnn::experimental::prim::Conv3dConfig config(
        /*weights_dtype=*/input_row_major.dtype(),
        /*output_layout=*/ttnn::Layout::ROW_MAJOR,
        /*T_out_block=*/1U,
        /*W_out_block=*/1U,
        /*H_out_block=*/1U,
        /*C_out_block=*/tt::constants::TILE_WIDTH,
        prepared_weight.c_in_block,
        dilation,
        kChannelAlignment,
        input_row_major.device()->compute_with_storage_grid_size());
    return ttnn::experimental::conv3d(
        input_row_major,
        prepared_weight.tensor,
        /*device=*/std::nullopt,
        bias_tile,
        config,
        input_row_major.dtype(),
        output_channels,
        kernel,
        stride,
        padding,
        dilation,
        /*padding_mode=*/"zeros",
        /*groups=*/1U,
        /*memory_config=*/std::nullopt,
        core::ComputeKernelConfig::precise());
}

ttnn::Tensor conv3d_input_grad(
    const ttnn::Tensor& grad_output, const PreparedWeight& transposed_weight, const GroupGeometry& geometry) {
    const uint32_t c_in_padded = geometry.c_in_padded();
    const uint32_t c_out_padded = geometry.c_out_padded();

    auto grad = to_row_major(grad_output);
    grad = pad_after(grad, /*dim=*/4, c_out_padded - geometry.C_out);

    // A stride-1 conv with padding q over the zero-upsampled dY (length in + 2p - span) holds dX[j - (q - span + p)]
    // at index j: q = span - p yields dX directly; for p > span, q = span and dX starts at offset p.
    Conv3dDims transposed_padding{};
    Conv3dDims crop_offset{};
    for (size_t i = 0; i < 3; ++i) {
        const uint32_t kernel_span = geometry.effective_kernel(i) - 1U;
        const uint32_t target_length = geometry.in_size[i] + 2U * geometry.padding[i] - kernel_span;
        grad = zero_upsample_spatial_dim(grad, i + 1, geometry.stride[i], target_length);
        if (geometry.padding[i] <= kernel_span) {
            transposed_padding[i] = kernel_span - geometry.padding[i];
            crop_offset[i] = 0U;
        } else {
            transposed_padding[i] = kernel_span;
            crop_offset[i] = geometry.padding[i];
        }
    }

    auto grad_input = run_ttnn_conv3d(
        grad,
        transposed_weight,
        std::nullopt,
        c_in_padded,
        geometry.kernel,
        /*stride=*/{1U, 1U, 1U},
        transposed_padding,
        geometry.dilation);

    if (crop_offset != Conv3dDims{0U, 0U, 0U} || c_in_padded != geometry.C_in) {
        grad_input = ttnn::slice(
            grad_input,
            ttsl::SmallVector<uint32_t>{0U, crop_offset[0], crop_offset[1], crop_offset[2], 0U},
            ttsl::SmallVector<uint32_t>{
                geometry.N,
                crop_offset[0] + geometry.in_size[0],
                crop_offset[1] + geometry.in_size[1],
                crop_offset[2] + geometry.in_size[2],
                geometry.C_in},
            ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U, 1U});
    }
    return grad_input;
}

// dW[co, ci, k] = sum_m dY^T[co, m] * Xpad[m * stride + k * dilation, ci]: one matmul per kernel offset.
ttnn::Tensor conv3d_weight_grad(
    const ttnn::Tensor& grad_output_t_tile,  // [C_out, M] TILE
    const ttnn::Tensor& padded_input,        // [N, D + 2p, H + 2p, W + 2p, C_in_total] ROW_MAJOR
    uint32_t input_channel_begin,
    const GroupGeometry& geometry) {
    const uint32_t M = geometry.num_output_positions();
    std::vector<ttnn::Tensor> per_offset_grads;
    per_offset_grads.reserve(geometry.kernel_volume());
    for (uint32_t kd = 0; kd < geometry.kernel[0]; ++kd) {
        for (uint32_t kh = 0; kh < geometry.kernel[1]; ++kh) {
            for (uint32_t kw = 0; kw < geometry.kernel[2]; ++kw) {
                const Conv3dDims offset{
                    kd * geometry.dilation[0], kh * geometry.dilation[1], kw * geometry.dilation[2]};
                ttsl::SmallVector<uint32_t> begins{0U, offset[0], offset[1], offset[2], input_channel_begin};
                ttsl::SmallVector<uint32_t> ends{
                    geometry.N,
                    offset[0] + (geometry.out_size[0] - 1U) * geometry.stride[0] + 1U,
                    offset[1] + (geometry.out_size[1] - 1U) * geometry.stride[1] + 1U,
                    offset[2] + (geometry.out_size[2] - 1U) * geometry.stride[2] + 1U,
                    input_channel_begin + geometry.C_in};
                ttsl::SmallVector<uint32_t> steps{1U, geometry.stride[0], geometry.stride[1], geometry.stride[2], 1U};

                auto window = ttnn::slice(padded_input, begins, ends, steps);  // [N, D_out, H_out, W_out, C_in]
                window = ttnn::reshape(window, ttnn::Shape({M, geometry.C_in}));
                auto grad_k = ttnn_fixed::matmul(grad_output_t_tile, to_tile(window));  // [C_out, C_in]
                grad_k = to_row_major(grad_k);
                per_offset_grads.push_back(ttnn::reshape(grad_k, ttnn::Shape({1U, geometry.C_out, geometry.C_in})));
            }
        }
    }

    auto grad_weight = concat_or_single(per_offset_grads, /*dim=*/0);
    grad_weight = ttnn::reshape(
        grad_weight,
        ttnn::Shape({geometry.kernel[0], geometry.kernel[1], geometry.kernel[2], geometry.C_out, geometry.C_in}));
    return ttnn::permute(grad_weight, ttsl::SmallVector<int64_t>{3, 4, 0, 1, 2});  // [C_out, C_in, kD, kH, kW]
}

Conv3dGeometry validate_and_build_geometry(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const Conv3dDims& stride,
    const Conv3dDims& padding,
    const Conv3dDims& dilation,
    uint32_t groups,
    const std::string& padding_mode) {
    if (input == nullptr || weight == nullptr) {
        throw std::invalid_argument("conv3d: input and weight must not be null");
    }
    const auto& input_value = input->get_value();
    const auto& weight_value = weight->get_value();
    const auto& input_shape = input_value.logical_shape();
    const auto& weight_shape = weight_value.logical_shape();

    if (input_shape.rank() != 5U) {
        throw std::invalid_argument(
            fmt::format("conv3d: input must have rank 5 with layout [N, D, H, W, C_in], got shape {}", input_shape));
    }
    if (weight_shape.rank() != 5U) {
        throw std::invalid_argument(fmt::format(
            "conv3d: weight must have rank 5 with layout [C_out, C_in, kD, kH, kW], got shape {}", weight_shape));
    }
    if (!ttnn::is_device_tensor(input_value) || !input_value.is_allocated() || !ttnn::is_device_tensor(weight_value) ||
        !weight_value.is_allocated()) {
        throw std::invalid_argument("conv3d:: input and weight must be allocated on device");
    }
    if (input_value.dtype() != ttnn::DataType::BFLOAT16 && input_value.dtype() != ttnn::DataType::FLOAT32) {
        throw std::invalid_argument(
            fmt::format("conv3d: input dtype must be bfloat16 or float32, got {}", input_value.dtype()));
    }
    if (weight_value.dtype() != input_value.dtype()) {
        throw std::invalid_argument(fmt::format(
            "conv3d: weight dtype {} must match input dtype {}", weight_value.dtype(), input_value.dtype()));
    }
    if (groups == 0U) {
        throw std::invalid_argument("conv3d: groups must be non-zero");
    }
    if (padding_mode != "zeros") {
        throw std::invalid_argument(fmt::format(
            R"(conv3d: only padding_mode "zeros" is supported by the autograd implementation, got "{}")",
            padding_mode));
    }

    Conv3dGeometry geometry;
    geometry.N = input_shape[0];
    geometry.in_size = {input_shape[1], input_shape[2], input_shape[3]};
    geometry.C_in = input_shape[4];
    geometry.C_out = weight_shape[0];
    geometry.groups = groups;
    geometry.kernel = {weight_shape[2], weight_shape[3], weight_shape[4]};
    geometry.stride = stride;
    geometry.padding = padding;
    geometry.dilation = dilation;

    if (geometry.C_in % groups != 0U) {
        throw std::invalid_argument(
            fmt::format("conv3d: input channels {} must be divisible by groups {}", geometry.C_in, groups));
    }
    if (geometry.C_out % groups != 0U) {
        throw std::invalid_argument(
            fmt::format("conv3d: out_channels {} must be divisible by groups {}", geometry.C_out, groups));
    }
    if (weight_shape[1] != geometry.C_in / groups) {
        throw std::invalid_argument(fmt::format(
            "conv3d: weight in_channels {} must equal input channels / groups = {} / {} = {}",
            weight_shape[1],
            geometry.C_in,
            groups,
            geometry.C_in / groups));
    }
    if (geometry.N == 0U || geometry.C_in == 0U || geometry.C_out == 0U || geometry.in_size[0] == 0U ||
        geometry.in_size[1] == 0U || geometry.in_size[2] == 0U) {
        throw std::invalid_argument(fmt::format(
            "conv3d: batch, channel, and spatial sizes must be non-zero, got input shape {} and weight shape {}",
            input_shape,
            weight_shape));
    }
    for (size_t i = 0; i < 3; ++i) {
        if (geometry.kernel[i] == 0U) {
            throw std::invalid_argument("conv3d: kernel size must be non-zero in every dimension");
        }
        if (stride[i] == 0U) {
            throw std::invalid_argument("conv3d: stride must be >= 1 in every dimension");
        }
        if (dilation[i] == 0U) {
            throw std::invalid_argument("conv3d: dilation must be >= 1 in every dimension");
        }
        const uint32_t padded_size = geometry.in_size[i] + 2U * padding[i];
        if (padded_size < geometry.effective_kernel(i)) {
            throw std::invalid_argument(fmt::format(
                "conv3d: effective kernel size {} exceeds padded input size {} in spatial dimension {}",
                geometry.effective_kernel(i),
                padded_size,
                i));
        }
        geometry.out_size[i] = (padded_size - geometry.effective_kernel(i)) / stride[i] + 1U;
    }

    if (bias != nullptr) {
        const auto& bias_value = bias->get_value();
        const auto& bias_shape = bias_value.logical_shape();
        if (bias_shape.volume() != geometry.C_out || bias_shape[-1] != geometry.C_out) {
            throw std::invalid_argument(fmt::format(
                "conv3d: bias must have volume {} with last dimension {}, got shape {}",
                geometry.C_out,
                geometry.C_out,
                bias_shape));
        }
        if (bias_value.dtype() != input_value.dtype()) {
            throw std::invalid_argument(fmt::format(
                "conv3d: bias dtype {} must match input dtype {}", bias_value.dtype(), input_value.dtype()));
        }
        if (!ttnn::is_device_tensor(bias_value)) {
            throw std::invalid_argument("conv3d: bias must be on device");
        }
    }

    // One core per C_in block of the kernel's activation; in dX that activation is dY, hence the C_out check.
    const GroupGeometry group = geometry.per_group();
    const auto grid = input_value.device()->compute_with_storage_grid_size();
    const uint32_t cores = static_cast<uint32_t>(grid.x * grid.y);
    const uint32_t c_in_block = geometry.c_in_block();
    if (group.c_in_padded() / c_in_block > cores) {
        throw std::invalid_argument(fmt::format(
            "conv3d: {} input channels per group need {} channel blocks of {} but the device has {} cores",
            group.C_in,
            group.c_in_padded() / c_in_block,
            c_in_block,
            cores));
    }
    if (input->get_requires_grad() && group.c_out_padded() / c_in_block > cores) {
        throw std::invalid_argument(fmt::format(
            "conv3d: the input gradient runs a transposed conv over {} output channels per group, needing {} "
            "channel blocks of {} but the device has {} cores; reduce C_out per group or set input requires_grad "
            "to false",
            group.C_out,
            group.c_out_padded() / c_in_block,
            c_in_block,
            cores));
    }
    return geometry;
}

struct GroupWeights {
    std::vector<PreparedWeight> forward;
    std::vector<PreparedWeight> transposed;  // empty until needed
};

std::vector<PreparedWeight> prepare_group_weights(
    const ttnn::Tensor& weight_row_major, const Conv3dGeometry& geometry, bool transposed) {
    const GroupGeometry group = geometry.per_group();
    const uint32_t c_in_padded = group.c_in_padded();
    const uint32_t c_out_padded = group.c_out_padded();
    std::vector<PreparedWeight> result;
    result.reserve(geometry.groups);
    for (uint32_t g = 0; g < geometry.groups; ++g) {
        auto group_weight = slice_weight_group(weight_row_major, geometry, g);
        result.push_back(
            transposed ? prepare_transposed_weight(group_weight, c_in_padded, c_out_padded)
                       : prepare_forward_weight(group_weight, c_out_padded));
    }
    return result;
}

std::vector<PreparedWeight> to_internal(const std::vector<ttnn::Tensor>& tensors, uint32_t c_in_block) {
    std::vector<PreparedWeight> result;
    result.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        result.push_back({tensor, c_in_block});
    }
    return result;
}

void validate_prepared_weight(
    const Conv3dPreparedWeight& prepared, const ttnn::Tensor& weight_value, const Conv3dGeometry& geometry) {
    const uint32_t groups = geometry.groups;
    if (prepared.weight_shape != weight_value.logical_shape() || prepared.groups != groups) {
        throw std::invalid_argument(fmt::format(
            "conv3d: prepared weight was built for shape {} with groups {}, but the call has shape {} with groups {}",
            prepared.weight_shape,
            prepared.groups,
            weight_value.logical_shape(),
            groups));
    }
    if (prepared.forward.size() != groups || (!prepared.transposed.empty() && prepared.transposed.size() != groups)) {
        throw std::invalid_argument(fmt::format(
            "conv3d: prepared weight holds {} forward and {} transposed group weights for groups {}",
            prepared.forward.size(),
            prepared.transposed.size(),
            groups));
    }
    const uint32_t expected_c_in_block = geometry.c_in_block();
    if (prepared.c_in_block != expected_c_in_block) {
        throw std::invalid_argument(fmt::format(
            "conv3d: prepared weight has C_in block {}, expected {} for this kernel",
            prepared.c_in_block,
            expected_c_in_block));
    }
    const GroupGeometry group = geometry.per_group();
    const uint32_t kvol = geometry.kernel_volume();
    auto check_form = [&](const ttnn::Tensor& tensor, uint32_t rows, uint32_t cols, const char* form, size_t g) {
        const auto& shape = tensor.logical_shape();
        if (!ttnn::is_device_tensor(tensor) || tensor.layout() != ttnn::Layout::TILE || shape.rank() != 2U ||
            shape[0] != rows || shape[1] != cols) {
            throw std::invalid_argument(fmt::format(
                "conv3d: prepared {} weight for group {} must be a device TILE tensor of shape [{}, {}], got shape {} "
                "layout {}",
                form,
                g,
                rows,
                cols,
                shape,
                tensor.layout()));
        }
    };
    for (size_t g = 0; g < prepared.forward.size(); ++g) {
        check_form(prepared.forward[g], kvol * group.c_in_padded(), group.c_out_padded(), "forward", g);
    }
    for (size_t g = 0; g < prepared.transposed.size(); ++g) {
        check_form(prepared.transposed[g], kvol * group.c_out_padded(), group.c_in_padded(), "transposed", g);
    }
}

autograd::TensorPtr conv3d_impl(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    GroupWeights weights,
    const Conv3dGeometry& geometry) {
    const auto& input_value = input->get_value();
    const GroupGeometry group = geometry.per_group();
    const uint32_t c_in_padded = group.c_in_padded();
    const uint32_t c_out_padded = group.c_out_padded();

    auto input_row_major = to_row_major(input_value);
    std::optional<ttnn::Tensor> bias_row;
    if (bias != nullptr) {
        bias_row = to_row_major(ttnn::reshape(bias->get_value(), ttnn::Shape({1U, geometry.C_out})));
    }

    auto group_outputs = map_groups(geometry.groups, [&](uint32_t g) {
        auto group_input = slice_channels(input_row_major, g * group.C_in, group.C_in);
        group_input = pad_after(group_input, /*dim=*/4, c_in_padded - group.C_in);

        std::optional<ttnn::Tensor> bias_tile;
        if (bias_row.has_value()) {
            auto group_bias = slice_dim(bias_row.value(), /*dim=*/1, g * group.C_out, group.C_out);
            bias_tile = to_tile(pad_after(group_bias, /*dim=*/1, c_out_padded - group.C_out));
        }

        auto output = run_ttnn_conv3d(
            group_input,
            weights.forward[g],
            bias_tile,
            c_out_padded,
            group.kernel,
            group.stride,
            group.padding,
            group.dilation);
        return slice_channels(output, 0U, group.C_out);
    });
    auto output = concat_or_single(group_outputs, /*dim=*/4);

    auto out = autograd::create_tensor(to_layout_of(output, input_value));

    // Only caller-supplied transposed forms are kept across backward calls; their validity is the caller's contract.
    auto caller_transposed = std::make_shared<const std::vector<PreparedWeight>>(std::move(weights.transposed));

    autograd::GradFunction grad = [input, weight, bias, out, geometry, caller_transposed]() {
        const auto& grad_output = out->get_grad();
        const auto& input_value = input->get_value();
        const auto& weight_value = weight->get_value();
        const GroupGeometry group = geometry.per_group();
        auto grad_output_row_major = to_row_major(grad_output);

        if (input->get_requires_grad()) {
            // Built from the weight's current value on every call, like every other op's closure reads its
            // parameters at backward time, so a retained graph never computes dX from a stale weight.
            std::vector<PreparedWeight> built;
            const std::vector<PreparedWeight>* transposed = caller_transposed.get();
            if (transposed->empty()) {
                built = prepare_group_weights(to_row_major(weight_value), geometry, /*transposed=*/true);
                transposed = &built;
            }
            auto group_grads = map_groups(geometry.groups, [&](uint32_t g) {
                return conv3d_input_grad(
                    slice_channels(grad_output_row_major, g * group.C_out, group.C_out), (*transposed)[g], group);
            });
            input->add_grad(to_layout_of(concat_or_single(group_grads, /*dim=*/4), input_value));
        }

        const bool needs_weight_grad = weight->get_requires_grad();
        const bool needs_bias_grad = bias != nullptr && bias->get_requires_grad();
        if (needs_weight_grad || needs_bias_grad) {
            const uint32_t M = geometry.num_output_positions();
            auto flatten_to_tile = [M](const ttnn::Tensor& row_major, uint32_t channels) {
                return to_tile(ttnn::reshape(row_major, ttnn::Shape({M, channels})));
            };
            std::optional<ttnn::Tensor> grad_output_flat;

            if (needs_weight_grad) {
                auto padded_input = to_row_major(input_value);
                for (size_t i = 0; i < 3; ++i) {
                    padded_input = pad_spatial_dim_both_sides(padded_input, i + 1, geometry.padding[i]);
                }
                auto group_grads = map_groups(geometry.groups, [&](uint32_t g) {
                    ttnn::Tensor group_flat;
                    if (geometry.groups == 1U) {
                        grad_output_flat = flatten_to_tile(grad_output_row_major, geometry.C_out);
                        group_flat = grad_output_flat.value();
                    } else {
                        group_flat = flatten_to_tile(
                            slice_channels(grad_output_row_major, g * group.C_out, group.C_out), group.C_out);
                    }
                    auto group_grad_output_t = ttnn::transpose(group_flat, -2, -1);
                    return conv3d_weight_grad(group_grad_output_t, padded_input, g * group.C_in, group);
                });
                weight->add_grad(to_layout_of(concat_or_single(group_grads, /*dim=*/0), weight_value));
            }

            if (needs_bias_grad) {
                if (!grad_output_flat.has_value()) {
                    grad_output_flat = flatten_to_tile(grad_output_row_major, geometry.C_out);
                }
                const auto& bias_value = bias->get_value();
                auto grad_bias = ttnn_fixed::sum_over_dim(grad_output_flat.value(), /*dim=*/0);  // [1, C_out]
                grad_bias = to_layout_of(grad_bias, bias_value);
                bias->add_grad(ttnn::reshape(grad_bias, bias_value.logical_shape()));
            }
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, input, weight, bias));
    return out;
}

}  // namespace

Conv3dPreparedWeight prepare_conv3d_weight(const ttnn::Tensor& weight, uint32_t groups, bool with_transposed) {
    const auto& shape = weight.logical_shape();
    if (shape.rank() != 5U) {
        throw std::invalid_argument(fmt::format(
            "conv3d: weight must have rank 5 with layout [C_out, C_in / groups, kD, kH, kW], got shape {}", shape));
    }
    if (groups == 0U || shape[0] % groups != 0U) {
        throw std::invalid_argument(
            fmt::format("conv3d: out_channels {} must be divisible by groups {} (groups >= 1)", shape[0], groups));
    }
    if (!ttnn::is_device_tensor(weight)) {
        throw std::invalid_argument("conv3d: weight must be on device");
    }
    Conv3dGeometry geometry;
    geometry.C_in = shape[1] * groups;
    geometry.C_out = shape[0];
    geometry.groups = groups;
    geometry.kernel = {shape[2], shape[3], shape[4]};

    auto weight_row_major = to_row_major(weight);
    Conv3dPreparedWeight prepared;
    prepared.weight_shape = shape;
    prepared.groups = groups;
    prepared.c_in_block = ttnn::operations::experimental::conv3d::default_c_in_block(geometry.kernel_volume());
    for (const auto& w : prepare_group_weights(weight_row_major, geometry, /*transposed=*/false)) {
        prepared.forward.push_back(w.tensor);
    }
    if (with_transposed) {
        for (const auto& w : prepare_group_weights(weight_row_major, geometry, /*transposed=*/true)) {
            prepared.transposed.push_back(w.tensor);
        }
    }
    return prepared;
}

autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const Conv3dDims& stride,
    const Conv3dDims& padding,
    const Conv3dDims& dilation,
    uint32_t groups,
    const std::string& padding_mode) {
    const Conv3dGeometry geometry =
        validate_and_build_geometry(input, weight, bias, stride, padding, dilation, groups, padding_mode);
    GroupWeights weights;
    weights.forward = prepare_group_weights(to_row_major(weight->get_value()), geometry, /*transposed=*/false);
    return conv3d_impl(input, weight, bias, std::move(weights), geometry);
}

autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const Conv3dPreparedWeight& prepared,
    const Conv3dDims& stride,
    const Conv3dDims& padding,
    const Conv3dDims& dilation,
    uint32_t groups,
    const std::string& padding_mode) {
    const Conv3dGeometry geometry =
        validate_and_build_geometry(input, weight, bias, stride, padding, dilation, groups, padding_mode);
    validate_prepared_weight(prepared, weight->get_value(), geometry);
    GroupWeights weights;
    weights.forward = to_internal(prepared.forward, prepared.c_in_block);
    weights.transposed = to_internal(prepared.transposed, prepared.c_in_block);
    return conv3d_impl(input, weight, bias, std::move(weights), geometry);
}

}  // namespace ttml::ops
