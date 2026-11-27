// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_op.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdint>
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

// Output channels are padded to the kernel's C_out block; activation channels only to its C_in block (16 or 32),
// since padding a 16-channel activation to a tile doubles its copy and its MACs for the same math.
constexpr uint32_t kCOutBlock = tt::constants::TILE_WIDTH;

// C_in is the activation in the forward pass and C_out in the backward pass, where dY feeds the transposed
// convolution, so each channel count is padded by role rather than by name.
struct Conv3dGeometry {
    uint32_t N = 0;
    Conv3dDims in_size{};   // D, H, W
    Conv3dDims out_size{};  // D_out, H_out, W_out
    uint32_t C_in = 0;
    uint32_t C_out = 0;
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
    [[nodiscard]] uint32_t c_in_block() const {
        return ttnn::operations::experimental::conv3d::default_c_in_block(kernel_volume());
    }
    [[nodiscard]] uint32_t padded_activation(uint32_t channels) const {
        return tt::round_up(channels, c_in_block());
    }
    [[nodiscard]] static uint32_t padded_output(uint32_t channels) {
        return tt::round_up(channels, kCOutBlock);
    }
};

// Every tensor crossing the public boundary is an allocated device tensor of a kernel-supported dtype.
void require_device_tensor(const ttnn::Tensor& tensor, const std::string& name) {
    if (!ttnn::is_device_tensor(tensor) || !tensor.is_allocated()) {
        throw std::invalid_argument(fmt::format("conv3d: {} must be an allocated device tensor", name));
    }
    if (tensor.dtype() != ttnn::DataType::BFLOAT16 && tensor.dtype() != ttnn::DataType::FLOAT32) {
        throw std::invalid_argument(
            fmt::format("conv3d: {} dtype must be bfloat16 or float32, got {}", name, tensor.dtype()));
    }
}

// Same as above, plus the tensor lives on the reference's device with the reference's dtype.
void require_device_tensor_like(
    const ttnn::Tensor& tensor, const std::string& name, const ttnn::Tensor& reference, const char* reference_name) {
    require_device_tensor(tensor, name);
    if (tensor.device() != reference.device()) {
        throw std::invalid_argument(fmt::format("conv3d: {} must be on the same device as {}", name, reference_name));
    }
    if (tensor.dtype() != reference.dtype()) {
        throw std::invalid_argument(fmt::format(
            "conv3d: {} dtype {} must match {} dtype {}", name, tensor.dtype(), reference_name, reference.dtype()));
    }
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

// ttnn's preparation pads C_in itself and stays on device for a device tensor; C_out must already be the launch
// count.
PreparedWeight prepare_weight(const ttnn::Tensor& raw_layout_weight) {
    const auto& shape = raw_layout_weight.logical_shape();
    const uint32_t c_in_block =
        ttnn::operations::experimental::conv3d::default_c_in_block(shape[2] * shape[3] * shape[4]);
    auto prepared = ttnn::operations::experimental::conv3d::prepare_conv3d_weights(
        raw_layout_weight, /*groups=*/1U, c_in_block, /*alignment=*/c_in_block, raw_layout_weight.device());
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

ttnn::Tensor slice_channels(const ttnn::Tensor& tensor, uint32_t count) {
    const auto& shape = tensor.logical_shape();
    if (count == shape[4]) {
        return tensor;
    }
    ttsl::SmallVector<uint32_t> begins(shape.rank(), 0U);
    ttsl::SmallVector<uint32_t> ends(shape.cbegin(), shape.cend());
    ttsl::SmallVector<uint32_t> steps(shape.rank(), 1U);
    ends[4] = count;
    return ttnn::slice(tensor, begins, ends, steps);
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

// The kernel tilizes one output block into TILE_HEIGHT matmul rows, so the block is sized to fill a tile: W first,
// then H, then D, keeping the product within one tile so the circular buffers stay single-tile. Ragged edges are
// clamped by the kernel.
Conv3dDims output_block(
    const ttnn::Tensor& input_row_major,
    const Conv3dDims& kernel,
    const Conv3dDims& stride,
    const Conv3dDims& padding,
    const Conv3dDims& dilation) {
    const auto& shape = input_row_major.logical_shape();
    Conv3dDims block{};
    uint32_t remaining = tt::constants::TILE_HEIGHT;
    for (size_t i = 3; i-- > 0;) {
        const uint32_t effective_kernel = dilation[i] * (kernel[i] - 1U) + 1U;
        const uint32_t out = (shape[i + 1] + 2U * padding[i] - effective_kernel) / stride[i] + 1U;
        block[i] = std::min(out, remaining);
        remaining /= block[i];
    }
    return block;
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
    const Conv3dDims block = output_block(input_row_major, kernel, stride, padding, dilation);
    const ttnn::experimental::prim::Conv3dConfig config(
        /*weights_dtype=*/input_row_major.dtype(),
        /*output_layout=*/ttnn::Layout::ROW_MAJOR,
        /*T_out_block=*/block[0],
        /*W_out_block=*/block[2],
        /*H_out_block=*/block[1],
        /*C_out_block=*/kCOutBlock,
        prepared_weight.c_in_block,
        dilation,
        /*alignment=*/prepared_weight.c_in_block,
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
    const ttnn::Tensor& grad_output_row_major,
    const PreparedWeight& transposed_weight,
    const Conv3dGeometry& geometry) {
    const uint32_t c_in_padded = geometry.padded_output(geometry.C_in);
    const uint32_t c_out_padded = geometry.padded_activation(geometry.C_out);

    auto grad = pad_after(grad_output_row_major, /*dim=*/4, c_out_padded - geometry.C_out);

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
    const ttnn::Tensor& padded_input,        // [N, D + 2p, H + 2p, W + 2p, C_in] ROW_MAJOR
    const Conv3dGeometry& geometry) {
    const uint32_t M = geometry.num_output_positions();
    std::vector<ttnn::Tensor> per_offset_grads;
    per_offset_grads.reserve(geometry.kernel_volume());
    for (uint32_t kd = 0; kd < geometry.kernel[0]; ++kd) {
        for (uint32_t kh = 0; kh < geometry.kernel[1]; ++kh) {
            for (uint32_t kw = 0; kw < geometry.kernel[2]; ++kw) {
                const Conv3dDims offset{
                    kd * geometry.dilation[0], kh * geometry.dilation[1], kw * geometry.dilation[2]};
                ttsl::SmallVector<uint32_t> begins{0U, offset[0], offset[1], offset[2], 0U};
                ttsl::SmallVector<uint32_t> ends{
                    geometry.N,
                    offset[0] + (geometry.out_size[0] - 1U) * geometry.stride[0] + 1U,
                    offset[1] + (geometry.out_size[1] - 1U) * geometry.stride[1] + 1U,
                    offset[2] + (geometry.out_size[2] - 1U) * geometry.stride[2] + 1U,
                    geometry.C_in};
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
    const Conv3dDims& dilation) {
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
    require_device_tensor(input_value, "input");
    require_device_tensor_like(weight_value, "weight", input_value, "input");

    Conv3dGeometry geometry;
    geometry.N = input_shape[0];
    geometry.in_size = {input_shape[1], input_shape[2], input_shape[3]};
    geometry.C_in = input_shape[4];
    geometry.C_out = weight_shape[0];
    geometry.kernel = {weight_shape[2], weight_shape[3], weight_shape[4]};
    geometry.stride = stride;
    geometry.padding = padding;
    geometry.dilation = dilation;

    if (weight_shape[1] != geometry.C_in) {
        throw std::invalid_argument(
            fmt::format("conv3d: weight in_channels {} must equal input channels {}", weight_shape[1], geometry.C_in));
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
        require_device_tensor_like(bias_value, "bias", input_value, "input");
        if (bias_shape.volume() != geometry.C_out || bias_shape[-1] != geometry.C_out) {
            throw std::invalid_argument(fmt::format(
                "conv3d: bias must have volume {} with last dimension {}, got shape {}",
                geometry.C_out,
                geometry.C_out,
                bias_shape));
        }
    }

    // One core per C_in block of the kernel's activation; in dX that activation is dY, hence the C_out check.
    const auto grid = input_value.device()->compute_with_storage_grid_size();
    const uint32_t cores = static_cast<uint32_t>(grid.x * grid.y);
    const uint32_t c_in_block = geometry.c_in_block();
    if (geometry.padded_activation(geometry.C_in) / c_in_block > cores) {
        throw std::invalid_argument(fmt::format(
            "conv3d: {} input channels need {} channel blocks of {} but the device has {} cores",
            geometry.C_in,
            geometry.padded_activation(geometry.C_in) / c_in_block,
            c_in_block,
            cores));
    }
    if (input->get_requires_grad() && geometry.padded_activation(geometry.C_out) / c_in_block > cores) {
        throw std::invalid_argument(fmt::format(
            "conv3d: the input gradient runs a transposed conv over {} output channels, needing {} channel blocks "
            "of {} but the device has {} cores; reduce C_out or set input requires_grad to false",
            geometry.C_out,
            geometry.padded_activation(geometry.C_out) / c_in_block,
            c_in_block,
            cores));
    }
    return geometry;
}

}  // namespace

autograd::TensorPtr conv3d(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const Conv3dDims& stride,
    const Conv3dDims& padding,
    const Conv3dDims& dilation) {
    const Conv3dGeometry geometry = validate_and_build_geometry(input, weight, bias, stride, padding, dilation);
    const auto& input_value = input->get_value();
    const uint32_t c_in_padded = geometry.padded_activation(geometry.C_in);
    const uint32_t c_out_padded = geometry.padded_output(geometry.C_out);

    auto input_row_major = pad_after(to_row_major(input_value), /*dim=*/4, c_in_padded - geometry.C_in);
    std::optional<ttnn::Tensor> bias_tile;
    if (bias != nullptr) {
        auto bias_row = to_row_major(ttnn::reshape(bias->get_value(), ttnn::Shape({1U, geometry.C_out})));
        bias_tile = to_tile(pad_after(bias_row, /*dim=*/1, c_out_padded - geometry.C_out));
    }

    auto output = run_ttnn_conv3d(
        input_row_major,
        prepare_forward_weight(weight->get_value(), c_out_padded),
        bias_tile,
        c_out_padded,
        geometry.kernel,
        geometry.stride,
        geometry.padding,
        geometry.dilation);
    output = slice_channels(output, geometry.C_out);

    auto out = autograd::create_tensor(to_layout_of(output, input_value));

    autograd::GradFunction grad = [input, weight, bias, out, geometry]() {
        const auto& grad_output = out->get_grad();
        const auto& input_value = input->get_value();
        const auto& weight_value = weight->get_value();
        auto grad_output_row_major = to_row_major(grad_output);

        if (input->get_requires_grad()) {
            // Built from the weight's current value on every call, like every other op's closure reads its
            // parameters at backward time, so a retained graph never computes dX from a stale weight.
            const auto transposed = prepare_transposed_weight(
                weight_value, geometry.padded_output(geometry.C_in), geometry.padded_activation(geometry.C_out));
            input->add_grad(to_layout_of(conv3d_input_grad(grad_output_row_major, transposed, geometry), input_value));
        }

        const bool needs_weight_grad = weight->get_requires_grad();
        const bool needs_bias_grad = bias != nullptr && bias->get_requires_grad();
        if (needs_weight_grad || needs_bias_grad) {
            const uint32_t M = geometry.num_output_positions();
            auto grad_output_flat = to_tile(ttnn::reshape(grad_output_row_major, ttnn::Shape({M, geometry.C_out})));

            if (needs_weight_grad) {
                auto padded_input = to_row_major(input_value);
                for (size_t i = 0; i < 3; ++i) {
                    padded_input = pad_spatial_dim_both_sides(padded_input, i + 1, geometry.padding[i]);
                }
                auto grad_output_t = ttnn::transpose(grad_output_flat, -2, -1);
                weight->add_grad(to_layout_of(conv3d_weight_grad(grad_output_t, padded_input, geometry), weight_value));
            }

            if (needs_bias_grad) {
                const auto& bias_value = bias->get_value();
                auto grad_bias = ttnn_fixed::sum_over_dim(grad_output_flat, /*dim=*/0);  // [1, C_out]
                grad_bias = to_layout_of(grad_bias, bias_value);
                bias->add_grad(ttnn::reshape(grad_bias, bias_value.logical_shape()));
            }
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, input, weight, bias));
    return out;
}

}  // namespace ttml::ops
