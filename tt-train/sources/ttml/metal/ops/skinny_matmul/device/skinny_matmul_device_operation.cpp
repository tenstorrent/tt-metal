// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "skinny_matmul_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::skinny_matmul::device {

namespace {

void check_tensor(const ttnn::Tensor& tensor, const std::string& name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "SkinnyMatmul requires {} on Device, got {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "SkinnyMatmul: {} has no device buffer", name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "SkinnyMatmul requires {} in Tile layout, got {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == dtype,
        "SkinnyMatmul requires {} of {} data type, got {}",
        name,
        enchantum::to_string(dtype),
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "SkinnyMatmul requires interleaved {}, got `{}`",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
    TT_FATAL(
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "SkinnyMatmul requires {} in DRAM", name);
}

void validate_inputs(const Params& args, const ttnn::Tensor& x, const ttnn::Tensor& w) {
    check_tensor(x, "x", tt::tt_metal::DataType::BFLOAT16);
    check_tensor(w, "w", tt::tt_metal::DataType::BFLOAT16);
    const auto& xs = x.logical_shape();
    const auto& ws = w.logical_shape();
    TT_FATAL(xs.rank() == 4 && ws.rank() == 4, "SkinnyMatmul: x {} and w {} must be 4D", xs, ws);
    TT_FATAL(ws[0] == 1 && ws[1] == 1, "SkinnyMatmul: w must be [1, 1, ., .], got {}", ws);
    if (args.x_transposed) {
        TT_FATAL(
            xs[0] == 1 && xs[1] == 1 && x.padded_shape() == xs,
            "SkinnyMatmul: a transposed x must be [1, 1, K', M] without padding, got {}",
            xs);
        TT_FATAL(
            args.transpose_b && args.transpose_out,
            "SkinnyMatmul: a transposed x needs w as [N, K] (transpose_b) and gives the transposed output");
    } else {
        TT_FATAL(
            !args.transpose_out,
            "SkinnyMatmul: the transposed output is only produced from a transposed x (the token-major reduce path "
            "has no harness case for it; review round 12)");
    }
    const uint32_t K = args.x_transposed ? xs[2] : xs[3];
    TT_FATAL(K % tt::constants::TILE_WIDTH == 0, "SkinnyMatmul: K must be a multiple of 32, got {}", K);
    const uint32_t wK = args.transpose_b ? ws[3] : ws[2];
    // x may be wider than w's K (the first wK columns are the product's K, the rest is the wide LoRA buffer's tail).
    TT_FATAL(
        wK % tt::constants::TILE_WIDTH == 0 && wK <= K,
        "SkinnyMatmul: K of w ({}) must be a multiple of 32 and at most x's width ({}) (transpose_b={})",
        wK,
        K,
        args.transpose_b);
    const auto geo = skinny_geometry(x, w, args.transpose_b, args.x_transposed);
    TT_FATAL(geo.Nt >= 1 && geo.Nt <= 2, "SkinnyMatmul: N must be <= 64 (got {} columns)", geo.n_logical);
    TT_FATAL(args.rows_per_item >= 1 && args.num_slices >= 1 && args.slice_tiles >= 1, "SkinnyMatmul: bad split");
    TT_FATAL(
        (args.num_slices - 1U) * args.slice_tiles < geo.Kt && args.num_slices * args.slice_tiles >= geo.Kt,
        "SkinnyMatmul: split ({} slices x {} tiles) does not cover Kt={}",
        args.num_slices,
        args.slice_tiles,
        geo.Kt);
}

}  // namespace

SkinnyGeometry skinny_geometry(const ttnn::Tensor& x, const ttnn::Tensor& w, bool transpose_b, bool x_transposed) {
    const auto& xp = x.padded_shape();
    const auto& wp = w.padded_shape();
    SkinnyGeometry g;
    // x transposed (iteration 40): [1, 1, K', M], tile (m, k) at page k * Mt + m: the row stride is Mt
    g.rows = x_transposed ? xp[3] / tt::constants::TILE_WIDTH : xp[0] * xp[1] * (xp[2] / tt::constants::TILE_HEIGHT);
    g.Kt = (transpose_b ? wp[3] : wp[2]) / tt::constants::TILE_WIDTH;
    g.Kx = x_transposed ? g.rows : xp[3] / tt::constants::TILE_WIDTH;
    g.Nt = (transpose_b ? wp[2] : wp[3]) / tt::constants::TILE_WIDTH;
    g.n_logical = transpose_b ? w.logical_shape()[2] : w.logical_shape()[3];
    return g;
}

// ---------------------------------------------------------------------------------------------------------------
// Phase A
// ---------------------------------------------------------------------------------------------------------------

void SkinnyMatmulPartialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_inputs(args, tensor_args.x, tensor_args.w);
}

partial::spec_return_value_t SkinnyMatmulPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto geo = skinny_geometry(tensor_args.x, tensor_args.w, args.transpose_b, args.x_transposed);
    return tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, geo.rows * tt::constants::TILE_HEIGHT, args.num_slices * geo.Nt * tt::constants::TILE_WIDTH},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tensor_args.x.memory_config()));
}

partial::tensor_return_value_t SkinnyMatmulPartialDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.x.device());
}

ttsl::hash::hash_t SkinnyMatmulPartialDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SkinnyMatmulPartialDeviceOperation>(
        args.transpose_b,
        args.x_transposed,
        args.rows_per_item,
        args.num_slices,
        args.slice_tiles,
        tensor_args.x.logical_shape(),
        tensor_args.w.logical_shape());
}

// ---------------------------------------------------------------------------------------------------------------
// Phase B
// ---------------------------------------------------------------------------------------------------------------

void SkinnyMatmulReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_inputs(args, tensor_args.x, tensor_args.w);
    check_tensor(tensor_args.partials, "partials", tt::tt_metal::DataType::FLOAT32);
    const auto geo = skinny_geometry(tensor_args.x, tensor_args.w, args.transpose_b, args.x_transposed);
    if (tensor_args.tail_out.has_value()) {
        const auto& t = *tensor_args.tail_out;
        check_tensor(t, "tail_out", tt::tt_metal::DataType::BFLOAT16);
        const auto& tp = t.padded_shape();
        // transposed output: the tiles go to the tile rows tail_col .. of a [1, 1, R', M] target
        const uint32_t t_rows = args.transpose_out ? tp[3] / tt::constants::TILE_WIDTH
                                                   : tp[0] * tp[1] * (tp[2] / tt::constants::TILE_HEIGHT);
        const uint32_t t_cols = args.transpose_out ? tp[0] * tp[1] * (tp[2] / tt::constants::TILE_HEIGHT)
                                                   : tp[3] / tt::constants::TILE_WIDTH;
        TT_FATAL(
            t_rows == geo.rows && args.tail_tiles >= geo.Nt && args.tail_col + args.tail_tiles <= t_cols,
            "SkinnyMatmul: tail_out {} must have the x rows and tile columns tail_col {} + tail_tiles {} <= {} (Nt {})",
            tp,
            args.tail_col,
            args.tail_tiles,
            t_cols,
            geo.Nt);
    } else {
        TT_FATAL(args.tail_tiles == 0, "SkinnyMatmul: tail_tiles without tail_out");
    }
    const auto& ps = tensor_args.partials.logical_shape();
    TT_FATAL(
        ps.rank() == 4 && ps[2] == geo.rows * tt::constants::TILE_HEIGHT &&
            ps[3] == args.num_slices * geo.Nt * tt::constants::TILE_WIDTH,
        "SkinnyMatmul: partials must be [1, 1, {}, {}], got {}",
        geo.rows * tt::constants::TILE_HEIGHT,
        args.num_slices * geo.Nt * tt::constants::TILE_WIDTH,
        ps);
}

reduce::spec_return_value_t SkinnyMatmulReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto geo = skinny_geometry(tensor_args.x, tensor_args.w, args.transpose_b, args.x_transposed);
    const auto& xs = tensor_args.x.logical_shape();
    const uint32_t M = args.x_transposed ? xs[3] : xs[2];
    return tt::tt_metal::TensorSpec(
        args.transpose_out ? ttnn::Shape{xs[0], xs[1], geo.n_logical, M} : ttnn::Shape{xs[0], xs[1], M, geo.n_logical},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tensor_args.x.memory_config()));
}

reduce::tensor_return_value_t SkinnyMatmulReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.x.device());
}

ttsl::hash::hash_t SkinnyMatmulReduceDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SkinnyMatmulReduceDeviceOperation>(
        args.transpose_b,
        args.x_transposed,
        args.transpose_out,
        args.num_slices,
        args.tail_col,
        args.tail_tiles,
        tensor_args.x.logical_shape(),
        tensor_args.w.logical_shape(),
        tensor_args.tail_out.has_value() ? tensor_args.tail_out->padded_shape() : ttnn::Shape{});
}

}  // namespace ttml::metal::ops::skinny_matmul::device

namespace ttnn::prim {

ttnn::Tensor ttml_skinny_matmul_partial(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    const ttml::metal::ops::skinny_matmul::device::SkinnySplit& split,
    bool transpose_b,
    bool x_transposed) {
    using OperationType = ttml::metal::ops::skinny_matmul::device::SkinnyMatmulPartialDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .transpose_b = transpose_b,
        .rows_per_item = split.rows_per_item,
        .num_slices = split.num_slices,
        .slice_tiles = split.slice_tiles,
        .x_transposed = x_transposed,
        .transpose_out = x_transposed};
    auto tensor_args = OperationType::tensor_args_t{.x = x, .w = w};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

ttnn::Tensor ttml_skinny_matmul_reduce(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    const ttnn::Tensor& partials,
    const ttml::metal::ops::skinny_matmul::device::SkinnySplit& split,
    bool transpose_b,
    const std::optional<ttnn::Tensor>& tail_out,
    uint32_t tail_col,
    uint32_t tail_tiles,
    bool x_transposed,
    bool transpose_out) {
    using OperationType = ttml::metal::ops::skinny_matmul::device::SkinnyMatmulReduceDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .transpose_b = transpose_b,
        .rows_per_item = split.rows_per_item,
        .num_slices = split.num_slices,
        .slice_tiles = split.slice_tiles,
        .tail_col = tail_col,
        .tail_tiles = tail_tiles,
        .x_transposed = x_transposed,
        .transpose_out = transpose_out};
    auto tensor_args = OperationType::tensor_args_t{.x = x, .w = w, .partials = partials, .tail_out = tail_out};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
