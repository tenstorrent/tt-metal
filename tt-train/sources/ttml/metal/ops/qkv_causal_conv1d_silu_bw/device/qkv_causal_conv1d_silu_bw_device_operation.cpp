// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_bw_device_operation.hpp"

#include <array>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

namespace {

constexpr std::string_view op_name = "qkv_causal_conv1d_silu_bw";

void check_tensor(const Tensor& reference, const Tensor& tensor, Layout layout, std::string_view name) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE && tensor.buffer() != nullptr,
        "{}: {} must be an allocated device tensor",
        op_name,
        name);
    TT_FATAL(tensor.layout() == layout, "{}: {} must be {}", op_name, name, enchantum::to_string(layout));
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{}: {} must be BFLOAT16", op_name, name);
    TT_FATAL(!tensor.is_sharded(), "{}: {} must use interleaved memory", op_name, name);
    TT_FATAL(tensor.device() == reference.device(), "{}: {} must be on the same device", op_name, name);
}

uint64_t channels(const QkvCausalConv1dSiluBackwardParams& attrs) {
    return static_cast<uint64_t>(attrs.q_width) + attrs.k_width + attrs.v_width;
}

}  // namespace

QkvCausalConv1dSiluBackwardOperation::program_factory_t QkvCausalConv1dSiluBackwardOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return QkvCausalConv1dSiluBackwardProgramFactory{};
}

void QkvCausalConv1dSiluBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr uint32_t tw = tt::constants::TILE_WIDTH;
    constexpr uint32_t th = tt::constants::TILE_HEIGHT;
    const uint64_t c = channels(attrs);
    TT_FATAL(attrs.q_width > 0 && attrs.k_width > 0 && attrs.v_width > 0, "{}: Q/K/V widths must be positive", op_name);
    TT_FATAL(
        attrs.q_width % tw == 0 && attrs.k_width % tw == 0 && attrs.v_width % tw == 0,
        "{}: Q/K/V widths must be tile aligned",
        op_name);
    TT_FATAL(attrs.sequence > 0 && attrs.sequence % th == 0, "{}: sequence must be positive and tile aligned", op_name);
    TT_FATAL(
        attrs.channel_chunk_size > 0 && attrs.channel_chunk_size % tw == 0 && attrs.channel_chunk_size <= c &&
            c % attrs.channel_chunk_size == 0,
        "{}: channel_chunk_size must be a tile-aligned divisor of Q+K+V",
        op_name);
    TT_FATAL(
        attrs.sequence_splits >= 1 && attrs.sequence_splits <= attrs.sequence / th,
        "{}: sequence_splits must be in [1, sequence / 32]",
        op_name);
    TT_FATAL(!attrs.output_mem_config.is_sharded(), "{}: output memory layout must be INTERLEAVED", op_name);
    TT_FATAL(!attrs.compute_kernel_config.packer_l1_acc, "{}: packer_l1_acc is unsupported", op_name);

    check_tensor(in.input, in.input, Layout::ROW_MAJOR, "input");
    check_tensor(in.input, in.history, Layout::ROW_MAJOR, "history");
    check_tensor(in.input, in.grad_q, Layout::TILE, "grad_q");
    check_tensor(in.input, in.grad_k, Layout::TILE, "grad_k");
    check_tensor(in.input, in.grad_v, Layout::TILE, "grad_v");
    for (const auto* tap : std::array{&in.tap0, &in.tap1, &in.tap2, &in.tap3}) {
        check_tensor(in.input, *tap, Layout::TILE, "tap");
        TT_FATAL(
            tap->logical_shape()[-1] == c && tap->logical_volume() == c,
            "{}: each tap must have logical volume Q+K+V on its last dimension",
            op_name);
    }
    TT_FATAL(
        in.input.logical_shape() == Shape({1, attrs.sequence, static_cast<uint32_t>(c)}),
        "{}: input must be [1,T,Q+K+V]",
        op_name);
    TT_FATAL(
        in.history.logical_shape() == Shape({1, 3, static_cast<uint32_t>(c)}),
        "{}: history must be [1,3,Q+K+V]",
        op_name);
    TT_FATAL(
        in.grad_q.logical_shape() == Shape({1, attrs.sequence, attrs.q_width}), "{}: grad_q must be [1,T,Q]", op_name);
    TT_FATAL(
        in.grad_k.logical_shape() == Shape({1, attrs.sequence, attrs.k_width}), "{}: grad_k must be [1,T,K]", op_name);
    TT_FATAL(
        in.grad_v.logical_shape() == Shape({1, attrs.sequence, attrs.v_width}), "{}: grad_v must be [1,T,V]", op_name);

    const auto* mesh = in.input.device();
    const uint64_t available_l1_bytes =
        mesh->l1_size_per_core() - mesh->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint64_t required_l1_bytes = qkv_conv_bw_l1_bytes(
        attrs.channel_chunk_size / tw, datatype_to_dataformat_converter(in.input.dtype()), in.input.element_size());
    TT_FATAL(
        required_l1_bytes <= available_l1_bytes,
        "{}: channel_chunk_size = {} needs {} bytes of L1 per core, only {} are available",
        op_name,
        attrs.channel_chunk_size,
        required_l1_bytes,
        available_l1_bytes);
}

QkvCausalConv1dSiluBackwardOperation::spec_return_value_t QkvCausalConv1dSiluBackwardOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto c = static_cast<uint32_t>(channels(attrs));
    return {
        TensorSpec(
            Shape({1, attrs.sequence, c}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config)),
        TensorSpec(
            Shape({qkv_conv_bw_tap_count, attrs.sequence_splits * tt::constants::TILE_HEIGHT, c}),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config))};
}

QkvCausalConv1dSiluBackwardOperation::tensor_return_value_t QkvCausalConv1dSiluBackwardOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto specs = compute_output_specs(attrs, in);
    return {
        ttnn::create_device_tensor(specs[0], in.input.device()),
        ttnn::create_device_tensor(specs[1], in.input.device())};
}

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device

namespace ttnn::prim {

std::vector<ttnn::Tensor> ttml_qkv_causal_conv1d_silu_bw(
    const ttnn::Tensor& grad_q,
    const ttnn::Tensor& grad_k,
    const ttnn::Tensor& grad_v,
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    const ttml::metal::ops::qkv_causal_conv1d_silu_bw::device::QkvCausalConv1dSiluBackwardParams& params) {
    using namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device;
    return ttnn::device_operation::launch<QkvCausalConv1dSiluBackwardOperation>(
        params,
        QkvCausalConv1dSiluBackwardInputs{
            .grad_q = grad_q,
            .grad_k = grad_k,
            .grad_v = grad_v,
            .input = input,
            .history = history,
            .tap0 = tap0,
            .tap1 = tap1,
            .tap2 = tap2,
            .tap3 = tap3});
}

}  // namespace ttnn::prim
