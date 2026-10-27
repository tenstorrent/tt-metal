// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_bw_device_operation.hpp"

#include <array>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>

#include "metal/common/program_utils.hpp"
#include "metal/common/tensor_validation.hpp"

using namespace tt::tt_metal;

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

namespace {

constexpr std::string_view op_name = "qkv_causal_conv1d_silu_bw";

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
    TT_FATAL(
        attrs.row_count > 0 && attrs.row_count % th == 0, "{}: row_count must be positive and tile aligned", op_name);
    TT_FATAL(
        attrs.channel_chunk_size > 0 && attrs.channel_chunk_size % tw == 0 && attrs.channel_chunk_size <= c &&
            c % attrs.channel_chunk_size == 0,
        "{}: channel_chunk_size must be a tile-aligned divisor of Q+K+V",
        op_name);
    TT_FATAL(
        attrs.row_splits >= 1 && attrs.row_splits <= attrs.row_count / th,
        "{}: row_splits must be in [1, row_count / 32]",
        op_name);
    TT_FATAL(!attrs.output_mem_config.is_sharded(), "{}: output memory layout must be INTERLEAVED", op_name);

    check_device_tensor(in.input, op_name, "input");
    const auto check = [&](const Tensor& tensor, std::string_view name, Layout layout = Layout::TILE) {
        check_device_tensor(tensor, op_name, name, {.layout = layout});
        check_same_device(tensor, in.input, op_name, name, "input");
    };
    check(in.history, "history", Layout::ROW_MAJOR);
    check(in.grad_q, "grad_q");
    check(in.grad_k, "grad_k");
    check(in.grad_v, "grad_v");
    for (const auto& [tap, name] :
         {std::pair{&in.tap0, "tap0"},
          std::pair{&in.tap1, "tap1"},
          std::pair{&in.tap2, "tap2"},
          std::pair{&in.tap3, "tap3"}}) {
        check(*tap, name);
        TT_FATAL(
            tap->logical_shape()[-1] == c && tap->logical_volume() == c,
            "{}: {} must have logical volume Q+K+V on its last dimension",
            op_name,
            name);
    }
    TT_FATAL(
        in.input.logical_shape() == Shape({1, attrs.row_count, static_cast<uint32_t>(c)}),
        "{}: input must be [1,T,Q+K+V]",
        op_name);
    TT_FATAL(
        in.history.logical_shape() == Shape({1, qkv_conv_bw::halo_rows, static_cast<uint32_t>(c)}),
        "{}: history must be [1,3,Q+K+V]",
        op_name);
    TT_FATAL(
        in.grad_q.logical_shape() == Shape({1, attrs.row_count, attrs.q_width}), "{}: grad_q must be [1,T,Q]", op_name);
    TT_FATAL(
        in.grad_k.logical_shape() == Shape({1, attrs.row_count, attrs.k_width}), "{}: grad_k must be [1,T,K]", op_name);
    TT_FATAL(
        in.grad_v.logical_shape() == Shape({1, attrs.row_count, attrs.v_width}), "{}: grad_v must be [1,T,V]", op_name);
    if (in.preallocated_grad_input.has_value()) {
        check(*in.preallocated_grad_input, "preallocated_grad_input");
        TT_FATAL(
            in.preallocated_grad_input->logical_shape() == Shape({1, attrs.row_count, static_cast<uint32_t>(c)}),
            "{}: preallocated_grad_input must be [1,T,Q+K+V]",
            op_name);
        // With row_splits > 1, split s reads the bottom faces of input tile row end(s) on its halo step while split
        // s + 1 has already written grad_input over them.
        TT_FATAL(
            in.preallocated_grad_input->buffer()->address() != in.input.buffer()->address(),
            "{}: preallocated_grad_input must not share a buffer with input",
            op_name);
    }

    const uint64_t l1_bytes = available_l1_bytes(*in.input.device());
    const uint64_t required_l1_bytes = qkv_conv_bw_l1_bytes(
        attrs.channel_chunk_size / tw, datatype_to_dataformat_converter(in.input.dtype()), attrs.compute_tap_grads);
    TT_FATAL(
        required_l1_bytes <= l1_bytes,
        "{}: channel_chunk_size = {} needs {} bytes of L1 per core, only {} are available",
        op_name,
        attrs.channel_chunk_size,
        required_l1_bytes,
        l1_bytes);
}

QkvCausalConv1dSiluBackwardOperation::spec_return_value_t QkvCausalConv1dSiluBackwardOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto c = static_cast<uint32_t>(channels(attrs));
    spec_return_value_t specs{
        in.preallocated_grad_input.has_value()
            ? in.preallocated_grad_input->tensor_spec()
            : TensorSpec(
                  Shape({1, attrs.row_count, c}),
                  TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config))};
    if (attrs.compute_tap_grads) {
        specs.emplace_back(
            Shape({qkv_conv_bw_tap_count, attrs.row_splits * tt::constants::TILE_HEIGHT, c}),
            TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config));
    }
    return specs;
}

QkvCausalConv1dSiluBackwardOperation::tensor_return_value_t QkvCausalConv1dSiluBackwardOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto specs = compute_output_specs(attrs, in);
    tensor_return_value_t outputs{
        in.preallocated_grad_input.has_value() ? *in.preallocated_grad_input
                                               : ttnn::create_device_tensor(specs[0], in.input.device())};
    for (size_t i = 1; i < specs.size(); ++i) {
        outputs.push_back(ttnn::create_device_tensor(specs[i], in.input.device()));
    }
    return outputs;
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
    const ttml::metal::ops::qkv_causal_conv1d_silu_bw::device::QkvCausalConv1dSiluBackwardParams& params,
    const std::optional<ttnn::Tensor>& preallocated_grad_input) {
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
            .tap3 = tap3,
            .preallocated_grad_input = preallocated_grad_input});
}

}  // namespace ttnn::prim
