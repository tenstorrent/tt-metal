// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_q_rope_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "mla_q_rope_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::mla_q_rope::device {

void MlaQRopeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "MlaQRope requires {} on device. Got {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "MlaQRope: {} buffer must be allocated.", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "MlaQRope requires {} TILE layout. Got {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "MlaQRope requires {} BFLOAT16. Got {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "MlaQRope requires {} INTERLEAVED. Got {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& q_in = tensor_args.q_in;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;
    const auto& trans = tensor_args.trans_mat;

    check_tensor(q_in, "q_in");
    check_tensor(cos, "cos_cache");
    check_tensor(sin, "sin_cache");
    check_tensor(trans, "trans_mat");

    TT_FATAL(
        q_in.device() == cos.device() && cos.device() == sin.device() && sin.device() == trans.device(),
        "MlaQRope: all tensors must be on the same device.");

    const auto q_shape = q_in.padded_shape();
    TT_FATAL(q_shape.rank() == 4U, "MlaQRope: q_in must be rank-4. Got {}", q_shape.rank());
    TT_FATAL(q_shape[1] >= 1U, "MlaQRope: q_in must have at least one head.");

    const uint32_t qk_head = args.qk_nope_dim + args.qk_rope_dim;
    TT_FATAL(
        q_shape[3] == qk_head,
        "MlaQRope: q_in dim 3 must equal qk_nope_dim + qk_rope_dim = {}. Got {}",
        qk_head,
        q_shape[3]);

    TT_FATAL(
        args.qk_nope_dim % TILE_WIDTH == 0,
        "MlaQRope: qk_nope_dim ({}) must be a multiple of TILE_WIDTH ({})",
        args.qk_nope_dim,
        TILE_WIDTH);
    TT_FATAL(args.qk_rope_dim != 0U, "MlaQRope: qk_rope_dim must be non-zero.");
    TT_FATAL(
        args.qk_rope_dim % TILE_WIDTH == 0,
        "MlaQRope: qk_rope_dim ({}) must be a multiple of TILE_WIDTH ({})",
        args.qk_rope_dim,
        TILE_WIDTH);
    TT_FATAL(
        q_shape[2] % TILE_HEIGHT == 0,
        "MlaQRope: S ({}) must be a multiple of TILE_HEIGHT ({})",
        q_shape[2],
        TILE_HEIGHT);

    const auto cos_shape = cos.padded_shape();
    TT_FATAL(cos_shape == sin.padded_shape(), "MlaQRope: cos and sin shapes must match.");
    TT_FATAL(
        cos_shape[0] == 1U && cos_shape[1] == 1U, "MlaQRope: cos/sin dims 0-1 must be 1. Got cos shape {}", cos_shape);
    TT_FATAL(
        cos_shape[2] == q_shape[2],
        "MlaQRope: cos/sin seq dim ({}) must match q_in seq ({})",
        cos_shape[2],
        q_shape[2]);
    TT_FATAL(
        cos_shape[3] == args.qk_rope_dim,
        "MlaQRope: cos/sin dim 3 must equal qk_rope_dim ({}). Got {}",
        args.qk_rope_dim,
        cos_shape[3]);

    const auto trans_shape = trans.padded_shape();
    TT_FATAL(
        trans_shape[0] == 1U && trans_shape[1] == 1U && trans_shape[2] == TILE_HEIGHT && trans_shape[3] == TILE_WIDTH,
        "MlaQRope: trans_mat must be [1, 1, {}, {}]. Got {}",
        TILE_HEIGHT,
        TILE_WIDTH,
        trans_shape);

    TT_FATAL(
        args.qk_rope_dim <= 128U,
        "MlaQRope: qk_rope_dim must be <= 128 (fp32 dest accumulation). Got {}",
        args.qk_rope_dim);
}

MlaQRopeDeviceOperation::spec_return_value_t MlaQRopeDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& q_in = tensor_args.q_in;
    return tt::tt_metal::TensorSpec(
        q_in.logical_shape(),
        tt::tt_metal::TensorLayout(q_in.dtype(), tt::tt_metal::Layout::TILE, q_in.memory_config()));
}

MlaQRopeDeviceOperation::tensor_return_value_t MlaQRopeDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs({}, tensor_args);
    return create_device_tensor(spec, tensor_args.q_in.device());
}

ttsl::hash::hash_t MlaQRopeDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<MlaQRopeDeviceOperation>(
        args, tensor_args.q_in.dtype(), tensor_args.q_in.logical_shape(), tensor_args.cos_cache.logical_shape());
}

MlaQRopeDeviceOperation::program_factory_t MlaQRopeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MlaQRopeProgramFactory{};
}

}  // namespace ttml::metal::ops::mla_q_rope::device

namespace ttnn::prim {

ttml::metal::ops::mla_q_rope::device::MlaQRopeDeviceOperation::tensor_return_value_t ttml_mla_q_rope(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim) {
    using OperationType = ttml::metal::ops::mla_q_rope::device::MlaQRopeDeviceOperation;

    auto attrs = OperationType::operation_attributes_t{
        .qk_nope_dim = qk_nope_dim,
        .qk_rope_dim = qk_rope_dim,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q_in = q_in,
        .cos_cache = cos_cache,
        .sin_cache = sin_cache,
        .trans_mat = trans_mat,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
