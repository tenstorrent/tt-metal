// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "q_rope_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "q_rope_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::q_rope_fw::device {

void QRopeFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "QRopeFw requires {} on device. Got {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "QRopeFw: {} buffer must be allocated.", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "QRopeFw requires {} TILE layout. Got {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "QRopeFw requires {} BFLOAT16. Got {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "QRopeFw requires {} INTERLEAVED. Got {}",
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
        "QRopeFw: all tensors must be on the same device.");

    const auto q_shape = q_in.padded_shape();
    TT_FATAL(q_shape.rank() == 4U, "QRopeFw: q_in must be rank-4. Got {}", q_shape.rank());
    TT_FATAL(q_shape[1] >= 1U, "QRopeFw: q_in must have at least one head.");

    const uint32_t qk_head = args.qk_nope_dim + args.qk_rope_dim;
    TT_FATAL(
        q_shape[3] == qk_head,
        "QRopeFw: q_in dim 3 must equal qk_nope_dim + qk_rope_dim = {}. Got {}",
        qk_head,
        q_shape[3]);

    TT_FATAL(
        args.qk_nope_dim % TILE_WIDTH == 0,
        "QRopeFw: qk_nope_dim ({}) must be a multiple of TILE_WIDTH ({})",
        args.qk_nope_dim,
        TILE_WIDTH);
    TT_FATAL(args.qk_rope_dim != 0U, "QRopeFw: qk_rope_dim must be non-zero.");
    TT_FATAL(
        args.qk_rope_dim % TILE_WIDTH == 0,
        "QRopeFw: qk_rope_dim ({}) must be a multiple of TILE_WIDTH ({})",
        args.qk_rope_dim,
        TILE_WIDTH);
    TT_FATAL(
        q_shape[2] % TILE_HEIGHT == 0,
        "QRopeFw: S ({}) must be a multiple of TILE_HEIGHT ({})",
        q_shape[2],
        TILE_HEIGHT);

    const auto cos_shape = cos.padded_shape();
    TT_FATAL(cos_shape == sin.padded_shape(), "QRopeFw: cos and sin shapes must match.");
    TT_FATAL(
        cos_shape[0] == 1U && cos_shape[1] == 1U, "QRopeFw: cos/sin dims 0-1 must be 1. Got cos shape {}", cos_shape);
    TT_FATAL(
        cos_shape[2] == q_shape[2], "QRopeFw: cos/sin seq dim ({}) must match q_in seq ({})", cos_shape[2], q_shape[2]);
    TT_FATAL(
        cos_shape[3] == args.qk_rope_dim,
        "QRopeFw: cos/sin dim 3 must equal qk_rope_dim ({}). Got {}",
        args.qk_rope_dim,
        cos_shape[3]);

    const auto trans_shape = trans.padded_shape();
    TT_FATAL(
        trans_shape[0] == 1U && trans_shape[1] == 1U && trans_shape[2] == TILE_HEIGHT && trans_shape[3] == TILE_WIDTH,
        "QRopeFw: trans_mat must be [1, 1, {}, {}]. Got {}",
        TILE_HEIGHT,
        TILE_WIDTH,
        trans_shape);

    TT_FATAL(args.qk_rope_dim <= 256U, "QRopeFw: qk_rope_dim must be <= 256. Got {}", args.qk_rope_dim);

    TT_FATAL(
        !args.fp32_dest_acc_en || args.qk_rope_dim <= 128U,
        "QRopeFw: fp32_dest_acc_en may only be true when qk_rope_dim <= 128. Got qk_rope_dim={}",
        args.qk_rope_dim);
}

QRopeFwDeviceOperation::spec_return_value_t QRopeFwDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& q_in = tensor_args.q_in;
    return ttnn::TensorSpec(
        q_in.logical_shape(),
        tt::tt_metal::TensorLayout(q_in.dtype(), tt::tt_metal::Layout::TILE, q_in.memory_config()));
}

QRopeFwDeviceOperation::tensor_return_value_t QRopeFwDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs({}, tensor_args);
    return create_device_tensor(spec, tensor_args.q_in.device());
}

ttsl::hash::hash_t QRopeFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<QRopeFwDeviceOperation>(
        args, tensor_args.q_in.dtype(), tensor_args.q_in.logical_shape(), tensor_args.cos_cache.logical_shape());
}

QRopeFwDeviceOperation::program_factory_t QRopeFwDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return QRopeFwProgramFactory{};
}

}  // namespace ttml::metal::ops::q_rope_fw::device

namespace ttnn::prim {

ttml::metal::ops::q_rope_fw::device::QRopeFwDeviceOperation::tensor_return_value_t ttml_q_rope_fw(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    bool fp32_dest_acc_en) {
    using OperationType = ttml::metal::ops::q_rope_fw::device::QRopeFwDeviceOperation;

    auto attrs = OperationType::operation_attributes_t{
        .qk_nope_dim = qk_nope_dim,
        .qk_rope_dim = qk_rope_dim,
        .fp32_dest_acc_en = fp32_dest_acc_en,
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
