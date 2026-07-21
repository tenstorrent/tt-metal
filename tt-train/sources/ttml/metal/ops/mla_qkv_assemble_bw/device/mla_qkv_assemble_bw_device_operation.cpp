// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_qkv_assemble_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "mla_qkv_assemble_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::mla_qkv_assemble_bw::device {

void MLAQKVAssembleBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "MLAQKVAssembleBw requires {} to be on device. Got storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "MLAQKVAssembleBw: {} buffer must be allocated.", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "MLAQKVAssembleBw requires {} to be in TILE layout. Got: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "MLAQKVAssembleBw requires {} dtype to be BFLOAT16. Got: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "MLAQKVAssembleBw requires {} memory layout to be INTERLEAVED. Got: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& dQ = tensor_args.dQ;
    const auto& dK = tensor_args.dK;
    const auto& dV = tensor_args.dV;
    check_tensor(dQ, "dQ");
    check_tensor(dK, "dK");
    check_tensor(dV, "dV");

    // The program launches on dQ.device() and feeds raw buffer addresses from all three inputs into
    // the TensorAccessors, so mixed-device inputs must fail here rather than dispatch invalid addresses.
    TT_FATAL(
        dQ.device() == dK.device() && dK.device() == dV.device(),
        "MLAQKVAssembleBw: dQ, dK, and dV must be on the same device.");

    const auto dQ_shape = dQ.padded_shape();
    const auto dK_shape = dK.padded_shape();
    const auto dV_shape = dV.padded_shape();

    TT_FATAL(dQ_shape.rank() == 4U, "MLAQKVAssembleBw: dQ must be rank-4");
    TT_FATAL(dK_shape.rank() == 4U, "MLAQKVAssembleBw: dK must be rank-4");
    TT_FATAL(dV_shape.rank() == 4U, "MLAQKVAssembleBw: dV must be rank-4");

    TT_FATAL(
        dQ_shape[0] == dK_shape[0] && dK_shape[0] == dV_shape[0],
        "MLAQKVAssembleBw: batch dim must match across dQ ({}), dK ({}), dV ({}).",
        dQ_shape[0],
        dK_shape[0],
        dV_shape[0]);
    TT_FATAL(
        dQ_shape[1] == args.n_heads && dK_shape[1] == args.n_heads && dV_shape[1] == args.n_heads,
        "MLAQKVAssembleBw: dim 1 (heads) must equal n_heads ({}). Got dQ={}, dK={}, dV={}.",
        args.n_heads,
        dQ_shape[1],
        dK_shape[1],
        dV_shape[1]);
    TT_FATAL(
        dQ_shape[2] == dK_shape[2] && dK_shape[2] == dV_shape[2],
        "MLAQKVAssembleBw: seq dim must match across dQ ({}), dK ({}), dV ({}).",
        dQ_shape[2],
        dK_shape[2],
        dV_shape[2]);

    const uint32_t qk_head = args.qk_nope_dim + args.qk_rope_dim;
    TT_FATAL(
        dQ_shape[3] == qk_head,
        "MLAQKVAssembleBw: dQ dim 3 must equal qk_nope_dim + qk_rope_dim = {}. Got {}",
        qk_head,
        dQ_shape[3]);
    TT_FATAL(
        dK_shape[3] == qk_head,
        "MLAQKVAssembleBw: dK dim 3 must equal qk_nope_dim + qk_rope_dim = {}. Got {}",
        qk_head,
        dK_shape[3]);
    TT_FATAL(
        dV_shape[3] == args.v_dim, "MLAQKVAssembleBw: dV dim 3 must equal v_dim = {}. Got {}", args.v_dim, dV_shape[3]);

    TT_FATAL(
        args.qk_nope_dim % TILE_WIDTH == 0,
        "MLAQKVAssembleBw: qk_nope_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.qk_nope_dim,
        TILE_WIDTH);
    TT_FATAL(
        args.qk_rope_dim % TILE_WIDTH == 0,
        "MLAQKVAssembleBw: qk_rope_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.qk_rope_dim,
        TILE_WIDTH);
    TT_FATAL(
        args.v_dim % TILE_WIDTH == 0,
        "MLAQKVAssembleBw: v_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.v_dim,
        TILE_WIDTH);
    TT_FATAL(
        dQ_shape[2] % TILE_HEIGHT == 0,
        "MLAQKVAssembleBw: S ({}) must be a multiple of TILE_HEIGHT ({}).",
        dQ_shape[2],
        TILE_HEIGHT);
}

MLAQKVAssembleBwDeviceOperation::spec_return_value_t MLAQKVAssembleBwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(3U);

    const auto& dQ = tensor_args.dQ;
    const auto& dK = tensor_args.dK;
    const auto dQ_shape = dQ.logical_shape();
    const uint32_t B = dQ_shape[0];
    const uint32_t S = dQ_shape[2];
    const uint32_t qk_head = args.qk_nope_dim + args.qk_rope_dim;

    const ttnn::Shape dq_pre_shape({B, 1U, S, args.n_heads * qk_head});
    const ttnn::Shape dkv_up_shape({B, 1U, S, args.n_heads * (args.qk_nope_dim + args.v_dim)});
    const ttnn::Shape dk_pe_shape({B, 1U, S, args.qk_rope_dim});

    // Each output inherits from its primary input source. Validation requires all three
    // inputs share dtype + memory_config, so these layouts are equal in practice.
    auto dq_layout = tt::tt_metal::TensorLayout(dQ.dtype(), tt::tt_metal::Layout::TILE, dQ.memory_config());
    auto dk_layout = tt::tt_metal::TensorLayout(dK.dtype(), tt::tt_metal::Layout::TILE, dK.memory_config());
    output_specs.emplace_back(dq_pre_shape, dq_layout);
    output_specs.emplace_back(dkv_up_shape, dk_layout);
    output_specs.emplace_back(dk_pe_shape, dk_layout);

    return output_specs;
}

MLAQKVAssembleBwDeviceOperation::tensor_return_value_t MLAQKVAssembleBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.dQ.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device)};
}

ttsl::hash::hash_t MLAQKVAssembleBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<MLAQKVAssembleBwDeviceOperation>(
        args,
        tensor_args.dQ.dtype(),
        tensor_args.dQ.logical_shape(),
        tensor_args.dK.logical_shape(),
        tensor_args.dV.logical_shape());
}

MLAQKVAssembleBwDeviceOperation::program_factory_t MLAQKVAssembleBwDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MLAQKVAssembleBwProgramFactory{};
}

}  // namespace ttml::metal::ops::mla_qkv_assemble_bw::device

namespace ttnn::prim {

ttml::metal::ops::mla_qkv_assemble_bw::device::MLAQKVAssembleBwDeviceOperation::tensor_return_value_t
ttml_mla_qkv_assemble_bw(
    const ttnn::Tensor& dQ,
    const ttnn::Tensor& dK,
    const ttnn::Tensor& dV,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    using OperationType = ttml::metal::ops::mla_qkv_assemble_bw::device::MLAQKVAssembleBwDeviceOperation;

    auto attrs = OperationType::operation_attributes_t{
        .n_heads = n_heads,
        .qk_nope_dim = qk_nope_dim,
        .qk_rope_dim = qk_rope_dim,
        .v_dim = v_dim,
    };
    auto tensor_args = OperationType::tensor_args_t{.dQ = dQ, .dK = dK, .dV = dV};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
