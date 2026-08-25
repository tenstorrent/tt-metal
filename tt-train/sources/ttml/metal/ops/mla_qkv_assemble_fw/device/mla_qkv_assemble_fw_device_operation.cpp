// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_qkv_assemble_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "mla_qkv_assemble_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::mla_qkv_assemble_fw::device {

void MLAQKVAssembleFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "MLAQKVAssembleFw requires {} to be on device. Got storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "MLAQKVAssembleFw: {} buffer must be allocated.", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "MLAQKVAssembleFw requires {} to be in TILE layout. Got: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "MLAQKVAssembleFw requires {} dtype to be BFLOAT16. Got: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "MLAQKVAssembleFw requires {} memory layout to be INTERLEAVED. Got: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& q_pre = tensor_args.q_pre;
    const auto& kv_up = tensor_args.kv_up;
    const auto& k_pe = tensor_args.k_pe;
    check_tensor(q_pre, "q_pre");
    check_tensor(kv_up, "kv_up");
    check_tensor(k_pe, "k_pe");

    // The program launches on kv_up.device() and feeds raw buffer addresses from all three inputs into
    // the TensorAccessors, so mixed-device inputs must fail here rather than dispatch invalid addresses.
    TT_FATAL(
        q_pre.device() == kv_up.device() && kv_up.device() == k_pe.device(),
        "MLAQKVAssembleFw: q_pre, kv_up, and k_pe must be on the same device.");

    const auto q_pre_shape = q_pre.padded_shape();
    const auto kv_up_shape = kv_up.padded_shape();
    const auto k_pe_shape = k_pe.padded_shape();

    TT_FATAL(q_pre_shape.rank() == 4U, "MLAQKVAssembleFw: q_pre must be rank-4. Got rank {}", q_pre_shape.rank());
    TT_FATAL(kv_up_shape.rank() == 4U, "MLAQKVAssembleFw: kv_up must be rank-4. Got rank {}", kv_up_shape.rank());
    TT_FATAL(k_pe_shape.rank() == 4U, "MLAQKVAssembleFw: k_pe must be rank-4. Got rank {}", k_pe_shape.rank());

    TT_FATAL(q_pre_shape[1] == 1U, "MLAQKVAssembleFw: q_pre dim 1 must be 1. Got {}", q_pre_shape[1]);
    TT_FATAL(kv_up_shape[1] == 1U, "MLAQKVAssembleFw: kv_up dim 1 must be 1. Got {}", kv_up_shape[1]);
    TT_FATAL(k_pe_shape[1] == 1U, "MLAQKVAssembleFw: k_pe dim 1 must be 1. Got {}", k_pe_shape[1]);

    TT_FATAL(
        q_pre_shape[0] == kv_up_shape[0] && kv_up_shape[0] == k_pe_shape[0],
        "MLAQKVAssembleFw: batch dim must match across q_pre ({}), kv_up ({}), k_pe ({}).",
        q_pre_shape[0],
        kv_up_shape[0],
        k_pe_shape[0]);
    TT_FATAL(
        q_pre_shape[2] == kv_up_shape[2] && kv_up_shape[2] == k_pe_shape[2],
        "MLAQKVAssembleFw: seq dim must match across q_pre ({}), kv_up ({}), k_pe ({}).",
        q_pre_shape[2],
        kv_up_shape[2],
        k_pe_shape[2]);

    const uint32_t qk_head_dim = args.qk_nope_dim + args.qk_rope_dim;
    const uint32_t expected_q_pre_w = args.n_heads * qk_head_dim;
    const uint32_t expected_kv_up_w = args.n_heads * (args.qk_nope_dim + args.v_dim);
    TT_FATAL(
        q_pre_shape[3] == expected_q_pre_w,
        "MLAQKVAssembleFw: q_pre dim 3 must equal n_heads * (qk_nope_dim + qk_rope_dim) = {}. Got {}",
        expected_q_pre_w,
        q_pre_shape[3]);
    TT_FATAL(
        kv_up_shape[3] == expected_kv_up_w,
        "MLAQKVAssembleFw: kv_up dim 3 must equal n_heads * (qk_nope_dim + v_dim) = {}. Got {}",
        expected_kv_up_w,
        kv_up_shape[3]);
    TT_FATAL(
        k_pe_shape[3] == args.qk_rope_dim,
        "MLAQKVAssembleFw: k_pe dim 3 must equal qk_rope_dim = {}. Got {}",
        args.qk_rope_dim,
        k_pe_shape[3]);

    TT_FATAL(
        args.qk_nope_dim % TILE_WIDTH == 0,
        "MLAQKVAssembleFw: qk_nope_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.qk_nope_dim,
        TILE_WIDTH);
    TT_FATAL(
        args.qk_rope_dim % TILE_WIDTH == 0,
        "MLAQKVAssembleFw: qk_rope_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.qk_rope_dim,
        TILE_WIDTH);
    TT_FATAL(
        args.v_dim % TILE_WIDTH == 0,
        "MLAQKVAssembleFw: v_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.v_dim,
        TILE_WIDTH);
    TT_FATAL(
        kv_up_shape[2] % TILE_HEIGHT == 0,
        "MLAQKVAssembleFw: S ({}) must be a multiple of TILE_HEIGHT ({}).",
        kv_up_shape[2],
        TILE_HEIGHT);
}

MLAQKVAssembleFwDeviceOperation::spec_return_value_t MLAQKVAssembleFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(3U);

    const auto& q_pre = tensor_args.q_pre;
    const auto& kv_up = tensor_args.kv_up;
    const auto kv_up_shape = kv_up.logical_shape();
    const uint32_t B = kv_up_shape[0];
    const uint32_t S = kv_up_shape[2];

    const uint32_t qk_head = args.qk_nope_dim + args.qk_rope_dim;

    const ttnn::Shape q_shape({B, args.n_heads, S, qk_head});
    const ttnn::Shape k_shape({B, args.n_heads, S, qk_head});
    const ttnn::Shape v_shape({B, args.n_heads, S, args.v_dim});

    // Each output inherits from its primary input source. Validation requires all three
    // inputs share dtype + memory_config, so these layouts are equal in practice.
    auto q_layout = tt::tt_metal::TensorLayout(q_pre.dtype(), tt::tt_metal::Layout::TILE, q_pre.memory_config());
    auto kv_layout = tt::tt_metal::TensorLayout(kv_up.dtype(), tt::tt_metal::Layout::TILE, kv_up.memory_config());
    output_specs.emplace_back(q_shape, q_layout);
    output_specs.emplace_back(k_shape, kv_layout);
    output_specs.emplace_back(v_shape, kv_layout);

    return output_specs;
}

MLAQKVAssembleFwDeviceOperation::tensor_return_value_t MLAQKVAssembleFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.kv_up.device();
    return {
        ttnn::create_device_tensor(specs[0], device),
        ttnn::create_device_tensor(specs[1], device),
        ttnn::create_device_tensor(specs[2], device)};
}

ttsl::hash::hash_t MLAQKVAssembleFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<MLAQKVAssembleFwDeviceOperation>(
        args,
        tensor_args.kv_up.dtype(),
        tensor_args.q_pre.logical_shape(),
        tensor_args.kv_up.logical_shape(),
        tensor_args.k_pe.logical_shape());
}

MLAQKVAssembleFwDeviceOperation::program_factory_t MLAQKVAssembleFwDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MLAQKVAssembleFwProgramFactory{};
}

}  // namespace ttml::metal::ops::mla_qkv_assemble_fw::device

namespace ttnn::prim {

ttml::metal::ops::mla_qkv_assemble_fw::device::MLAQKVAssembleFwDeviceOperation::tensor_return_value_t
ttml_mla_qkv_assemble_fw(
    const ttnn::Tensor& q_pre,
    const ttnn::Tensor& kv_up,
    const ttnn::Tensor& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    using OperationType = ttml::metal::ops::mla_qkv_assemble_fw::device::MLAQKVAssembleFwDeviceOperation;

    auto attrs = OperationType::operation_attributes_t{
        .n_heads = n_heads,
        .qk_nope_dim = qk_nope_dim,
        .qk_rope_dim = qk_rope_dim,
        .v_dim = v_dim,
    };
    auto tensor_args = OperationType::tensor_args_t{.q_pre = q_pre, .kv_up = kv_up, .k_pe = k_pe};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
