// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_kv_assemble_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "mla_kv_assemble_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::mla_kv_assemble_fw::device {

void MLAKVAssembleFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "MLAKVAssembleFw requires {} to be on device. Got storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "MLAKVAssembleFw: {} buffer must be allocated.", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "MLAKVAssembleFw requires {} to be in TILE layout. Got: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "MLAKVAssembleFw requires {} dtype to be BFLOAT16. Got: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "MLAKVAssembleFw requires {} memory layout to be INTERLEAVED. Got: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& kv_up = tensor_args.kv_up;
    const auto& k_pe = tensor_args.k_pe;
    check_tensor(kv_up, "kv_up");
    check_tensor(k_pe, "k_pe");

    TT_FATAL(kv_up.device() == k_pe.device(), "MLAKVAssembleFw: kv_up and k_pe must be on the same device.");

    const auto kv_up_shape = kv_up.padded_shape();
    const auto k_pe_shape = k_pe.padded_shape();

    TT_FATAL(kv_up_shape.rank() == 4U, "MLAKVAssembleFw: kv_up must be rank-4. Got rank {}", kv_up_shape.rank());
    TT_FATAL(k_pe_shape.rank() == 4U, "MLAKVAssembleFw: k_pe must be rank-4. Got rank {}", k_pe_shape.rank());

    TT_FATAL(kv_up_shape[1] == 1U, "MLAKVAssembleFw: kv_up dim 1 must be 1. Got {}", kv_up_shape[1]);
    TT_FATAL(k_pe_shape[1] == 1U, "MLAKVAssembleFw: k_pe dim 1 must be 1. Got {}", k_pe_shape[1]);

    TT_FATAL(
        kv_up_shape[0] == k_pe_shape[0],
        "MLAKVAssembleFw: batch dim must match across kv_up ({}), k_pe ({}).",
        kv_up_shape[0],
        k_pe_shape[0]);
    TT_FATAL(
        kv_up_shape[2] == k_pe_shape[2],
        "MLAKVAssembleFw: seq dim must match across kv_up ({}), k_pe ({}).",
        kv_up_shape[2],
        k_pe_shape[2]);

    const uint32_t expected_kv_up_w = args.n_heads * (args.qk_nope_dim + args.v_dim);
    TT_FATAL(
        kv_up_shape[3] == expected_kv_up_w,
        "MLAKVAssembleFw: kv_up dim 3 must equal n_heads * (qk_nope_dim + v_dim) = {}. Got {}",
        expected_kv_up_w,
        kv_up_shape[3]);
    TT_FATAL(
        k_pe_shape[3] == args.qk_rope_dim,
        "MLAKVAssembleFw: k_pe dim 3 must equal qk_rope_dim = {}. Got {}",
        args.qk_rope_dim,
        k_pe_shape[3]);

    TT_FATAL(
        args.qk_nope_dim % TILE_WIDTH == 0,
        "MLAKVAssembleFw: qk_nope_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.qk_nope_dim,
        TILE_WIDTH);
    TT_FATAL(
        args.qk_rope_dim % TILE_WIDTH == 0,
        "MLAKVAssembleFw: qk_rope_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.qk_rope_dim,
        TILE_WIDTH);
    TT_FATAL(
        args.v_dim % TILE_WIDTH == 0,
        "MLAKVAssembleFw: v_dim ({}) must be a multiple of TILE_WIDTH ({}).",
        args.v_dim,
        TILE_WIDTH);
    TT_FATAL(
        kv_up_shape[2] % TILE_HEIGHT == 0,
        "MLAKVAssembleFw: S ({}) must be a multiple of TILE_HEIGHT ({}).",
        kv_up_shape[2],
        TILE_HEIGHT);
}

MLAKVAssembleFwDeviceOperation::spec_return_value_t MLAKVAssembleFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(2U);

    const auto& kv_up = tensor_args.kv_up;
    const auto kv_up_shape = kv_up.logical_shape();
    const uint32_t B = kv_up_shape[0];
    const uint32_t S = kv_up_shape[2];

    const uint32_t qk_head = args.qk_nope_dim + args.qk_rope_dim;

    const ttnn::Shape k_shape({B, args.n_heads, S, qk_head});
    const ttnn::Shape v_shape({B, args.n_heads, S, args.v_dim});

    auto kv_layout = tt::tt_metal::TensorLayout(kv_up.dtype(), tt::tt_metal::Layout::TILE, kv_up.memory_config());
    output_specs.emplace_back(k_shape, kv_layout);
    output_specs.emplace_back(v_shape, kv_layout);

    return output_specs;
}

MLAKVAssembleFwDeviceOperation::tensor_return_value_t MLAKVAssembleFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.kv_up.device();
    return {create_device_tensor(specs[0], device), create_device_tensor(specs[1], device)};
}

ttsl::hash::hash_t MLAKVAssembleFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<MLAKVAssembleFwDeviceOperation>(
        args, tensor_args.kv_up.dtype(), tensor_args.kv_up.logical_shape(), tensor_args.k_pe.logical_shape());
}

MLAKVAssembleFwDeviceOperation::program_factory_t MLAKVAssembleFwDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MLAKVAssembleFwProgramFactory{};
}

}  // namespace ttml::metal::ops::mla_kv_assemble_fw::device

namespace ttnn::prim {

ttml::metal::ops::mla_kv_assemble_fw::device::MLAKVAssembleFwDeviceOperation::tensor_return_value_t
ttml_mla_kv_assemble_fw(
    const ttnn::Tensor& kv_up,
    const ttnn::Tensor& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    using OperationType = ttml::metal::ops::mla_kv_assemble_fw::device::MLAKVAssembleFwDeviceOperation;

    auto attrs = OperationType::operation_attributes_t{
        .n_heads = n_heads,
        .qk_nope_dim = qk_nope_dim,
        .qk_rope_dim = qk_rope_dim,
        .v_dim = v_dim,
    };
    auto tensor_args = OperationType::tensor_args_t{.kv_up = kv_up, .k_pe = k_pe};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
