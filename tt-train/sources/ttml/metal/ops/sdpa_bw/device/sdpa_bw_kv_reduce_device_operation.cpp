// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_bw_kv_reduce_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::sdpa_bw::device {

void SDPABackwardKVReduceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& gk = tensor_args.grad_key_partial;
    const auto& gv = tensor_args.grad_value_partial;
    const auto& key = tensor_args.key;
    const auto& value = tensor_args.value;
    const DeviceTensorRequirements fp32_tile{.dtypes = {tt::tt_metal::DataType::FLOAT32}};
    check_device_tensor(gk, "SDPABackwardKVReduce", "grad_key_partial", fp32_tile);
    check_device_tensor(gv, "SDPABackwardKVReduce", "grad_value_partial", fp32_tile);
    check_device_tensor(key, "SDPABackwardKVReduce", "key");
    check_device_tensor(value, "SDPABackwardKVReduce", "value");

    const auto gk_shape = gk.logical_shape();
    const auto gv_shape = gv.logical_shape();
    const auto k_shape = key.logical_shape();
    const auto v_shape = value.logical_shape();
    const uint32_t G = args.heads_per_group;
    TT_FATAL(G >= 1, "heads_per_group must be >= 1, got {}", G);
    TT_FATAL(
        gk_shape[0] == k_shape[0] && gk_shape[1] == k_shape[1] * G && gk_shape[2] == k_shape[2] &&
            gk_shape[3] == k_shape[3],
        "grad_key_partial {} must be key {} with heads x {}",
        gk_shape,
        k_shape,
        G);
    TT_FATAL(
        gv_shape[0] == v_shape[0] && gv_shape[1] == v_shape[1] * G && gv_shape[2] == v_shape[2] &&
            gv_shape[3] == v_shape[3],
        "grad_value_partial {} must be value {} with heads x {}",
        gv_shape,
        v_shape,
        G);
    TT_FATAL(
        k_shape[2] % tt::constants::TILE_HEIGHT == 0 && k_shape[3] % tt::constants::TILE_WIDTH == 0 &&
            v_shape[3] % tt::constants::TILE_WIDTH == 0,
        "key {} and value {} must be tile aligned in S and D",
        k_shape,
        v_shape);
}

SDPABackwardKVReduceDeviceOperation::spec_return_value_t SDPABackwardKVReduceDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    auto like = [](const ttnn::Tensor& t) {
        return tt::tt_metal::TensorSpec(
            t.logical_shape(), tt::tt_metal::TensorLayout(t.dtype(), tt::tt_metal::Layout::TILE, t.memory_config()));
    };
    return {like(tensor_args.key), like(tensor_args.value)};
}

SDPABackwardKVReduceDeviceOperation::tensor_return_value_t SDPABackwardKVReduceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto [k_spec, v_spec] = compute_output_specs(args, tensor_args);
    return {
        ttnn::create_device_tensor(k_spec, tensor_args.key.device()),
        ttnn::create_device_tensor(v_spec, tensor_args.value.device())};
}

}  // namespace ttml::metal::ops::sdpa_bw::device

namespace ttnn::prim {

ttml::metal::ops::sdpa_bw::device::SDPABackwardKVReduceDeviceOperation::tensor_return_value_t ttml_sdpa_kv_bw_reduce(
    const ttnn::Tensor& grad_key_partial,
    const ttnn::Tensor& grad_value_partial,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const uint32_t heads_per_group) {
    using OperationType = ttml::metal::ops::sdpa_bw::device::SDPABackwardKVReduceDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{.heads_per_group = heads_per_group};
    auto tensor_args = OperationType::tensor_args_t{
        .grad_key_partial = grad_key_partial, .grad_value_partial = grad_value_partial, .key = key, .value = value};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
