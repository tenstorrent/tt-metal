// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_bw_kv_program_factory.hpp"

#include <optional>
#include <tt-metalium/program_descriptors.hpp>

#include "metal/common/program_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation_types.hpp"
#include "metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.hpp"
#include "ring_sdpa_bw_kv_device_operation.hpp"
#include "ring_sdpa_bw_kv_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {

TTML_ASSERT_DESCRIPTOR_OVERRIDE(RingSDPABwKVDeviceOperation, RingSDPABwKVProgramFactory);

namespace {

namespace sdpa_kv = ttml::metal::ops::sdpa_bw::device::kv;
using ttml::metal::ops::sdpa_bw::device::SDPABackwardKVProgramFactory;

// Calls `fn(attrs, tensor_args, return_value)` with the SDPA backward (key and value) arguments for one device at this
// ring step. No explicit mask tensor is needed: the kernels generate the causal mask internally.
template <typename Fn>
decltype(auto) with_sdpa_bw_kv_args(
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    AttentionMaskType mask_type,
    Fn&& fn) {
    auto& grad_key = std::get<0>(tensor_return_value);
    auto& grad_value = std::get<1>(tensor_return_value);
    const std::optional<ttnn::Tensor> no_tensor = std::nullopt;
    const sdpa_kv::operation_attributes_t sdpa_attrs{.mask_type = mask_type, .dropout_probability = 0.0F};
    const sdpa_kv::tensor_args_t sdpa_tensor_args{
        .grad_output = tensor_args.grad_output,
        .query = tensor_args.query,
        .key = tensor_args.key,
        .value = tensor_args.value,
        .attn_mask = no_tensor,
        .intermediates = tensor_args.intermediates,
        .u_scaler = tensor_args.u_scaler,
        .preallocated_grad_key = grad_key,
        .preallocated_grad_value = grad_value};
    sdpa_kv::tensor_return_value_t sdpa_return_value{grad_key, grad_value};
    return fn(sdpa_attrs, sdpa_tensor_args, sdpa_return_value);
}

}  // namespace

tt::tt_metal::ProgramDescriptor RingSDPABwKVProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    TT_FATAL(mesh_coordinate.has_value(), "Ring SDPA backward builds one program per device and needs its coordinate");
    const auto mask_type = mask_for_device_at_step(operation_attributes, *mesh_coordinate);
    if (!mask_type.has_value()) {
        return {};
    }
    return with_sdpa_bw_kv_args(
        tensor_args, tensor_return_value, *mask_type, [](const auto& attrs, const auto& args, auto& return_value) {
            return SDPABackwardKVProgramFactory::create_descriptor(attrs, args, return_value);
        });
}

void RingSDPABwKVProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_coordinate) {
    TT_FATAL(mesh_coordinate.has_value(), "Ring SDPA backward builds one program per device and needs its coordinate");
    // Only devices that run this step have a program.
    const auto mask_type = mask_for_device_at_step(operation_attributes, *mesh_coordinate);
    TT_FATAL(mask_type.has_value(), "Ring SDPA backward has a program on a device that sits this step out");
    with_sdpa_bw_kv_args(
        tensor_args, tensor_return_value, *mask_type, [&](const auto& attrs, const auto& args, auto& return_value) {
            SDPABackwardKVProgramFactory::override_runtime_arguments(
                program, attrs, args, return_value, mesh_coordinate);
        });
}

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv
