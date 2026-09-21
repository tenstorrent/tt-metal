// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_shift_fused_device_operation.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttml::metal::ops::ring_shift_fused {

void RingShiftFusedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& inputs = tensor_args.inputs;
    TT_FATAL(!inputs.empty(), "ring_shift_fused: no tensors to shift");
    TT_FATAL(inputs.size() <= 15U, "ring_shift_fused: at most 15 tensors per launch (the handshake page), got {}",
             inputs.size());
    auto* mesh_device = inputs.front().device();
    TT_FATAL(mesh_device != nullptr, "ring_shift_fused: the tensors must be on a mesh device");
    for (size_t t = 0; t < inputs.size(); ++t) {
        const auto& tensor = inputs[t];
        TT_FATAL(tensor.device() == mesh_device, "ring_shift_fused: tensor {} is on another device", t);
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED &&
                tensor.memory_config().buffer_type() == ttnn::BufferType::DRAM,
            "ring_shift_fused: tensor {} must be interleaved in DRAM", t);
        TT_FATAL(tensor.buffer() != nullptr && tensor.buffer()->num_pages() > 0, "ring_shift_fused: tensor {} is empty", t);
    }
    TT_FATAL(
        tensor_args.preallocated_outputs.empty() || tensor_args.preallocated_outputs.size() == inputs.size(),
        "ring_shift_fused: {} preallocated outputs for {} inputs", tensor_args.preallocated_outputs.size(),
        inputs.size());
    for (size_t t = 0; t < tensor_args.preallocated_outputs.size(); ++t) {
        TT_FATAL(
            tensor_args.preallocated_outputs[t].tensor_spec() == inputs[t].tensor_spec(),
            "ring_shift_fused: preallocated output {} does not match its input", t);
    }
    TT_FATAL(
        !attrs.send_sockets.empty() && attrs.send_sockets.size() == attrs.recv_sockets.size(),
        "ring_shift_fused: the sockets come in sender/receiver pairs, got {} and {}", attrs.send_sockets.size(),
        attrs.recv_sockets.size());
    for (size_t i = 0; i < attrs.send_sockets.size(); ++i) {
        TT_FATAL(
            attrs.send_sockets[i].get_config_buffer() != nullptr && attrs.recv_sockets[i].get_config_buffer() != nullptr,
            "ring_shift_fused: socket pair {} has no config buffer", i);
    }
}

RingShiftFusedDeviceOperation::spec_return_value_t RingShiftFusedDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    spec_return_value_t specs;
    specs.reserve(tensor_args.inputs.size());
    for (const auto& tensor : tensor_args.inputs) {
        specs.push_back(tensor.tensor_spec());
    }
    return specs;
}

RingShiftFusedDeviceOperation::tensor_return_value_t RingShiftFusedDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if (!tensor_args.preallocated_outputs.empty()) {
        return tensor_args.preallocated_outputs;
    }
    const auto specs = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.inputs.front().device();
    tensor_return_value_t outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(ttnn::create_device_tensor(spec, device));
    }
    return outputs;
}

ttsl::hash::hash_t RingShiftFusedDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto hash = ttsl::hash::hash_objects(tensor_args.inputs.size(), attrs.send_sockets.size());
    for (size_t i = 0; i < attrs.send_sockets.size(); ++i) {
        hash = ttsl::hash::hash_objects(
            hash, attrs.send_sockets[i].get_config_buffer()->address(),
            attrs.recv_sockets[i].get_config_buffer()->address());
    }
    for (const auto& tensor : tensor_args.inputs) {
        hash = ttsl::hash::hash_objects(hash, tensor.logical_shape(), tensor.dtype(), tensor.layout());
    }
    return hash;
}

}  // namespace ttml::metal::ops::ring_shift_fused

namespace ttnn::prim {

ttml::metal::ops::ring_shift_fused::RingShiftFusedDeviceOperation::tensor_return_value_t ttml_ring_shift_fused(
    const std::vector<ttnn::Tensor>& inputs,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& send_sockets,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& recv_sockets,
    const std::vector<ttnn::Tensor>& preallocated_outputs) {
    using OperationType = ttml::metal::ops::ring_shift_fused::RingShiftFusedDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t(send_sockets, recv_sockets);
    auto tensor_args = OperationType::tensor_args_t{.inputs = inputs, .preallocated_outputs = preallocated_outputs};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
