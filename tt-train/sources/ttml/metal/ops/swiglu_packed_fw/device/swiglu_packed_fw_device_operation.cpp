// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "swiglu_packed_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_packed_fw::device {

void SwigluPackedFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SwigluPackedFw requires {} on Device. Storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "SwigluPackedFw: {} buffer is null", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "SwigluPackedFw requires TILE layout. {} layout: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SwigluPackedFw requires BFLOAT16. {} dtype: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SwigluPackedFw requires INTERLEAVED. {} layout: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& packed = tensor_args.packed;
    check_tensor(packed, "packed");

    const auto& packed_padded = packed.padded_shape();
    TT_FATAL(packed_padded.rank() == 4U, "SwigluPackedFw: packed must be 4D, got rank {}", packed_padded.rank());
    const uint32_t two_tiles_w = 2U * tt::constants::TILE_WIDTH;
    TT_FATAL(
        packed_padded[-1] % two_tiles_w == 0U,
        "SwigluPackedFw: packed last padded dim {} must be a multiple of {} so each half is tile-aligned",
        packed_padded[-1],
        two_tiles_w);
    TT_FATAL(
        packed.logical_shape()[-1] % 2U == 0U,
        "SwigluPackedFw: packed last logical dim {} must be even so the gate|up split is well-defined",
        packed.logical_shape()[-1]);

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = tensor_args.preallocated_output.value();
        check_tensor(out, "preallocated_output");
        const auto& out_padded = out.padded_shape();
        TT_FATAL(
            out_padded[-1] * 2U == packed_padded[-1] && out_padded[-2] == packed_padded[-2] &&
                out_padded[0] == packed_padded[0] && out_padded[1] == packed_padded[1],
            "SwigluPackedFw: preallocated_output padded shape {} must be packed {} with the last dim halved",
            out_padded,
            packed_padded);
    }
}

spec_return_value_t SwigluPackedFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    const auto& packed = tensor_args.packed;
    const auto& in_shape = packed.logical_shape();
    const ttnn::Shape out_shape({in_shape[0], in_shape[1], in_shape[2], in_shape[-1] / 2U});
    return ttnn::TensorSpec(
        out_shape, tt::tt_metal::TensorLayout(packed.dtype(), tt::tt_metal::Layout::TILE, packed.memory_config()));
}

tensor_return_value_t SwigluPackedFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.packed.device());
}

ttsl::hash::hash_t SwigluPackedFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    const auto& out_memcfg = tensor_args.preallocated_output.has_value()
                                 ? tensor_args.preallocated_output->memory_config()
                                 : packed.memory_config();
    return tt::tt_metal::operation::hash_operation<SwigluPackedFwDeviceOperation>(
        args, packed.dtype(), packed.logical_shape(), packed.padded_shape(), out_memcfg);
}

}  // namespace ttml::metal::ops::swiglu_packed_fw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_packed_fw::device::SwigluPackedFwDeviceOperation::tensor_return_value_t ttml_swiglu_packed_fw(
    const ttnn::Tensor& packed, const std::optional<ttnn::Tensor>& preallocated_output) {
    using Op = ttml::metal::ops::swiglu_packed_fw::device::SwigluPackedFwDeviceOperation;

    const auto tensor_args = Op::tensor_args_t{
        .packed = packed,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<Op>(Op::operation_attributes_t{}, tensor_args);
}

}  // namespace ttnn::prim
