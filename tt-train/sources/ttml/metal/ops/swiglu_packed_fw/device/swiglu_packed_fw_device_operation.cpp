// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "swiglu_packed_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_packed_fw::device {

namespace {

// The output is packed with the gate|up halves collapsed into one, so every shape of the output is
// the corresponding shape of packed with the last dim halved.
ttnn::Shape halve_last_dim(const ttnn::Shape& shape) {
    return ttnn::Shape({shape[0], shape[1], shape[2], shape[-1] / 2U});
}

}  // namespace

void SwigluPackedFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
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

    // Call only after check_tensor has established DEVICE storage on both tensors.
    auto check_same_device = [](const ttnn::Tensor& tensor, const ttnn::Tensor& reference, const std::string& name) {
        TT_FATAL(
            tensor.device() == reference.device(),
            "SwigluPackedFw: {} is on a different device than packed. The program is created on "
            "packed's device and the kernels are handed {}'s raw buffer address, so a foreign "
            "buffer would be addressed as if it were local.",
            name,
            name);
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
        packed.logical_shape()[-1] % two_tiles_w == 0U,
        "SwigluPackedFw: packed last logical dim {} must be a multiple of {} so the gate|up split "
        "lands on the tile boundary where the kernel splits the padded row",
        packed.logical_shape()[-1],
        two_tiles_w);

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = tensor_args.preallocated_output.value();
        check_tensor(out, "preallocated_output");
        check_same_device(out, packed, "preallocated_output");
        const auto expected_padded = halve_last_dim(packed_padded);
        TT_FATAL(
            out.padded_shape() == expected_padded,
            "SwigluPackedFw: preallocated_output padded shape {} must be {} (packed {} with the last dim halved)",
            out.padded_shape(),
            expected_padded,
            packed_padded);
        const auto expected_logical = halve_last_dim(packed.logical_shape());
        TT_FATAL(
            out.logical_shape() == expected_logical,
            "SwigluPackedFw: preallocated_output logical shape {} must be {}; its spec is returned "
            "to the caller as the output spec",
            out.logical_shape(),
            expected_logical);
    }
}

spec_return_value_t SwigluPackedFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    const auto& packed = tensor_args.packed;
    return tt::tt_metal::TensorSpec(
        halve_last_dim(packed.logical_shape()),
        tt::tt_metal::TensorLayout(packed.dtype(), tt::tt_metal::Layout::TILE, packed.memory_config()));
}

tensor_return_value_t SwigluPackedFwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.packed.device());
}

ttsl::hash::hash_t SwigluPackedFwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    const auto& out_memcfg = tensor_args.preallocated_output.has_value()
                                 ? tensor_args.preallocated_output->memory_config()
                                 : packed.memory_config();
    return tt::tt_metal::operation::hash_operation<SwigluPackedFwDeviceOperation>(
        args, packed.dtype(), packed.logical_shape(), packed.padded_shape(), packed.memory_config(), out_memcfg);
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
