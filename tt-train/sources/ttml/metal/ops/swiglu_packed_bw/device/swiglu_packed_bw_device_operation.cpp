// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "swiglu_packed_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_packed_bw::device {

void SwigluPackedBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "SwigluPackedBw requires {} on Device. Storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "SwigluPackedBw: {} buffer is null", name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "SwigluPackedBw requires TILE layout. {} layout: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SwigluPackedBw requires BFLOAT16. {} dtype: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SwigluPackedBw requires INTERLEAVED. {} layout: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    // Call only after check_tensor has established DEVICE storage on both tensors.
    auto check_same_device = [](const ttnn::Tensor& tensor, const ttnn::Tensor& reference, const std::string& name) {
        TT_FATAL(
            tensor.device() == reference.device(),
            "SwigluPackedBw: {} is on a different device than packed. The program is created on "
            "packed's device and the kernels are handed {}'s raw buffer address, so a foreign "
            "buffer would be addressed as if it were local.",
            name,
            name);
    };

    const auto& packed = tensor_args.packed;
    const auto& dL_dh = tensor_args.dL_dh;
    check_tensor(packed, "packed");
    check_tensor(dL_dh, "dL_dh");
    check_same_device(dL_dh, packed, "dL_dh");

    const auto& packed_padded = packed.padded_shape();
    const auto& dh_padded = dL_dh.padded_shape();
    TT_FATAL(packed_padded.rank() == 4U, "SwigluPackedBw: packed must be 4D, got rank {}", packed_padded.rank());
    const uint32_t two_tiles_w = 2U * tt::constants::TILE_WIDTH;
    TT_FATAL(
        packed_padded[-1] % two_tiles_w == 0U,
        "SwigluPackedBw: packed last padded dim {} must be a multiple of {} so each half is tile-aligned",
        packed_padded[-1],
        two_tiles_w);
    TT_FATAL(
        packed.logical_shape()[-1] % two_tiles_w == 0U,
        "SwigluPackedBw: packed last logical dim {} must be a multiple of {} so the gate|up split "
        "lands on the tile boundary where the kernel splits the padded row",
        packed.logical_shape()[-1],
        two_tiles_w);
    TT_FATAL(
        dh_padded[-1] * 2U == packed_padded[-1] && dh_padded[-2] == packed_padded[-2] &&
            dh_padded[0] == packed_padded[0] && dh_padded[1] == packed_padded[1],
        "SwigluPackedBw: dL_dh padded shape {} must be packed {} with the last dim halved",
        dh_padded,
        packed_padded);
    TT_FATAL(
        dL_dh.logical_shape()[-1] * 2U == packed.logical_shape()[-1],
        "SwigluPackedBw: dL_dh last logical dim {} must be half packed's {}",
        dL_dh.logical_shape()[-1],
        packed.logical_shape()[-1]);

    if (tensor_args.preallocated_dL_dpacked.has_value()) {
        const auto& out = tensor_args.preallocated_dL_dpacked.value();
        check_tensor(out, "preallocated_dL_dpacked");
        check_same_device(out, packed, "preallocated_dL_dpacked");
        TT_FATAL(
            out.padded_shape() == packed_padded,
            "SwigluPackedBw: preallocated_dL_dpacked padded shape {} must match packed {}",
            out.padded_shape(),
            packed_padded);
        TT_FATAL(
            out.logical_shape() == packed.logical_shape(),
            "SwigluPackedBw: preallocated_dL_dpacked logical shape {} must match packed {}; its "
            "spec is returned to the caller as the output spec",
            out.logical_shape(),
            packed.logical_shape());
    }
}

spec_return_value_t SwigluPackedBwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_dL_dpacked.has_value()) {
        return tensor_args.preallocated_dL_dpacked->tensor_spec();
    }

    const auto& packed = tensor_args.packed;
    return tt::tt_metal::TensorSpec(
        packed.logical_shape(),
        tt::tt_metal::TensorLayout(packed.dtype(), tt::tt_metal::Layout::TILE, packed.memory_config()));
}

tensor_return_value_t SwigluPackedBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_dL_dpacked.has_value()) {
        return tensor_args.preallocated_dL_dpacked.value();
    }
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.packed.device());
}

ttsl::hash::hash_t SwigluPackedBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    const auto& out_memcfg = tensor_args.preallocated_dL_dpacked.has_value()
                                 ? tensor_args.preallocated_dL_dpacked->memory_config()
                                 : packed.memory_config();
    return tt::tt_metal::operation::hash_operation<SwigluPackedBwDeviceOperation>(
        args,
        packed.dtype(),
        packed.logical_shape(),
        packed.padded_shape(),
        packed.memory_config(),
        tensor_args.dL_dh.logical_shape(),
        tensor_args.dL_dh.memory_config(),
        out_memcfg);
}

}  // namespace ttml::metal::ops::swiglu_packed_bw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_packed_bw::device::SwigluPackedBwDeviceOperation::tensor_return_value_t ttml_swiglu_packed_bw(
    const ttnn::Tensor& packed, const ttnn::Tensor& dL_dh, const std::optional<ttnn::Tensor>& preallocated_dL_dpacked) {
    using Op = ttml::metal::ops::swiglu_packed_bw::device::SwigluPackedBwDeviceOperation;

    const auto tensor_args = Op::tensor_args_t{
        .packed = packed,
        .dL_dh = dL_dh,
        .preallocated_dL_dpacked = preallocated_dL_dpacked,
    };

    return ttnn::device_operation::launch<Op>(Op::operation_attributes_t{}, tensor_args);
}

}  // namespace ttnn::prim
