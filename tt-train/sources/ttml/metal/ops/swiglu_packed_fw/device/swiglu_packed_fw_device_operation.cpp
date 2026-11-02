// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "metal/common/tensor_validation.hpp"
#include "metal/ops/common/swiglu_packed_common.hpp"
#include "swiglu_packed_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_packed_fw::device {

namespace {
constexpr std::string_view kOp = "SwigluPackedFw";
}  // namespace

void SwigluPackedFwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    check_device_tensor(packed, kOp, "packed");
    swiglu_packed::validate_packed(packed, kOp);

    const uint32_t half = packed.logical_shape()[-1] / 2U;
    TT_FATAL(
        args.out_cols == 0 || (args.out_cols >= half && args.out_cols % tt::constants::TILE_WIDTH == 0),
        "{}: out_cols {} must be a multiple of 32 and at least the half width {}",
        kOp,
        args.out_cols,
        half);
    if (args.transposed_output) {
        const auto& ps = packed.logical_shape();
        TT_FATAL(
            args.out_cols != 0 && ps[0] * ps[1] == 1,
            "{}: the transposed output needs out_cols and a [1, 1, R, 2I] input, got out_cols {} for {}",
            kOp,
            args.out_cols,
            ps);
    }
    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = *tensor_args.preallocated_output;
        check_device_tensor(out, kOp, "preallocated_output");
        check_same_device(out, packed, kOp, "preallocated_output", "packed");
        if (args.transposed_output) {
            TT_FATAL(
                out.logical_shape()[-2] == args.out_cols && out.logical_shape()[-1] == packed.logical_shape()[-2],
                "{}: preallocated_output {} must be [.., out_cols {}, R]",
                kOp,
                out.logical_shape(),
                args.out_cols);
        } else if (args.out_cols == 0) {
            swiglu_packed::validate_half_of_packed(out, packed, "preallocated_output", kOp);
        } else {
            TT_FATAL(
                out.logical_shape()[-1] == args.out_cols && out.logical_shape()[-2] == packed.logical_shape()[-2],
                "{}: preallocated_output {} must be [.., R, out_cols {}]",
                kOp,
                out.logical_shape(),
                args.out_cols);
        }
    }
}

spec_return_value_t SwigluPackedFwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    const auto& packed = tensor_args.packed;
    auto shape = swiglu_packed::halve_last_dim(packed.logical_shape());
    if (args.out_cols != 0) {
        shape[-1] = args.out_cols;
    }
    if (args.transposed_output) {
        shape = ttnn::Shape({shape[0], shape[1], shape[3], shape[2]});
    }
    return tt::tt_metal::TensorSpec(
        shape, tt::tt_metal::TensorLayout(packed.dtype(), tt::tt_metal::Layout::TILE, packed.memory_config()));
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
    const ttnn::Tensor& packed,
    const std::optional<ttnn::Tensor>& preallocated_output,
    uint32_t out_cols,
    bool transposed_output) {
    using Op = ttml::metal::ops::swiglu_packed_fw::device::SwigluPackedFwDeviceOperation;

    const auto tensor_args = Op::tensor_args_t{
        .packed = packed,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{.out_cols = out_cols, .transposed_output = transposed_output}, tensor_args);
}

}  // namespace ttnn::prim
