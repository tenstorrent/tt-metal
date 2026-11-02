// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_packed_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "metal/common/tensor_validation.hpp"
#include "metal/ops/common/swiglu_packed_common.hpp"
#include "swiglu_packed_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::swiglu_packed_bw::device {

namespace {
constexpr std::string_view kOp = "SwigluPackedBw";
}  // namespace

void SwigluPackedBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& packed = tensor_args.packed;
    const auto& dL_dh = tensor_args.dL_dh;
    check_device_tensor(packed, kOp, "packed");
    check_device_tensor(dL_dh, kOp, "dL_dh");
    check_same_device(dL_dh, packed, kOp, "dL_dh", "packed");
    TT_FATAL(!(args.transposed_grad && args.transposed_io), "{}: transposed_grad and transposed_io are exclusive", kOp);
    // dL_dh may be wider than the half width (the wide LoRA input buffer's gradient): only its first I columns
    // are read. Transposed (iteration 38): [1, 1, >= I + 32, R], rows [I, I + 32) the scaled LoRA tail.
    // Transposed IO (iteration 39): packed^T [1, 1, 2I, T] and dL_dh = dh^T wide [1, 1, >= I + 32, T].
    if (args.transposed_io) {
        swiglu_packed::validate_packed_transposed(packed, kOp);
        const auto half = swiglu_packed::halve_rows(packed.logical_shape());
        const auto& g = dL_dh.logical_shape();
        TT_FATAL(
            g.rank() == 4 && g[0] * g[1] == 1 && g[-1] == half[-1] && g[-2] >= half[-2] + tt::constants::TILE_HEIGHT &&
                g[-2] % tt::constants::TILE_HEIGHT == 0 && dL_dh.padded_shape() == g,
            "{}: transposed dL_dh {} must be [1, 1, >= I + 32, T] (no padding) for packed^T {}",
            kOp,
            g,
            packed.logical_shape());
        TT_FATAL(tensor_args.lora_a.has_value(), "{}: the transposed IO needs lora_a", kOp);
        const auto& a = tensor_args.lora_a->logical_shape();
        check_device_tensor(*tensor_args.lora_a, kOp, "lora_a");
        TT_FATAL(
            a.rank() == 4 && a[0] == 1 && a[1] == 1 && a[2] <= tt::constants::TILE_HEIGHT && a[3] == half[-2],
            "{}: lora_a {} must be [1, 1, <= 32, I {}]",
            kOp,
            a,
            half[-2]);
    } else {
        swiglu_packed::validate_packed(packed, kOp);
    }
    if (args.transposed_io) {
    } else if (args.transposed_grad) {
        const auto half = swiglu_packed::halve_last_dim(packed.logical_shape());
        const auto& g = dL_dh.logical_shape();
        TT_FATAL(
            half[0] * half[1] == 1 && g.rank() == 4 && g[0] * g[1] == 1 && g[-1] == half[-2] &&
                g[-2] >= half[-1] + tt::constants::TILE_HEIGHT && g[-2] % tt::constants::TILE_HEIGHT == 0,
            "{}: transposed dL_dh {} must be [1, 1, >= I + 32, R] for packed {}",
            kOp,
            g,
            packed.logical_shape());
        TT_FATAL(tensor_args.lora_a.has_value(), "{}: the transposed gradient needs lora_a", kOp);
        const auto& a = tensor_args.lora_a->logical_shape();
        check_device_tensor(*tensor_args.lora_a, kOp, "lora_a");
        TT_FATAL(
            a.rank() == 4 && a[0] == 1 && a[1] == 1 && a[2] <= tt::constants::TILE_HEIGHT && a[3] == half[-1],
            "{}: lora_a {} must be [1, 1, <= 32, I {}]",
            kOp,
            a,
            half[-1]);
    } else {
        const auto half = swiglu_packed::halve_last_dim(packed.logical_shape());
        const auto& g = dL_dh.logical_shape();
        TT_FATAL(
            g.rank() == half.rank() && g[-1] >= half[-1] && g[-1] % tt::constants::TILE_WIDTH == 0,
            "{}: dL_dh {} must be [.., R, >= I {}] (a multiple of 32 wide)",
            kOp,
            g,
            half[-1]);
        for (int d = 0; d + 1 < static_cast<int>(g.rank()); ++d) {
            TT_FATAL(g[d] == half[d], "{}: dL_dh {} must match packed {} in every dim but the last", kOp, g, half);
        }
    }

    if (tensor_args.preallocated_dL_dpacked.has_value()) {
        const auto& out = *tensor_args.preallocated_dL_dpacked;
        check_device_tensor(out, kOp, "preallocated_dL_dpacked");
        check_same_device(out, packed, kOp, "preallocated_dL_dpacked", "packed");
        swiglu_packed::validate_same_as_packed(out, packed, "preallocated_dL_dpacked", kOp);
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
        tensor_args.dL_dh.dtype(),
        tensor_args.dL_dh.logical_shape(),
        tensor_args.dL_dh.padded_shape(),
        tensor_args.dL_dh.memory_config(),
        out_memcfg);
}

}  // namespace ttml::metal::ops::swiglu_packed_bw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_packed_bw::device::SwigluPackedBwDeviceOperation::tensor_return_value_t ttml_swiglu_packed_bw(
    const ttnn::Tensor& packed,
    const ttnn::Tensor& dL_dh,
    const std::optional<ttnn::Tensor>& preallocated_dL_dpacked,
    const std::optional<ttnn::Tensor>& lora_a,
    bool transposed_grad,
    float lowrank_scale,
    bool transposed_io) {
    using Op = ttml::metal::ops::swiglu_packed_bw::device::SwigluPackedBwDeviceOperation;

    const auto tensor_args = Op::tensor_args_t{
        .packed = packed,
        .dL_dh = dL_dh,
        .preallocated_dL_dpacked = preallocated_dL_dpacked,
        .lora_a = lora_a,
    };

    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .transposed_grad = transposed_grad, .lowrank_scale = lowrank_scale, .transposed_io = transposed_io},
        tensor_args);
}

}  // namespace ttnn::prim
