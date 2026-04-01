// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_dispatch_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::moe_dispatch {

void MoeDispatchDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(args.sorted_hidden.storage_type() == tt::tt_metal::StorageType::DEVICE, "sorted_hidden on device");
    TT_FATAL(args.sorted_hidden.layout() == tt::tt_metal::Layout::TILE, "sorted_hidden TILE");
    TT_FATAL(args.w_up.storage_type() == tt::tt_metal::StorageType::DEVICE, "w_up on device");
    TT_FATAL(args.w_up.layout() == tt::tt_metal::Layout::TILE, "w_up TILE");
    TT_FATAL(
        attrs.expert_counts_per_device.size() == attrs.expert_offsets_per_device.size(),
        "counts/offsets per-device size mismatch");
}

MoeDispatchDeviceOperation::spec_return_value_t MoeDispatchDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto sh = args.sorted_hidden.logical_shape();
    const auto wu = args.w_up.logical_shape();
    auto layout = tt::tt_metal::TensorLayout(
        args.sorted_hidden.dtype(),
        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
        args.sorted_hidden.memory_config());

    uint32_t EP = static_cast<uint32_t>(attrs.expert_counts_per_device.size());
    uint32_t E = EP > 0 ? static_cast<uint32_t>(attrs.expert_counts_per_device[0].size()) : 0;
    uint32_t max_total_rows = 0;
    for (uint32_t owner = 0; owner < EP; owner++) {
        uint32_t total = 0;
        for (uint32_t e_local = 0; e_local < attrs.E_local; e_local++) {
            uint32_t ge = owner * attrs.E_local + e_local;
            if (ge >= E)
                break;
            for (uint32_t d = 0; d < EP; d++) {
                total += attrs.expert_counts_per_device[d][ge] / 32;
            }
        }
        max_total_rows = std::max(max_total_rows, total);
    }
    uint32_t out_N = max_total_rows * 32;
    if (out_N == 0)
        out_N = 32;
    uint32_t D = sh[3];

    ttnn::Shape out_shape({sh[0], sh[1], out_N, wu[3]});
    ttnn::Shape dispatch_shape({sh[0], sh[1], out_N, D});
    return {ttnn::TensorSpec(out_shape, layout), ttnn::TensorSpec(dispatch_shape, layout)};
}

MoeDispatchDeviceOperation::tensor_return_value_t MoeDispatchDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto specs = compute_output_specs(attrs, args);
    return {
        create_device_tensor(specs[0], args.sorted_hidden.device()),
        create_device_tensor(specs[1], args.sorted_hidden.device()),
    };
}

ttsl::hash::hash_t MoeDispatchDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    uint32_t num_ep = static_cast<uint32_t>(attrs.expert_counts_per_device.size());
    uint32_t E = num_ep > 0 ? static_cast<uint32_t>(attrs.expert_counts_per_device[0].size()) : 0;
    return tt::tt_metal::operation::hash_operation<MoeDispatchDeviceOperation>(
        args.sorted_hidden.padded_shape(), args.w_up.padded_shape(), attrs.E_local, attrs.cluster_axis, num_ep, E);
}

}  // namespace ttml::metal::ops::moe_dispatch

namespace ttnn::prim {

std::vector<ttnn::Tensor> ttml_moe_dispatch(
    const ttnn::Tensor& sorted_hidden,
    const ttnn::Tensor& w_up,
    uint32_t cluster_axis,
    const std::vector<std::vector<uint32_t>>& expert_offsets_per_device,
    const std::vector<std::vector<uint32_t>>& expert_counts_per_device,
    uint32_t E_local) {
    using Op = ttml::metal::ops::moe_dispatch::MoeDispatchDeviceOperation;
    auto attrs = Op::operation_attributes_t{
        .cluster_axis = cluster_axis,
        .E_local = E_local,
        .expert_counts_per_device = expert_counts_per_device,
        .expert_offsets_per_device = expert_offsets_per_device,
    };
    auto args = Op::tensor_args_t{
        .sorted_hidden = sorted_hidden,
        .w_up = w_up,
    };
    return ttnn::device_operation::launch<Op>(attrs, args);
}

}  // namespace ttnn::prim
