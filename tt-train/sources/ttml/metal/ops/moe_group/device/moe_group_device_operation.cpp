// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_group_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "moe_group_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::moe_group::device {

void MoeGroupDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto check =
        [](const ttnn::Tensor& t, const char* name, tt::tt_metal::Layout layout, tt::tt_metal::DataType dtype) {
            TT_FATAL(t.storage_type() == tt::tt_metal::StorageType::DEVICE, "moe_group: {} must be on device", name);
            TT_FATAL(t.buffer() != nullptr, "moe_group: {} buffer is null", name);
            TT_FATAL(
                t.layout() == layout,
                "moe_group: {} must be {} layout, got {}",
                name,
                enchantum::to_string(layout),
                enchantum::to_string(t.layout()));
            TT_FATAL(
                t.dtype() == dtype,
                "moe_group: {} must be {}, got {}",
                name,
                enchantum::to_string(dtype),
                enchantum::to_string(t.dtype()));
        };

    check(args.dispatched, "dispatched", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::BFLOAT16);
    check(args.metadata, "metadata", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT16);
    check(args.local_expert_ids, "local_expert_ids", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT16);

    const auto& ds = args.dispatched.logical_shape();
    TT_FATAL(ds.rank() == 4U, "moe_group: dispatched must be 4D [D,B,S,H]");
    TT_FATAL(
        ds[0] == attrs.d && ds[1] == attrs.b && ds[2] == attrs.s && ds[3] == attrs.h,
        "moe_group: dispatched shape mismatch");

    const auto& ms = args.metadata.logical_shape();
    TT_FATAL(ms.rank() == 4U, "moe_group: metadata must be 4D [D,B,S,K]");
    TT_FATAL(ms[3] == attrs.k, "moe_group: metadata K mismatch");

    const auto& ls = args.local_expert_ids.logical_shape();
    TT_FATAL(ls.rank() == 1U && ls[0] == attrs.e_local, "moe_group: local_expert_ids shape mismatch");
}

spec_return_value_t MoeGroupDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto dram = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};

    // grouped: [1, 1, T_cap, H]  TILE  bf16
    ttnn::TensorSpec grouped_spec(
        ttnn::Shape{1U, 1U, attrs.t_cap, attrs.h},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, dram));

    // counts: [1, 1, 1, E_local]  ROW_MAJOR  uint32
    ttnn::TensorSpec counts_spec(
        ttnn::Shape{1U, 1U, 1U, attrs.e_local},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, tt::tt_metal::Layout::ROW_MAJOR, dram));

    // offsets: [1, 1, 1, E_local+1]  ROW_MAJOR  uint32
    ttnn::TensorSpec offsets_spec(
        ttnn::Shape{1U, 1U, 1U, attrs.e_local + 1U},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, tt::tt_metal::Layout::ROW_MAJOR, dram));

    // plan: [1, 1, 1, T_cap]  ROW_MAJOR  uint32
    ttnn::TensorSpec plan_spec(
        ttnn::Shape{1U, 1U, 1U, attrs.t_cap},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, tt::tt_metal::Layout::ROW_MAJOR, dram));

    return {grouped_spec, counts_spec, offsets_spec, plan_spec};
}

tensor_return_value_t MoeGroupDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto specs = compute_output_specs(attrs, args);
    auto* device = args.dispatched.device();
    return {
        create_device_tensor(specs[0], device),
        create_device_tensor(specs[1], device),
        create_device_tensor(specs[2], device),
        create_device_tensor(specs[3], device),
    };
}

ttsl::hash::hash_t MoeGroupDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return tt::tt_metal::operation::hash_operation<MoeGroupDeviceOperation>(
        attrs, args.dispatched.dtype(), args.dispatched.logical_shape());
}

}  // namespace ttml::metal::ops::moe_group::device

namespace ttnn::prim {

ttml::metal::ops::moe_group::device::MoeGroupDeviceOperation::tensor_return_value_t ttml_moe_group(
    const ttnn::Tensor& dispatched,
    const ttnn::Tensor& metadata,
    const ttnn::Tensor& local_expert_ids,
    uint32_t e_local,
    uint32_t k) {
    using Op = ttml::metal::ops::moe_group::device::MoeGroupDeviceOperation;

    const auto& ds = dispatched.logical_shape();
    uint32_t d = ds[0], b = ds[1], s = ds[2], h = ds[3];
    // Upper bound includes per-core padding (3 slots per core per expert for 16B alignment)
    // plus 32-row tail padding per expert. Use grid size to compute num_total_cores.
    auto grid = dispatched.device()->compute_with_storage_grid_size();
    uint32_t num_total_cores = grid.x * grid.y;
    uint32_t t_cap = std::min(e_local, k) * d * b * s + e_local * (32U + 3U * num_total_cores);

    auto attrs = Op::operation_attributes_t{
        .e_local = e_local,
        .k = k,
        .d = d,
        .b = b,
        .s = s,
        .h = h,
        .t_cap = t_cap,
    };
    auto tensor_args = Op::tensor_args_t{
        .dispatched = dispatched,
        .metadata = metadata,
        .local_expert_ids = local_expert_ids,
    };

    return ttnn::device_operation::launch<Op>(attrs, tensor_args);
}

}  // namespace ttnn::prim
