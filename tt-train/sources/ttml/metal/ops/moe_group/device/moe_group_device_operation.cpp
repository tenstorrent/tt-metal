// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_group_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

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
    check(args.scores, "scores", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::BFLOAT16);
    check(args.local_expert_ids, "local_expert_ids", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT16);

    const auto& ds = args.dispatched.logical_shape();
    TT_FATAL(ds.rank() == 4U, "moe_group: dispatched must be 4D [D,B,S,H]");
    TT_FATAL(
        ds[0] == attrs.d && ds[1] == attrs.b && ds[2] == attrs.s && ds[3] == attrs.h,
        "moe_group: dispatched shape mismatch");

    const auto& ms = args.metadata.logical_shape();
    TT_FATAL(ms.rank() == 4U, "moe_group: metadata must be 4D [D,B,S,K]");
    TT_FATAL(
        ms[0] == attrs.d && ms[1] == attrs.b && ms[2] == attrs.s && ms[3] == attrs.k,
        "moe_group: metadata shape [{},{},{},{}] does not match dispatched [D={},B={},S={},K={}]",
        ms[0],
        ms[1],
        ms[2],
        ms[3],
        attrs.d,
        attrs.b,
        attrs.s,
        attrs.k);

    const auto& ss = args.scores.logical_shape();
    TT_FATAL(ss.rank() == 4U, "moe_group: scores must be 4D [D,B,S,K]");
    TT_FATAL(
        ss[0] == attrs.d && ss[1] == attrs.b && ss[2] == attrs.s && ss[3] == attrs.k,
        "moe_group: scores shape [{},{},{},{}] does not match metadata [D={},B={},S={},K={}]",
        ss[0],
        ss[1],
        ss[2],
        ss[3],
        attrs.d,
        attrs.b,
        attrs.s,
        attrs.k);

    const auto& ls = args.local_expert_ids.logical_shape();
    TT_FATAL(ls.rank() == 1U && ls[0] == attrs.e_local, "moe_group: local_expert_ids shape mismatch");

    TT_FATAL(attrs.e_local > 0U, "moe_group: e_local must be > 0");
    TT_FATAL(attrs.k > 0U, "moe_group: k must be > 0");
    TT_FATAL(attrs.h > 0U, "moe_group: h must be > 0");
    TT_FATAL(attrs.d > 0U && attrs.b > 0U && attrs.s > 0U, "moe_group: D, B, S must all be > 0");
}

spec_return_value_t MoeGroupDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto dram = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};

    // grouped: [1, 1, T_cap, H]  TILE  bf16
    ttnn::TensorSpec grouped_spec(
        ttnn::Shape{1U, 1U, attrs.t_cap, attrs.h},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, dram));

    // grouped_scores: [1, 1, 1, T_cap]  ROW_MAJOR  bf16
    ttnn::TensorSpec grouped_scores_spec(
        ttnn::Shape{1U, 1U, 1U, attrs.t_cap},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR, dram));

    // k_slot: [1, 1, 1, T_cap]  ROW_MAJOR  uint16
    ttnn::TensorSpec k_slot_spec(
        ttnn::Shape{1U, 1U, 1U, attrs.t_cap},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT16, tt::tt_metal::Layout::ROW_MAJOR, dram));

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

    return {grouped_spec, grouped_scores_spec, k_slot_spec, counts_spec, offsets_spec, plan_spec};
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
        create_device_tensor(specs[4], device),
        create_device_tensor(specs[5], device),
    };
}

}  // namespace ttml::metal::ops::moe_group::device

namespace ttnn::prim {

ttml::metal::ops::moe_group::device::MoeGroupDeviceOperation::tensor_return_value_t ttml_moe_group(
    const ttnn::Tensor& dispatched,
    const ttnn::Tensor& metadata,
    const ttnn::Tensor& scores,
    const ttnn::Tensor& local_expert_ids,
    uint32_t e_local,
    uint32_t k) {
    using Op = ttml::metal::ops::moe_group::device::MoeGroupDeviceOperation;

    const auto& ds = dispatched.logical_shape();
    uint32_t d = ds[0], b = ds[1], s = ds[2], h = ds[3];
    // Upper bound includes per-core padding ((cursor_align-1) slots per core per
    // expert to keep per-core write addresses L1-aligned for ALL three side
    // tensors written per active row: plan (uint32), grouped_scores (bf16),
    // k_slot (uint16). cursor_align = max element-count alignment across the
    // three dtypes = L1_ALIGN_BYTES / sizeof(uint16_t) = 8 on WH/BH today.
    // Plus 32-row tile padding per expert.
    auto grid = dispatched.device()->compute_with_storage_grid_size();
    uint32_t num_total_cores = grid.x * grid.y;
    uint32_t l1_align_bytes = tt::tt_metal::hal::get_l1_alignment();
    uint32_t cursor_align = l1_align_bytes / sizeof(uint16_t);
    uint32_t t_cap_unaligned = std::min(e_local, k) * d * b * s +
                               e_local * (tt::constants::TILE_HEIGHT + (cursor_align - 1U) * num_total_cores);
    uint32_t t_cap = tt::round_up(t_cap_unaligned, tt::constants::TILE_HEIGHT);

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
        .scores = scores,
        .local_expert_ids = local_expert_ids,
    };

    return ttnn::device_operation::launch<Op>(attrs, tensor_args);
}

}  // namespace ttnn::prim
