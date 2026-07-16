// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ungroup_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "moe_ungroup_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::moe_ungroup::device {

void MoeUngroupDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto check =
        [](const ttnn::Tensor& t, const char* name, tt::tt_metal::Layout layout, tt::tt_metal::DataType dtype) {
            TT_FATAL(t.storage_type() == tt::tt_metal::StorageType::DEVICE, "moe_ungroup: {} must be on device", name);
            TT_FATAL(t.buffer() != nullptr, "moe_ungroup: {} buffer is null", name);
            TT_FATAL(
                t.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                "moe_ungroup: {} must be in DRAM, got {}",
                name,
                enchantum::to_string(t.buffer()->buffer_type()));
            TT_FATAL(
                t.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                "moe_ungroup: {} must be INTERLEAVED, got {}",
                name,
                enchantum::to_string(t.memory_config().memory_layout()));
            TT_FATAL(
                t.layout() == layout,
                "moe_ungroup: {} must be {} layout, got {}",
                name,
                enchantum::to_string(layout),
                enchantum::to_string(t.layout()));
            TT_FATAL(
                t.dtype() == dtype,
                "moe_ungroup: {} must be {}, got {}",
                name,
                enchantum::to_string(dtype),
                enchantum::to_string(t.dtype()));
        };

    check(args.expert_out, "expert_out", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check(args.plan, "plan", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);
    check(args.offsets, "offsets", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);
    check(args.grouped_scores, "grouped_scores", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::BFLOAT16);

    const auto& es = args.expert_out.logical_shape();
    TT_FATAL(es.rank() == 4U, "moe_ungroup: expert_out must be 4D [1,1,T_cap,H]");
    TT_FATAL(
        es[0] == 1U && es[1] == 1U && es[2] == attrs.t_cap && es[3] == attrs.h,
        "moe_ungroup: expert_out shape mismatch");

    const auto& ps = args.plan.logical_shape();
    TT_FATAL(
        ps.rank() == 4U && ps[0] == 1U && ps[1] == 1U && ps[2] == 1U && ps[3] == attrs.t_cap,
        "moe_ungroup: plan must be [1,1,1,T_cap={}], got rank={} shape={}",
        attrs.t_cap,
        ps.rank(),
        ps);

    const auto& os = args.offsets.logical_shape();
    TT_FATAL(
        os.rank() == 4U && os[0] == 1U && os[1] == 1U && os[2] == 1U && os[3] == attrs.e_local + 1U,
        "moe_ungroup: offsets must be [1,1,1,E_local+1={}], got rank={} shape={}",
        attrs.e_local + 1U,
        os.rank(),
        os);

    const auto& gss = args.grouped_scores.logical_shape();
    TT_FATAL(
        gss.rank() == 4U && gss[0] == 1U && gss[1] == 1U && gss[2] == 1U && gss[3] == attrs.t_cap,
        "moe_ungroup: grouped_scores must be [1,1,1,T_cap={}], got rank={} shape={}",
        attrs.t_cap,
        gss.rank(),
        gss);

    TT_FATAL(attrs.e_local > 0U, "moe_ungroup: e_local must be > 0");
    TT_FATAL(attrs.h > 0U, "moe_ungroup: H must be > 0");
    TT_FATAL(attrs.t_cap % tt::constants::TILE_HEIGHT == 0U, "moe_ungroup: T_cap must be a multiple of 32");
}

spec_return_value_t MoeUngroupDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto dram = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};

    // ungrouped: [D, B, S, H]  ROW_MAJOR  bf16
    return ttnn::TensorSpec(
        ttnn::Shape{attrs.d, attrs.b, attrs.s, attrs.h},
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR, dram));
}

tensor_return_value_t MoeUngroupDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto spec = compute_output_specs(attrs, args);
    auto* device = args.expert_out.device();
    return create_device_tensor(spec, device);
}

}  // namespace ttml::metal::ops::moe_ungroup::device

namespace ttnn::prim {

ttml::metal::ops::moe_ungroup::device::MoeUngroupDeviceOperation::tensor_return_value_t ttml_moe_ungroup(
    const ttnn::Tensor& expert_out,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& grouped_scores,
    uint32_t e_local,
    uint32_t d,
    uint32_t b,
    uint32_t s) {
    using Op = ttml::metal::ops::moe_ungroup::device::MoeUngroupDeviceOperation;

    const auto& es = expert_out.logical_shape();
    uint32_t t_cap = es[2];
    uint32_t h = es[3];

    auto attrs = Op::operation_attributes_t{
        .e_local = e_local,
        .d = d,
        .b = b,
        .s = s,
        .h = h,
        .t_cap = t_cap,
    };
    auto tensor_args = Op::tensor_args_t{
        .expert_out = expert_out,
        .plan = plan,
        .offsets = offsets,
        .grouped_scores = grouped_scores,
    };

    return ttnn::device_operation::launch<Op>(attrs, tensor_args);
}

}  // namespace ttnn::prim
