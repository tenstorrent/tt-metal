// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tuple>

#include "chunk_gated_delta_rule_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device {

namespace {

// The g/beta gather reads head h as column h of a single tile column, so HV must fit one tile.
constexpr uint32_t kMaxValueHeads = tt::constants::TILE_WIDTH;

void check_tensor(const ttnn::Tensor& tensor, const char* name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE && tensor.buffer() != nullptr,
        "chunk_gated_delta_rule_bw: {} must be allocated on device",
        name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "chunk_gated_delta_rule_bw: {} must be TILE, got {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == dtype,
        "chunk_gated_delta_rule_bw: {} dtype {} must be {}",
        name,
        enchantum::to_string(tensor.dtype()),
        enchantum::to_string(dtype));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "chunk_gated_delta_rule_bw: {} must be DRAM interleaved",
        name);
}

}  // namespace

void ChunkGatedDeltaRuleBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto dtype = tensor_args.q.dtype();
    TT_FATAL(
        dtype == tt::tt_metal::DataType::BFLOAT16 || dtype == tt::tt_metal::DataType::FLOAT32,
        "chunk_gated_delta_rule_bw: dtype must be BFLOAT16 or FLOAT32, got {}",
        enchantum::to_string(dtype));
    check_tensor(tensor_args.q, "q", dtype);
    check_tensor(tensor_args.k, "k", dtype);
    check_tensor(tensor_args.v, "v", dtype);
    check_tensor(tensor_args.g, "g", dtype);
    check_tensor(tensor_args.beta, "beta", dtype);
    check_tensor(tensor_args.d_out, "d_out", dtype);
    if (tensor_args.initial_state.has_value()) {
        check_tensor(*tensor_args.initial_state, "initial_state", dtype);
    }
    if (tensor_args.d_final_state.has_value()) {
        check_tensor(*tensor_args.d_final_state, "d_final_state", dtype);
    }

    const auto& q_shape = tensor_args.q.logical_shape();
    const auto qs = as_rank4(q_shape);
    const auto vs = as_rank4(tensor_args.v.logical_shape());
    const auto gs = as_rank4(tensor_args.g.logical_shape());
    const auto ds = as_rank4(tensor_args.d_out.logical_shape());
    TT_FATAL(
        qs.rank() == 4 && vs.rank() == 4 && gs.rank() == 4 && ds.rank() == 4 && qs[1] == 1 && vs[1] == 1 &&
            gs[1] == 1 && ds[1] == 1,
        "chunk_gated_delta_rule_bw: q/k must be [B,T,HK*K], v [B,T,HV*V], g/beta [B,T,HV] (or [B,1,T,X]), d_out "
        "[B,T,HV*V] or [B*HV,1,T,V]; got q={} v={} g={} d_out={}",
        q_shape,
        tensor_args.v.logical_shape(),
        tensor_args.g.logical_shape(),
        tensor_args.d_out.logical_shape());
    const uint32_t B = qs[0];
    const uint32_t T = qs[2];
    const uint32_t HV = gs[3];
    TT_FATAL(
        tensor_args.k.logical_shape() == q_shape && tensor_args.beta.logical_shape() == tensor_args.g.logical_shape() &&
            vs[0] == B && vs[2] == T && gs[0] == B && gs[2] == T,
        "chunk_gated_delta_rule_bw: shapes disagree: q={} k={} v={} g={} beta={}",
        q_shape,
        tensor_args.k.logical_shape(),
        tensor_args.v.logical_shape(),
        tensor_args.g.logical_shape(),
        tensor_args.beta.logical_shape());
    TT_FATAL(
        HV >= 1 && HV <= kMaxValueHeads, "chunk_gated_delta_rule_bw: HV={} must be in [1, {}]", HV, kMaxValueHeads);

    const uint32_t C = args.chunk_size;
    TT_FATAL(C == 32U || C == 64U, "chunk_gated_delta_rule_bw: chunk_size must be 32 or 64, got {}", C);
    TT_FATAL(
        T % C == 0U,
        "chunk_gated_delta_rule_bw: T={} must be a multiple of chunk_size={} (whole token tiles move as pages)",
        T,
        C);

    // A page is [32 tokens x 32 dims] of one head; a head dim off the tile grid would address pages
    // straddling two heads.
    const uint32_t K = args.key_head_dim;
    TT_FATAL(vs[3] % HV == 0U, "chunk_gated_delta_rule_bw: v width {} is not a multiple of HV={}", vs[3], HV);
    const uint32_t V = vs[3] / HV;
    TT_FATAL(
        K > 0U && K % tt::constants::TILE_WIDTH == 0U && V % tt::constants::TILE_WIDTH == 0U,
        "chunk_gated_delta_rule_bw: K={} and V={} must be multiples of 32",
        K,
        V);
    TT_FATAL(qs[3] % K == 0U, "chunk_gated_delta_rule_bw: q width {} is not a multiple of K={}", qs[3], K);
    const uint32_t HK = qs[3] / K;
    TT_FATAL(
        HK >= 1U && HV % HK == 0U,
        "chunk_gated_delta_rule_bw: value heads HV={} must be a multiple of key heads HK={}",
        HV,
        HK);
    TT_FATAL(
        ds == vs || ds == ttnn::Shape({B * HV, 1U, T, V}),
        "chunk_gated_delta_rule_bw: d_out must be token-major {} or head-major [{}, 1, {}, {}], got {}",
        vs,
        B * HV,
        T,
        V,
        tensor_args.d_out.logical_shape());

    const ttnn::Shape state_shape({B, HV, K, V});
    for (const auto& [state, name] :
         {std::pair{&tensor_args.initial_state, "initial_state"},
          std::pair{&tensor_args.d_final_state, "d_final_state"}}) {
        if (state->has_value()) {
            TT_FATAL(
                (*state)->logical_shape() == state_shape,
                "chunk_gated_delta_rule_bw: {} must be {}, got {}",
                name,
                state_shape,
                (*state)->logical_shape());
        }
    }

    const bool has_h = tensor_args.h.has_value();
    TT_FATAL(
        tensor_args.v_new.has_value() == has_h && tensor_args.a.has_value() == has_h,
        "chunk_gated_delta_rule_bw: h, v_new and a are the saved forward and must be given together");
    if (has_h) {
        const uint32_t BH = B * HV;
        const ttnn::Shape h_shape({BH, T / C, K, V});
        const ttnn::Shape v_new_shape({BH, 1U, T, V});
        const ttnn::Shape a_shape({BH, 1U, T, C});
        for (const auto& [tensor, name, shape] :
             {std::tuple{&*tensor_args.h, "h", &h_shape},
              std::tuple{&*tensor_args.v_new, "v_new", &v_new_shape},
              std::tuple{&*tensor_args.a, "a", &a_shape}}) {
            check_tensor(*tensor, name, tt::tt_metal::DataType::FLOAT32);
            TT_FATAL(
                tensor->logical_shape() == *shape,
                "chunk_gated_delta_rule_bw: {} must be {}, got {}",
                name,
                *shape,
                tensor->logical_shape());
        }
    }
}

spec_return_value_t ChunkGatedDeltaRuleBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Geometry geo = make_geometry(args, tensor_args);
    const auto dtype = tensor_args.q.dtype();
    const auto layout = [&](tt::tt_metal::DataType dt, const tt::tt_metal::MemoryConfig& mem) {
        return tt::tt_metal::TensorLayout(dt, tt::tt_metal::Layout::TILE, mem);
    };
    const auto& out_mem = args.output_memory_config;

    spec_return_value_t specs;
    // dq/dk: one [B,1,T,HK*K] plane per value head of a key head; the caller sums the G planes.
    const ttnn::Shape dqk_shape({geo.G * geo.B, 1U, geo.T, geo.HK * geo.K});
    specs.emplace_back(dqk_shape, layout(dtype, out_mem));
    specs.emplace_back(dqk_shape, layout(dtype, out_mem));
    specs.emplace_back(tensor_args.v.logical_shape(), layout(dtype, out_mem));
    specs.emplace_back(tensor_args.g.logical_shape(), layout(dtype, out_mem));
    specs.emplace_back(tensor_args.g.logical_shape(), layout(dtype, out_mem));
    specs.emplace_back(
        ttnn::Shape({1U, 1U, geo.scratch_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}),
        layout(tt::tt_metal::DataType::FLOAT32, ttnn::DRAM_MEMORY_CONFIG));
    if (tensor_args.initial_state.has_value()) {
        specs.emplace_back(ttnn::Shape({geo.B, geo.HV, geo.K, geo.V}), layout(dtype, out_mem));
    }
    return specs;
}

tensor_return_value_t ChunkGatedDeltaRuleBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    auto* device = tensor_args.q.device();
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, device));
    }
    return outputs;
}

ttsl::hash::hash_t ChunkGatedDeltaRuleBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<ChunkGatedDeltaRuleBackwardDeviceOperation>(
        args.chunk_size,
        args.key_head_dim,
        args.compute_kernel_config,
        args.output_memory_config,
        tensor_args.q.dtype(),
        tensor_args.q.logical_shape(),
        tensor_args.v.logical_shape(),
        tensor_args.g.logical_shape(),
        tensor_args.d_out.logical_shape(),
        tensor_args.initial_state.has_value(),
        tensor_args.d_final_state.has_value(),
        tensor_args.has_saved_forward());
}

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device

namespace ttnn::prim {

ttml::metal::ops::chunk_gated_delta_rule_bw::device::ChunkGatedDeltaRuleBackwardDeviceOperation::tensor_return_value_t
ttml_chunk_gated_delta_rule_bw(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& d_out,
    const std::optional<ttnn::Tensor>& initial_state,
    const std::optional<ttnn::Tensor>& d_final_state,
    const std::optional<ttnn::Tensor>& h,
    const std::optional<ttnn::Tensor>& v_new,
    const std::optional<ttnn::Tensor>& a,
    uint32_t chunk_size,
    uint32_t key_head_dim,
    float scale,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using OperationType =
        ttml::metal::ops::chunk_gated_delta_rule_bw::device::ChunkGatedDeltaRuleBackwardDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .chunk_size = chunk_size,
        .key_head_dim = key_head_dim,
        .scale = scale,
        .compute_kernel_config = compute_kernel_config,
        .output_memory_config = output_memory_config};
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .d_out = d_out,
        .initial_state = initial_state,
        .d_final_state = d_final_state,
        .h = h,
        .v_new = v_new,
        .a = a};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
