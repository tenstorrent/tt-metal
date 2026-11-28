// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_composite.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"
#include "ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {
namespace {

using ttnn::Tensor;
constexpr auto kF32 = tt::tt_metal::DataType::FLOAT32;
constexpr auto kBF16 = tt::tt_metal::DataType::BFLOAT16;

// fp32 matmul with fp32 accumulation, batched over the two leading dims.
Tensor mm(const Tensor& a, const Tensor& b) {
    return ttnn_fixed::matmul(a, b, /*transpose_a=*/false, /*transpose_b=*/false);
}

// [B, 1, T, Hs*D] -> [B*Hs*G, T/C, C, D] bf16 TILE; every one of the Hs heads is repeated G times (GVA).
Tensor heads_chunked(const Tensor& flat, uint32_t B, uint32_t T, uint32_t Hs, uint32_t D, uint32_t G, uint32_t C) {
    auto x = std::get<0>(ttnn::experimental::nlp_create_qkv_heads(
        flat,
        std::nullopt,
        /*num_q_heads=*/Hs,
        /*num_kv_heads=*/0U,
        /*transpose_k_heads=*/false,
        /*kv_tied=*/false,
        /*memory_config=*/std::nullopt,
        /*optional_output_tensors=*/std::nullopt));  // [B, Hs, T, D]
    if (G > 1U) {
        x = ttnn::repeat_interleave(x, G, /*dim=*/1);  // [B, Hs*G, T, D]
    }
    return ttnn::reshape(x, ttnn::Shape{B * Hs * G, T / C, C, D});
}

// [BH, NC, R, W] -> [BH, 1, R, W]: chunk i.
Tensor chunk_at(const Tensor& x, uint32_t i) {
    const auto& s = x.logical_shape();
    return ttnn::slice(
        x,
        ttsl::SmallVector<uint32_t>{0U, i, 0U, 0U},
        ttsl::SmallVector<uint32_t>{s[0], i + 1U, s[2], s[3]},
        ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});
}

// The constant fp32 inputs of chunk_gdn_prep: the [1, 1, C, C] identity, inclusive lower ones and ones, and the three
// 16x16 quadrant masks of a 32x32 tile packed into [1, 1, 32, 96] (top-left, bottom-right, bottom-left).  Uploaded
// once per device, so a forward does no host writes (which trace capture forbids).
struct PrepConstants {
    Tensor eye;
    Tensor tril;
    Tensor ones;
    Tensor masks;
};

Tensor device_constant(const std::string& key, uint32_t rows, uint32_t cols, float (*value)(uint32_t, uint32_t)) {
    return autograd::ctx().get_device_constant(key, [&] {
        std::vector<float> data(static_cast<size_t>(rows) * cols);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                data[i * cols + j] = value(i, j);
            }
        }
        return core::from_vector<float, kF32>(data, ttnn::Shape{1U, 1U, rows, cols}, &autograd::ctx().get_device());
    });
}

PrepConstants prep_constants(uint32_t C) {
    const auto key = [C](const char* name) { return fmt::format("chunk_gated_delta_rule/{}/{}", name, C); };
    return {
        device_constant(key("eye"), C, C, [](uint32_t i, uint32_t j) { return i == j ? 1.0F : 0.0F; }),
        device_constant(key("tril"), C, C, [](uint32_t i, uint32_t j) { return j <= i ? 1.0F : 0.0F; }),
        device_constant(key("ones"), C, C, [](uint32_t, uint32_t) { return 1.0F; }),
        device_constant(
            "chunk_gated_delta_rule/masks",
            32U,
            96U,
            [](uint32_t i, uint32_t j) {
                const bool lo_i = i < 16U;
                const bool lo_j = j % 32U < 16U;
                const uint32_t quadrant = j / 32U;
                const bool on = quadrant == 0U ? (lo_i && lo_j) : quadrant == 1U ? (!lo_i && !lo_j) : (!lo_i && lo_j);
                return on ? 1.0F : 0.0F;
            }),
    };
}

// What chunk_gdn_prep runs with in ttnn's own chunked forward.
ttnn::DeviceComputeKernelConfig prep_kernel_config() {
    auto config = core::ComputeKernelConfig::precise();
    config.packer_l1_acc = false;
    return config;
}

void validate(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    uint32_t K,
    const std::optional<Tensor>& h0,
    uint32_t C,
    tt::tt_metal::DataType output_dtype) {
    // Placement first, so a host, unallocated or foreign-device tensor fails here and not inside a ttnn op.
    constexpr const char* kOp = "chunk_gated_delta_rule";
    const metal::DeviceTensorRequirements bf16_tile{};  // defaults: BFLOAT16, TILE, interleaved
    metal::check_device_tensor(q, kOp, "q", bf16_tile);
    metal::check_device_tensor(k, kOp, "k", bf16_tile);
    metal::check_device_tensor(v, kOp, "v", bf16_tile);
    // The gates may be BF16 or FP32; beta must match g.
    metal::check_device_tensor(g, kOp, "g", {.dtypes = {kBF16, kF32}});
    metal::check_device_tensor(beta, kOp, "beta", {.dtypes = {g.dtype()}});
    for (const auto& [tensor, name] :
         {std::pair{&k, "k"}, std::pair{&v, "v"}, std::pair{&g, "g"}, std::pair{&beta, "beta"}}) {
        metal::check_same_device(*tensor, q, kOp, name, "q");
    }
    if (h0.has_value()) {
        metal::check_device_tensor(*h0, kOp, "initial_state", {.dtypes = {kF32}});
        metal::check_same_device(*h0, q, kOp, "initial_state", "q");
    }

    for (const auto* t : {&q, &k, &v, &g, &beta}) {
        TT_FATAL(t->logical_shape().rank() == 4U, "chunk_gated_delta_rule: inputs must be rank 4");
    }
    TT_FATAL(C == 32U || C == 64U, "chunk_gated_delta_rule: chunk_size must be 32 or 64, got {}", C);
    TT_FATAL(
        output_dtype == kF32 || output_dtype == kBF16,
        "chunk_gated_delta_rule: the output dtype must be FLOAT32 or BFLOAT16, got {}",
        output_dtype);
    TT_FATAL(
        K > 0U && K % 32U == 0U, "chunk_gated_delta_rule: key_head_dim must be a positive multiple of 32, got {}", K);

    const auto& qs = q.logical_shape();
    const auto& vs = v.logical_shape();
    const auto& gs = g.logical_shape();
    const uint32_t B = qs[0], T = qs[2], HV = gs[3], V = vs[3] / std::max(HV, 1U);
    TT_FATAL(
        B >= 1U && T >= C && T % C == 0U,
        "chunk_gated_delta_rule: T={} must be a multiple of chunk_size={} and >= it",
        T,
        C);
    TT_FATAL(qs[1] == 1U && qs[3] % K == 0U, "chunk_gated_delta_rule: q must be [B,1,T,HK*K], got {}", qs);
    TT_FATAL(k.logical_shape() == qs, "chunk_gated_delta_rule: k {} must match q {}", k.logical_shape(), qs);
    const uint32_t HK = qs[3] / K;
    TT_FATAL(HV >= 1U && HV % HK == 0U, "chunk_gated_delta_rule: HK={} must divide HV={}", HK, HV);
    TT_FATAL(
        vs[0] == B && vs[1] == 1U && vs[2] == T && vs[3] == V * HV && V % 32U == 0U,
        "chunk_gated_delta_rule: v must be [B,1,T,HV*V] with V a multiple of 32, got {}",
        vs);
    TT_FATAL(
        gs == ttnn::Shape({B, 1U, T, HV}) && beta.logical_shape() == gs,
        "chunk_gated_delta_rule: g and beta must be [B,1,T,HV], got {} and {}",
        gs,
        beta.logical_shape());
    if (h0.has_value()) {
        TT_FATAL(
            h0->logical_shape() == ttnn::Shape({B, HV, K, V}),
            "chunk_gated_delta_rule: initial_state must be [B,HV,K,V]=[{},{},{},{}], got {}",
            B,
            HV,
            K,
            V,
            h0->logical_shape());
    }
}

}  // namespace

ChunkGatedDeltaRuleCompositeResult chunk_gated_delta_rule_fw_composite(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    uint32_t key_head_dim,
    const std::optional<ttnn::Tensor>& initial_state,
    uint32_t chunk_size,
    std::optional<float> scale,
    tt::tt_metal::DataType output_dtype,
    bool save_for_backward) {
    const uint32_t K = key_head_dim;
    const uint32_t C = chunk_size;
    validate(q, k, v, g, beta, K, initial_state, C, output_dtype);

    const uint32_t B = q.logical_shape()[0];
    const uint32_t T = q.logical_shape()[2];
    const uint32_t HK = q.logical_shape()[3] / K;
    const uint32_t HV = g.logical_shape()[3];
    const uint32_t V = v.logical_shape()[3] / HV;
    const uint32_t G = HV / HK;
    const uint32_t BH = B * HV;
    const uint32_t NC = T / C;
    const float q_scale = scale.value_or(1.0F / std::sqrt(static_cast<float>(K)));

    auto* device = q.device();

    // The state-independent per-chunk terms come from ttnn's chunk_gdn_prep, the first phase of ttnn's chunked
    // forward.  It reads q and k head-major [BH, NC, C, K] bf16 (GVA repeated), v flat [B, T, HV*V] and the gates
    // token-major [B, T, HV], and returns, all fp32 with gc the chunk-local inclusive cumsum of g:
    //   v_beta [BH, NC, C, V]    v * beta
    //   nkd [BH, NC, C, K]       -(k * beta * exp(gc))
    //   q_decay [BH, NC, C, K]   q * exp(gc)
    //   intra [BH, NC, C, C]     (q k^T) * L, L[i, j] = exp(gc_i - gc_j) for i >= j, else 0
    //   k_dec_t [BH, NC, K, C]   (k * exp(gc_last - gc))^T
    //   dl [BH, NC, 32, 32]      exp(gc_last) on the diagonal
    //   t_inv [BH, NC, C, C]     (I + N)^-1, N = strict_lower((k * beta) k^T * L)
    // q goes in unscaled, so the output is scaled instead.
    const auto consts = prep_constants(C);
    const auto prep = ttnn::prim::chunk_gdn_prep(
        heads_chunked(q, B, T, HK, K, G, C),
        heads_chunked(k, B, T, HK, K, G, C),
        ttnn::reshape(v, ttnn::Shape{B, T, HV * V}),
        ttnn::reshape(g, ttnn::Shape{B, T, HV}),
        ttnn::reshape(beta, ttnn::Shape{B, T, HV}),
        consts.eye,
        consts.tril,
        consts.ones,
        consts.masks,
        C,
        ttnn::DRAM_MEMORY_CONFIG,
        prep_kernel_config(),
        /*v_flat=*/true,
        HV);
    // o per chunk [BH, NC, C, V], from the unscaled q -> [BH, 1, T, V] in output_dtype.
    const auto finish_output = [&](const Tensor& o_chunks) {
        auto out = ttnn::reshape(ttnn::multiply(o_chunks, q_scale), ttnn::Shape{BH, 1U, T, V});
        return out.dtype() == output_dtype ? out : ttnn::typecast(out, output_dtype);
    };

    // Without the saved state, ttnn's scan runs the recurrence and the output in one launch.  Its o differs from
    // the saved path's by rounding.
    if (!save_for_backward) {
        const Tensor s0 = initial_state.has_value() ? ttnn::reshape(*initial_state, ttnn::Shape{BH, K, V})
                                                    : core::zeros(ttnn::Shape{BH, K, V}, device, kF32);
        const auto scan = ttnn::prim::chunk_gdn_scan(
            prep[0],
            prep[1],
            prep[2],
            prep[3],
            prep[4],
            prep[5],
            prep[6],
            s0,
            C,
            /*output_final_state=*/true,
            ttnn::DRAM_MEMORY_CONFIG,
            prep_kernel_config());
        return {
            .output = finish_output(scan[0]),
            .final_state = ttnn::reshape(scan[1], ttnn::Shape{B, HV, K, V}),
        };
    }

    const auto& q_decay = prep[2];
    const auto& intra = prep[3];
    const auto& k_dec_t = prep[4];
    const auto& t_inv = prep[6];
    const auto u = mm(t_inv, prep[0]);      // T_inv v_beta, [BH, NC, C, V]
    const auto neg_w = mm(t_inv, prep[1]);  // -T_inv (k * beta * exp(gc)), [BH, NC, C, K]
    const auto decay_last = ttnn::slice(
        prep[5],
        ttsl::SmallVector<uint32_t>{0U, 0U, 0U, 0U},
        ttsl::SmallVector<uint32_t>{BH, NC, 1U, 1U},
        ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});  // exp(gc_last), [BH, NC, 1, 1]

    // The only sequential part: the state S [BH, 1, K, V] across the chunks.  The output does not feed the
    // recurrence, so it is two batched matmuls on the collected states and corrected values after the loop.
    Tensor state = initial_state.has_value() ? ttnn::reshape(*initial_state, ttnn::Shape{BH, 1U, K, V})
                                             : core::zeros(ttnn::Shape{BH, 1U, K, V}, device, kF32);
    std::vector<Tensor> h_parts, v_new_parts;
    h_parts.reserve(NC);
    v_new_parts.reserve(NC);
    for (uint32_t i = 0; i < NC; ++i) {
        h_parts.push_back(state);
        v_new_parts.push_back(ttnn::add(chunk_at(u, i), mm(chunk_at(neg_w, i), state)));  // [BH, 1, C, V]
        state = ttnn::add(ttnn::multiply(state, chunk_at(decay_last, i)), mm(chunk_at(k_dec_t, i), v_new_parts.back()));
    }
    auto h = ttnn::concat(h_parts, /*dim=*/1);          // [BH, NC, K, V]
    auto v_new = ttnn::concat(v_new_parts, /*dim=*/1);  // [BH, NC, C, V]

    return {
        .output = finish_output(ttnn::add(mm(q_decay, h), mm(intra, v_new))),
        .final_state = ttnn::reshape(state, ttnn::Shape{B, HV, K, V}),
        .h = std::move(h),
        .v_new = ttnn::reshape(v_new, ttnn::Shape{BH, 1U, T, V}),
        .a = ttnn::reshape(t_inv, ttnn::Shape{BH, 1U, T, C}),
    };
}

}  // namespace ttml::ops
