// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_composite.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {
namespace {

using ttnn::Tensor;
constexpr auto kF32 = tt::tt_metal::DataType::FLOAT32;
constexpr auto kBF16 = tt::tt_metal::DataType::BFLOAT16;

// fp32 matmul with fp32 accumulation, batched over the two leading dims.
Tensor mm(const Tensor& a, const Tensor& b, bool transpose_b = false) {
    return ttnn_fixed::matmul(a, b, /*transpose_a=*/false, transpose_b);
}

Tensor to_rm_f32(const Tensor& x) {
    return ttnn::to_layout(ttnn::typecast(x, kF32), ttnn::Layout::ROW_MAJOR);
}

// [B, 1, T, Hs*D] -> [B*Hs*G, T/C, C, D] fp32 TILE; every one of the Hs heads is repeated G times (GVA).
Tensor heads_chunked(const Tensor& flat, uint32_t B, uint32_t T, uint32_t Hs, uint32_t D, uint32_t G, uint32_t C) {
    auto x = ttnn::reshape(to_rm_f32(flat), ttnn::Shape{B, T, Hs, D});
    x = ttnn::permute(x, ttsl::SmallVector<int64_t>{0, 2, 1, 3});  // [B, Hs, T, D]
    if (G > 1U) {
        x = ttnn::repeat_interleave(x, G, /*dim=*/1);  // [B, Hs*G, T, D]
    }
    x = ttnn::reshape(x, ttnn::Shape{B * Hs * G, T / C, C, D});
    return ttnn::to_layout(x, ttnn::Layout::TILE);
}

// [B, 1, T, HV] -> [B*HV, T/C, 1, C] fp32 TILE: one row of C gate values per chunk.
Tensor gate_rows(const Tensor& flat, uint32_t B, uint32_t T, uint32_t HV, uint32_t C) {
    auto x = ttnn::reshape(to_rm_f32(flat), ttnn::Shape{B, T, HV});
    x = ttnn::permute(x, ttsl::SmallVector<int64_t>{0, 2, 1});  // [B, HV, T]
    x = ttnn::reshape(x, ttnn::Shape{B * HV, T / C, 1U, C});
    return ttnn::to_layout(x, ttnn::Layout::TILE);
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

Tensor concat_or_single(std::vector<Tensor>& parts, int dim) {
    return parts.size() == 1U ? parts.front() : ttnn::concat(parts, dim);
}

// [C, C] constants, built once per call on the host: identity, strictly lower ones, and the additive mask that is 0 on
// and below the diagonal and -1e30 above it (exp of the masked difference of cumulative gates is then exactly 0).
struct Constants {
    Tensor eye;
    Tensor strict_lower;
    Tensor upper_mask;
};

Constants make_constants(uint32_t C, ttnn::distributed::MeshDevice* device) {
    std::vector<float> eye(C * C, 0.0F), strict(C * C, 0.0F), mask(C * C, 0.0F);
    for (uint32_t i = 0; i < C; ++i) {
        for (uint32_t j = 0; j < C; ++j) {
            eye[i * C + j] = i == j ? 1.0F : 0.0F;
            strict[i * C + j] = i > j ? 1.0F : 0.0F;
            mask[i * C + j] = i >= j ? 0.0F : -1e30F;
        }
    }
    const ttnn::Shape shape{1U, 1U, C, C};
    auto make = [&](const std::vector<float>& data) {
        return core::from_vector<float, ttnn::DataType::FLOAT32>(data, shape, device);
    };
    return {make(eye), make(strict), make(mask)};
}

// (I + N)^-1 for strictly lower triangular N [BH, NC, C, C].  Row i of X = I - N X depends only on rows < i, so
// X <- I - N X is exact after C - 1 rounds (the same arithmetic as forward substitution, batched over all chunks).
Tensor unit_lower_inverse(const Tensor& n, const Tensor& eye, uint32_t C) {
    auto x = ttnn::subtract(eye, n);  // exact for rows 0 and 1
    for (uint32_t round = 2U; round < C; ++round) {
        x = ttnn::subtract(eye, mm(n, x));
    }
    return x;
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
    bool output_head_major,
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

    auto* device = &autograd::ctx().get_device();
    const auto c = make_constants(C, device);

    // Per-head chunked layouts, [BH, NC, C, D] (GVA repeated), and the gates as rows [BH, NC, 1, C].
    const auto qh = ttnn::multiply(heads_chunked(q, B, T, HK, K, G, C), q_scale);
    const auto kh = heads_chunked(k, B, T, HK, K, G, C);
    const auto vh = heads_chunked(v, B, T, HV, V, 1U, C);
    const auto g_row = gate_rows(g, B, T, HV, C);
    const auto beta_row = gate_rows(beta, B, T, HV, C);

    // Chunk-local inclusive cumulative gate, as a row and as a column.
    const auto gc_row = ttnn::cumsum(g_row, /*dim=*/-1);
    const auto gc_col = ttnn::transpose(gc_row, -2, -1);  // [BH, NC, C, 1]
    const auto beta_col = ttnn::transpose(beta_row, -2, -1);

    // L[i, j] = exp(gc_i - gc_j) for i >= j, else exactly 0.
    const auto decay = ttnn::exp(ttnn::add(ttnn::subtract(gc_col, gc_row), c.upper_mask));  // [BH, NC, C, C]

    const auto k_beta = ttnn::multiply(kh, beta_col);
    const auto v_beta = ttnn::multiply(vh, beta_col);

    // N = strict_lower((k*beta) k^T * L);  T_inv = (I + N)^-1.
    const auto n = ttnn::multiply(ttnn::multiply(mm(k_beta, kh, /*transpose_b=*/true), decay), c.strict_lower);
    const auto t_inv = unit_lower_inverse(n, c.eye, C);  // [BH, NC, C, C]

    const auto exp_gc = ttnn::exp(gc_col);                                       // [BH, NC, C, 1]
    const auto w = mm(t_inv, ttnn::multiply(k_beta, exp_gc));                    // [BH, NC, C, K]
    const auto u = mm(t_inv, v_beta);                                            // [BH, NC, C, V]
    const auto intra = ttnn::multiply(mm(qh, kh, /*transpose_b=*/true), decay);  // [BH, NC, C, C]
    const auto q_decay = ttnn::multiply(qh, exp_gc);                             // [BH, NC, C, K]
    const auto gc_last = ttnn::slice(
        gc_col,
        ttsl::SmallVector<uint32_t>{0U, 0U, C - 1U, 0U},
        ttsl::SmallVector<uint32_t>{BH, NC, C, 1U},
        ttsl::SmallVector<uint32_t>{1U, 1U, 1U, 1U});  // [BH, NC, 1, 1]
    const auto decay_last = ttnn::exp(gc_last);
    // (k * exp(gc_last - gc))^T, [BH, NC, K, C]
    const auto k_dec_t = ttnn::transpose(ttnn::multiply(kh, ttnn::exp(ttnn::subtract(gc_last, gc_col))), -2, -1);

    // The only sequential part: the state S [BH, 1, K, V] across the chunks.
    Tensor state = initial_state.has_value() ? ttnn::reshape(*initial_state, ttnn::Shape{BH, 1U, K, V})
                                             : core::zeros(ttnn::Shape{BH, 1U, K, V}, device, kF32);
    std::vector<Tensor> h_parts, v_new_parts, out_parts;
    for (uint32_t i = 0; i < NC; ++i) {
        if (save_for_backward) {
            h_parts.push_back(state);
        }
        const auto v_new = ttnn::subtract(chunk_at(u, i), mm(chunk_at(w, i), state));  // [BH, 1, C, V]
        out_parts.push_back(ttnn::add(mm(chunk_at(q_decay, i), state), mm(chunk_at(intra, i), v_new)));
        state = ttnn::add(ttnn::multiply(state, chunk_at(decay_last, i)), mm(chunk_at(k_dec_t, i), v_new));
        if (save_for_backward) {
            v_new_parts.push_back(v_new);
        }
    }

    auto out = concat_or_single(out_parts, /*dim=*/2);  // [BH, 1, T, V]
    if (!output_head_major) {
        // [BH, 1, T, V] -> [B, 1, T, HV*V]
        auto o = ttnn::reshape(ttnn::to_layout(out, ttnn::Layout::ROW_MAJOR), ttnn::Shape{B, HV, T, V});
        o = ttnn::permute(o, ttsl::SmallVector<int64_t>{0, 2, 1, 3});  // [B, T, HV, V]
        out = ttnn::to_layout(ttnn::reshape(o, ttnn::Shape{B, 1U, T, HV * V}), ttnn::Layout::TILE);
    }
    if (out.dtype() != output_dtype) {
        out = ttnn::typecast(out, output_dtype);
    }

    ChunkGatedDeltaRuleCompositeResult result{
        .output = std::move(out),
        .final_state = ttnn::reshape(state, ttnn::Shape{B, HV, K, V}),
        .h = std::nullopt,
        .v_new = std::nullopt,
        .a = std::nullopt,
    };
    if (save_for_backward) {
        result.h = concat_or_single(h_parts, /*dim=*/1);             // [BH, NC, K, V]
        result.v_new = concat_or_single(v_new_parts, /*dim=*/2);     // [BH, 1, T, V]
        result.a = ttnn::reshape(t_inv, ttnn::Shape{BH, 1U, T, C});  // [BH, 1, T, C]
    }
    return result;
}

}  // namespace ttml::ops
