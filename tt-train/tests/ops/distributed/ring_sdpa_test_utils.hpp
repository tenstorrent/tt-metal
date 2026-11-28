// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/distributed/cp_sharding.hpp"
#include "core/tt_tensor_utils.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"

// Host SDPA references shared by the ring SDPA tests. Q is (B, H, S_q, D); K and V are (B, G, S_k, D) with
// H a multiple of G, so query head h reads K/V head h / (H / G).

namespace ttml::test_utils::ring_sdpa {

struct SdpaForward {
    xt::xarray<float> output;
    xt::xarray<float> lse;            // (B, H, S_q)
    xt::xarray<float> probabilities;  // (B, H, S_q, S_k)
};

struct SdpaGrads {
    xt::xarray<float> dq;
    xt::xarray<float> dk;
    xt::xarray<float> dv;
};

inline SdpaForward sdpa_forward(
    const xt::xarray<float>& q, const xt::xarray<float>& k, const xt::xarray<float>& v, bool causal = false) {
    const size_t B = q.shape()[0], H = q.shape()[1], S_q = q.shape()[2], D = q.shape()[3];
    const size_t S_k = k.shape()[2];
    const size_t group = H / k.shape()[1];
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    SdpaForward ref{
        xt::zeros<float>({B, H, S_q, D}), xt::zeros<float>({B, H, S_q}), xt::zeros<float>({B, H, S_q, S_k})};
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / group;
            for (size_t i = 0; i < S_q; ++i) {
                const size_t visible = causal ? i + 1 : S_k;
                std::vector<double> scores(visible);
                double max_score = -std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < visible; ++j) {
                    double dot = 0.0;
                    for (size_t d = 0; d < D; ++d) {
                        dot += static_cast<double>(q(b, h, i, d)) * k(b, g, j, d);
                    }
                    scores[j] = dot * scale;
                    max_score = std::max(max_score, scores[j]);
                }
                double sum = 0.0;
                for (size_t j = 0; j < visible; ++j) {
                    sum += std::exp(scores[j] - max_score);
                }
                ref.lse(b, h, i) = static_cast<float>(max_score + std::log(sum));
                for (size_t j = 0; j < visible; ++j) {
                    const double p = std::exp(scores[j] - max_score) / sum;
                    ref.probabilities(b, h, i, j) = static_cast<float>(p);
                    for (size_t d = 0; d < D; ++d) {
                        ref.output(b, h, i, d) += static_cast<float>(p * v(b, g, j, d));
                    }
                }
            }
        }
    }
    return ref;
}

// `o` is the forward output the backward is given; the softmax backward uses u = rowsum(dO * o) from it.
inline SdpaGrads sdpa_backward(
    const xt::xarray<float>& q,
    const xt::xarray<float>& k,
    const xt::xarray<float>& v,
    const xt::xarray<float>& o,
    const xt::xarray<float>& probabilities,
    const xt::xarray<float>& d_o) {
    const size_t B = q.shape()[0], H = q.shape()[1], S_q = q.shape()[2], D = q.shape()[3];
    const size_t S_k = k.shape()[2];
    const size_t group = H / k.shape()[1];
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    SdpaGrads grads{xt::zeros<float>(q.shape()), xt::zeros<float>(k.shape()), xt::zeros<float>(v.shape())};
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            const size_t g = h / group;
            for (size_t i = 0; i < S_q; ++i) {
                double u = 0.0;
                for (size_t d = 0; d < D; ++d) {
                    u += static_cast<double>(d_o(b, h, i, d)) * o(b, h, i, d);
                }
                for (size_t j = 0; j < S_k; ++j) {
                    const double p = probabilities(b, h, i, j);
                    double dp = 0.0;
                    for (size_t d = 0; d < D; ++d) {
                        dp += static_cast<double>(d_o(b, h, i, d)) * v(b, g, j, d);
                        grads.dv(b, g, j, d) += static_cast<float>(p * d_o(b, h, i, d));
                    }
                    const double ds = p * (dp - u) * scale;
                    for (size_t d = 0; d < D; ++d) {
                        grads.dq(b, h, i, d) += static_cast<float>(ds * k(b, g, j, d));
                        grads.dk(b, g, j, d) += static_cast<float>(ds * q(b, h, i, d));
                    }
                }
            }
        }
    }
    return grads;
}

// Moves the sequence axis of a (B, H, S, D) tensor between sample order and the order the CP devices hold it in.
inline xt::xarray<float> permute_cp_order(
    const xt::xarray<float>& x, core::distributed::CpSharding sharding, uint32_t cp_size, bool to_device) {
    const size_t B = x.shape()[0], H = x.shape()[1], S = x.shape()[2], D = x.shape()[3];
    xt::xarray<float> out = xt::empty<float>(x.shape());
    for (size_t bh = 0; bh < B * H; ++bh) {
        for (size_t pos = 0; pos < S; ++pos) {
            const size_t global = core::distributed::cp_global_position(sharding, pos, S, cp_size);
            const size_t from = to_device ? global : pos;
            const size_t to = to_device ? pos : global;
            std::copy_n(x.data() + (bh * S + from) * D, D, out.data() + (bh * S + to) * D);
        }
    }
    return out;
}

// CP on the first non-trivial axis of the open mesh, and TP on the second when there are two.
inline void initialize_cp_parallelism_context() {
    const auto shape = autograd::ctx().get_device().shape();
    uint32_t non_trivial_axes = 0U;
    for (uint32_t i = 0; i < shape.dims(); ++i) {
        non_trivial_axes += shape[i] > 1U ? 1U : 0U;
    }
    autograd::ctx().initialize_parallelism_context(
        {.enable_ddp = false, .enable_tp = non_trivial_axes > 1U, .cp = autograd::ContextParallelConfig{}});
}

// Joins the per-device CP shards of a (B, H, S_local, D) tensor and returns the sequence in sample order.
inline xt::xarray<float> gather_cp_sequence(
    const ttnn::Tensor& tensor, uint32_t cp_axis, core::distributed::CpSharding sharding) {
    const auto shards = core::to_xtensor<float>(tensor, core::IdentityComposer{});
    const auto mesh_shape = tensor.device()->shape();
    const uint32_t ring_size = mesh_shape[cp_axis];
    uint32_t stride = 1U;
    for (uint32_t i = mesh_shape.dims(); i-- > cp_axis + 1U;) {
        stride *= mesh_shape[i];
    }
    const auto& first = shards[0];
    const size_t B = first.shape()[0], H = first.shape()[1], S_local = first.shape()[2], D = first.shape()[3];
    xt::xarray<float> out = xt::empty<float>({B, H, S_local * ring_size, D});
    for (uint32_t r = 0; r < ring_size; ++r) {
        const auto& shard = shards[r * stride];
        for (size_t bh = 0; bh < B * H; ++bh) {
            std::copy_n(
                shard.data() + bh * S_local * D,
                S_local * D,
                out.data() + (bh * S_local * ring_size + r * S_local) * D);
        }
    }
    return permute_cp_order(out, sharding, ring_size, /*to_device=*/false);
}

}  // namespace ttml::test_utils::ring_sdpa
