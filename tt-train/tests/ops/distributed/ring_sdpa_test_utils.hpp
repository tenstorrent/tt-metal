// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

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

}  // namespace ttml::test_utils::ring_sdpa
