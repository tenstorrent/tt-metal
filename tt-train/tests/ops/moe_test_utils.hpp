// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "xtensor/containers/xadapt.hpp"

namespace ttml::test_utils::moe {

// How the dispatched [D,B,S,H] tensor is filled.
enum class DispatchedPattern {
    UniformRandom,      // uniform in [-1, 1] from the shared random_data helper
    RowIndexBroadcast,  // row (d,b,s) := float(d*B*S + b*S + s), broadcast across H.
                        // Round-trips exactly through bf16 for small grids (<=2^7 rows)
                        // and makes reordering tests trivially debuggable (out row N == N).
};

struct MoeHostInputConfig {
    uint32_t D;
    uint32_t B;
    uint32_t S;
    uint32_t H;
    uint32_t E;  // total experts (metadata range)
    uint32_t K;
    DispatchedPattern dispatched_pattern = DispatchedPattern::UniformRandom;
    // Non-null → dispatched and scores are bf16-roundtripped on host so the
    // reference path sees the same values as the device. Null for pure-profile
    // tests that don't compare against a reference.
    ttnn::distributed::MeshDevice* roundtrip_device = nullptr;
    uint32_t seed = 42;
};

struct MoeHostInputs {
    xt::xarray<float> dispatched;   // [D,B,S,H]
    xt::xarray<uint32_t> metadata;  // [D,B,S,K]
    xt::xarray<float> scores;       // [D,B,S,K]
};

struct MoeDeviceInputs {
    ttnn::Tensor dispatched_bf16;
    ttnn::Tensor metadata_u16;
    ttnn::Tensor scores_bf16;
    ttnn::Tensor leids_u16;
};

inline xt::xarray<float> bf16_roundtrip(const xt::xarray<float>& x, ttnn::distributed::MeshDevice* device) {
    auto t = ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(x, device, ttnn::Layout::ROW_MAJOR);
    return ttml::core::to_xtensor(t);
}

inline xt::xarray<float> make_dispatched_row_index_broadcast(uint32_t D, uint32_t B, uint32_t S, uint32_t H) {
    xt::xarray<float> out = xt::zeros<float>({D, B, S, H});
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                const float v = static_cast<float>(d * B * S + b * S + s);
                for (uint32_t h = 0; h < H; ++h) out(d, b, s, h) = v;
            }
        }
    }
    return out;
}

// Top-K routing metadata: per (d,b,s) row, K distinct expert ids drawn from [0, E)
// via shuffle. Matches the contract moe_group expects.
inline xt::xarray<uint32_t> make_metadata_shuffle(
    uint32_t D, uint32_t B, uint32_t S, uint32_t K, uint32_t E, uint32_t seed) {
    xt::xarray<uint32_t> out = xt::zeros<uint32_t>({D, B, S, K});
    std::mt19937 rng(seed);
    std::vector<uint32_t> all(E);
    for (uint32_t e = 0; e < E; ++e) all[e] = e;
    for (uint32_t d = 0; d < D; ++d) {
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < S; ++s) {
                std::shuffle(all.begin(), all.end(), rng);
                for (uint32_t ki = 0; ki < K; ++ki) out(d, b, s, ki) = all[ki];
            }
        }
    }
    return out;
}

inline MoeHostInputs make_moe_host_inputs(const MoeHostInputConfig& cfg) {
    MoeHostInputs in;
    switch (cfg.dispatched_pattern) {
        case DispatchedPattern::UniformRandom:
            in.dispatched = ttml::test_utils::make_uniform_xarray<float>(
                std::array<std::size_t, 4>{cfg.D, cfg.B, cfg.S, cfg.H}, -1.0F, 1.0F, cfg.seed + 5);
            break;
        case DispatchedPattern::RowIndexBroadcast:
            in.dispatched = make_dispatched_row_index_broadcast(cfg.D, cfg.B, cfg.S, cfg.H);
            break;
    }
    in.scores = ttml::test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 4>{cfg.D, cfg.B, cfg.S, cfg.K}, 0.0F, 0.5F, cfg.seed + 13);
    in.metadata = make_metadata_shuffle(cfg.D, cfg.B, cfg.S, cfg.K, cfg.E, cfg.seed + 1);

    if (cfg.roundtrip_device != nullptr) {
        in.dispatched = bf16_roundtrip(in.dispatched, cfg.roundtrip_device);
        in.scores = bf16_roundtrip(in.scores, cfg.roundtrip_device);
    }
    return in;
}

inline MoeDeviceInputs to_device_inputs(
    const MoeHostInputs& host, const std::vector<uint16_t>& leids, ttnn::distributed::MeshDevice* device) {
    xt::xarray<uint16_t> md16 = xt::cast<uint16_t>(host.metadata);
    xt::xarray<uint16_t> leids_xt = xt::adapt(leids, std::vector<size_t>{leids.size()});
    return MoeDeviceInputs{
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(host.dispatched, device, ttnn::Layout::ROW_MAJOR),
        ttml::core::from_xtensor<uint16_t, ttnn::DataType::UINT16>(md16, device, ttnn::Layout::ROW_MAJOR),
        ttml::core::from_xtensor<float, ttnn::DataType::BFLOAT16>(host.scores, device, ttnn::Layout::ROW_MAJOR),
        ttml::core::from_xtensor<uint16_t, ttnn::DataType::UINT16>(leids_xt, device, ttnn::Layout::ROW_MAJOR),
    };
}

}  // namespace ttml::test_utils::moe
