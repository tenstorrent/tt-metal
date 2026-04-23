// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_matmul.hpp"

#include <atomic>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/narrow/narrow.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::metal {

namespace {

constexpr uint32_t kTileH = 32U;
constexpr uint32_t kTileHW = 32U * 32U;  // elements per tile-page

std::atomic<std::uint64_t> g_fast_path_calls{0U};
std::atomic<std::uint64_t> g_slow_path_calls{0U};

// Fast path is legal iff every tile-aligned offset satisfies the narrow
// bank-alignment check. Since row_lo is always a multiple of TILE_H, the
// worst case is row_lo == TILE_H; if TILE_H * H ≡ 0 (mod num_banks * TILE_HW)
// then every larger tile-aligned offset also satisfies it.
bool can_use_fast_path(uint32_t H, uint32_t K, uint32_t N, uint32_t num_banks) {
    const uint64_t P = static_cast<uint64_t>(num_banks) * kTileHW;
    const bool x_ok = (static_cast<uint64_t>(kTileH) * H) % P == 0U;
    const bool out_ok = (static_cast<uint64_t>(kTileH) * N) % P == 0U;
    const bool w_ok = (static_cast<uint64_t>(K) * N) % P == 0U;
    return x_ok && out_ok && w_ok;
}

ttnn::Tensor run_fast_path(
    const ttnn::Tensor& X,
    const ttnn::Tensor& W,
    const std::vector<uint32_t>& offsets_host,
    uint32_t E,
    uint32_t T_cap,
    uint32_t K,
    uint32_t N) {
    auto* device = X.device();
    const auto dram_mem_cfg = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};
    auto out =
        ttnn::zeros(ttnn::Shape({1U, 1U, T_cap, N}), X.dtype(), ttnn::Layout::TILE, std::ref(*device), dram_mem_cfg);

    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t row_lo = offsets_host[e];
        const uint32_t row_hi = offsets_host[e + 1U];
        if (row_hi == row_lo) {
            continue;
        }
        const uint32_t len = row_hi - row_lo;

        auto X_e = ttnn::narrow(X, /*dim=*/-2, /*start=*/static_cast<int32_t>(row_lo), /*length=*/len);
        auto W_e_3d = ttnn::narrow(W, /*dim=*/0, /*start=*/static_cast<int32_t>(e), /*length=*/1U);
        auto W_e = ttnn::view(W_e_3d, ttnn::Shape({1U, 1U, K, N}));
        auto out_e = ttnn::narrow(out, /*dim=*/-2, /*start=*/static_cast<int32_t>(row_lo), /*length=*/len);

        ttnn_fixed::matmul(
            X_e,
            W_e,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*output_tensor=*/std::optional<ttnn::Tensor>(out_e));
    }
    return out;
}

ttnn::Tensor run_slow_path(
    const ttnn::Tensor& X,
    const ttnn::Tensor& W,
    const std::vector<uint32_t>& offsets_host,
    uint32_t E,
    uint32_t K,
    uint32_t N) {
    std::vector<ttnn::Tensor> parts;
    parts.reserve(E);

    const ttsl::SmallVector<uint32_t> w_step = {1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> x_step = {1U, 1U, 1U, 1U};

    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t row_lo = offsets_host[e];
        const uint32_t row_hi = offsets_host[e + 1U];
        if (row_hi < row_lo) {
            throw std::runtime_error("sparse_matmul: offsets are not monotonic.");
        }
        if (row_hi == row_lo) {
            continue;
        }

        const ttsl::SmallVector<uint32_t> x_start = {0U, 0U, row_lo, 0U};
        const ttsl::SmallVector<uint32_t> x_end = {1U, 1U, row_hi, K};
        auto X_e = ttnn::slice(X, x_start, x_end, x_step);

        const ttsl::SmallVector<uint32_t> w_start = {e, 0U, 0U};
        const ttsl::SmallVector<uint32_t> w_end = {e + 1U, K, N};
        auto W_e_3d = ttnn::slice(W, w_start, w_end, w_step);
        auto W_e = W_e_3d.reshape(ttnn::Shape({1U, 1U, K, N}));

        parts.push_back(ttnn_fixed::matmul(X_e, W_e, /*transpose_a=*/false, /*transpose_b=*/false));
    }

    if (parts.empty()) {
        throw std::runtime_error(
            "sparse_matmul: all experts empty (T_cap == 0); caller must avoid this degenerate case.");
    }
    if (parts.size() == 1U) {
        return parts.front();
    }
    return ttnn::concat(parts, /*dim=*/2);
}

}  // namespace

SparseMatmulCounters get_sparse_matmul_counters() {
    return {g_fast_path_calls.load(std::memory_order_relaxed), g_slow_path_calls.load(std::memory_order_relaxed)};
}

void reset_sparse_matmul_counters() {
    g_fast_path_calls.store(0U, std::memory_order_relaxed);
    g_slow_path_calls.store(0U, std::memory_order_relaxed);
}

ttnn::Tensor sparse_matmul(const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& offsets) {
    const auto x_shape = X.logical_shape();
    const auto w_shape = W.logical_shape();

    const uint32_t T_cap = x_shape[-2];
    const uint32_t K = x_shape[-1];
    const uint32_t E = w_shape[0];
    const uint32_t N = w_shape[-1];

    if (w_shape[-2] != K) {
        throw std::runtime_error("sparse_matmul: W inner dim must equal X's K.");
    }

    const auto offsets_host = offsets.to_vector<uint32_t>();
    if (offsets_host.size() != E + 1U) {
        throw std::runtime_error("sparse_matmul: offsets size must be E_local + 1.");
    }
    if (offsets_host.back() != T_cap) {
        throw std::runtime_error("sparse_matmul: offsets[-1] must equal T_cap.");
    }

    auto* device = X.device();
    const uint32_t num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint32_t H = K;  // X's last dim; by alignment, constraint is on kTileH*H.

    if (can_use_fast_path(H, K, N, num_banks)) {
        g_fast_path_calls.fetch_add(1U, std::memory_order_relaxed);
        return run_fast_path(X, W, offsets_host, E, T_cap, K, N);
    }

    g_slow_path_calls.fetch_add(1U, std::memory_order_relaxed);
    return run_slow_path(X, W, offsets_host, E, K, N);
}

}  // namespace ttml::metal
