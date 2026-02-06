// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "weight_utils.hpp"

#include <fmt/format.h>

#include <cstring>
#include <random>
#include <stdexcept>

namespace ttml::serialization {

std::vector<float> pad_and_resize_flat(
    const std::vector<float>& flat, int64_t rows, int64_t cols, int64_t target_rows, int64_t target_cols) {
    // If dimensions match, return as is
    if (rows == target_rows && cols == target_cols) {
        return flat;
    }

    // Create output tensor with target dimensions
    std::vector<float> out(static_cast<size_t>(target_rows * target_cols), 0.0f);

    // Copy data from source to target, handling both row and column differences
    int64_t copy_rows = std::min(rows, target_rows);
    int64_t copy_cols = std::min(cols, target_cols);

    for (int64_t r = 0; r < copy_rows; ++r) {
        for (int64_t c = 0; c < copy_cols; ++c) {
            out[r * target_cols + c] = flat[r * cols + c];
        }
    }

    // Initialize random number generator once if we need to fill additional space
    bool need_random_fill = (target_rows > rows) || (target_cols > cols);
    std::mt19937 gen;
    std::normal_distribution<float> dist(0.0f, 0.02f);  // Small random values

    if (need_random_fill) {
        std::random_device rd;
        gen.seed(rd());
    }

    // For additional rows (if target_rows > rows), use small random initialization
    // instead of zeros to avoid dead neurons
    if (target_rows > rows) {
        for (int64_t r = copy_rows; r < target_rows; ++r) {
            for (int64_t c = 0; c < target_cols; ++c) {
                out[r * target_cols + c] = dist(gen);
            }
        }
    }

    // For additional columns (if target_cols > cols), use small random initialization
    if (target_cols > cols) {
        for (int64_t r = 0; r < copy_rows; ++r) {
            for (int64_t c = copy_cols; c < target_cols; ++c) {
                out[r * target_cols + c] = dist(gen);
            }
        }
    }

    return out;
}

std::vector<float> unpermute_proj_rows(const std::vector<float>& w, int64_t rows, int64_t cols, int64_t n_heads) {
    // Reorder rows within each head: [0..D/2-1, D/2..D-1] → interleave → [0, D/2, 1, D/2+1, ...]
    if (rows % n_heads != 0) {
        throw std::runtime_error(
            fmt::format("unpermute_proj_rows: rows {} not divisible by n_heads {}", rows, n_heads));
    }
    const int64_t D = rows / n_heads;  // rows per head
    if (D % 2 != 0) {
        throw std::runtime_error(fmt::format("unpermute_proj_rows: rows per head {} must be even", D));
    }

    std::vector<float> out(w.size());
    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t head_row0 = h * D;
        const int64_t half = D / 2;
        for (int64_t i = 0; i < half; ++i) {
            const int64_t src_even = head_row0 + i;
            const int64_t src_odd = head_row0 + half + i;
            const int64_t dst_even = head_row0 + (2 * i);
            const int64_t dst_odd = head_row0 + (2 * i + 1);

            std::memcpy(&out[dst_even * cols], &w[src_even * cols], sizeof(float) * cols);
            std::memcpy(&out[dst_odd * cols], &w[src_odd * cols], sizeof(float) * cols);
        }
    }
    return out;
}

std::vector<float> unpermute_norm_weights(const std::vector<float>& w) {
    // For RMSNorm weights: reshape to (2, head_dim/2), transpose, then flatten
    // This converts from non-meta format ([x1,x2,...,y1,y2...]) to the meta-style
    // format expected by TTML ([x1,y1,x2,y2,...])
    const int64_t total_size = w.size();
    const int64_t head_dim = total_size;

    if (head_dim % 2 != 0) {
        throw std::runtime_error(fmt::format("unpermute_norm_weights: head_dim {} must be even", head_dim));
    }

    const int64_t half = head_dim / 2;
    std::vector<float> out(total_size);

    // Reshape to (2, half), transpose to (half, 2), then flatten
    // Original layout: w[i*half + j] where i in [0,1], j in [0,half)
    // After transpose: out[j*2 + i] where j in [0,half), i in [0,1]
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < half; ++j) {
            out[j * 2 + i] = w[i * half + j];
        }
    }

    return out;
}

std::vector<float> transpose_flat(const std::vector<float>& x, int64_t r, int64_t c) {
    std::vector<float> y(x.size());
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < c; ++j) {
            y[(size_t)j * r + i] = x[(size_t)i * c + j];
        }
    }
    return y;
}

std::vector<float> strict_copy_linear(
    const std::vector<float>& src,
    int64_t src_r,
    int64_t src_c,
    int64_t tgt_r,
    int64_t tgt_c,
    const std::string& dbg,
    bool verbose) {
    if (src_r == tgt_r && src_c == tgt_c) {
        return src;
    }
    if (src_c == tgt_r && src_r == tgt_c) {
        if (verbose) {
            fmt::print("[{}] transposing weights\n", dbg);
        }
        return transpose_flat(src, src_r, src_c);
    }
    throw std::runtime_error(fmt::format(
        "[{}] shape mismatch: src=({}x{}), src^T=({}x{}), tgt=({}x{})", dbg, src_r, src_c, src_c, src_r, tgt_r, tgt_c));
}

}  // namespace ttml::serialization
