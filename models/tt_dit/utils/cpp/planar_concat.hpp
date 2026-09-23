// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace tt_dit_planar {

// Per-shard view of an input shard
struct ShardView {
    const uint8_t* data;
    int r;
    int c;
};

enum class DimOrder {
    CHWT,  // shard memory layout: (h_per, w_per, T) — T innermost
    CTHW,  // shard memory layout: (T, h_per, w_per) — W innermost
};

// Scatter all shards of one component into one plane region of `out`
void scatter_component(
    const std::vector<ShardView>& shards,
    DimOrder dim_order,
    uint8_t* out,
    int T,
    int plane_offset,
    int plane_W,
    int row_stride,
    int h_per,
    int w_per);

// Top-level entry: schedule Y/Cb/Cr scatters in one batch for maximum thread-pool parallelism
void planar_concat(
    const std::vector<ShardView>& y_shards,
    int y_h_per,
    int y_w_per,
    const std::vector<ShardView>& cb_shards,
    int uv_h_per,
    int uv_w_per,
    const std::vector<ShardView>& cr_shards,
    DimOrder dim_order,
    int T,
    int H,
    int W,
    int out_H,
    int out_W,
    uint8_t* out);

// Optional knob — set the size of the static thread pool
void set_thread_pool_size(int n_threads);

}  // namespace tt_dit_planar
