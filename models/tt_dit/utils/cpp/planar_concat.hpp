// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace tt_dit_planar {

// Per-shard view of an input shard.  `data` is the raw uint8 byte pointer;
// `h_per`, `w_per`, `T` are the per-shard logical dimensions.  Mesh
// coordinates `(r, c)` index the shard into the destination plane.
struct ShardView {
    const uint8_t* data;
    int r;
    int c;
};

enum class DimOrder {
    CHWT,  // shard memory layout: (h_per, w_per, T) — T innermost
    CTHW,  // shard memory layout: (T, h_per, w_per) — W innermost
};

// Scatter all shards of one component into one plane region of `out`.
//
// `plane_offset` is the byte offset into each output frame's row at which
// this component's plane begins (Y: 0; Cb: H*W; Cr: H*W + (H/2)*(W/2)).
// `plane_W` is the full destination width of this plane (W for Y, W/2 for UV).
// `row_stride` is the bytes per output frame row (= H*W + 2*(H/2)*(W/2)).
// `T` is the temporal extent (= number of output rows).
// `h_per`, `w_per` are per-shard heights and widths of THIS component.
//
// Tasks are dispatched across the static thread pool.  Returns when all
// scatters of this component are complete.
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

// Top-level entry: schedule Y/Cb/Cr scatters in one batch for maximum
// thread-pool parallelism (96 tasks for a 4×8 mesh).  Blocks until all
// tasks finish.
// ``out_H``/``out_W`` are the logical output dims. When the VAE pads a global
// right/bottom tail (``out_H < H`` and/or ``out_W < W``), each shard's write
// extent is clamped to the logical bound while its source is still addressed
// with the full padded per-shard dims — so the padded tail is never written
// and ``out`` is sized for the logical (cropped) frame, no separate trim pass.
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

// Optional knob — set the size of the static thread pool.  No-op after the
// pool has been created (first call to scatter_component / planar_concat).
void set_thread_pool_size(int n_threads);

}  // namespace tt_dit_planar
