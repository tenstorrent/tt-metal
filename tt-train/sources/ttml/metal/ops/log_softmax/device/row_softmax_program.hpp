// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

// Device program shared by the fused log_softmax and the fused cross-entropy. Both reduce each row of
// the last axis with the collapse-resistant statistics of arXiv:2610.31642, Section 2: one online pass
// keeps per-lane maximum, first-maximum index and the exponential tail without that maximum, combined
// into the row maximum m, first maximum k and log1p(tail) l in FP32.
//
//   LogSoftmaxForward:    y = (x - m) - l, plus the row state (m, k, l).
//   LogSoftmaxBackward:   dx from upstream dy and the row state; dx[k] rebuilt from the others (Eq. 7).
//   CrossEntropyForward:  loss = -((x[t] - m) - l), one value per row.
//   CrossEntropyBackward: dx = s * softmax(x), with dx[t] = s * expm1((x[t] - m) - l), where s is the scaler
//                         times the row's upstream gradient when one is given.
//
// Vocabulary-parallel cross-entropy (the vocabulary sharded across TP devices, shard r owning global columns
// [first_v, first_v + width) with first_v = r * width):
//   VocabParallelStats:    per shard and row, one FP32 tile row with columns (m, first_v + k, tail without
//                          x[k], x[t] or 0 when the target is on another shard): no log1p, no output tensor.
//   VocabParallelCombine:  input = those tiles all-gathered along the last axis ([.., tile width * shards]); combines
//                          the shards as lanes (one global first maximum, lowest shard on ties), l = log1p(T),
//                          loss = -((x[t] - M) - l) per row and the row state (M, l) as two column-vector tiles.
//   VocabParallelBackward: dx = s * exp(x - M - l) from that state, no statistics pass; the target's shard
//                          substitutes dx[t] = s * expm1((x[t] - M) - l) (Eq. 7 in closed form when t = k).
namespace ttml::metal::ops::row_softmax {

enum class Kind : uint8_t {
    LogSoftmaxForward,
    LogSoftmaxBackward,
    CrossEntropyForward,
    CrossEntropyBackward,
    VocabParallelStats,
    VocabParallelCombine,
    VocabParallelBackward,
};

// Whether the kernels support the tensor's tile. Every kernel takes the tile geometry from the tensor spec, but the
// SFPU passes (column broadcasts, row reductions, per-lane statistics) walk a DST slot as one full tile of four
// faces, so only the default tile qualifies.
bool supports_tile(const tt::tt_metal::Tile& tile);

// Whether the cross-entropy backward kernels can fold `grad`, the loss's upstream gradient, into their per-row scale:
// one value ([1, 1, 1, 1]) or one value per row of `input`, as a TILE tensor in interleaved DRAM on the same device
// and in the same tiles. Otherwise the caller multiplies the unscaled gradient by it.
bool can_fuse_upstream(const ttnn::Tensor& input, const ttnn::Tensor& grad);

// Work split. A unit is one tile row; with few rows, each row's columns are split over `chunks` cores that
// exchange partial statistics, one per lane (tile column).
struct Layout {
    uint32_t width_tiles;
    uint32_t rows;    // Tile rows.
    uint32_t chunks;  // Cores sharing one row.
    uint32_t chunk_tiles;
};
Layout layout_for(const ttnn::Tensor& input, Kind kind = Kind::LogSoftmaxForward);
// Whether the program exchanges partial statistics between the cores sharing a row (and needs a scratch).
bool needs_exchange(const Layout& layout, Kind kind);

struct Options {
    Kind kind;
    ttnn::DataType upstream_dtype = ttnn::DataType::BFLOAT16;  // Upstream gradient dtype, when there is one.
    float scaler = 1.F;                                        // Cross-entropy backward only.
    // Cross-entropy backward: upstream gradient folded into the row scale s. 0: none, 1: one value
    // ([1, 1, 1, 1]), 2: one value per row ([N, C, H, 1], one tile per tile row of the input).
    uint32_t upstream_kind = 0;
    uint32_t target_page_bytes = 0;    // Cross-entropy: bytes in one target page.
    uint32_t target_tiled_height = 0;  // Cross-entropy: tile rows per target page.
    uint32_t first_v = 0;  // Vocabulary-parallel stats/backward: global column of this shard's first column.
};

// Buffers the program reads and writes. Null entries are unused by the kind; a null scratch aliases
// the input (never read without a split).
struct Buffers {
    tt::tt_metal::Buffer* input = nullptr;
    tt::tt_metal::Buffer* upstream = nullptr;
    tt::tt_metal::Buffer* state = nullptr;
    tt::tt_metal::Buffer* target = nullptr;
    tt::tt_metal::Buffer* output = nullptr;
    tt::tt_metal::Buffer* scratch = nullptr;
};

// Program for one device. Buffer addresses are Buffer* runtime args: a program-cache hit re-patches them, and
// every other runtime arg follows from the program hash.
tt::tt_metal::ProgramDescriptor make_descriptor(
    const Options& options, const ttnn::Tensor& input, const Buffers& buffers);

// The DRAM scratch through which the cores sharing a row exchange partial statistics, one page per (row, chunk)
// unit, owned by the cached workload. nullopt when the kind does not exchange (see needs_exchange).
std::optional<tt::tt_metal::WorkloadBuffer> make_scratch(const ttnn::Tensor& input, Kind kind);

// A workload that runs `descriptor` on every range of `tensor_coords` and owns `scratch`, if any.
tt::tt_metal::WorkloadDescriptor replicated_workload(
    tt::tt_metal::ProgramDescriptor descriptor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    std::optional<tt::tt_metal::WorkloadBuffer> scratch);

}  // namespace ttml::metal::ops::row_softmax
