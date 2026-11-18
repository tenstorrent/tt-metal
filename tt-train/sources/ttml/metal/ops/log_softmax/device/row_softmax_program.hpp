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

// Device program of the fused log_softmax. It reduces each row of the last axis with the
// collapse-resistant statistics of arXiv:2610.31642, Section 2: one online pass
// keeps per-lane maximum, first-maximum index and the exponential tail without that maximum, combined
// into the row maximum m, first maximum k and log1p(tail) l in FP32.
//
//   LogSoftmaxForward:    y = (x - m) - l, plus the row state (m, k, l).
//   LogSoftmaxBackward:   dx from upstream dy and the row state; dx[k] rebuilt from the others (Eq. 7).
namespace ttml::metal::ops::row_softmax {

enum class Kind : uint8_t { LogSoftmaxForward, LogSoftmaxBackward };

// Whether the kernels support the tensor's tile. Every kernel takes the tile geometry from the tensor spec, but the
// SFPU passes (column broadcasts, row reductions, per-lane statistics) walk a DST slot as one full tile of four
// faces, so only the default tile qualifies.
bool supports_tile(const tt::tt_metal::Tile& tile);

// Work split. A unit is one tile row; with few rows, each row's columns are split over `chunks` cores that
// exchange partial statistics, one per lane (tile column).
struct Layout {
    uint32_t width_tiles;
    uint32_t rows;    // Tile rows.
    uint32_t chunks;  // Cores sharing one row.
    uint32_t chunk_tiles;
};
Layout layout_for(const ttnn::Tensor& input);

struct Options {
    Kind kind;
    ttnn::DataType upstream_dtype = ttnn::DataType::BFLOAT16;  // Upstream gradient dtype, when there is one.
};

// Buffers the program reads and writes. Null entries are unused by the kind; a null scratch aliases
// the input (never read without a split).
struct Buffers {
    tt::tt_metal::Buffer* input = nullptr;
    tt::tt_metal::Buffer* upstream = nullptr;
    tt::tt_metal::Buffer* state = nullptr;
    tt::tt_metal::Buffer* output = nullptr;
    tt::tt_metal::Buffer* scratch = nullptr;
};

// Program for one device. Buffer addresses are Buffer* runtime args: a program-cache hit re-patches them, and
// every other runtime arg follows from the program hash.
tt::tt_metal::ProgramDescriptor make_descriptor(
    const Options& options, const ttnn::Tensor& input, const Buffers& buffers);

// The DRAM scratch through which the cores sharing a row exchange partial statistics, one page per (row, chunk)
// unit, owned by the cached workload. nullopt when the row is not split.
std::optional<tt::tt_metal::WorkloadBuffer> make_scratch(const ttnn::Tensor& input);

// A workload that runs `descriptor` on every range of `tensor_coords` and owns `scratch`, if any.
tt::tt_metal::WorkloadDescriptor replicated_workload(
    tt::tt_metal::ProgramDescriptor descriptor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    std::optional<tt::tt_metal::WorkloadBuffer> scratch);

}  // namespace ttml::metal::ops::row_softmax
