// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fast multithreaded host packing of row-major bfloat16 into Bfp8_b / Bfp4_b tiles, byte-identical to the generic
// encode_tensor_data + pack_as_bfp*_tiles path for the standard 32x32 tile (16x16 faces) with zero padding.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::bfp_fast_pack {

// One packing job: `batch` matrices, each logically rows x cols (padded to multiples of 32 with zeros). With transpose
// the input matrices are stored cols x rows (e.g. an HF [out, in] weight packed as [in, out]).
struct Geometry {
    size_t batch = 1;
    size_t rows = 0;
    size_t cols = 0;
    bool transpose = false;
    // Source strides in elements (non-transposed input only); 0 = dense. Lets a strided view (e.g. one device's
    // column shard of a larger row-major tensor) be packed without first copying it.
    size_t row_stride = 0;
    size_t batch_stride = 0;
};

// Number of uint32 words the packed Bfp8_b (bits=8) or Bfp4_b (bits=4) tiles occupy.
size_t packed_words(const Geometry& g, int bits);

// Row-major bf16 bits -> Bfp8_b/Bfp4_b tiles in tile order, written to `dst` (packed_words words). nthreads == 0 picks
// min(32, hardware threads), the measured sweet spot on a 64-core EPYC.
void pack_bf16(const Geometry& g, const uint16_t* src, int bits, uint32_t* dst, unsigned nthreads = 0);

// Same, into a freshly allocated huge-page backed buffer that the packing threads touch first (no zero-fill pass).
HostBuffer pack_bf16_to_host_buffer(const Geometry& g, const uint16_t* src, int bits);

// 8 or 4 when a host tensor with `spec` can be produced by pack_bf16 from row-major bf16 with zero padding: Bfp8_b /
// Bfp4_b, TILE, standard 32x32 tile with 16x16 faces, and the plain physical layout (each matrix padded to multiples
// of 32, nothing else). nullopt otherwise, or when TT_METAL_DISABLE_FAST_BFP_PACK=1.
std::optional<int> fast_bfp_bits(const TensorSpec& spec);

}  // namespace tt::tt_metal::bfp_fast_pack
