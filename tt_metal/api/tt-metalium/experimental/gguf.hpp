// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

namespace tt::tt_metal::experimental {

// Experimental GGUF host API; subject to change.

// True for the expanded GGUF layouts (DataType::GGUF_*_EXPANDED).
bool is_gguf_expanded(DataType dtype);

// The GGUF format whose blocks from_gguf takes and to_gguf returns for a GGUF dtype: itself, or the format an
// expanded layout is made from.
DataType gguf_source_dtype(DataType dtype);

// One GGUF format. A page holds the record of 32 columns of N by 32 * group_tiles rows of K: their GGUF bytes, with
// the bits regrouped for the Blackhole decoder (scale fields, then or after the codes parts of page_tiles K tiles
// each). An expanded layout's page holds the same weights in more bytes (block_bytes); the other fields are those of
// the decoder's page it fills.
struct GgufTraits {
    uint32_t format_id;    // the decoder's format number (ckernel::gguf::Format)
    uint32_t block_elems;  // elements per GGUF block
    uint32_t block_bytes;  // bytes per GGUF block (expanded layouts: page bytes per block_elems rows of a column)
    uint32_t group_tiles;  // K tiles per page
    uint32_t page_tiles;   // K tiles per codes part
    uint32_t code_bytes;   // bytes per codes part
    uint32_t scale_rows;   // 64-byte rows of scale fields per page
    uint32_t fp16_rows;    // the last scale rows: fp16 fields, given fp32 copies in the decoder's L1 page
    uint32_t code_offset;  // bytes of grid levels before the codes in the decoder's L1 page (codebook formats)

    uint32_t page_height() const { return 32 * group_tiles; }
    uint32_t page_bytes() const { return 32 * page_height() * block_bytes / block_elems; }

    // Bytes of the decoder's L1 page of page_tiles K tiles (ckernel::gguf::page_layout).
    uint32_t l1_page_bytes() const;
};

const GgufTraits& gguf_traits(DataType dtype);

// Records (K groups of one N tile) per page of a GGUF tensor with K rows: Q4_0 and Q4_1 store up to 4 consecutive K
// groups of an N tile in one page (the largest of 4, 2, 1 that divides the groups), so that a page is one contiguous
// read of several records; the other formats one.
uint32_t gguf_records_per_page(DataType dtype, uint64_t k);

// A GGUF tensor of `spec` (a GGUF dtype, shape [K, N]) from its blocks as a GGUF file stores them: N rows of
// K / block_elems blocks. Only bits move; the pages take exactly the blocks' bytes.
HostTensor from_gguf(ttsl::Span<const std::byte> blocks, const TensorSpec& spec);

// The blocks of a GGUF tensor, the exact inverse of from_gguf.
std::vector<std::byte> to_gguf(const HostTensor& tensor);

}  // namespace tt::tt_metal::experimental
