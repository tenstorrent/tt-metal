// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/cache_tensor_layout_spec.hpp>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::internal::disaggregation {

namespace {

// The NdShardSpec shard_shape, aligned to the tensor's logical rank. The addresser reads the
// intra-device layout (seq granule, feature width) off this instead of bespoke spec fields.
const Shape& shard_shape_of(const TensorSpec& tensor) {
    const auto& nd = tensor.memory_config().nd_shard_spec();
    TT_FATAL(nd.has_value(), "KvLayoutSpec.tensor must be ND-sharded (its shard spec carries the layout)");
    return nd->shard_shape;
}

}  // namespace

std::optional<uint32_t> KvLayoutSpec::sequence_axis() const {
    if (!has_sequence()) {
        return std::nullopt;
    }
    const auto& shape = tensor.logical_shape();
    const Shape& shard = shard_shape_of(tensor);
    const uint32_t rank = static_cast<uint32_t>(shape.rank());
    // The sequence axis is the one the allocation tiles at the DRAM token block (kTile) while its full
    // extent is larger — i.e. chopped into many blocks. Feature axes are full-width; batch/head are 1.
    for (uint32_t i = 0; i < rank; ++i) {
        const uint32_t g = static_cast<uint32_t>(shard[i]);
        if (g == kTile && static_cast<uint32_t>(shape[i]) > kTile) {
            return i;
        }
    }
    return std::nullopt;
}

uint64_t feature_width(const TensorSpec& tensor) {
    const auto& shape = tensor.logical_shape();
    const Shape& shard = shard_shape_of(tensor);
    const uint32_t rank = static_cast<uint32_t>(shape.rank());
    // Feature axes are the ones whose shard granule spans the full logical extent (the seq axis is
    // tiled at the token block; batch/slot and head-shard axes have granule 1).
    uint64_t f = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (static_cast<uint32_t>(shard[i]) == static_cast<uint32_t>(shape[i])) {
            f *= static_cast<uint64_t>(shape[i]);
        }
    }
    return f;
}

uint32_t chunk_size_bytes(const TensorSpec& tensor, uint32_t tokens_per_chunk) {
    const uint64_t f = feature_width(tensor);
    if (tensor.data_type() == DataType::BFLOAT8_B) {
        return static_cast<uint32_t>((tokens_per_chunk / kTile) * (f / kTile) * kBfp8TileBytes);
    }
    return static_cast<uint32_t>(tokens_per_chunk * f * kBf16Bytes);
}

uint32_t num_dram_banks(tt::ARCH arch) {
    // Mirrors the DRAM channel count in the SoC arch descriptor (umd .../soc_descs/*.yaml): the `dram:`
    // block lists 8 channels for Blackhole and 6 for Wormhole B0. (SocDescriptor::get_num_dram_channels()
    // is the live equivalent, but it needs a SocArchDescriptor loaded from yaml — not host-constructible
    // from a bare ARCH without pulling driver/cluster deps into the addresser, so we key off the arch.)
    switch (arch) {
        case tt::ARCH::BLACKHOLE: return 8;
        case tt::ARCH::WORMHOLE_B0: return 6;
        default: TT_THROW("num_dram_banks: no DRAM bank count encoded for arch {}", static_cast<int>(arch));
    }
}

std::span<const uint32_t> optimal_bank_order(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::BLACKHOLE: return kOptimalDramBankOrder;
        default: TT_THROW("optimal_bank_order: no NOC-local bank order encoded for arch {}", static_cast<int>(arch));
    }
}

}  // namespace tt::tt_metal::internal::disaggregation
