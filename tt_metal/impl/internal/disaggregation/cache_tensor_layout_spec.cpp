// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/cache_tensor_layout_spec.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/soc_arch_descriptor.hpp>

namespace tt::tt_metal::internal::disaggregation {

using tt::constants::BFLOAT8_B_TILE_HW;
using tt::constants::TILE_WIDTH;

namespace {

// NOC-local DRAM bank order: a flash-op / migration locality CHOICE (blaze's OPTIMAL_DRAM_BANK_ORDER),
// NOT an arch fact the SoC descriptor exposes — so it is defined here (the one place), keyed by arch.
inline constexpr std::array<uint32_t, 8> kBlackholeOptimalDramBankOrder = {1, 3, 2, 0, 5, 7, 6, 4};

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
    // The sequence axis is the one the allocation tiles at the DRAM token block (TILE_WIDTH) while its full
    // extent is larger — i.e. chopped into many blocks. Feature axes are full-width; batch/head are 1.
    for (uint32_t i = 0; i < rank; ++i) {
        const uint32_t g = static_cast<uint32_t>(shard[i]);
        if (g == TILE_WIDTH && static_cast<uint32_t>(shape[i]) > TILE_WIDTH) {
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
        return static_cast<uint32_t>((tokens_per_chunk / TILE_WIDTH) * (f / TILE_WIDTH) * BFLOAT8_B_TILE_HW);
    }
    return static_cast<uint32_t>(tokens_per_chunk * f * tt::datum_size(tt::DataFormat::Float16_b));
}

uint32_t num_dram_banks(tt::ARCH arch) {
    // The DRAM channel count IS an arch fact — read it from the SoC arch descriptor (the `dram:` block of
    // umd .../soc_descs/<arch>.yaml). SocArchDescriptor is host-constructible from a bare ARCH; no
    // device/cluster needed. get_dram_cores() is channel-major, so its outer size is the bank count.
    return static_cast<uint32_t>(tt::umd::SocArchDescriptor(arch).get_dram_cores().size());
}

std::span<const uint32_t> optimal_bank_order(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::BLACKHOLE: return kBlackholeOptimalDramBankOrder;
        default: TT_THROW("optimal_bank_order: no NOC-local bank order encoded for arch {}", static_cast<int>(arch));
    }
}

}  // namespace tt::tt_metal::internal::disaggregation
