// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/strong_type.hpp>
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal::internal::disaggregation {

// KvLayoutSpec — the model-facing, declarative definition of ONE cached KV/state tensor.
//
// It is the "common language" a model uses to specify its KV cache layout; migration, tiering, and
// weights services consume it. Deliberately RESIDENCE-AGNOSTIC: it answers *what a tensor is* and
// *how its sequence coordinate is interpreted*. Deriving physical addresses (the old per-model
// `locate` / to_chunk_map), the resharding planner, and tiering are downstream KV Manager concerns.
//
// DESIGN — why this is small. A KV tensor is already almost fully described by a native distributed
// `TensorSpec`. The mesh tensor carries:
//   * shape + dtype                         -> TensorLayout
//   * intra-device bank striping            -> MemoryConfig / NdShardSpec (shard_shape, grid,
//                                              orientation, shard_distribution_strategy)
//   * mesh placement + device coords        -> TensorTopology (per-axis Shard/Replicate, MeshShape,
//                                              get_device_coord() -> replica / device groups)
// So this spec adds ONLY what the tensor cannot carry:
//   (1) how the sequence coordinate is interpreted  -> TemporalPolicy (which ALSO says whether a
//                                                      sequence axis exists: Rolling/None have none)
//   (2) slot-direct vs paged addressing             -> AddressingMode
// WHICH axis is the sequence is NOT stored: the allocation enforces that the sequence axis is the
// token-block-tiled NdShardSpec axis, so it is derived from the tensor (see sequence_axis()).
//
// Everything mesh/bank/shard-flavoured is READ OFF the tensor, never re-declared here. The earlier
// skeleton's Distribution part duplicated TensorTopology::placements(); its SpDim/MeshCols/MeshRows
// duplicated the TensorTopology MeshShape; its GqaGroup/head_shard_axis/idx_cp were placement facts
// already in the topology; its MemLayout.{chunk_n_tokens,num_banks,bf16_chunk_bytes,bank_order}
// duplicated NdShardSpec / the device dram grid / the tensor dtype. All removed.
//
// Generation-policy scalars that genuinely are NOT in the tensor (they live in the flash op /
// migration / prefill engine) are inputs to the DOWNSTREAM addresser, not part of this definition —
// see GenerationPolicy below.

// ---------------------------------------------------------------------------------------------
// Strong types — semantic values that must never be interchanged (see cpp coding standards).
// ---------------------------------------------------------------------------------------------

// Sliding-window width W, in tokens.
using WindowTokens = ttsl::StrongType<uint32_t, struct WindowTokensTag>;
// The span of one block-local ATTENTION chunk, in tokens (temporal::BlockLocal). A retention semantic,
// NOT a layout quantity — distinct from the DRAM/flash "chunk" strides (KChunkSize, DeviceChunkSize,
// and chunk_size_bytes' token-block granule), which is why it has its own name.
using BlockLocalSpan = ttsl::StrongType<uint32_t, struct BlockLocalSpanTag>;

// ---------------------------------------------------------------------------------------------
// (1) TemporalPolicy — retention window over the prefix, keyed by the universal `prefix_len`.
// THE irreducible semantic: nothing in TensorSpec / TensorTopology / NdShardSpec encodes attention
// or recurrence temporality. Each alternative fixes (extent, address-map, update) for the sequence.
// ---------------------------------------------------------------------------------------------
namespace temporal {

// [0, prefix_len) — grows with the prefix. Full KV, MLA latent.
struct Dense {};

// [prefix_len - width, prefix_len) — a rolling ring; addressed by (position mod width). SWA.
struct Window {
    WindowTokens width;
};

// The current chunk only; resets at chunk boundaries (not a trailing window). Llama-4 iRoPE local.
struct BlockLocal {
    BlockLocalSpan span;
};

// The last (kernel_size - 1) inputs — a tiny conv / token-shift ring (Mamba conv1d, RWKV
// token-shift). Tagless: the ring depth is the conv-state tensor's OWN axis (KDA allocates it as
// [B, kernel_size - 1, width]), so it is derived from the tensor when needed, not stored here —
// like the sequence axis. has_sequence() is false (no per-token axis).
struct Rolling {};

// Computed once at prefill and frozen; addressed by SOURCE position, never grows. Enc-dec /
// vision cross-attention.
struct Static {};

// No sequence axis at all — a fixed-size recurrent summary addressed by a sparse checkpoint tag
// (the prefix_len it was snapshotted at). Mamba SSM state, KDA state matrix.
struct None {};

}  // namespace temporal

using TemporalPolicy = std::variant<
    temporal::Dense,
    temporal::Window,
    temporal::BlockLocal,
    temporal::Rolling,
    temporal::Static,
    temporal::None>;

// ---------------------------------------------------------------------------------------------
// (2) AddressingMode — how a logical position resolves to a physical slot.
// ---------------------------------------------------------------------------------------------

enum class AddressingMode : uint8_t {
    Slot = 0,   // direct: physical = f(slot, coord), no runtime indirection
    Paged = 1,  // a block table maps logical position -> physical block (enables sharing / packing)
};

// ---------------------------------------------------------------------------------------------
// The composed spec — the tensor plus the three things the tensor cannot carry.
// ---------------------------------------------------------------------------------------------

struct KvLayoutSpec {
    // The logical tensor as a native distributed mesh TensorSpec. Carries shape + dtype +
    // MemoryConfig/NdShardSpec (intra-device bank striping) + TensorTopology (mesh placement, extents,
    // device coords). Distribution, mesh geometry, and bank layout are read from HERE, not re-declared.
    TensorSpec tensor;

    // How the sequence coordinate is interpreted (part 1). ALSO encodes whether a sequence axis exists
    // at all: Rolling/None are fixed-size recurrent/conv summaries with no per-token axis.
    TemporalPolicy temporal;

    // Slot-direct vs paged (block-table) indirection (part 2).
    AddressingMode addressing = AddressingMode::Slot;

    // Target architecture. The arch-specific layout constants — the DRAM bank count and the OPTIMAL
    // NOC-local bank-order permutation — derive from this (see num_dram_banks / optimal_bank_order);
    // they are read from the SoC arch descriptor, not hardcoded per call.
    tt::ARCH arch = tt::ARCH::Invalid;

    // Whether this tensor has a per-token sequence axis — read from `temporal`, not stored.
    bool has_sequence() const {
        return !std::holds_alternative<temporal::Rolling>(temporal) &&
               !std::holds_alternative<temporal::None>(temporal);
    }

    // The sequence (temporal) axis index into `tensor`'s logical shape, DERIVED from the tensor: the
    // token-block-tiled NdShardSpec axis (shard granule == the DRAM token block and < its extent).
    // nullopt when !has_sequence(). The allocation enforces this tiling, so the axis is recoverable
    // without being declared (a plain shape-dim index, like metal's own NdShardSpec / MeshMapper dims).
    std::optional<uint32_t> sequence_axis() const;
};

// A model's KV cache is an ORDERED LIST of co-resident specs (K/V, indexer, one per layer-type).
// Per-layer heterogeneity (e.g. gpt-oss sliding vs full layers) is expressed as DISTINCT specs — each
// layer-type is its own tensor with its own TensorSpec/TemporalPolicy — replacing the earlier
// per-layer {temporal,bank_scheme,k_chunk}_by_layer override maps on a single spec.
using KvCacheModel = std::vector<KvLayoutSpec>;

// ---------------------------------------------------------------------------------------------
// GenerationPolicy — inputs the DOWNSTREAM addresser (to_chunk_map) needs that the tensor genuinely
// cannot carry because they live in the flash op / migration / prefill engine, NOT the allocated
// tensor. Deliberately OUTSIDE KvLayoutSpec (which is residence-agnostic): these are passed to the
// address derivation alongside the spec and the tensor's TensorTopology.
// ---------------------------------------------------------------------------------------------

// The sequence-shard (CP) stride, in tokens: how many tokens each seq-shard device owns per round
// before the sequence rotates to the next device. Residence-NEUTRAL — decode's CP round-robin and the
// prefill block-cyclic per-device window are the same quantity (the prefill compute-chunk period is
// just this * the seq-shard mesh extent, so it is NOT a separate field). Lives in the flash op /
// migration / prefill engine, not the tensor.
using DeviceChunkSize = ttsl::StrongType<uint32_t, struct DeviceChunkSizeTag>;
// K-chunk (flash op block) size, in tokens — the effective SDPA k-chunk / block_size.
using KChunkSize = ttsl::StrongType<uint32_t, struct KChunkSizeTag>;
// Number of DRAM banks a single head/group fans out over (BLOCK / CYCLIC height-sharding).
using BanksPerHead = ttsl::StrongType<uint32_t, struct BanksPerHeadTag>;
// Origin offset of the first sequence-shard (CP) device on the mesh axis.
using SpOrigin = ttsl::StrongType<uint32_t, struct SpOriginTag>;
// index_k column-split degree (BLOCK_CYCLIC): how many devices a sparse-index row is split across.
using IdxCp = ttsl::StrongType<uint32_t, struct IdxCpTag>;
// GQA group index -> mesh ROW block (BLOCK K/V per-group placement).
using GqaGroup = ttsl::StrongType<uint32_t, struct GqaGroupTag>;

// DRAM / tile constants (device + dtype facts the addresser needs; not stored on the spec).
inline constexpr std::array<uint32_t, 8> kOptimalDramBankOrder = {1, 3, 2, 0, 5, 7, 6, 4};
inline constexpr uint32_t kNumDramBanks = 8;
inline constexpr uint32_t kTile = 32;
inline constexpr uint32_t kBfp8TileBytes = 1088;  // 32x32 bfloat8_b tile
inline constexpr uint32_t kBf16Bytes = 2;

// Bank ordering permutation: the OPTIMAL NOC-local order, or the portable identity round-robin. A
// generation-policy selector (which permutation the flash op / migration used), not a tensor property.
enum class BankOrder : uint8_t {
    Identity = 0,  // portable round-robin (page_id % num_banks)
    Optimal = 1,   // NOC-local permutation co-locating each bank with its consuming cores
};

// How a chunk index maps to (bank, per-bank offset). The round-robin cases (Natural / BlockCyclic)
// coincide with the tensor's NdShardSpec::shard_distribution_strategy; MlaShard / Block / Cyclic carry
// op-specific arithmetic beyond it. A derivation selector, not a tensor property — and residence-neutral:
// the prefill write layout IS BlockCyclic with the block ordinal = the tensor's ND-shard row-major ravel
// (which folds the user-major [num_users*num_layers] batch axis). It is not a distinct scheme.
enum class BankScheme : uint8_t {
    Natural = 0,
    MlaShard = 1,
    Block = 2,
    Cyclic = 3,
    BlockCyclic = 4,
};

struct GenerationPolicy {
    BankScheme bank_scheme = BankScheme::Natural;
    BankOrder bank_order = BankOrder::Optimal;
    KChunkSize k_chunk_size{128};
    // Seq-shard (CP) stride; default k_chunk_size * num_banks. For the prefill write layout this is the
    // per-device window (compute-chunk period / seq-shard mesh extent) — same field, no prefill variant.
    std::optional<DeviceChunkSize> device_chunk_size;
    BanksPerHead banks_per_head{kNumDramBanks};  // BLOCK/CYCLIC per-head fan-out
    uint32_t num_blocks = kNumDramBanks;         // OPTIMAL indexer permutation block count
    SpOrigin sp_origin{0};                       // CP device origin on the seq-shard mesh axis
    IdxCp idx_cp{1};                             // index_k column-split (BLOCK_CYCLIC)
    std::optional<GqaGroup> group;               // GQA group -> mesh ROW block (BLOCK K/V)
    // num_banks and the mesh geometry (sp_dim / mesh cols/rows / head-shard axis / device coords) are
    // NOT fields here: num_banks is a device fact (dram grid), the geometry is read from the tensor's
    // TensorTopology, and the seq axis + feature width from its NdShardSpec.
};

// Bytes for one chunk of `tokens_per_chunk` tokens. The feature width and dtype sizing are read from
// the tensor (NdShardSpec shard_shape's full-width axes + TensorLayout) — no feature_dim arg, no sizing
// flag. When tokens_per_chunk == the shard's seq granule this is exactly metal's NdShard shard/page
// byte size; other granules scale it linearly. (No stored InnerFootprint: F is the shard feature width.)
uint32_t chunk_size_bytes(const TensorSpec& tensor, uint32_t tokens_per_chunk);

// F — the per-token FEATURE width, derived from the tensor: the product of the NdShardSpec shard_shape
// axes that span their full logical extent (the seq axis is tiled at the token block; batch/head axes
// have granule 1; only the feature axes are full-width).
uint64_t feature_width(const TensorSpec& tensor);

// DRAM bank count for an arch — mirrors the DRAM channel count in the SoC arch descriptor
// (umd .../soc_descs/*.yaml). BLACKHOLE=8, WORMHOLE_B0=6; other archs TT_THROW until their descriptor
// value is added. Replaces threading num_dram_banks through every call: it is an arch fact.
uint32_t num_dram_banks(tt::ARCH arch);

// The OPTIMAL NOC-local DRAM bank-order permutation for an arch (index i -> physical bank id).
// Currently only BLACKHOLE is encoded; other archs TT_THROW until their NOC ordering is added.
std::span<const uint32_t> optimal_bank_order(tt::ARCH arch);

}  // namespace tt::tt_metal::internal::disaggregation
