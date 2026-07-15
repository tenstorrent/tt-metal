// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/flags.hpp>

namespace tensor_accessor {

/**
 * @brief Encodes a fundamental configuration of a tensor accessor, which must be available at compile time.
 * Specifies whether the tensor is sharded, stored in DRAM, and which arguments should be passed as compile-time vs
 * runtime arguments.
 */
enum class ArgConfig : uint8_t {
    None = 0,
    Sharded = 1 << 0,
    IsDram = 1 << 1,
    RuntimeRank = 1 << 2,
    RuntimeNumBanks = 1 << 3,
    RuntimeTensorShape = 1 << 4,
    RuntimeShardShape = 1 << 5,
    RuntimeBankCoords = 1 << 6,
    RuntimePageSize = 1 << 7,
    Runtime =
        RuntimeRank | RuntimeNumBanks | RuntimeTensorShape | RuntimeShardShape | RuntimeBankCoords | RuntimePageSize
};

using ArgsConfig = Flags<ArgConfig>;
constexpr ArgsConfig operator|(ArgConfig a, ArgConfig b) noexcept { return ArgsConfig(a) | b; }
constexpr ArgsConfig operator|(ArgConfig a, ArgsConfig b) noexcept { return ArgsConfig(a) | b; }

// The num_banks word (whether it is a compile-time arg or a common runtime arg) doubles as the carrier for the
// shard-contiguous distribution flag: the bank count lives in the low bits and the shard-contiguous flag in the top
// bit. num_banks is tiny (number of DRAM/L1 banks), so this never collides with a real bank count. This lets the
// distribution strategy be selected -- at compile time or per dispatch -- without adding an arg/slot or an ArgConfig
// bit.
//
// pack/unpack are the single source of truth for this layout, so the host encoder and the device decoders
// cannot drift. Callers must ensure num_banks < ShardContiguousBit (always true for real bank counts).
inline constexpr uint32_t ShardContiguousBit = 1u << 31;
constexpr uint32_t pack_num_banks(uint32_t num_banks, bool is_shard_contiguous) {
    return num_banks | (is_shard_contiguous ? ShardContiguousBit : 0u);
}
constexpr uint32_t unpack_num_banks(uint32_t packed_num_banks) { return packed_num_banks & ~ShardContiguousBit; }
constexpr bool unpack_is_shard_contiguous(uint32_t packed_num_banks) {
    return (packed_num_banks & ShardContiguousBit) != 0;
}

}  // namespace tensor_accessor
