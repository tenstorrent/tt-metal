// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-only header. Device (kernel/firmware) code must use api/buffer_types.hpp from tt_metal/hw/inc instead:
// the two sides are compiled by different compilers and must not share this file. The device header mirrors
// TensorMemoryLayout and BufferType; keep their enumerator values in sync with the ones below.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#error "tt-metalium/buffer_types.hpp is host-only; include \"api/buffer_types.hpp\" in device code"
#endif

#include <cstdint>  // IWYU pragma: keep

namespace tt::tt_metal {

enum class TensorMemoryLayout {
    INTERLEAVED = 0,
    HEIGHT_SHARDED = 2,
    WIDTH_SHARDED = 3,
    BLOCK_SHARDED = 4,
    ND_SHARDED = 5,  // The ND_SHARDED value is reserved for cases where the ND sharding pattern cannot be represented
                     // by any of the legacy 2D sharding strategies. For more info, see the end of
                     // tech_reports/tensor_sharding/tensor_sharding.md
};

enum class ShardOrientation {
    ROW_MAJOR = 0,
    COL_MAJOR,
};

enum class ShardDistributionStrategy {
    // Distribute each shard to each of the cores in a linearized list in a round-robin manner.
    ROUND_ROBIN_1D = 0,
    // Distribute a 2D grid of shards to a 2D grid of cores with one to one mapping.
    GRID_2D = 1,
    // Distribute shards contiguously over a linearized list of cores: adjacent shards go to the
    // same core until it is full. Shard s lands on core (s / shards_per_core) at slot
    // (s % shards_per_core). Requires a uniform shards_per_core (num_shards % num_cores == 0).
    CONTIGUOUS_1D = 2,
};

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY,
    L1_SMALL,
    TRACE,
};

}  // namespace tt::tt_metal
