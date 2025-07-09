// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt {
namespace tt_metal {

enum class TensorMemoryLayout {
    INTERLEAVED = 0,
    HEIGHT_SHARDED = 2,
    WIDTH_SHARDED = 3,
    BLOCK_SHARDED = 4,
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
};

enum class ShardMode {
    PHYSICAL,  // TODO: Deprecate this option to treat shard shape as physical
    LOGICAL,
};

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY,
    L1_SMALL,
    TRACE,
};

}  // namespace tt_metal
}  // namespace tt
