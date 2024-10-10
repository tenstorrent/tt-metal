// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt {

namespace tt_metal {

enum class TensorMemoryLayout {
    INTERLEAVED = 0,
    SINGLE_BANK,
    HEIGHT_SHARDED,
    WIDTH_SHARDED,
    BLOCK_SHARDED,
};

enum class ShardOrientation {
    ROW_MAJOR = 0,
    COL_MAJOR,
};

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY,
    L1_SMALL,
    TRACE,
};

} // namespace tt_metal

} // namespace tt
