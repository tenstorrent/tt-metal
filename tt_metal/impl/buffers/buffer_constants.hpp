// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ostream>

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

inline std::ostream& operator<<(std::ostream& os, TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: os << "INTERLEAVED"; break;
        case TensorMemoryLayout::SINGLE_BANK: os << "SINGLE_BANK"; break;
        case TensorMemoryLayout::HEIGHT_SHARDED: os << "HEIGHT_SHARDED"; break;
        case TensorMemoryLayout::WIDTH_SHARDED: os << "WIDTH_SHARDED"; break;
        case TensorMemoryLayout::BLOCK_SHARDED: os << "BLOCK_SHARDED"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, ShardOrientation orientation) {
    switch (orientation) {
        case ShardOrientation::ROW_MAJOR: os << "ROW_MAJOR"; break;
        case ShardOrientation::COL_MAJOR: os << "COL_MAJOR"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, BufferType buffer) {
    switch (buffer) {
        case BufferType::DRAM: os << "DRAM"; break;
        case BufferType::L1: os << "L1"; break;
        case BufferType::SYSTEM_MEMORY: os << "SYSTEM_MEMORY"; break;
        case BufferType::L1_SMALL: os << "L1_SMALL"; break;
        case BufferType::TRACE: os << "TRACE"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

} // namespace tt_metal

} // namespace tt
