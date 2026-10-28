// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The enums below do not need it. Kept because includers of this header may rely on <cstdint> transitively
// (preserved from the pre-split shared header).
#include <cstdint>  // IWYU pragma: keep

// Buffer enums for device (kernel) code.
//
// This is the device-side counterpart of the host header tt-metalium/buffer_types.hpp. The two files are
// deliberately separate: one is compiled by the host compiler, the other by the RISC-V kernel compiler, and
// neither may include the other. Enumerator values are passed from host to device as compile-time args, so
// every enumerator here must match the host header exactly; only the enums kernels use are mirrored.

namespace tt::tt_metal {

enum class TensorMemoryLayout {
    INTERLEAVED = 0,
    HEIGHT_SHARDED = 2,
    WIDTH_SHARDED = 3,
    BLOCK_SHARDED = 4,
    ND_SHARDED = 5,
};

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY,
    L1_SMALL,
    TRACE,
};

}  // namespace tt::tt_metal
