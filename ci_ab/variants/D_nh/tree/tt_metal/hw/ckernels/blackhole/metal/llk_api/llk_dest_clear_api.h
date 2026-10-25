// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"

// The unpacker writes DEST only from a 32-bit source, so a build with none has the math thread as DEST's one writer.
#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK)
constexpr bool llk_unpack_may_write_dest() {
    for (const std::uint32_t format : unpack_src_format) {
        const std::uint32_t masked = ckernel::masked_data_format(format);
        if (format != static_cast<std::uint32_t>(DataFormat::Invalid) &&
            (masked == static_cast<std::uint32_t>(DataFormat::Float32) ||
             masked == static_cast<std::uint32_t>(DataFormat::Int32))) {
            return true;
        }
    }
    return false;
}
#else
constexpr bool llk_unpack_may_write_dest() { return true; }
#endif
