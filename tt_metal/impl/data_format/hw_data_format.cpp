// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/data_format/hw_data_format.hpp"

namespace tt::tt_metal {

namespace {

// Quasar HW DataFormat codes (mirror of the relevant entries in
// tensix_types.h. A few host DataFormat
// enumerators use a value that differs from the HW encoding to keep host enum
// values unique / avoid collisions, so device compilation needs the real HW
// code. Keep these in sync with tensix_types.h.
constexpr hw_format_t kHwInt16 = 9;        // host Int16 is 13 (UInt16 owns 9 on host)
constexpr hw_format_t kHwMxFp4_2x_B = 24;  // host MxFp4_2x_B is 29 (UInt32 owns 24 on host)
constexpr hw_format_t kHwMxInt8 = 2;       // host MxInt8 is 12 (Bfp8 owns 2 on host)
constexpr hw_format_t kHwMxInt4 = 3;       // host MxInt4 is 16 (Bfp4 owns 3 on host)
constexpr hw_format_t kHwMxInt2 = 11;      // host MxInt2 is 17 (Bfp2 owns 11 on host)

}  // namespace

hw_format_t host_data_format_to_hw(DataFormat f) {
    switch (f) {
        case DataFormat::Int16: return kHwInt16;
        case DataFormat::MxFp4_2x_B: return kHwMxFp4_2x_B;
        case DataFormat::MxInt8: return kHwMxInt8;
        case DataFormat::MxInt4: return kHwMxInt4;
        case DataFormat::MxInt2: return kHwMxInt2;
        default: return static_cast<hw_format_t>(f);
    }
}

}  // namespace tt::tt_metal
