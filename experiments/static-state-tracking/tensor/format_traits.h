// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// sst::tensor::format_traits
// ----------------------------------------------------------------------------
// The tile data format. The prototype is Float16_b-only; this is the skeleton
// to grow from — add more formats (and any per-format trait helpers they need)
// here as the experiment expands.
//
// SST-local mirror of tt::DataFormat.
// ----------------------------------------------------------------------------

#include <cstdint>

namespace sst::tensor {

enum class DataFormat : uint8_t {
    Float16_b = 5,
    // Add more tt::DataFormat mirrors here as formats are supported, e.g.
    //   Float32 = 0, Bfp8_b = 6, …  (values must match tt::DataFormat).
};

}  // namespace sst::tensor
