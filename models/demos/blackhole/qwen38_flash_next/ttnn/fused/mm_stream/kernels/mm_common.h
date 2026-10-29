// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// x52 streamed decode matmul: constants shared by the hub, reader, compute and writer kernels.
#include <cstdint>
namespace mm_stream {
constexpr uint32_t MAX_COLS = 4;  // output tiles per core (the fp32 DEST half holds 4)
}
