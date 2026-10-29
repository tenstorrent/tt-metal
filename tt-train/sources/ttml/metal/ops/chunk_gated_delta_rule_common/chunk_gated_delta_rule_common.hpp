// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Constants and tile math shared by the forward and backward chunk_gated_delta_rule ops.  Included by
// the host factories and by the kernels, so it has no dependencies beyond <cstdint>.

#pragma once

#include <cstdint>

namespace chunk_gated_delta_rule_common {

// L1 below the circular-buffer region that get_max_worker_l1_unreserved_size() still counts: the
// kernel-config ring buffer the kernel binaries live in, plus slack.  Every L1 fit check subtracts it.
constexpr uint32_t kL1KernelConfigReserve = 80U * 1024U;

// Element offset of (row r, col c) inside a 32x32 tile stored as four 16x16 faces in (face_r, face_c)
// row-major order.
constexpr uint32_t tile_elem_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256U + ((r & 15U) << 4) + (c & 15U);
}

}  // namespace chunk_gated_delta_rule_common
