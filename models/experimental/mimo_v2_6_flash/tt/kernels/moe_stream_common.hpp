// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared constants for the MiMo MoE decode streaming kernels (tt/moe_stream.py).
#pragma once

#include <cstdint>

namespace mimo_moe {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kFaceDim = 16;
constexpr uint32_t kBf16FaceBytes = kFaceDim * kFaceDim * 2;  // 512 B: face 1 of a bf16 32x32 tile starts here

// The positions in the deduplicated stream's stick [count, ids...] of its n distinct experts, by ascending expert id.
// The stick lists them in user order, so a user's experts sit where the other users put theirs; the down stream sums
// each user's experts in this order instead, so a user's output is the same whatever else the step selected. Its
// reader (w_down) and writer (h) both walk this order, so compute pairs each h with its own w_down.
template <uint32_t capacity, typename Ids>
inline void experts_by_id(Ids ids, uint32_t n, uint32_t (&order)[capacity]) {
    for (uint32_t p = 0; p < n; p++) {
        const uint32_t id = ids[1 + p];
        uint32_t q = p;
        for (; q > 0 && ids[1 + order[q - 1]] > id; q--) {
            order[q] = order[q - 1];
        }
        order[q] = p;
    }
}

}  // namespace mimo_moe
