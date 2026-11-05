// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K3.1 expert-parallel all-to-all (dispatch + combine) prototype: shared routing decisions.
//
// Roles (one Tensix core each):
//   ROLE_EAST / ROLE_WEST : owns ONE fabric connection (link LINK, toward higher / lower line index),
//                           sends every row that must travel in that direction on that link.
//   ROLE_SELF             : local copies (rows whose destination is this device) + completion wait.
//
// Reader (NCRISC) and writer (BRISC) on the same core run the SAME decision function over the SAME
// metadata so the data CB stays in lock-step without passing per-row descriptors.
#pragma once

#include <cstdint>

namespace k31ep {

constexpr uint32_t ROLE_EAST = 0;
constexpr uint32_t ROLE_WEST = 1;
constexpr uint32_t ROLE_SELF = 2;

constexpr uint32_t MODE_DISPATCH = 0;
constexpr uint32_t MODE_COMBINE = 1;

// Per-token metadata row: 16 x uint16 expert ids, then 16 x bf16 router weights = 64 B.
constexpr uint32_t META_BYTES = 64;

// Bitmask of devices (line positions) hit by one token's top-K expert ids.
template <uint32_t K, uint32_t EPD>
FORCE_INLINE uint32_t hit_mask(const volatile tt_l1_ptr uint16_t* ids) {
    uint32_t m = 0;
    for (uint32_t k = 0; k < K; ++k) {
        m |= 1u << (ids[k] / EPD);
    }
    return m;
}

// Dispatch: does this core handle token t, and how far (multicast range) must it go?
// The token row is multicast ONCE per direction, to hops [1, far], so every hit device in that
// direction gets exactly one copy and every link carries the row at most once.
// RING (1D_RING fabric, N even): direction E covers ring hops 1..N/2-1, W covers 1..N/2-1, and the antipodal
// hop N/2 goes E for even t / W for odd t (load balance). Completion flags then cover N/2 hops both ways.
template <uint32_t ROLE, uint32_t MY, uint32_t L, uint32_t LINK, uint32_t N = 0, bool RING = false>
FORCE_INLINE bool dispatch_need(uint32_t t, uint32_t mask, uint32_t& far) {
    far = 0;
    if constexpr (ROLE == ROLE_SELF) {
        return (mask >> MY) & 1u;
    } else {
        if ((t % L) != LINK) {
            return false;
        }
        if constexpr (RING) {
            const uint32_t maxh = (N / 2) - ((((t / L) & 1u) == (ROLE == ROLE_EAST ? 0u : 1u)) ? 0u : 1u);
            for (uint32_t k = maxh; k >= 1; --k) {
                const uint32_t d = (ROLE == ROLE_EAST) ? (MY + k) % N : (MY + N - k) % N;
                if ((mask >> d) & 1u) {
                    far = k;
                    break;
                }
            }
        } else if constexpr (ROLE == ROLE_EAST) {
            const uint32_t hi = 31u - __builtin_clz(mask);
            far = hi > MY ? hi - MY : 0;
        } else {
            const uint32_t lo = __builtin_ctz(mask);
            far = lo < MY ? MY - lo : 0;
        }
        return far > 0;
    }
}

// Combine: row (s, t) = this device's (already router-weighted, locally reduced) partial output for
// source device s's token t. Sent back to s by unicast iff the token hit this device.
template <uint32_t ROLE, uint32_t MY, uint32_t L, uint32_t LINK, uint32_t N = 0, bool RING = false>
FORCE_INLINE bool combine_need(uint32_t s, uint32_t t, uint32_t mask, uint32_t& hops) {
    hops = 0;
    if (((mask >> MY) & 1u) == 0) {
        return false;
    }
    if constexpr (ROLE == ROLE_SELF) {
        return s == MY;
    } else {
        if ((t % L) != LINK) {
            return false;
        }
        if constexpr (RING) {
            if (s == MY) {
                return false;
            }
            const uint32_t he = (s + N - MY) % N;
            bool east = he < N / 2 || (he == N / 2 && (((t / L) & 1u) == 0));
            if (east == (ROLE == ROLE_EAST)) {
                hops = east ? he : N - he;
                return true;
            }
            return false;
        } else if constexpr (ROLE == ROLE_EAST) {
            if (s > MY) {
                hops = s - MY;
                return true;
            }
        } else {
            if (s < MY) {
                hops = MY - s;
                return true;
            }
        }
        return false;
    }
}

}  // namespace k31ep
