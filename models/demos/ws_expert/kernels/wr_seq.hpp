// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert store-and-forward ring schedule, shared by the forwarder kernels (wr_land.cpp, wr_fwd.cpp) and mirrored by
// ws_sf.ring_schedule on the host.
//
// One directed ring (direction dir, link l) runs through all N chips. Expert e is owned by chip e % NCHIP; its group
// cores j = NL s + l (s < GC / NL) travel on link l's two directed rings: every slice goes both ways from the owner and
// stops after reach(e, s, dir) hops, the two reaches adding up to N - 1 (on an even ring the half-way chip alternates
// between the directions slice by slice, so both directions carry the same load). The order on every ring is expert-
// major (then slice, then packet), the same on every chip, so a forwarder knows each packet's destination from its
// position in the stream and the packets carry no metadata. Packet kinds at a chip h hops from the owner:
//   INJ  h == 0: the owner reads the packet from DRAM and sends it one hop;
//   FWD  0 < h < reach: lands it on this chip's group core and sends it one hop on;
//   TERM h == reach: lands it only.
#pragma once

#include <cstdint>

namespace wr {

constexpr uint32_t INJ = 0, FWD = 1, TERM = 2;

// words of a forwarder's L1 word tensor (16 B apart)
constexpr uint32_t W_ARR = 0;     // packets landed in the staging slots (fabric atomic incs from upstream)
constexpr uint32_t W_DFREE = 1;   // staging slots downstream has freed (fabric inline writes from downstream's pair)
constexpr uint32_t W_RELAY = 2;   // the opposite-direction forwarder's freed count, to send to its upstream chip
constexpr uint32_t W_LANDED = 3;  // packets this core's BRISC has written into the group cores (read by its NCRISC)

FORCE_INLINE uint32_t reach(uint32_t e, uint32_t s, uint32_t dir, uint32_t N) {
    const uint32_t kf = ((e + s) & 1) ? N / 2 : (N - 1) / 2;
    return dir == 0 ? kf : N - 1 - kf;
}

// f(kind, e, n, c, u) for every packet of ring (dir, l) at this chip, in stream order: expert e (the n-th of its group),
// group core c = (e % G) GC + j, packet u of the core's slice. hop[o]: hops from owner chip o to this chip along dir.
template <uint32_t NEXP, uint32_t NCHIP, uint32_t G, uint32_t GC, uint32_t NL, uint32_t NCH, typename F>
FORCE_INLINE void walk(const uint32_t* hop, uint32_t dir, uint32_t l, uint32_t N, F&& f) {
    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t h = hop[e % NCHIP];
        if (h > N / 2) {
            continue;
        }
        const uint32_t g = e % G, n = e / G;
        for (uint32_t s = 0; s < GC / NL; ++s) {
            const uint32_t r = reach(e, s, dir, N);
            if (r == 0 || h > r) {
                continue;
            }
            const uint32_t kind = h == 0 ? INJ : (h < r ? FWD : TERM);
            const uint32_t c = g * GC + NL * s + l;
            for (uint32_t u = 0; u < NCH; ++u) {
                f(kind, e, n, c, u);
            }
        }
    }
}

}  // namespace wr
