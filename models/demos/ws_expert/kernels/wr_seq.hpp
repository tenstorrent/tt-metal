// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert store-and-forward ring order, shared by the forwarder kernels (wr_land.cpp, wr_fwd.cpp).
//
// One directed ring (a cycle through all chips, one link, one direction) has one forwarder core per chip. Its packet
// order comes from the host (ws_sf.conveyor) as a table per chip and ring: entry i is the i-th packet this chip handles
// on the ring, in time order, kind | u << 2 | c << 6 | e << 14:
//   INJ  this chip owns expert e: the packet is read from DRAM and sent one hop;
//   FWD  the packet arrives from upstream, is landed on group core c and sent one hop on;
//   TERM the packet arrives and is landed only (its last chip).
// u: packet of core c's slice (CHT tiles from tile u CHT); the expert is the n = e / G-th of group c / GC.
// The incoming packets (FWD, TERM) fill the staging slots in table order; the outgoing ones (INJ, FWD) go to the next
// chip's staging slots in table order, which is exactly the next chip's incoming order.
#pragma once

#include <cstdint>

namespace wr {

constexpr uint32_t INJ = 0, FWD = 1, TERM = 2;

// words of a forwarder's L1 word tensor (16 B apart)
constexpr uint32_t W_ARR = 0;     // packets landed in the staging slots (fabric atomic incs from upstream)
constexpr uint32_t W_DFREE = 1;   // staging slots downstream has freed (fabric inline writes from downstream's pair)
constexpr uint32_t W_RELAY = 2;   // the opposite-direction forwarder's freed count, to send to its upstream chip
constexpr uint32_t W_LANDED = 3;  // packets this core's BRISC has written into the group cores (read by its NCRISC)
constexpr uint32_t W_TABLE = 4;   // the order table is in CB 2 (BRISC -> NCRISC)

struct Pkt {
    uint32_t kind, u, c, e;
};

FORCE_INLINE Pkt decode(uint32_t w) { return Pkt{w & 3, (w >> 2) & 15, (w >> 6) & 0xFF, w >> 14}; }

}  // namespace wr
