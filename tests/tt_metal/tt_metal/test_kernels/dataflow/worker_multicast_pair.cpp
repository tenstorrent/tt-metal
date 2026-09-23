// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

// Worker-rectangle multicast coverage. Two issues from one sender:
//  1. an ordinary multicast to a rectangle that excludes the sender
//     ({peer} .. {peer}, one destination);
//  2. a loopback multicast (sender included) to the rectangle spanning the
//     sender and the peer (two destinations).
// The rectangle coordinates arrive as NoC worker coordinates; the address
// backend turns them into whatever the transport needs (an XY multicast
// address, or the ATT descriptor the V3 issue path decodes through the map).
void kernel_main() {
    const uint32_t src_addr = get_arg(args::src_addr);
    const uint32_t dst_addr_unicast_rect = get_arg(args::dst_addr_unicast_rect);
    const uint32_t dst_addr_loopback_rect = get_arg(args::dst_addr_loopback_rect);
    const uint32_t size_bytes = get_arg(args::size_bytes);
    const uint32_t self_x = get_arg(args::self_x);
    const uint32_t self_y = get_arg(args::self_y);
    const uint32_t peer_x = get_arg(args::peer_x);
    const uint32_t peer_y = get_arg(args::peer_y);

    Noc noc;
    UnicastEndpoint src;
    MulticastEndpoint dst;

    // 1. Peer-only rectangle: one destination, sender excluded.
    noc.async_write_multicast<NocOptions::DEFAULT>(
        src,
        dst,
        size_bytes,
        /*num_dsts=*/1,
        {.addr = src_addr},
        {.noc_x_start = peer_x,
         .noc_y_start = peer_y,
         .noc_x_end = peer_x,
         .noc_y_end = peer_y,
         .addr = dst_addr_unicast_rect});

    // 2. Sender + peer rectangle with the sender included: two destinations.
    const uint32_t x_start = self_x < peer_x ? self_x : peer_x;
    const uint32_t x_end = self_x < peer_x ? peer_x : self_x;
    const uint32_t y_start = self_y < peer_y ? self_y : peer_y;
    const uint32_t y_end = self_y < peer_y ? peer_y : self_y;
    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
        src,
        dst,
        size_bytes,
        /*num_dsts=*/2,
        {.addr = src_addr},
        {.noc_x_start = x_start,
         .noc_y_start = y_start,
         .noc_x_end = x_end,
         .noc_y_end = y_end,
         .addr = dst_addr_loopback_rect});

    noc.async_write_barrier();
}
