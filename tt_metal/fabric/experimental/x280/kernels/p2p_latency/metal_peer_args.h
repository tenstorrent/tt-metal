// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
/* Runtime-argument slots shared by p2p_latency Metal peer kernels. */
#ifndef X280_P2P_LATENCY_METAL_PEER_ARGS_H
#define X280_P2P_LATENCY_METAL_PEER_ARGS_H

#include "../shared/p2p_latency_frontend.h"
#include "../shared/ubench_layout_cookie.h"

#define P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE UBENCH_LAYOUT_COOKIE('P', '2', 'M', 3)

/* Each target's hang guard, in its own wall-clock ticks: a wait longer than
 * this ends the phase TIMEOUT, with no sample stored for it. */
#define P2P_LATENCY_TENSIX_HANG_GUARD_TICKS UINT64_C(3000000000)
/* Conservative raw-tick scheduling rate inferred from ETH_CLOCK_CYCLE_1MS. */
#define P2P_LATENCY_ERISC_NOMINAL_CLOCK_HZ UINT64_C(1000000000)
#define P2P_LATENCY_ERISC_HANG_GUARD_TICKS (UINT64_C(30) * P2P_LATENCY_ERISC_NOMINAL_CLOCK_HZ)
#define P2P_LATENCY_DRISC_HANG_GUARD_TICKS UINT64_C(30000000000)

enum p2p_latency_metal_peer_arg_slot {
    /* The shared layout cookie is checked before any following slot. */
    P2P_LATENCY_METAL_PEER_ARG_LAYOUT_COOKIE = UBENCH_LAYOUT_COOKIE_SLOT,
    P2P_LATENCY_METAL_PEER_ARG_SELF = 1,
    P2P_LATENCY_METAL_PEER_ARG_FIRST_DIRECTION = 2,
    P2P_LATENCY_METAL_PEER_ARG_PAYLOAD_SIZE = 3,
    P2P_LATENCY_METAL_PEER_ARG_WARMUP_COUNT = 4,
    P2P_LATENCY_METAL_PEER_ARG_SAMPLE_COUNT = 5,
    P2P_LATENCY_METAL_PEER_ARG_RUN_NONCE = 6,
    P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_LO = 7,
    P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_HI = 8,
    P2P_LATENCY_METAL_PEER_ARG_OWN_INBOUND_ADDR = 9,
    P2P_LATENCY_METAL_PEER_ARG_OWN_STAGING_ADDR = 10,
    P2P_LATENCY_METAL_PEER_ARG_OWN_RESULT_ADDR = 11,
    /* Peer-local byte address split into low/high u32 words. */
    P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_LO = 12,
    P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_HI = 13,
    P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_X = 14,
    P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_Y = 15,
    P2P_LATENCY_METAL_PEER_ARG_COUNT = 16
};

UBENCH_LAYOUT_ASSERT(
    (int)P2P_LATENCY_METAL_PEER_ARG_LAYOUT_COOKIE == 0,
    "Metal peer layout discriminator is the safely readable prefix");
UBENCH_LAYOUT_ASSERT((int)P2P_LATENCY_METAL_PEER_ARG_COUNT == 16, "Metal peer runtime-argument schema has 16 slots");

#endif /* X280_P2P_LATENCY_METAL_PEER_ARGS_H */
