// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host packing for the shared p2p_latency Metal-peer runtime schema.

#ifndef X280_SUITES_P2P_LATENCY_METAL_PEER_ARGUMENTS_HPP
#define X280_SUITES_P2P_LATENCY_METAL_PEER_ARGUMENTS_HPP

#include "metal_peer_args.h"
#include "p2p_latency_frontend.h"
#include "x280/device/translated_coord.hpp"

#include <cstdint>
#include <vector>

namespace x280::harness::arguments {

struct MetalPeerRuntimeArgs {
    uint32_t layout_cookie;
    p2p_latency_common_args_t common;
    uint32_t own_inbound_addr;
    uint32_t own_staging_addr;
    uint32_t own_result_addr;
    uint64_t peer_inbound_addr;
    TranslatedCoord peer_noc;
};

// Serializes one already-validated placement into the fixed 16-word Metal
// runtime schema. It chooses no target, address, coordinate, or clock value.
inline std::vector<uint32_t> pack_metal_peer_runtime_args(const MetalPeerRuntimeArgs& args) {
    std::vector<uint32_t> packed(P2P_LATENCY_METAL_PEER_ARG_COUNT, 0u);
    packed[P2P_LATENCY_METAL_PEER_ARG_LAYOUT_COOKIE] = args.layout_cookie;
    packed[P2P_LATENCY_METAL_PEER_ARG_SELF] = args.common.self;
    packed[P2P_LATENCY_METAL_PEER_ARG_FIRST_DIRECTION] = args.common.first_direction;
    packed[P2P_LATENCY_METAL_PEER_ARG_PAYLOAD_SIZE] = args.common.payload_size;
    packed[P2P_LATENCY_METAL_PEER_ARG_WARMUP_COUNT] = args.common.warmup_count;
    packed[P2P_LATENCY_METAL_PEER_ARG_SAMPLE_COUNT] = args.common.sample_count;
    packed[P2P_LATENCY_METAL_PEER_ARG_RUN_NONCE] = args.common.run_nonce;
    packed[P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_LO] = static_cast<uint32_t>(args.common.warmup_max_poll_cycles);
    packed[P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_HI] =
        static_cast<uint32_t>(args.common.warmup_max_poll_cycles >> 32);
    packed[P2P_LATENCY_METAL_PEER_ARG_OWN_INBOUND_ADDR] = args.own_inbound_addr;
    packed[P2P_LATENCY_METAL_PEER_ARG_OWN_STAGING_ADDR] = args.own_staging_addr;
    packed[P2P_LATENCY_METAL_PEER_ARG_OWN_RESULT_ADDR] = args.own_result_addr;
    packed[P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_LO] = static_cast<uint32_t>(args.peer_inbound_addr);
    packed[P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_HI] = static_cast<uint32_t>(args.peer_inbound_addr >> 32);
    packed[P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_X] = args.peer_noc.x();
    packed[P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_Y] = args.peer_noc.y();
    return packed;
}

}  // namespace x280::harness::arguments

#endif  // X280_SUITES_P2P_LATENCY_METAL_PEER_ARGUMENTS_HPP
