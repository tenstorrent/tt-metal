// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Compile-time target selection for the shared Metal-peer kernel.

#ifndef X280_P2P_LATENCY_METAL_PEER_TARGET_HPP
#define X280_P2P_LATENCY_METAL_PEER_TARGET_HPP

#include <cstdint>

#include "metal_peer_args.h"

#if (                                                                             \
    defined(P2P_LATENCY_TARGET_TENSIX) + defined(P2P_LATENCY_TARGET_IDLE_ERISC) + \
    defined(P2P_LATENCY_TARGET_DRISC)) != 1
#error "p2p_latency Metal peer requires exactly one target define"
#endif

#if defined(P2P_LATENCY_TARGET_TENSIX)

#include "backends/metal_noc_peer_tile.hpp"
#include "../shared/tensix_live_timer.h"

namespace p2p_latency {

struct TargetPolicy {
    static constexpr uint32_t BACKEND_ID = P2P_LATENCY_BACKEND_ID_TENSIX_TENSIX_PEER_TILE;

    // Reads this target's local counter; ticks are not cross-core timestamps.
    static inline uint64_t read_clock() { return get_timestamp(); }

    // Concurrent-safe performance read; deadline reads keep their own seam.
    static inline uint64_t read_measurement_clock() {
        uint64_t tick;
        return tensix_live_timer_read(TENSIX_TIMER_LOCAL_LOW_ADDR, 8, &tick) ? tick : UINT64_MAX;
    }
};

struct MetalPeerTarget {
    using Backend = backend_metal_noc_peer_tile::Backend<TargetPolicy>;
    static constexpr uint64_t HANG_GUARD_TICKS = P2P_LATENCY_TENSIX_HANG_GUARD_TICKS;

    static bool enter_transport_mode() { return true; }

    static bool leave_transport_mode() { return true; }
};

}  // namespace p2p_latency

#elif defined(P2P_LATENCY_TARGET_IDLE_ERISC)

#include "internal/ethernet/tt_eth_api.h"
#include "backends/metal_noc_peer_tile.hpp"

namespace p2p_latency {

struct TargetPolicy {
    static constexpr uint32_t BACKEND_ID = P2P_LATENCY_BACKEND_ID_ERISC_ERISC_PEER_TILE;

    // Reads this target's local counter; ticks are not cross-core timestamps.
    static inline uint64_t read_clock() { return eth_read_wall_clock(); }
    static inline uint64_t read_measurement_clock() { return eth_read_wall_clock(); }
};

struct MetalPeerTarget {
    using Backend = backend_metal_noc_peer_tile::Backend<TargetPolicy>;
    static constexpr uint64_t HANG_GUARD_TICKS = P2P_LATENCY_ERISC_HANG_GUARD_TICKS;

    static bool enter_transport_mode() { return true; }

    static bool leave_transport_mode() { return true; }
};

}  // namespace p2p_latency

#else

#include "experimental/drisc_mode.h"

#include "internal/tt-1xx/risc_common.h"
#include "backends/metal_noc_peer_tile.hpp"

namespace p2p_latency {

struct TargetPolicy {
    static constexpr uint32_t BACKEND_ID = P2P_LATENCY_BACKEND_ID_DRISC_DRISC_PEER_TILE;

    // Reads this target's local counter; ticks are not cross-core timestamps.
    static inline uint64_t read_clock() { return get_timestamp(); }
    static inline uint64_t read_measurement_clock() { return get_timestamp(); }
};

struct MetalPeerTarget {
    using Backend = backend_metal_noc_peer_tile::Backend<TargetPolicy>;
    static constexpr uint64_t HANG_GUARD_TICKS = P2P_LATENCY_DRISC_HANG_GUARD_TICKS;

    // DRISC firmware owns the NIU modes and leaves NIU 0 in stream mode on every DRISC Metal
    // owns (experimental/drisc_mode.h), so a NOC0 kernel can initiate without switching.
    static bool enter_transport_mode() { return noc_index == 0u && !experimental::drisc_is_noc2axi_mode(0); }

    static bool leave_transport_mode() { return true; }
};

}  // namespace p2p_latency

#endif

#endif  // X280_P2P_LATENCY_METAL_PEER_TARGET_HPP
