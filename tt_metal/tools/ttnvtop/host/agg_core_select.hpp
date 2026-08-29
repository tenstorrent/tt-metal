// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Which idle ethernet core does the ttnvtop aggregator run on, and is it
// actually ours to take?  (PLAN_ETH_AGGREGATOR.md 3.4)
//
// "Free of links" is not the same as "free of users". Two different owners can
// have a claim on an inactive ethernet core, and which of them applies depends
// on the dispatch core type:
//
//   WORKER dispatch (the default -- DispatchCoreConfig() is DispatchCoreType::WORKER,
//     and on WH/BH resolve_dispatch_core_type returns whatever the config says):
//     inactive eth cores are NOT in the fast-dispatch pool. dispatch_core_manager.cpp
//     only adds them under `if (resolve_dispatch_core_type(...) == CoreType::ETH)`.
//     Nothing else allocates them, so taking one is safe.
//
//   ETH dispatch: dispatch_core_manager adds EVERY core from
//     get_inactive_ethernet_cores() to the available pool. Taking one without
//     reserving it risks fast dispatch later placing a dispatch kernel on the same
//     core. This is not exotic -- 8-chip Wormhole systems require ETH dispatch to
//     run 2 command queues (see programming_examples/distributed/4_...).
//
// There is no reservation API we can use on Wormhole. ServiceCoreManager looks like
// the right registry -- dispatch_core_manager already drops its claimed cores from
// the pool -- but claim() is gated:
//     TT_FATAL(cluster.is_ubb_galaxy() || cluster.arch() == tt::ARCH::BLACKHOLE,
//              "Service core claims are only supported on Blackhole and UBB Galaxy");
// so it is unavailable on exactly the system that has the problem.
//
// The eventual fix is the pattern the real-time profiler already uses fifteen lines
// below the eth branch in dispatch_core_manager.cpp: reserve from the BACK of the
// pool at construction time, because dispatch consumes from the FRONT. That is an
// upstream change.
//
// Until then this refuses to run under ETH dispatch rather than gambling that
// dispatch will not reach the core we picked. A monitor that silently corrupts a
// dispatch kernel is a far worse outcome than a monitor that declines to start.

#pragma once

#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"

namespace ttnvtop {

struct EthCoreChoice {
    bool ok = false;
    tt::tt_metal::CoreCoord core{};
    std::string reason;  // populated when !ok, for a loud failure at init
};

// Ethernet channels that no shipped Wormhole cluster descriptor ever routes, on
// either an MMIO or a remote chip (PLAN_ETH_AGGREGATOR.md 2). Preferring these
// keeps the aggregator away from channels that a recabling could bring into use.
// Preference only -- never a requirement, since Blackhole harvests channels and
// the free set differs there (2.1).
inline bool is_never_routed_channel(uint32_t channel) {
    switch (channel) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 10:
        case 11:
        case 12:
        case 13: return true;
        default: return false;
    }
}

inline EthCoreChoice select_aggregator_eth_core(tt::tt_metal::IDevice* device) {
    EthCoreChoice choice;

    auto& ctx = tt::tt_metal::MetalContext::instance();
    const auto& dispatch_core_config = ctx.get_dispatch_core_manager().get_dispatch_core_config();
    const auto core_type = tt::tt_metal::resolve_dispatch_core_type(
        tt::tt_metal::MetalEnvAccessor(ctx.get_env()).impl(), device->id(), dispatch_core_config);

    if (core_type == tt::CoreType::ETH) {
        choice.reason = "fast dispatch is configured for ETH cores on device " + std::to_string(device->id()) +
                        ", so every inactive ethernet core is in the dispatch pool "
                        "(dispatch_core_manager.cpp: `if (resolve_dispatch_core_type(...) == CoreType::ETH)`). "
                        "There is no reservation API on this architecture -- ServiceCoreManager::claim() is "
                        "Blackhole/UBB-Galaxy only -- so the aggregator would be sharing a core with a dispatch "
                        "kernel. Refusing to start. Run with WORKER dispatch, or reserve an eth core from the "
                        "back of the dispatch pool the way the real-time profiler reserves a tensix.";
        return choice;
    }

    const auto inactive = device->get_inactive_ethernet_cores();
    if (inactive.empty()) {
        choice.reason = "device " + std::to_string(device->id()) + " has no inactive ethernet core";
        return choice;
    }

    // Logical eth coords are (0, channel) -- see CoordinateManager's
    // CoreCoord(0, eth_channel, CoreType::ETH, CoordSystem::LOGICAL).
    bool have_fallback = false;
    tt::tt_metal::CoreCoord fallback{};
    for (const auto& c : inactive) {
        if (is_never_routed_channel(c.y)) {
            choice.ok = true;
            choice.core = c;
            return choice;
        }
        if (!have_fallback || c.y < fallback.y) {
            fallback = c;
            have_fallback = true;
        }
    }
    choice.ok = true;
    choice.core = fallback;
    return choice;
}

}  // namespace ttnvtop
