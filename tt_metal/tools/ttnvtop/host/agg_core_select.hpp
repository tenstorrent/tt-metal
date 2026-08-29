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

#include <algorithm>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/risc_type.hpp>

namespace ttnvtop {

// AGGREGATOR LIFECYCLE -- READ BEFORE LAUNCHING TWICE.
//
// The aggregator kernel never returns; that is how it persists (3.5). The
// consequence is that it CANNOT be replaced by a second launch. The idle-erisc
// firmware loop that consumes launch messages only runs between kernels, so a new
// RUN_MSG_GO is never seen -- and worse, ConfigureDeviceWithProgram writes the new
// binary and runtime args straight over the live kernel, which keeps running on
// corrupted state and keeps pushing plausible-looking WRONG telemetry.
//
// Measured 2026-08-29 by launching twice on one core:
//     sender markers state=0x00000000 sweeps=0 head=0 cores=0   <- 2nd never started
//     landed magic=0x00000000 sweeps=5400 lost=5586 cores=0 cap=2240  <- 1st corrupted
//
// So a launcher MUST either use a core with no aggregator on it, or reset the core
// first. Never launch onto a live one. Two callers in one process, or two processes,
// must not pick the same core -- hence rank_aggregator_eth_cores() below.

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

// Inactive eth cores in preference order: never-routed channels first, then the
// rest, each group by ascending channel. Deterministic, so independent callers
// agree on the ordering and can take distinct ranks rather than colliding.
inline std::vector<tt::tt_metal::CoreCoord> rank_aggregator_eth_cores(tt::tt_metal::IDevice* device) {
    std::vector<tt::tt_metal::CoreCoord> ranked(
        device->get_inactive_ethernet_cores().begin(), device->get_inactive_ethernet_cores().end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        const bool na = is_never_routed_channel(a.y);
        const bool nb = is_never_routed_channel(b.y);
        if (na != nb) {
            return na;
        }
        return a.y < b.y;
    });
    return ranked;
}

// `rank` selects among the available cores: 0 is the best choice, 1 the next, and so
// on. Callers that must not collide (a second aggregator, or a test that cannot
// relaunch onto a live kernel) take successive ranks.
inline EthCoreChoice select_aggregator_eth_core(tt::tt_metal::IDevice* device, size_t rank = 0) {
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

    // Logical eth coords are (0, channel) -- see CoordinateManager's
    // CoreCoord(0, eth_channel, CoreType::ETH, CoordSystem::LOGICAL).
    const auto ranked = rank_aggregator_eth_cores(device);
    if (ranked.size() <= rank) {
        choice.reason = "device " + std::to_string(device->id()) + " has " + std::to_string(ranked.size()) +
                        " inactive ethernet core(s), need rank " + std::to_string(rank);
        return choice;
    }
    choice.ok = true;
    choice.core = ranked[rank];
    return choice;
}

// Stop an aggregator: the ONLY way, short of closing the device.
//
// The kernel never returns, so there is no go-signal or mailbox handshake that ends
// it -- the firmware loop that would read one is not running. Asserting the RISC
// reset is what stops it. The core then holds no running firmware until device init
// reloads it, which is acceptable because the aggregator's lifetime is one
// device-open epoch anyway (3.5).
//
// Call this before relaunching on a core that may already hold an aggregator.
// Launching over a live one does NOT replace it: the new launch message is never
// consumed and ConfigureDeviceWithProgram corrupts the running kernel's binary and
// args underneath it, after which it keeps pushing plausible-looking wrong telemetry.
//
// RiscType::ALL, deliberately. RiscType::ERISC0 and ERISC1 are aliases for BRISC and
// TRISC0 in UMD (risc_type.hpp has a standing "Consider having separate entries"
// TODO), so naming them invites a copy-paste that means something else elsewhere.
inline void stop_aggregator(tt::tt_metal::IDevice* device, tt::tt_metal::CoreCoord logical_eth_core) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto virtual_core = device->virtual_core_from_logical_core(logical_eth_core, tt::CoreType::ETH);
    const tt_cxy_pair core(device->id(), virtual_core);

    // Assert only. This stops the kernel and RELEASES ITS FABRIC CONNECTION, which is
    // the property that matters -- but it leaves the core with no running firmware,
    // so the core is NOT relaunchable afterwards until device init reloads it.
    //
    // Deasserting does not bring it back: measured 2026-08-29, an assert+deassert pair
    // leaves the next launch on that core silently never starting (the new kernel's
    // markers stay at the zero the host wrote). Restarting an ERISC needs its reset
    // vector pointed at the firmware, which is device-init's job, not ours. Do not
    // add a deassert here on the assumption that it restores the firmware.
    //
    // The aggregator's lifetime is one device-open epoch regardless (3.5), so
    // "stopped until the device is reopened" is the intended semantics, not a gap.
    cluster.assert_risc_reset_at_core(core, tt::umd::RiscType::ALL);
}

}  // namespace ttnvtop
