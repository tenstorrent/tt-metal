// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric unicast on a Keraunos test line, with the sender and receiver on hart 1 (DM1) of the end chips' router CCEs,
// co-launched with the routers through the fabric program extension, and chosen links backed by the NOC: the extension
// sets the kind word of the link record on both ends of each chosen link (PLAN-0008 Phase 1b, 1c; decisions
// 62db7e799795-D2 and -D3). Run with TT_METAL_SLOW_DISPATCH_MODE=1. On Mock the run checks the launch messages and
// skips: UMD MockChip reads and writes are no-ops.

#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// Links of the line backed by the NOC, each by its two ends' fabric chip ids. Empty: every link is Ethernet.
using NocLinks = std::vector<std::pair<uint32_t, uint32_t>>;

// Called once by run_line_unicast(), on the test's thread, after the devices are open with fabric and before any
// traffic. It reads the open cluster through MetalContext::instance() (get_control_plane(), get_cluster()) and may use
// gtest assertions; after a fatal failure in it, run_line_unicast() returns and the fixture's TearDown closes the
// devices. Not called on Mock or when the run skips before opening.
using AfterOpen = std::function<void()>;

// A worker beside a router: the router's Ethernet channel and core, on DM1.
struct Worker {
    FabricNodeId node;
    ChipId chip;
    chan_id_t router_chan;
    tt::tt_metal::CoreCoord logical;
    tt::tt_metal::CoreCoord virtual_core;
};

// The line's two ends: the sender on fabric chip 0, the receiver on the last fabric chip, each beside its router that
// faces the other end.
struct Line {
    Worker sender;
    Worker receiver;
    uint32_t num_hops;
};

// The line's ends of the open cluster, from the control plane. Fatal when a chip has no router toward the other end.
Line line_ends();

class KeraunosCoLaunchedWorkers : public BaseFabricFixture {
protected:
    // Opens the devices with 1D fabric, sends kNumPackets from the line's first chip to its last with the links in
    // `noc_links` on the NOC and the rest on Ethernet, then checks the link records, both workers' results and that
    // every router terminates, and the path each link took from the routers' NOC staging words. `after_open`, when
    // set, runs between the open and the first check (AfterOpen). Skips off Quasar, when a link in `noc_links` is not
    // on the open line, and on Mock after checking the launch messages.
    void run_line_unicast(const NocLinks& noc_links, const AfterOpen& after_open = {});

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
    void TearDown() override;
};

}  // namespace tt::tt_fabric::fabric_router_tests
