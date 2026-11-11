// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric unicast on a Keraunos test line, with the sender and receiver on hart 1 (DM1) of the end chips' router CCEs,
// co-launched with the routers through the fabric program extension. Run with TT_METAL_SLOW_DISPATCH_MODE=1. On Mock
// the run checks the launch messages and skips: UMD MockChip reads and writes are no-ops.

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// A worker beside a router: the router's Ethernet channel and core, on DM1.
struct Worker {
    FabricNodeId node;
    ChipId chip;
    chan_id_t router_chan;
    CoreCoord logical;
    CoreCoord virtual_core;
};

// The line's two ends: the sender on fabric chip 0, the receiver on the last fabric chip, each beside its router that
// faces the other end.
struct Line {
    Worker sender;
    Worker receiver;
    uint32_t num_hops;
};

Line line_ends();

class KeraunosCoLaunchedWorkers : public BaseFabricFixture {
protected:
    // Opens the devices with 1D fabric, the line's workers co-launched beside its end routers, sends kNumPackets from
    // the line's first chip to its last, then checks both workers' results and that every router terminates. Skips off
    // Quasar, and on Mock after checking the launch messages.
    void run_line_unicast();

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
    void TearDown() override;
};

}  // namespace tt::tt_fabric::fabric_router_tests
