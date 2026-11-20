// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric unicast on a Keraunos test line, with the sender and receiver on CCE1 of the end routers' HSIO tiles, launched
// as ordinary programs once fabric is up. Run with TT_METAL_SLOW_DISPATCH_MODE=1. On Mock
// the run compiles the kernels and skips: UMD MockChip reads and writes are no-ops.

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// A worker beside a router: the router's Ethernet channel, and the worker's core, CCE1 of the router's tile
// (BaseFabricFixture::get_worker_beside_router).
struct Worker {
    FabricNodeId node;
    ChipId chip;
    chan_id_t router_chan;
    FabricTestWorker core;
};

// The line's two ends: the sender on fabric chip 0, the receiver on the last fabric chip, each beside its router that
// faces the other end.
struct Line {
    Worker sender;
    Worker receiver;
    uint32_t num_hops;
};

// The line's ends of the open cluster, from the control plane. Fatal when a chip has no router toward the other end, or
// its SoC descriptor lists no idle CCE1 on that router's tile.
Line line_ends();

class KeraunosCoLaunchedWorkers : public BaseFabricFixture {
protected:
    // Opens the devices with 1D fabric, sends kNumPackets from the line's first chip to its last, then checks both
    // workers' results and that every router terminates. Skips off Quasar, and on Mock after compiling the workers'
    // kernels.
    void run_line_unicast();

    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
    void TearDown() override;
};

}  // namespace tt::tt_fabric::fabric_router_tests
