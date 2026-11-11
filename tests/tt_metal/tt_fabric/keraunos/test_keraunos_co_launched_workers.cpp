// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric unicast on the Keraunos test lines in this directory, with the sender and receiver on hart 1 (DM1) of the end
// chips' router CCEs, co-launched with the routers through the fabric program extension: one hop on the two-chip
// descriptors, two hops through chip M on the three-chip ones. Run with TT_METAL_SLOW_DISPATCH_MODE=1. On Mock the
// test checks the launch messages and skips: UMD MockChip reads and writes are no-ops.

#include <gtest/gtest.h>

#include "co_launched_line.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

TEST_F(KeraunosCoLaunchedWorkers, LineUnicast) { run_line_unicast(); }

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
