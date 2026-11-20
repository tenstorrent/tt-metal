// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric unicast on the Keraunos test lines of vcluster/manifests/, with the sender and receiver on hart 1 (DM1) of the
// end chips' router CCEs, co-launched with the routers through the fabric program extension: one hop on
// pair2_eth.yaml, two hops through chip M on line3_eth.yaml. Run with TT_METAL_SLOW_DISPATCH_MODE=1. On Mock the
// test checks the launch messages and skips: UMD MockChip reads and writes are no-ops.
//
// KeraunosCoLaunchedWorkersNocLinks runs the same line with chosen links backed by the NOC: the fabric program
// extension sets the kind word of the link record on both ends of each chosen link (PLAN-0008 Phase 1b, 1c; decisions
// 62db7e799795-D2 and -D3).

#include <gtest/gtest.h>

#include "co_launched_line.hpp"

namespace tt::tt_fabric::fabric_router_tests {
namespace {

TEST_F(KeraunosCoLaunchedWorkers, LineUnicast) { run_line_unicast({}); }

class KeraunosCoLaunchedWorkersNocLinks : public KeraunosCoLaunchedWorkers,
                                          public ::testing::WithParamInterface<NocLinks> {};

TEST_P(KeraunosCoLaunchedWorkersNocLinks, LineUnicast) { run_line_unicast(GetParam()); }

// Fabric chips 0, 1, 2 are A, M, B on the three-chip line.
INSTANTIATE_TEST_SUITE_P(
    Links,
    KeraunosCoLaunchedWorkersNocLinks,
    ::testing::Values(NocLinks{{0, 1}}, NocLinks{{0, 1}, {1, 2}}),
    [](const ::testing::TestParamInfo<NocLinks>& info) { return info.param.size() == 1 ? "AM" : "AM_MB"; });

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
