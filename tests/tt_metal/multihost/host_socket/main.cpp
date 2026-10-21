// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <tt-metalium/distributed_context.hpp>

#include "common/multihost_test_tools.hpp"

// multihost::common::multihost_main minus its fault-tolerance gate: nothing here
// revokes or shrinks a communicator, so these tests run on any MPI build, not only
// the ULFM one. That is what lets the same binary be driven by the system OpenMPI
// with pml/osc ucx for the RDMA numbers.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int local_rc = RUN_ALL_TESTS();
    const auto& context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    fmt::print("Rank {}: local rc = {}\n", *context->rank(), local_rc);
    ASSERT_EQ_ALL_RANKS(local_rc, context);
    return local_rc;
}
