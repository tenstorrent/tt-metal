// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);
    using Rank = tt::tt_metal::distributed::multihost::Rank;
    using Tag = tt::tt_metal::distributed::multihost::Tag;

    ::testing::InitGoogleTest(&argc, argv);

    // Run tests on every rank
    int local_rc = RUN_ALL_TESTS();

    // Propagate the worst return code to all ranks
    int global_rc = local_rc;

    return global_rc;
}
