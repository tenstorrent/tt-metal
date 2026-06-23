// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Host-side guard for matmul_block's caller_owns_pack_target contract. The device helper
// enforces the same predicate via static_assert; testing it here locks the invariant so a
// future relaxation (e.g. dropping the packer_l1_acc or Interm requirement) fails CI even
// if no caller happens to instantiate the now-broken combination. No device required.

#include <gtest/gtest.h>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_constraints.hpp"

namespace {

using compute_kernel_lib::caller_owns_pack_target_supported;

// caller_owns disabled: the contract is inert — every layout/accum/target combination is fine.
TEST(MatmulBlockCallerOwnsConstraint, DisabledIsAlwaysSupported) {
    for (bool trm : {false, true}) {
        for (bool l1_acc : {false, true}) {
            for (bool interm : {false, true}) {
                EXPECT_TRUE(caller_owns_pack_target_supported(
                    /*caller_owns_pack_target=*/false, trm, l1_acc, interm));
            }
        }
    }
}

// The single supported caller_owns config: TileRowMajor + packer_l1_acc + Interm.
TEST(MatmulBlockCallerOwnsConstraint, ValidConfigSupported) {
    EXPECT_TRUE(caller_owns_pack_target_supported(
        /*caller_owns_pack_target=*/true,
        /*is_tile_row_major=*/true,
        /*packer_l1_acc=*/true,
        /*last_block_is_interm=*/true));
}

// Each single-axis deviation must be rejected — at runtime these deadlock (orphaned reload
// wait_front) or, for SubblockMajor, corrupt output.
TEST(MatmulBlockCallerOwnsConstraint, InvalidConfigsRejected) {
    EXPECT_FALSE(caller_owns_pack_target_supported(true, /*is_tile_row_major=*/false, true, true));
    EXPECT_FALSE(caller_owns_pack_target_supported(true, true, /*packer_l1_acc=*/false, true));
    EXPECT_FALSE(caller_owns_pack_target_supported(true, true, true, /*last_block_is_interm=*/false));
}

// Mirror the helper's compile-time enforcement so the predicate is also pinned at build time.
static_assert(caller_owns_pack_target_supported(false, false, false, false));
static_assert(caller_owns_pack_target_supported(true, true, true, true));
static_assert(!caller_owns_pack_target_supported(true, false, true, true));
static_assert(!caller_owns_pack_target_supported(true, true, false, true));
static_assert(!caller_owns_pack_target_supported(true, true, true, false));

}  // namespace
