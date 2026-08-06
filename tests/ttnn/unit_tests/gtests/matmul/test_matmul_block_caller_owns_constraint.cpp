// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Host-side guard for matmul_block's caller_owns_pack_target contract. The device helper
// enforces the same predicate via static_assert; testing it here locks the invariant so a
// future relaxation (e.g. dropping the packer_l1_acc or target requirement) fails CI even
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
                for (bool plain_out : {false, true}) {
                    EXPECT_TRUE(caller_owns_pack_target_supported(
                        /*caller_owns_pack_target=*/false, trm, l1_acc, interm, plain_out));
                }
            }
        }
    }
}

// The two supported caller_owns targets: fixed Interm, or fixed scratch + plain Out.
TEST(MatmulBlockCallerOwnsConstraint, ValidConfigsSupported) {
    EXPECT_TRUE(caller_owns_pack_target_supported(
        /*caller_owns_pack_target=*/true,
        /*is_tile_row_major=*/true,
        /*packer_l1_acc=*/true,
        /*last_block_is_interm=*/true,
        /*last_block_is_plain_out=*/false));
    EXPECT_TRUE(caller_owns_pack_target_supported(
        /*caller_owns_pack_target=*/true,
        /*is_tile_row_major=*/true,
        /*packer_l1_acc=*/true,
        /*last_block_is_interm=*/false,
        /*last_block_is_plain_out=*/true));
}

// Each single-axis deviation must be rejected — at runtime these deadlock (orphaned reload
// wait_front) or, for SubblockMajor, corrupt output.
TEST(MatmulBlockCallerOwnsConstraint, InvalidConfigsRejected) {
    EXPECT_FALSE(caller_owns_pack_target_supported(true, /*is_tile_row_major=*/false, true, true, false));
    EXPECT_FALSE(caller_owns_pack_target_supported(true, true, /*packer_l1_acc=*/false, true, false));
    EXPECT_FALSE(caller_owns_pack_target_supported(
        true, true, true, /*last_block_is_interm=*/false, /*last_block_is_plain_out=*/false));
}

// Mirror the helper's compile-time enforcement so the predicate is also pinned at build time.
static_assert(caller_owns_pack_target_supported(false, false, false, false, false));
static_assert(caller_owns_pack_target_supported(true, true, true, true, false));
static_assert(caller_owns_pack_target_supported(true, true, true, false, true));
static_assert(!caller_owns_pack_target_supported(true, false, true, true, false));
static_assert(!caller_owns_pack_target_supported(true, true, false, true, false));
static_assert(!caller_owns_pack_target_supported(true, true, true, false, false));

}  // namespace
