// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The activation policy is mathematical: it applies after the compiler-bound
// target ingress transform above. Codegen supplies class codes from that
// target record, never from an activation name.

static_assert(
    TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT <= 3,
    "BH BF16 target special finalizer has invalid structural compare count");
