// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Goes in LLK_LIB in Init and Uninit
// Set or clear extended state mask bits
// If set, check that set mask covers all ones currently set in state
// sstanisic todo: implement llk_san_extended_state_mask
// template <bool clear = false, typename... Ts>
// llk_san_extended_state_mask(Ts... args)
// {
//     LLK_ASSERT(false, "not implemented");
// }
