// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"

// DEST_SRCA_MM is Blackhole-only (tt-llk experimental/llk_math_dest_srca_mm.h). The Compute API keeps the dest_srca_mm
// calls on every arch, so their entry points reject them here.

template <std::uint32_t in0_rows>
inline void llk_math_dest_srca_mm_init() {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Quasar");
}

inline void llk_math_dest_srca_mm_uninit() { LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Quasar"); }

inline void llk_math_dest_srca_mm_page_post([[maybe_unused]] const std::uint32_t dst_index) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Quasar");
}

inline void llk_math_dest_srca_mm_codes_post(
    [[maybe_unused]] const std::uint32_t dst_index, [[maybe_unused]] const std::uint32_t codes_index) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Quasar");
}

inline void llk_math_dest_srca_mm_page_wait(
    [[maybe_unused]] const std::uint32_t page_operand, [[maybe_unused]] const std::uint32_t dst_index) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Quasar");
}

template <std::uint32_t in0_rows>
inline void llk_math_dest_srca_mm(
    [[maybe_unused]] const std::uint32_t src_index,
    [[maybe_unused]] const std::uint32_t dst_index,
    [[maybe_unused]] const std::uint32_t src_rows) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Quasar");
}
