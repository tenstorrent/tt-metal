// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"

// DEST_SRCA_MM is Blackhole-only (tt-llk experimental/llk_unpack_dest_srca_mm.h). The Compute API keeps the
// dest_srca_mm calls on every arch, so their entry points reject them here.

inline void llk_unpack_dest_srca_mm_init([[maybe_unused]] const std::uint32_t in0_operand) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Wormhole");
}

inline void llk_unpack_dest_srca_mm_page_init([[maybe_unused]] const std::uint32_t page_operand) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Wormhole");
}

inline void llk_unpack_dest_srca_mm_page(
    [[maybe_unused]] const std::uint32_t page_operand, [[maybe_unused]] const std::uint32_t tile_index) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Wormhole");
}

inline void llk_unpack_dest_srca_mm(
    [[maybe_unused]] const std::uint32_t in0_operand, [[maybe_unused]] const std::uint32_t in0_tile_index) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Wormhole");
}

inline void llk_unpack_dest_srca_mm_codes_page(
    [[maybe_unused]] const std::uint32_t page_operand,
    [[maybe_unused]] const std::uint32_t tile_index,
    [[maybe_unused]] const std::uint32_t codes,
    [[maybe_unused]] const std::uint32_t exp,
    [[maybe_unused]] const std::uint32_t words_offset,
    [[maybe_unused]] const std::uint32_t words,
    [[maybe_unused]] const std::int32_t words_shift = 0) {
    LLK_ASSERT(false, "dest_srca_mm is Blackhole-only; not supported on Wormhole");
}
