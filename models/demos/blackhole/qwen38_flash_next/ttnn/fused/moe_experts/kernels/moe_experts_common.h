// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared constants of the routed-expert program's three kernels (reader, compute, writer): the shape of the Qwen3.8
// expert (hidden 2560 = 80 tiles, intermediate 640 = 20 tiles), the cached BF4 block geometry and the metadata page
// the reader publishes (the experts of this die the two token rows selected).
#pragma once

#include <cstdint>

namespace moe_experts {

constexpr uint32_t HT = 80;            // hidden tiles = K of the gate/up matmul
constexpr uint32_t NT = 20;            // intermediate tiles = gate/up columns = K of the down matmul
constexpr uint32_t TILE_BYTES = 2048;  // bf16 tile
constexpr uint32_t BF4_TILE_BYTES = 576;
constexpr uint32_t BLOCK_TILES = 40;  // a weight block: two 20-tile transactions
constexpr uint32_t TXN_TILES = 20;
constexpr uint32_t BANK_BLOCKS_PER_EXPERT = 10;  // gate/up blocks of one (expert, bank)
constexpr uint32_t W2_BLOCKS_PER_EXPERT = 5;     // down blocks of one (expert, ring core)
constexpr uint32_t EXPERTS_PER_DEVICE = 64;
constexpr uint32_t TOP_K = 10;
constexpr uint32_t MAX_ACTIVE = 20;  // experts of this die two rows can select: 2 x 10
// runtime-arg layout of a core's weight block lists: (bank id, block index in the bank's (expert) piece) pairs
constexpr uint32_t MAX_G_BLOCKS = 8;
constexpr uint32_t MAX_W_BLOCKS = 4;  // a core's down units: at most 2, of 2 + 1 or 2 + 2 blocks
constexpr uint32_t MAX_W_UNITS = 2;

// The metadata page (uint32 words): word 0 = active experts n; entry j at 1 + 4 j = {local expert, row mask, k slot of
// row 0, k slot of row 1}.
constexpr uint32_t MD_ENTRY_WORDS = 4;
constexpr uint32_t MD_WORDS = 1 + MAX_ACTIVE * MD_ENTRY_WORDS;
constexpr uint32_t MD_PAGE_BYTES = 384;
static_assert(MD_WORDS * 4 <= MD_PAGE_BYTES, "metadata page");

// More than 2 rows (up to 8): the experts of this die 8 rows can select (at most the 64 of the die), and an entry is
// {local expert, row mask, k slots packed 4 bits per row} = 3 words.
constexpr uint32_t WIDE_MAX_ROWS = 8;
constexpr uint32_t WIDE_MAX_ACTIVE = 64;
constexpr uint32_t WIDE_ENTRY_WORDS = 3;
constexpr uint32_t WIDE_MD_WORDS = 1 + WIDE_MAX_ACTIVE * WIDE_ENTRY_WORDS;
constexpr uint32_t WIDE_MD_PAGE_BYTES = 832;
static_assert(WIDE_MD_WORDS * 4 <= WIDE_MD_PAGE_BYTES, "wide metadata page");

// rows <= 2 keeps the original layout (4-word entries, 20 experts); more rows use the wide one.
constexpr bool wide_rows(uint32_t rows) { return rows > 2; }
constexpr uint32_t max_active(uint32_t rows) { return wide_rows(rows) ? WIDE_MAX_ACTIVE : MAX_ACTIVE; }
constexpr uint32_t entry_words(uint32_t rows) { return wide_rows(rows) ? WIDE_ENTRY_WORDS : MD_ENTRY_WORDS; }

// Bytes of an hidden / in2 tile that hold rows 0 .. rows - 1: face 0 whole, then 32 B (16 bf16) per row of face 1.
constexpr uint32_t tile_row_bytes(uint32_t rows) { return 512 + 32 * rows; }

}  // namespace moe_experts
