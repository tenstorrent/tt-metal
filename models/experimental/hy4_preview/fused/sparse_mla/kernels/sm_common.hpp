// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared by every sparse-MLA kernel: the selection table, which S-block serves which chunk, the merge tree, and
// where a packed KV row sits in DRAM. Free of host-only and kernel-only dependencies.
//
// Selection table (uint32, row-major, one row per page, `TOPK` entries wide):
//   row 0      : [n_0 .. n_{U-1}]  number of valid ids of each user on THIS chip (0 .. TOPK)
//   row 1 + u  : the ids of user u: rows of this chip's slab of the user's KV slot, entries [0, n_u) valid
// Chunk c of user u is the list entries [c * CHUNK, (c + 1) * CHUNK) and is served by S-block (u + c) % GROUPS, so
// users with at most one chunk each run on different S-blocks at the same time.
#pragma once

#include <cstdint>

namespace sm {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t MAX_USERS = 8;
constexpr uint32_t MERGE_STEPS = 3;  // log2(GROUPS), GROUPS == 8

constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// First chunk of user u that S-block b serves; the following ones are GROUPS apart.
template <uint32_t GROUPS>
constexpr uint32_t first_chunk(uint32_t b, uint32_t u) {
    return (b + GROUPS - (u % GROUPS)) % GROUPS;
}

// f(u, c, valid_rows) for every chunk S-block b serves, users ascending, chunks ascending.
template <uint32_t USERS, uint32_t GROUPS, uint32_t CHUNK, typename F>
inline __attribute__((always_inline)) void for_each_chunk(const uint32_t* n, uint32_t b, F&& f) {
    for (uint32_t u = 0; u < USERS; ++u) {
        const uint32_t nch = ceil_div(n[u], CHUNK);
        for (uint32_t c = first_chunk<GROUPS>(b, u); c < nch; c += GROUPS) {
            const uint32_t left = n[u] - c * CHUNK;
            f(u, c, left < CHUNK ? left : CHUNK);
        }
    }
}

// Number of chunks S-block b serves for user u.
template <uint32_t GROUPS, uint32_t CHUNK>
constexpr uint32_t chunks_here(uint32_t n_u, uint32_t b, uint32_t u) {
    const uint32_t nch = ceil_div(n_u, CHUNK);
    const uint32_t first = first_chunk<GROUPS>(b, u);
    return first < nch ? (nch - first + GROUPS - 1) / GROUPS : 0;
}

// ---- converters of an S-block ----
// An S-block has NCONV row reader / converter cores; converter v takes the slabs v, v + NCONV, ... of every chunk
// of the S-block, so the slabs of a chunk are read and converted side by side.
template <uint32_t NCONV>
constexpr uint32_t own_slabs(uint32_t slabs, uint32_t v) {
    return slabs > v ? (slabs - v + NCONV - 1) / NCONV : 0;
}
constexpr uint32_t CTRL_END = 0xFFFFFFFFu;  // reader -> converter: no more chunks

// ---- which core am I ----
// The layout is fixed (layout.py sblocks): two halves of the worker grid starting at logical columns X0_LEFT and
// X0_RIGHT; in each half 4 S-blocks of 2 rows; column x0 holds the block's two reader / converter cores, columns
// x0 + 1 .. x0 + 4 its 8 attention cores (head group g at column x0 + 1 + g % 4, row 2 * (block % 4) + g / 4).
// Every core finds its role from its own logical position, so the kernels need no per-core runtime arguments
// (per-core arguments are written to each core separately at every launch).
struct Role {
    uint32_t block;
    uint32_t slot;  // converter index on a reader core, head group on an attention core
};
#if defined(KERNEL_BUILD)
template <uint32_t X0_LEFT, uint32_t X0_RIGHT>
inline __attribute__((always_inline)) Role role_from_position() {
    const uint32_t x = get_absolute_logical_x();
    const uint32_t y = get_absolute_logical_y();
    const bool right = x >= X0_RIGHT;
    const uint32_t x0 = right ? X0_RIGHT : X0_LEFT;
    return Role{(right ? 4u : 0u) + y / 2, x == x0 ? y % 2 : (x - x0 - 1) + 4 * (y % 2)};
}
#endif
// NoC coordinates, one entry of 4 words per S-block in the common runtime arguments:
// [reader 0 x, reader 0 y, attention group 0 x, attention group 0 y]. Reader v is at (x, y + v), attention group
// g at (x + g % 4, y + g / 4): the host checks that the device's coordinates are laid out this way.
constexpr uint32_t COORD_WORDS = 4;

// Any receive due on S-block b for any user? If not, the merge slots are not used at all.
template <uint32_t USERS, uint32_t GROUPS, uint32_t CHUNK>
constexpr bool any_receive(const uint32_t* n, uint32_t b);

// ---- merge tree of one user over the S-blocks that hold its chunks ----
// Relative index r = (b - u) mod GROUPS; active S-blocks are r < n_act = min(chunks, GROUPS). r = 0 is the root:
// the user's partial ends on S-block u. Step s merges r + 2^s into r for every r that is a multiple of 2^(s + 1).
template <uint32_t GROUPS>
constexpr uint32_t rel(uint32_t b, uint32_t u) {
    return first_chunk<GROUPS>(b, u);
}
template <uint32_t GROUPS, uint32_t CHUNK>
constexpr uint32_t active_blocks(uint32_t n_u) {
    const uint32_t nch = ceil_div(n_u, CHUNK);
    return nch < GROUPS ? nch : GROUPS;
}
// Does relative block r receive a partial at step s?
constexpr bool receives(uint32_t r, uint32_t n_act, uint32_t s) {
    return r < n_act && (r & ((2u << s) - 1)) == 0 && r + (1u << s) < n_act;
}
// The step at which relative block r (not the root, active) sends its partial: its lowest set bit.
constexpr uint32_t send_step(uint32_t r) {
    uint32_t s = 0;
    while (((r >> s) & 1u) == 0) {
        ++s;
    }
    return s;
}
constexpr uint32_t num_receives(uint32_t r, uint32_t n_act) {
    uint32_t k = 0;
    for (uint32_t s = 0; s < MERGE_STEPS; ++s) {
        k += receives(r, n_act, s) ? 1 : 0;
    }
    return k;
}

template <uint32_t USERS, uint32_t GROUPS, uint32_t CHUNK>
constexpr bool any_receive(const uint32_t* n, uint32_t b) {
    for (uint32_t u = 0; u < USERS; ++u) {
        if (num_receives(rel<GROUPS>(b, u), active_blocks<GROUPS, CHUNK>(n[u])) != 0) {
            return true;
        }
    }
    return false;
}

// Counts are handed to the compute kernels two per word (a count is at most TOPK < 65536).
constexpr uint32_t COUNT_WORDS = MAX_USERS / 2;

// ---- where a packed row is in DRAM ----
// The prefill stack's KV storage is [slots, 1, R, 656] fp8_e4m3 ROW_MAJOR, one row per page, pages padded to
// `stride` bytes, ND-sharded [1, 1, 32, 656] ROUND_ROBIN_1D over the chip's `BANKS` DRAM banks, slot-major
// (tensor_accessor.h get_bank_and_offset: shard = page / 32, bank = shard % BANKS, slot in bank = shard / BANKS).
// `page` = slot * R + row. Returns (bank, byte offset inside the tensor's allocation of that bank).
template <uint32_t BANKS, uint32_t SHARD_ROWS>
constexpr uint32_t kv_row_bank(uint32_t page) {
    return (page / SHARD_ROWS) % BANKS;
}
template <uint32_t BANKS, uint32_t SHARD_ROWS>
constexpr uint32_t kv_row_offset(uint32_t page, uint32_t stride) {
    return (((page / SHARD_ROWS) / BANKS) * SHARD_ROWS + page % SHARD_ROWS) * stride;
}

// Element offset of column 0 of `row` in a 32x32 tile (four 16x16 faces).
constexpr uint32_t tile_col0_offset(uint32_t row) { return (row / 16) * 2 * 256 + (row % 16) * 16; }

}  // namespace sm
