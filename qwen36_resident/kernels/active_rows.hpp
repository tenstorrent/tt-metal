// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-user rows mode (named compile-time arg rows_mode; every kernel of a launch gets it). A launch's rows are
// independent users: row u of the token state runs this step only when bit u of the active word is set.
// - Attention and the GDN heads skip inactive rows: no state, KV or conv history is read or written for them, and
//   their mixer outputs stay zero, so their activations in the streamers stay finite.
// - The streamers still compute every row; inactive rows' results are ignored.
// - The conv history is in canonical order: slot j holds the j-th oldest input, and every step writes
//   (h1, h2, y). There is no ring shared by the rows.
// - Preserve word 1 (a masked launch beside rows with live state, e.g. a new user's first token): the history is
//   written back for the active rows only, face row by face row, so inactive rows keep theirs.
//   Otherwise whole tiles are written, and inactive rows must hold no state.
// Off (0): every row runs every step, and the history is the shared ring (streamer_common.hpp conv_step).
#pragma once

#include <stdint.h>

constexpr bool rows_mode = get_named_compile_time_arg_val("rows_mode") != 0;
// row groups of the launch (ResidentModel groups=): groups x batch rows share one weight stream. 1 = one group.
constexpr uint32_t groups = get_named_compile_time_arg_val("groups");
static_assert(groups >= 1 && groups <= 4, "a launch carries 1 to 4 row groups");
static_assert(groups == 1 || rows_mode, "row groups are a rows-mode feature");
constexpr uint32_t kTokActive = 9;      // token state word: bit u = row u runs this step (rows mode)
constexpr uint32_t kTokPreserve = 10;   // token state word: 1 = inactive rows hold live state (rows mode)
constexpr uint32_t kTokHeadBytes = 48;  // token state words 0 .. 11, read at once
// the rows' positions: with one row group words 1 .. batch of the head; with more (whose rows outnumber the head's
// free words) one word per global row g * batch + u at byte tok_pos_off of the token state (past its x0 / rope)
constexpr uint32_t tok_pos_off = get_named_compile_time_arg_val("tok_pos_off");
// per row group: the mixers' row-buffer block (row_tiles batch x 32 tiles) and the streamers' o_in block (o_tiles)
constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
constexpr uint32_t o_tiles = get_named_compile_time_arg_val("o_tiles");

// the rows that run this step, in order (rows mode: the set bits of mask; otherwise all of them)
template <uint32_t B>
struct ActiveRows {
    uint32_t n = 0;
    uint32_t row[B];

    FORCE_INLINE explicit ActiveRows(uint32_t mask) {
        for (uint32_t u = 0; u < B; u++) {
            if (!rows_mode || ((mask >> u) & 1)) {
                row[n++] = u;
            }
        }
    }
};

// row groups: the groups with an active row this step (rows mode; bit g of the mask has rows g * batch_rows ..). A step
// runs only those groups' streamer work, hub sub-rounds and transfers; with no active row at all group 0 runs (an idle
// launch, e.g. the trace warm-up). Every kernel derives the same list from the token state's active word.
struct ActiveGroups {
    uint32_t n = 0;
    uint32_t g[groups];
    uint32_t mask = 0;

    FORCE_INLINE ActiveGroups(uint32_t active, uint32_t batch_rows) {
        for (uint32_t gr = 0; gr < groups; gr++) {
            const uint32_t rows = batch_rows >= 32 ? ~0u : ((1u << batch_rows) - 1);
            if (groups == 1 || ((active >> (gr * batch_rows)) & rows) != 0) {
                g[n++] = gr;
                mask |= 1u << gr;
            }
        }
        if (n == 0) {
            g[n++] = 0;
            mask = 1;
        }
    }
    FORCE_INLINE bool has(uint32_t gr) const { return (mask >> gr) & 1; }
};
