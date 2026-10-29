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
constexpr uint32_t kTokActive = 9;      // token state word: bit u = row u runs this step (rows mode)
constexpr uint32_t kTokPreserve = 10;   // token state word: 1 = inactive rows hold live state (rows mode)
constexpr uint32_t kTokHeadBytes = 48;  // token state words 0 .. 11, read at once

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
