// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-load integrity probe with per-byte-lane fault attribution.
//
// The kernel pads its own .text up to KERNEL_BYTES with a known pattern, then at runtime
// verifies that pattern where it actually landed in L1 and publishes what it found. If the
// dispatcher wrote the binary incorrectly -- truncated, torn by a config-buffer wrap, shifted,
// or bit-flipped in transit -- the host sees exactly which words and which BYTE LANES differ.
//
// Pattern design, and why it is a hybrid:
//
//   * PAD_FILL (0xAAAAAAAA / 0x55555555) covers most words. Run as a complement pair those
//     toggle every bit on every reload -- maximum switching activity, and stuck-at bits in a
//     given lane show up at once. But a uniform fill cannot detect a word-offset or a
//     duplicated chunk: 0xAA shifted by N is still 0xAA.
//
//   * Every MARKER_STRIDE-th word is therefore an index marker (MARKER_MAGIC ^ i) instead of
//     the fill. Markers make offsets, duplications and torn copies visible, while costing only
//     1/256 of the payload -- 99.6% of the bytes still carry the pure lane-toggling fill.
//
// Fault attribution: the kernel knows the expected value of every word, so rather than only
// hashing, it compares word by word and accumulates PER-LANE error counts and the OR of all
// differing bits per lane. That is what turns "a kernel load went bad" into "lane 3 is bad,
// here are the bits, starting at word N".

#include <cstdint>
#include <cstddef>
#include "api/dataflow/dataflow_api.h"

#define EMPTY_KERNEL_BYTES 64

#if KERNEL_BYTES > EMPTY_KERNEL_BYTES

#define PAD_WORDS ((KERNEL_BYTES - EMPTY_KERNEL_BYTES) / 4)

// One index marker every 256 words (1 KB). Keep in sync with the host.
#define MARKER_STRIDE 256u
#define MARKER_MAGIC 0xC0DE0000u

// The single source of truth for what word i must contain. Used both to generate the pad at
// compile time and to verify it at run time, so the two can never drift apart.
static constexpr uint32_t expected_word(uint32_t i) {
    if ((i % MARKER_STRIDE) == 0u) {
        return MARKER_MAGIC ^ i;
    }
#ifdef PAD_FILL
    return PAD_FILL;
#else
    // No fill requested: index-derived pattern, every word distinct.
    return 0xA5C3963Cu ^ (i * 2654435761u);
#endif
}

template <size_t N>
struct Pad {
    uint32_t v[N];
};

template <size_t N>
static constexpr Pad<N> make_pad() {
    Pad<N> p{};
    for (size_t i = 0; i < N; i++) {
        p.v[i] = expected_word(static_cast<uint32_t>(i));
    }
    return p;
}

// Forced into .text so it inflates the *kernel binary*, which is what the dispatcher copies
// into the kernel config buffer. (This is why the assembler emits "ignoring changed section
// attributes for .text" -- data placed into an already-executable section. Harmless: the
// bytes land in .text, which is the whole intent.)
[[gnu::section(".text"), gnu::used, gnu::aligned(4)]]
static constexpr Pad<PAD_WORDS> kernel_pad = make_pad<PAD_WORDS>();

#endif

// Result layout in L1 (see the host test for the mirror of this).
//   0 hash            4 tag(fill)        8  first_bad_expected   12..15 per-lane bad counts
//   1 words checked   5 bad word count   9  lane diff OR (packed)
//   2 first word      6 first_bad_index  10 marker mismatches
//   3 last word       7 first_bad_value  11 (reserved)
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t expect_words = get_arg_val<uint32_t>(1);
    const uint32_t tag = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

#if KERNEL_BYTES > EMPTY_KERNEL_BYTES
    uint32_t hash = 0x811C9DC5u;
    uint32_t bad = 0;
    uint32_t bad_marker = 0;
    uint32_t first_bad_idx = 0xFFFFFFFFu;
    uint32_t first_bad_val = 0;
    uint32_t first_bad_exp = 0;
    uint32_t lane_bad[4] = {0, 0, 0, 0};
    uint32_t lane_diff[4] = {0, 0, 0, 0};

    const uint32_t words = (expect_words < PAD_WORDS) ? expect_words : PAD_WORDS;
    for (uint32_t i = 0; i < words; i++) {
        const uint32_t w = kernel_pad.v[i];
        const uint32_t e = expected_word(i);

        hash ^= (w & 0xFFu);
        hash *= 16777619u;
        hash ^= ((w >> 8) & 0xFFu);
        hash *= 16777619u;
        hash ^= ((w >> 16) & 0xFFu);
        hash *= 16777619u;
        hash ^= ((w >> 24) & 0xFFu);
        hash *= 16777619u;

        const uint32_t diff = w ^ e;
        if (diff != 0u) {
            if (first_bad_idx == 0xFFFFFFFFu) {
                first_bad_idx = i;
                first_bad_val = w;
                first_bad_exp = e;
            }
            bad++;
            if ((i % MARKER_STRIDE) == 0u) {
                bad_marker++;  // a bad marker means offset / duplication / torn copy
            }
            // Attribute the fault to byte lanes: lane 3 is bits 24..31.
            for (uint32_t l = 0; l < 4; l++) {
                const uint32_t lane = (diff >> (l * 8)) & 0xFFu;
                if (lane != 0u) {
                    lane_bad[l]++;
                    lane_diff[l] |= lane;
                }
            }
        }
    }

    out[0] = hash;
    out[1] = words;
    out[2] = kernel_pad.v[0];
    out[3] = kernel_pad.v[PAD_WORDS - 1];
    out[5] = bad;
    out[6] = first_bad_idx;
    out[7] = first_bad_val;
    out[8] = first_bad_exp;
    out[9] = lane_diff[0] | (lane_diff[1] << 8) | (lane_diff[2] << 16) | (lane_diff[3] << 24);
    out[10] = bad_marker;
    out[11] = 0;
    out[12] = lane_bad[0];
    out[13] = lane_bad[1];
    out[14] = lane_bad[2];
    out[15] = lane_bad[3];
#else
    for (uint32_t i = 0; i < 16; i++) {
        out[i] = 0;
    }
#endif
    out[4] = tag;  // written last: proves THIS dispatch ran and the record above is complete
}
