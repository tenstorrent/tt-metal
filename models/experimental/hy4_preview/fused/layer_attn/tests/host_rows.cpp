// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Host check of the 16-row form of program P2 (tests/test_rows_host.py compiles and runs this): the parties of X4 as
// the kernels run them, built on the kernels' own header (kernels/attn_rows.hpp), against each other for every row
// mask. Per mask, on a chip with `sources` column partners and `row_sources` row partners:
//   attention cores  (sm_attn_io.cpp, SM_LAYER_ROWS) one count in byte p of a group's word per live row 8 p + b
//   coordinator      (attn_coord16.cpp) per sub-pass: needs field p of C_GROUP[g] == live_count; fills the payload
//                    with for_each_live; hands the op over with live_count partials at landing offset p (0 bytes
//                    when the sub-pass has no live row); the row op likewise once field p of C_ROW is complete
//   sender           (collective_sender.cpp, hand-over words) counts an arrival on the landing iff bytes != 0, in
//                    the order of the hand-overs
//   relay            (attn_relay16.cpp) the n-th arrival is sub-pass arrival_subpass(mask, n); bit
//                    arrival_bit(p, source) to the merge cores for_each_live(mask, p) names; expects arrivals(mask)
//   merge core i     (attn_merge16_dm.cpp) for p with live(mask, p, i): waits for byte p of A_LOCAL, for the bits
//                    arrival_bit(p, s) of every source, reads the partial at landing slot p, place(mask, p, i);
//                    writes its result at row staging place and counts byte p of C_ROW; counts A_CTX once
//   post hub         (attn_post_hub.cpp) waits for live_rows(mask) counts of A_CTX
// Checked: every wait is met exactly (no count or bit is waited for that nobody sends, none is left over), every
// partial a merge core reads is the one of its own row and sub-pass from the right source, a payload never exceeds
// its slot, and the merge kernel is told as many chains as the data-movement kernel runs.
// Output: "masks <n> waits <n> partials <n> checksum <n>", or the first disagreement and exit 1.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../kernels/attn_rows.hpp"

namespace {
constexpr uint32_t PART = 8704, GROUP_BYTES = 8 * PART;
struct Tag {
    int source = -1, p = -1, i = -1;  // which chip's partial of which row
};
[[noreturn]] void fail(uint32_t mask, const char* what, long a = 0, long b = 0) {
    std::printf("mask %#06x: %s (%ld, %ld)\n", mask, what, a, b);
    std::exit(1);
}
}  // namespace

int main(int argc, char** argv) {
    const uint32_t sources = argc > 1 ? std::atoi(argv[1]) : 7;
    uint64_t waits = 0, partials = 0, checksum = 0;
    for (uint32_t mask = 0; mask < (1u << hy4r::TOKEN_ROWS); ++mask) {
        // ---- attention cores: counts into a group's word on the sender core, and A_LOCAL of merge core b ----
        uint32_t c_group = 0;
        uint32_t a_local[hy4r::SUB] = {0};
        for (uint32_t b = 0; b < hy4r::SUB; ++b) {
            for (uint32_t p = 0; p < 2; ++p) {
                if ((mask >> (b + 8 * p)) & 1u) {  // (sm_attn_io.cpp: a live row, with rows to attend or empty)
                    c_group += 1u << (8 * p);
                    a_local[b] += 1u << (8 * p);
                }
            }
        }
        // ---- every source chip's coordinator and sender, for the op towards this chip ----
        // landing[source] = the landing core's slots: [sub-pass][place] -> tag; arrival counter per landing
        std::vector<std::vector<Tag>> landing(sources, std::vector<Tag>(hy4r::SUBPASSES * hy4r::SUB));
        std::vector<uint32_t> arrived(sources, 0);
        std::vector<std::vector<uint32_t>> order(sources);  // sub-pass of each arrival, in order
        for (uint32_t s = 0; s < sources; ++s) {
            uint32_t word = c_group;  // the same mask on every chip
            for (uint32_t p = 0; p < hy4r::SUBPASSES; ++p) {
                const uint32_t n = hy4r::live_count(mask, p);
                if (hy4r::field(word, p) < n) {
                    fail(mask, "the coordinator waits for partials no attention core counts", p, n);
                }
                ++waits;
                word -= n * hy4r::one(p);
                std::vector<Tag> staging(hy4r::SUB);
                uint32_t filled = 0;
                hy4r::for_each_live(mask, p, [&](uint32_t i, uint32_t k) {
                    if (k >= hy4r::SUB || staging[k].i != -1) {
                        fail(mask, "a staging place is used twice or is outside the slot", p, k);
                    }
                    staging[k] = Tag{static_cast<int>(s), static_cast<int>(p), static_cast<int>(i)};
                    ++filled;
                });
                const uint32_t bytes = n * PART, offset = p * GROUP_BYTES;
                if (filled != n || bytes > GROUP_BYTES) {
                    fail(mask, "the payload is not the live rows", filled, n);
                }
                if (bytes != 0) {  // the sender: copy, count
                    for (uint32_t k = 0; k < bytes / PART; ++k) {
                        landing[s][offset / GROUP_BYTES * hy4r::SUB + k] = staging[k];
                    }
                    ++arrived[s];
                    order[s].push_back(p);
                }
            }
            if (word != 0) {
                fail(mask, "counts are left in a group's word", word);
            }
        }
        // ---- the relays of this chip: one per source ----
        uint32_t colmask[hy4r::SUB] = {0};
        for (uint32_t s = 0; s < sources; ++s) {
            if (arrived[s] != hy4r::arrivals(mask)) {
                fail(
                    mask,
                    "the relay expects another number of arrivals than the sender counts",
                    arrived[s],
                    hy4r::arrivals(mask));
            }
            for (uint32_t n = 0; n < arrived[s]; ++n) {
                const uint32_t p = hy4r::arrival_subpass(mask, n);
                if (p != order[s][n]) {
                    fail(mask, "the relay takes an arrival for another sub-pass", p, order[s][n]);
                }
                hy4r::for_each_live(mask, p, [&](uint32_t i, uint32_t) { colmask[i] += hy4r::arrival_bit(p, s); });
                ++waits;
            }
            if (hy4r::arrival_subpass(mask, arrived[s]) != hy4r::SUBPASSES) {
                fail(mask, "arrival_subpass names a sub-pass beyond the last arrival");
            }
        }
        // ---- the merge cores ----
        uint32_t c_row = 0, a_ctx = 0;
        std::vector<Tag> row_staging(hy4r::SUBPASSES * hy4r::SUB);
        for (uint32_t i = 0; i < hy4r::SUB; ++i) {
            uint32_t ran = 0;
            for (uint32_t p = 0; p < hy4r::SUBPASSES; ++p) {
                if (!hy4r::live(mask, p, i)) {
                    continue;
                }
                if (hy4r::field(a_local[i], p) != 1) {
                    fail(mask, "a merge core waits for a local partial nobody counts", i, p);
                }
                a_local[i] -= hy4r::one(p);
                ++waits;
                for (uint32_t s = 0; s < sources; ++s) {
                    if ((colmask[i] & hy4r::arrival_bit(p, s)) == 0) {
                        fail(mask, "a merge core waits for an arrival bit nobody sets", i, s);
                    }
                    colmask[i] -= hy4r::arrival_bit(p, s);
                    ++waits;
                    const Tag t = landing[s][p * hy4r::SUB + hy4r::place(mask, p, i)];
                    if (t.source != static_cast<int>(s) || t.p != static_cast<int>(p) || t.i != static_cast<int>(i)) {
                        fail(mask, "a merge core reads another row's partial", i, t.i);
                    }
                    ++partials;
                    checksum = checksum * 1000003u + (mask * 131u + s * 17u + p * 8u + i);
                }
                const uint32_t k = hy4r::place(mask, p, i);
                if (row_staging[p * hy4r::SUB + k].i != -1) {
                    fail(mask, "two merge cores write the same place of the row staging", p, k);
                }
                row_staging[p * hy4r::SUB + k] = Tag{0, static_cast<int>(p), static_cast<int>(i)};
                c_row += hy4r::one(p);
                ++a_ctx;
                ++ran;
            }
            if (ran != hy4r::chains(mask, i)) {
                fail(mask, "the merge kernel is told another number of chains", ran, hy4r::chains(mask, i));
            }
            if (a_local[i] != 0 || colmask[i] != 0) {
                fail(mask, "a merge core leaves counts or arrival bits", a_local[i], colmask[i]);
            }
        }
        // ---- the coordinator's row op, and the post hub ----
        for (uint32_t p = 0; p < hy4r::SUBPASSES; ++p) {
            const uint32_t n = hy4r::live_count(mask, p);
            if (hy4r::field(c_row, p) != n) {
                fail(mask, "the coordinator's row wait does not match the merge cores' counts", p, n);
            }
            for (uint32_t k = 0; k < n; ++k) {
                if (row_staging[p * hy4r::SUB + k].i == -1) {
                    fail(mask, "a hole in the row payload", p, k);
                }
            }
            ++waits;
        }
        if (a_ctx != hy4r::live_rows(mask)) {
            fail(mask, "the post hub waits for another number of merge cores", a_ctx, hy4r::live_rows(mask));
        }
        ++waits;
    }
    std::printf(
        "masks %u waits %llu partials %llu checksum %llu\n",
        1u << hy4r::TOKEN_ROWS,
        static_cast<unsigned long long>(waits),
        static_cast<unsigned long long>(partials),
        static_cast<unsigned long long>(checksum));
    return 0;
}
