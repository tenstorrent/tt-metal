// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-core summary every ND stage kernel writes to its result shard. The host parses the same layout in
// nd_stage_lib.py (the W_* constants); keep the two in step.

#pragma once

#include <cstdint>

namespace nd_stage {

constexpr uint32_t MAGIC_CHECKER = 0x4E445354;  // "NDST"
constexpr uint32_t MAGIC_L1 = 0x4E444C31;       // "NDL1"
constexpr uint32_t MAGIC_NOC = 0x4E444E43;      // "NDNC"
constexpr uint32_t MAGIC_DEST = 0x4E444453;     // "NDDS"
constexpr uint32_t NO_ITER = 0xFFFFFFFF;
constexpr uint32_t MAX_SAMPLES = 4;

enum Word : uint32_t {
    W_MAGIC = 0,
    W_ITERS_DONE,
    W_ITERS_CHECKED,
    W_BAD_ITERS,
    W_BAD_WORDS,
    W_FIRST_BAD_ITER,
    W_FIRST_BAD_WORD,
    W_FIRST_GOT,
    W_FIRST_REF,
    W_LAST_BAD_ITER,
    W_XOR_UNION,
    W_ROW_MASK,
    W_COL_MASK,
    W_TILE_MASK,
    W_ARC_START,
    W_ARC_END,
    W_CYCLES_LO,
    W_CYCLES_HI,
    W_N_SAMPLES,
    W_CORE_ID,                   // the host's index for this core, so it can check which shard came from which core
    W_BIT_HIST,                  // 8 words: a saturating u8 count per element bit position of every flipped bit
    W_SAMPLES = W_BIT_HIST + 8,  // MAX_SAMPLES x (iter, word, got, ref)
    W_COUNT = W_SAMPLES + 4 * MAX_SAMPLES,
};

// Counters live in RISC-local memory and reach L1 once, in finish(): an L1 read-modify-write would go via
// the data cache, and the magic is written last so a kernel that never finished leaves none.
class Result {
public:
    Result(uint32_t addr, uint32_t magic, uint32_t core_id) : addr_(addr), magic_(magic) {
        for (uint32_t i = 0; i < W_COUNT; ++i) {
            w_[i] = 0;
        }
        w_[W_CORE_ID] = core_id;
        w_[W_FIRST_BAD_ITER] = NO_ITER;
        w_[W_LAST_BAD_ITER] = NO_ITER;
        w_[W_ARC_START] = arc_scratch();
        t0_ = get_timestamp();
    }

    // word: index into the compared region; words_per_tile 0 means the region has no tile structure.
    template <uint32_t elem_shift>
    void record_mismatch(uint32_t iter, uint32_t word, uint32_t got, uint32_t want, uint32_t words_per_tile) {
        const uint32_t x = got ^ want;
        ++w_[W_BAD_WORDS];
        if (w_[W_FIRST_BAD_ITER] == NO_ITER) {
            w_[W_FIRST_BAD_ITER] = iter;
            w_[W_FIRST_BAD_WORD] = word;
            w_[W_FIRST_GOT] = got;
            w_[W_FIRST_REF] = want;
        }
        w_[W_LAST_BAD_ITER] = iter;
        if (words_per_tile != 0) {
            w_[W_TILE_MASK] |= 1u << ((word / words_per_tile) & 31);
            const uint32_t in_tile = word % words_per_tile;
            if constexpr (elem_shift == 0) {
                mark_element(in_tile);
            } else {
                if (x & 0xFFFF) {
                    mark_element(in_tile * 2);
                }
                if (x >> 16) {
                    mark_element(in_tile * 2 + 1);
                }
            }
        }
        // 16-bit elements fold both halves so the union and histogram read as element bit positions.
        if constexpr (elem_shift == 0) {
            w_[W_XOR_UNION] |= x;
            count_bits(x);
        } else {
            w_[W_XOR_UNION] |= (x & 0xFFFF) | (x >> 16);
            count_bits(x & 0xFFFF);
            count_bits(x >> 16);
        }
        const uint32_t n = w_[W_N_SAMPLES];
        if (n < MAX_SAMPLES) {
            w_[W_SAMPLES + 4 * n + 0] = iter;
            w_[W_SAMPLES + 4 * n + 1] = word;
            w_[W_SAMPLES + 4 * n + 2] = got;
            w_[W_SAMPLES + 4 * n + 3] = want;
            w_[W_N_SAMPLES] = n + 1;
        }
    }

    void end_iteration(bool checked, bool bad, uint32_t iter) {
        w_[W_ITERS_DONE] = iter + 1;
        if (checked) {
            ++w_[W_ITERS_CHECKED];
            if (bad) {
                ++w_[W_BAD_ITERS];
            }
        }
    }

    void finish() {
        const uint64_t cycles = get_timestamp() - t0_;
        w_[W_CYCLES_LO] = static_cast<uint32_t>(cycles);
        w_[W_CYCLES_HI] = static_cast<uint32_t>(cycles >> 32);
        w_[W_ARC_END] = arc_scratch();
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr_);
        for (uint32_t i = 1; i < W_COUNT; ++i) {
            out[i] = w_[i];
        }
        out[W_MAGIC] = magic_;
    }

private:
    void count_bits(uint32_t x) {
        for (uint32_t b = 0; x != 0; ++b, x >>= 1) {
            if (x & 1) {
                uint32_t& word = w_[W_BIT_HIST + b / 4];
                const uint32_t shift = (b % 4) * 8;
                if (((word >> shift) & 0xFF) != 0xFF) {
                    word += 1u << shift;
                }
            }
        }
    }

    // Tile element e -> (row, col): four 16x16 faces in row-major face order, each row-major.
    void mark_element(uint32_t e) {
        const uint32_t face = e >> 8;
        const uint32_t row = (face >> 1) * 16 + ((e >> 4) & 15);
        const uint32_t col = (face & 1) * 16 + (e & 15);
        w_[W_ROW_MASK] |= 1u << row;
        w_[W_COL_MASK] |= 1u << col;
    }

    // The word ARC firmware writes for the Tensix dynamic throttle (odd = throttled).
    static uint32_t arc_scratch() {
#ifdef MEM_L1_ARC_FW_SCRATCH
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
#else
        return 0;
#endif
    }

    uint32_t addr_;
    uint32_t magic_;
    uint64_t t0_;
    uint32_t w_[W_COUNT];
};

}  // namespace nd_stage
