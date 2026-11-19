// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill megakernel (PMK) — the SEGMENT TABLE and the cache-write constants shared by every phase kernel that needs a
// row's slot / position (tt/pmk_cache.py SegTable is the host-side twin: the two encode the same words).
//
// A PMK replay runs T packed rows (v1: one request of <= T rows at rows [0, n_valid); v2: K segments back to back,
// fresh or continuation, each starting on a SEG_ALIGN = 128-row boundary). The table is ONE ROW_MAJOR uint32 page
// (DRAM or L1) the host writes before every replay, so one captured program serves every (slot, position, length)
// mix. Every kernel that needs it does one NoC read of table_bytes(NSEG_MAX) into its scratch, then reads fields
// through SegTable.
//
//   header (8 words): MAGIC, n_seg, t_rows, align (128), block (64: latent page rows), pool (128: pooled-key page
//                     tokens), flags, 0
//   segment k (8 words at 8 + 8 k): slot          request slot = decode row (SEG_NONE: an empty entry)
//                                   row0          first packed row (multiple of align)
//                                   n_valid       token rows: [row0, row0 + n_valid)
//                                   pos0          position of row0 in the request (chunk start; 0 = fresh)
//                                   page_row      row of the page table that holds the slot's pages (== slot unless
//                                                 the caller's table is ordered otherwise)
//                                   flags         F_SAMPLE (the head samples the segment's last valid row), F_FRESH
//                                                 (KDA state from zero), F_CONT (attends the slot's cached prefix)
//                                   n_rows        the segment's padded span (multiple of align): rows [row0 + n_valid,
//                                                 row0 + n_rows) are padding
//                                   0
// Rows outside every segment are padding (pad pages follow the NULL-block rule: never a slot's page).
#pragma once

#include <cstdint>

namespace pmk {
constexpr uint32_t SEG_MAGIC = 0x31474553u;  // "SEG1"
constexpr uint32_t SEG_NONE = 0xFFFFFFFFu;
constexpr uint32_t NULL_BLOCK = 0;  // vLLM's null block: never a request's page (tt/prefill_tp.py NULL_BLOCK)
constexpr uint32_t HDR_WORDS = 8, SEG_WORDS = 8;
constexpr uint32_t H_MAGIC = 0, H_NSEG = 1, H_TROWS = 2, H_ALIGN = 3, H_BLOCK = 4, H_POOL = 5, H_FLAGS = 6;
constexpr uint32_t S_SLOT = 0, S_ROW0 = 1, S_NVALID = 2, S_POS0 = 3, S_PTROW = 4, S_FLAGS = 5, S_NROWS = 6;
constexpr uint32_t F_SAMPLE = 1, F_FRESH = 2, F_CONT = 4;
constexpr uint32_t TILE_BF16 = 2048, TILE_FP32 = 4096;

constexpr uint32_t table_bytes(uint32_t nseg_max) { return 4 * (HDR_WORDS + SEG_WORDS * nseg_max); }
constexpr uint32_t align64(uint32_t b) { return (b + 63) & ~63u; }

// byte offset of row r's left 16-element face row inside a 32 x 32 bf16 tile (the right half is + 512) — the same as
// kernels/mla/mla_common.hpp mla::row_off (repeated here so this header stays free of the MLA includes)
constexpr uint32_t row_off_bf16(uint32_t r) { return (r >> 4) * 1024 + (r & 15) * 32; }

// kernel-side accessor over the table's words in L1 (loaded by the caller, after its read barrier + invalidate)
struct SegTable {
    volatile tt_l1_ptr uint32_t* w;

    uint32_t n() const { return w[H_NSEG]; }
    uint32_t t_rows() const { return w[H_TROWS]; }
    bool ok() const { return w[H_MAGIC] == SEG_MAGIC; }
    uint32_t seg(uint32_t k, uint32_t field) const { return w[HDR_WORDS + SEG_WORDS * k + field]; }
    // the segment whose padded span holds packed row r, else SEG_NONE (rows outside every segment are padding)
    uint32_t seg_of_row(uint32_t r) const {
        const uint32_t n_ = n();
        for (uint32_t k = 0; k < n_; ++k) {
            const uint32_t row0 = seg(k, S_ROW0);
            if (seg(k, S_SLOT) != SEG_NONE && r >= row0 && r - row0 < seg(k, S_NROWS)) {
                return k;
            }
        }
        return SEG_NONE;
    }
    // packed row r of segment k carries a token (r inside the padded span assumed)
    bool valid(uint32_t k, uint32_t r) const { return r - seg(k, S_ROW0) < seg(k, S_NVALID); }
    // the request position of packed row r of segment k
    uint32_t pos(uint32_t k, uint32_t r) const { return seg(k, S_POS0) + (r - seg(k, S_ROW0)); }
};

// Hand-off rule (prefill_megakernel.md §4b): block counters are monotonic per 32-row block; a consumer waits until the
// counter is >= k, never on equality, and never resets it (reset once at layer start by the family).
FORCE_INLINE void wait_min(uint32_t sem_id, uint32_t k) {
    if (k == 0) {
        return;
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id)), k);
}
// WAIT_MODE of a write kernel: 0 = no wait (standalone unit / the producer finished in an earlier program), 1 = one
// wait for WAIT_COUNT at phase start (e.g. the KDA scan chains' final-state counter), 2 = per 32-row source block
// (the latent / [k | g] rows land block by block: page i waits for its last row-tile's block count)
constexpr uint32_t WAIT_NONE = 0, WAIT_ALL = 1, WAIT_BLOCKS = 2;

// the table's words: a DRAM / L1 tensor page read into `l1` (SEGTAB_CB defined: already resident in that CB at
// SEGTAB_OFF, e.g. beside the stack's ARG CB 30 on the owners; no read)
template <typename Acc>
FORCE_INLINE volatile tt_l1_ptr uint32_t* load_segtab(const Acc& tab, uint32_t l1) {
#if defined(SEGTAB_CB)
    (void)tab;
    (void)l1;
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(SEGTAB_CB) + SEGTAB_OFF);
#else
    noc_async_read_page(0, tab, l1);
    noc_async_read_barrier();
    invalidate_l1_cache();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
#endif
}
}  // namespace pmk
