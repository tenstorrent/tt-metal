// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The MTP draft chained behind a verify pass on device (tt/spec_chain.py; host mirror reference/spec_chain.py), one
// core, three modes. Every buffer is a one-page interleaved DRAM tensor, read and written whole (the result page in
// 64-byte sections, one per writing mode).
//
// Mode 0, the chain step (end of the accept trace, after SampleVerify):
// - verify: the verify's packed upload, int32 [verify_rows * B]: tokens, positions P, and row chain_row, the chain
//   row: word 0 source, 1 columns, 2 limit, 3 .. 3 + S the source's slots, 3 + S relay (mode 2).
// - accept: SampleVerify's output page, uint32 [accept_words]: group 0's committed count m and tokens.
// - argmax: the verify's greedy tokens, uint32 [B].
// Writes the draft's packed upload (mtp, int32 [mtp_rows * B], reference.spec_inputs.pack_mtp's rows: tokens; fill,
// step2, step3 ring cache positions; fill, step2, step3 RoPE indices), idle unless the chain row names a source with
// 1 <= m <= columns and the bonus P[slots[m - 1]] + 1 below the limit; then, as pack_mtp(plan_mtp(...)): slot
// slots[c] (c < m) takes committed token c with its fill at P[slots[c]], and the anchor slots[m - 1] the steps at
// the bonus and the bonus + 1. Writes the result header: the accept page, the argmax and the positions of slots
// 0 .. S - 1, and the status (drafting, anchor, bonus).
//
// Mode 1, the drafts step (end of the MTP draft trace): reads mtp (written by mode 0 or copied by the host) and each
// MTP pass's greedy tokens (heads, uint32 [B]); the anchor is the slot with a step2 position. Writes the result's
// drafts section: the anchor (0xFFFFFFFF: none) and the anchor's token of each pass.
//
// Mode 2, the relay step (first in the verify trace, a chained verify): reads verify as the host copied it, the
// result page the lane's last pass left, and the accept step's control page (ctrl, uint32 [ctrl_words], group 0 the
// source's: drafts at word ctrl_drafts). Unless the chain row's relay word is set it writes nothing. Otherwise, when
// the page's chain drafted (status drafting 1 and the drafts step's anchor the chain's) and its count m and the chain
// row's columns n + 1 are in range, column c of the source (slot slots[c]) takes the last committed token (c = 0,
// accept token m - 1) or draft c at position bonus + c, and ctrl group 0's draft c - 1 the same draft; else the
// columns go idle (token 0, position -1, draft 0) and the chain row's source is cleared (no draft behind the pass).
// Writes verify and ctrl back whole.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE uint64_t dram_addr(uint32_t base, uint32_t page_size, uint32_t offset = 0) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_size};
    return get_noc_addr(0, g, offset);
}

constexpr uint32_t align64(uint32_t bytes) { return (bytes + 63) & ~63u; }

constexpr uint32_t kNone = 0xFFFFFFFFu;

}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);  // slots of a verify pass
    constexpr uint32_t D = get_compile_time_arg_val(3);  // drafts (MTP passes)
    constexpr uint32_t verify_rows = get_compile_time_arg_val(4);
    constexpr uint32_t chain_row = get_compile_time_arg_val(5);
    constexpr uint32_t mtp_rows = get_compile_time_arg_val(6);
    constexpr uint32_t accept_words = get_compile_time_arg_val(7);
    constexpr uint32_t ring = get_compile_time_arg_val(8);  // MTP_RING_TOKENS
    constexpr uint32_t argmax_at = get_compile_time_arg_val(9);
    constexpr uint32_t status_at = get_compile_time_arg_val(10);
    constexpr uint32_t drafts_at = get_compile_time_arg_val(11);
    constexpr uint32_t result_words = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(13);
    constexpr uint32_t positions_at = get_compile_time_arg_val(14);
    constexpr uint32_t chain_relay = get_compile_time_arg_val(15);  // the chain row's relay word
    constexpr uint32_t ctrl_words = get_compile_time_arg_val(16);
    constexpr uint32_t ctrl_drafts = get_compile_time_arg_val(17);  // group 0's first draft word
    static_assert(S <= B && B * 4 <= 1024 && 3 + S <= B && mtp_rows == 7 && chain_row < verify_rows);
    static_assert(chain_relay == 3 + S && chain_relay < B && ctrl_drafts + D <= ctrl_words);
    static_assert(accept_words <= argmax_at && argmax_at + S <= positions_at && positions_at + S <= status_at);
    static_assert(status_at + 4 <= drafts_at && 1 + D <= accept_words);
    static_assert(drafts_at + 1 + D <= result_words && drafts_at % 16 == 0 && result_words % 16 == 0);
    constexpr uint32_t result_bytes = result_words * 4;
    constexpr uint32_t mtp_bytes = mtp_rows * B * 4;
    constexpr uint32_t row_bytes = B * 4;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t base = get_write_ptr(cb_scratch);

    if constexpr (mode == 0) {
        const uint32_t verify_addr = get_arg_val<uint32_t>(0);
        const uint32_t accept_addr = get_arg_val<uint32_t>(1);
        const uint32_t argmax_addr = get_arg_val<uint32_t>(2);
        const uint32_t mtp_addr = get_arg_val<uint32_t>(3);
        const uint32_t result_addr = get_arg_val<uint32_t>(4);
        constexpr uint32_t verify_bytes = verify_rows * B * 4;
        constexpr uint32_t accept_bytes = accept_words * 4;
        constexpr uint32_t header_bytes = drafts_at * 4;
        const uint32_t verify_l1 = base;
        const uint32_t accept_l1 = verify_l1 + align64(verify_bytes);
        const uint32_t argmax_l1 = accept_l1 + align64(accept_bytes);
        const uint32_t mtp_l1 = argmax_l1 + align64(row_bytes);
        const uint32_t header_l1 = mtp_l1 + align64(mtp_bytes);
        noc_async_read(dram_addr(verify_addr, verify_bytes), verify_l1, verify_bytes);
        noc_async_read(dram_addr(accept_addr, accept_bytes), accept_l1, accept_bytes);
        noc_async_read(dram_addr(argmax_addr, row_bytes), argmax_l1, row_bytes);
        noc_async_read_barrier();

        const int32_t* pos = reinterpret_cast<const int32_t*>(verify_l1) + B;
        const int32_t* chain = reinterpret_cast<const int32_t*>(verify_l1) + chain_row * B;
        const uint32_t* accept = reinterpret_cast<const uint32_t*>(accept_l1);
        const uint32_t* argmax = reinterpret_cast<const uint32_t*>(argmax_l1);
        volatile tt_l1_ptr int32_t* mtp = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mtp_l1);
        volatile tt_l1_ptr uint32_t* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header_l1);

        // pack_mtp's idle draft: tokens 0, cache positions -1, RoPE indices 0.
        for (uint32_t i = 0; i < mtp_rows * B; ++i) {
            mtp[i] = (i >= B && i < 4 * B) ? -1 : 0;
        }
        for (uint32_t i = 0; i < drafts_at; ++i) {
            header[i] = 0;
        }
        for (uint32_t i = 0; i < accept_words; ++i) {
            header[i] = accept[i];
        }
        for (uint32_t s = 0; s < S; ++s) {
            header[argmax_at + s] = argmax[s];
            header[positions_at + s] = static_cast<uint32_t>(pos[s]);
        }
        header[status_at + 1] = kNone;

        auto cache = [](int32_t t) -> int32_t { return t >= 0 ? static_cast<int32_t>(ring + t % ring) : -1; };
        const uint32_t m = accept[0];
        const uint32_t columns = static_cast<uint32_t>(chain[1]);
        if (chain[0] != 0 && m >= 1 && m <= columns && columns <= S) {
            const uint32_t anchor = static_cast<uint32_t>(chain[3 + m - 1]);
            const int32_t bonus = pos[anchor] + 1;
            if (bonus < chain[2]) {
                for (uint32_t c = 0; c < m; ++c) {
                    const uint32_t s = static_cast<uint32_t>(chain[3 + c]);
                    const int32_t t = pos[s];
                    mtp[s] = static_cast<int32_t>(accept[1 + c]);
                    mtp[B + s] = cache(t);
                    mtp[4 * B + s] = t > 0 ? t : 0;
                }
                mtp[2 * B + anchor] = cache(bonus);
                mtp[3 * B + anchor] = cache(bonus + 1);
                mtp[5 * B + anchor] = bonus;
                mtp[6 * B + anchor] = bonus + 1;
                header[status_at] = 1;
                header[status_at + 1] = anchor;
                header[status_at + 2] = static_cast<uint32_t>(bonus);
            }
        }
        noc_async_write(mtp_l1, dram_addr(mtp_addr, mtp_bytes), mtp_bytes);
        noc_async_write(header_l1, dram_addr(result_addr, result_bytes), header_bytes);
        noc_async_write_barrier();
    } else if constexpr (mode == 2) {
        const uint32_t verify_addr = get_arg_val<uint32_t>(0);
        const uint32_t result_addr = get_arg_val<uint32_t>(1);
        const uint32_t ctrl_addr = get_arg_val<uint32_t>(2);
        constexpr uint32_t verify_bytes = verify_rows * B * 4;
        constexpr uint32_t ctrl_bytes = ctrl_words * 4;
        const uint32_t verify_l1 = base;
        const uint32_t result_l1 = verify_l1 + align64(verify_bytes);
        const uint32_t ctrl_l1 = result_l1 + align64(result_bytes);
        noc_async_read(dram_addr(verify_addr, verify_bytes), verify_l1, verify_bytes);
        noc_async_read(dram_addr(result_addr, result_bytes), result_l1, result_bytes);
        noc_async_read(dram_addr(ctrl_addr, ctrl_bytes), ctrl_l1, ctrl_bytes);
        noc_async_read_barrier();

        volatile tt_l1_ptr int32_t* tok = reinterpret_cast<volatile tt_l1_ptr int32_t*>(verify_l1);
        volatile tt_l1_ptr int32_t* pos = tok + B;
        volatile tt_l1_ptr int32_t* chain = tok + chain_row * B;
        const uint32_t* page = reinterpret_cast<const uint32_t*>(result_l1);
        volatile tt_l1_ptr uint32_t* ctrl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_l1);
        if (chain[chain_relay] != 0) {
            const uint32_t columns = static_cast<uint32_t>(chain[1]);
            const uint32_t m = page[0];  // group 0's committed count (the accept page starts the result page)
            const bool ok = page[status_at] == 1 && page[drafts_at] == page[status_at + 1] && columns >= 1 &&
                            columns <= 1 + D && m >= 1 && m <= 1 + D;
            const int32_t bonus = static_cast<int32_t>(page[status_at + 2]);
            const uint32_t n = columns < S ? columns : S;
            for (uint32_t c = 0; c < n; ++c) {
                const uint32_t s = static_cast<uint32_t>(chain[3 + c]);
                if (s >= S) {
                    continue;  // not a slot of the pass (the host lays the source's columns into slots 0 .. S - 1)
                }
                const uint32_t token = ok ? (c == 0 ? page[m] : page[drafts_at + c]) : 0;
                tok[s] = static_cast<int32_t>(token);
                pos[s] = ok ? bonus + static_cast<int32_t>(c) : -1;
                if (c > 0) {
                    ctrl[ctrl_drafts + c - 1] = ok ? page[drafts_at + c] : 0;
                }
            }
            if (!ok) {
                chain[0] = 0;
            }
            noc_async_write(verify_l1, dram_addr(verify_addr, verify_bytes), verify_bytes);
            noc_async_write(ctrl_l1, dram_addr(ctrl_addr, ctrl_bytes), ctrl_bytes);
            noc_async_write_barrier();
        }
    } else {
        const uint32_t mtp_addr = get_arg_val<uint32_t>(0);
        const uint32_t result_addr = get_arg_val<uint32_t>(1);
        constexpr uint32_t drafts_bytes = (result_words - drafts_at) * 4;
        const uint32_t mtp_l1 = base;
        const uint32_t heads_l1 = mtp_l1 + align64(mtp_bytes);
        const uint32_t out_l1 = heads_l1 + D * align64(row_bytes);
        noc_async_read(dram_addr(mtp_addr, mtp_bytes), mtp_l1, mtp_bytes);
        for (uint32_t k = 0; k < D; ++k) {
            const uint32_t head_addr = get_arg_val<uint32_t>(2 + k);
            noc_async_read(dram_addr(head_addr, row_bytes), heads_l1 + k * align64(row_bytes), row_bytes);
        }
        noc_async_read_barrier();

        const int32_t* step2 = reinterpret_cast<const int32_t*>(mtp_l1) + 2 * B;
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
        uint32_t anchor = kNone;
        for (uint32_t s = 0; s < S; ++s) {
            if (step2[s] >= 0) {
                anchor = s;
                break;
            }
        }
        for (uint32_t i = 0; i < result_words - drafts_at; ++i) {
            out[i] = 0;
        }
        out[0] = anchor;
        for (uint32_t k = 0; k < D; ++k) {
            const uint32_t* head = reinterpret_cast<const uint32_t*>(heads_l1 + k * align64(row_bytes));
            out[1 + k] = anchor == kNone ? kNone : head[anchor];
        }
        noc_async_write(out_l1, dram_addr(result_addr, result_bytes, drafts_at * 4), drafts_bytes);
        noc_async_write_barrier();
    }
}
