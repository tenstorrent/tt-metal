// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The draft chained behind a verify pass on device (tt/spec_chain.py; host mirror reference/spec_chain.py), one core,
// five modes. Every buffer is a one-page interleaved DRAM tensor, read and written whole (the result page in 64-byte
// sections, one per writing mode).
//
// A pass's S slots form G draft groups of GS = S / G slots (G = 1: one group, the whole pass), each with at most one
// request (its source) and its own drafter state; group g's request commits accept group g. With adaptive groups
// (reference/spec_decode.py ADAPTIVE: G = 4 groups of up to GS = 4 slots over S = 8) a pass's groups are all of one
// width w (2 or 4) that travels with them: the chain block's source word and the description's drafting word hold w,
// and group g holds slots g * w .. g * w + w - 1 (otherwise w = GS). Per group g:
// - the chain row's block at g * (4 + GS): source, columns n + 1, limit, the source's slots [GS] (-1 past its
//   columns), relay; the MTP history's slots (kernels/mtp_history.cpp) follow the G blocks;
// - the result page's status at status_at + 4 g (drafting, anchor, bonus, unused) and drafts section at
//   drafts_at + g * (1 + D) (the drafts step's anchor, the drafts);
// - every layer: the commit's description in mtp row g, the group's slots g * GS .. g * GS + GS - 1, its anchor the
//   last of them; DFlash: its block rows g * dblock .., its ring rows g * dring ...
//
// Mode 0, the chain step (end of the accept trace, after the accept step):
// - verify: the verify's packed upload, int32 [verify_rows * B]: tokens, positions P, and row chain_row, the chain row.
// - accept: the accept page, uint32 [accept_words]: group g's committed count m and tokens at g * out_words.
// - argmax: the verify's greedy tokens, uint32 [B].
// Writes the draft's packed upload, idle except for the groups whose block names a source with 1 <= m <= columns and
// the bonus P[slots[m - 1]] + 1 below the limit. One layer (planned == 0, G == 1): reference.spec_inputs pack_mtp's
// rows (tokens; fill, step2, step3 ring cache positions; fill, step2, step3 RoPE indices) as pack_mtp(plan_mtp(...)):
// slot slots[c] (c < m) takes committed token c with its fill at P[slots[c]], and the anchor slots[m - 1] the steps at
// the bonus and the bonus + 1. Planned (every layer, or one layer of one-draft groups): pack_mtp_layers' description
// of each group's commit (drafting, m, A = P[slots[0]], the slots, the committed tokens). Writes the result header:
// the accept page, the argmax and the positions of slots 0 .. S - 1, and each group's status.
//
// Mode 1, the drafts step (end of the draft trace): reads the draft's upload (written by mode 0 or copied by the host)
// and each MTP pass's greedy tokens (heads, uint32 [B]); a group's anchor is the slot with a step2 position (one
// layer) or its last slot when its description drafts (planned). Writes each group's drafts section: the anchor
// (0xFFFFFFFF: none) and the anchor's token of each pass.
//
// Mode 2, the relay step (first in the verify trace, a chained verify): reads verify as the host copied it, the
// result page the lane's last pass left, and the accept step's control page (ctrl, uint32 [ctrl_words], group g's
// drafts at g * group_words + ctrl_drafts). For each group whose block's relay word is set: when the page's group
// drafted (status drafting 1 and the drafts step's anchor the chain's) and its count m and the block's columns n + 1
// are in range, column c of the source (slot slots[c]) takes the last committed token (c = 0, accept token m - 1) or
// draft c at position bonus + c, and ctrl group g's draft c - 1 the same draft; else the columns go idle (token 0,
// position -1, draft 0) and the block's source is cleared. Writes verify and ctrl back whole when any group relays.
//
// Mode 3, the plan step (start of the MTP draft trace, every layer): per group, plan_mtp_layers from its description
// (mtp row g) and its slots of the stash page (each slot's position and layer-0 token in the last draft's pass). The
// rows run are the m committed ones (A .. A + m - 1) after up to max(0, layers - m) stash rows (A - 1, A - 2, .. while
// the group's stash slots hold them), right-aligned in the group's slots. Writes the passes' inputs: ring cache
// positions (int32 [B]), S serial positions (int32 [B], slot j's alone), RoPE indices, hidden table rows (step slot
// s: s; stash slot k: table_rows + k; idle slot j: table_rows + j, its own stash row) and layer-0 tokens (uint32
// [1, B]); and, when a group drafts, the new stash page (the drafting groups' slots rewritten).
//
// Mode 4, the tokens step (before MTP pass `layer`, every layer; the groups' widths from the descriptions, an idle
// group's GS): that pass's tokens, slot j of a group taking the
// layer-0 token of the group's slot j + layer, or past its last slot the group anchor's greedy token of an earlier
// pass (a draft).
//
// With DFlash drafting (drafter 1, tt/dflash.py) modes 0 and 1 read and write the DFlash draft's packed upload
// (reference.spec_chain.DFlashLayout, dwords words) instead of mtp: mode 0 writes each group's commit (the committed
// columns' slots as context rows at P[slots[c]] into the group's ring rows, the last committed token at the bonus as
// the anchor at word g * dblock, word g * dblock + 1 the anchor slot + 1), the other groups idle; mode 1 takes a
// group's anchor from its word g * dblock + 1 and its drafts from the drafter's GreedyHead tokens (one head row, group
// g's draft k at word g * D + k).

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
    constexpr uint32_t D = get_compile_time_arg_val(3);  // drafts per request (MTP passes)
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
    constexpr uint32_t chain_relay = get_compile_time_arg_val(15);  // a chain block's relay word
    constexpr uint32_t ctrl_words = get_compile_time_arg_val(16);
    constexpr uint32_t ctrl_drafts = get_compile_time_arg_val(17);  // a control group's first draft word
    constexpr uint32_t layers = get_compile_time_arg_val(18);       // MTP layers drafting (1, or one per draft)
    constexpr uint32_t table_rows = get_compile_time_arg_val(19);   // the hidden table's step rows (mode 3)
    constexpr uint32_t layer = get_compile_time_arg_val(20);        // the pass whose tokens mode 4 writes
    constexpr uint32_t drafter = get_compile_time_arg_val(21);      // 0: MTP, 1: DFlash
    constexpr uint32_t dwords = get_compile_time_arg_val(22);       // DFlash: the packed upload's words
    constexpr uint32_t dring = get_compile_time_arg_val(23);        // DFlash: a group's context ring rows
    constexpr uint32_t dblock = get_compile_time_arg_val(24);       // DFlash: a group's block rows
    constexpr uint32_t dheads = get_compile_time_arg_val(25);       // DFlash: query heads per block row
    constexpr int32_t dlast = get_compile_time_arg_val(26);         // DFlash: the RoPE tables' last row
    constexpr uint32_t G = get_compile_time_arg_val(27);            // draft groups of a pass
    constexpr uint32_t out_words = get_compile_time_arg_val(28);    // an accept page group's words
    constexpr uint32_t group_words = get_compile_time_arg_val(29);  // a control page group's words
    // The MTP drafter's planned program (reference.spec_decode.planned_drafts: per-group commit descriptions, the plan
    // step, the hidden gather): every MTP layer, or the one layer of one-draft groups. 0: the one-layer program.
    constexpr uint32_t planned = get_compile_time_arg_val(30);
    // Mode 3's serial position rows: row k holds the slots j with j % SR == k (reference/spec_decode.py SERIAL_WRITES).
    constexpr uint32_t SR = get_compile_time_arg_val(31);
    constexpr uint32_t GS = S / G;                                  // a draft group's slots
    constexpr uint32_t cgw = 4 + GS;                                // a chain block's words
    constexpr uint32_t draft_words = 1 + D;                         // a group's drafts section
    static_assert(G >= 1 && (G * GS == S || (G == 4 && GS == 4 && S == 8)) && D + 1 == GS);
    static_assert(G * out_words <= accept_words);
    static_assert(S <= B && B * 4 <= 1024 && mtp_rows == 7 && chain_row < verify_rows && G <= mtp_rows);
    static_assert(SR >= 1 && S % SR == 0 && GS % SR == 0);
    // The history's slots follow the G chain blocks (in the upload's next row once the blocks fill this one).
    static_assert(chain_relay == 3 + GS && G * cgw <= B && ctrl_drafts + D <= group_words);
    static_assert(G * group_words <= ctrl_words && 3 + 2 * GS <= B && 2 * S <= B && B <= table_rows);
    static_assert(layer < layers + (layers == 1) && (planned == 1 || drafter == 1 || G == 1));
    static_assert(planned <= 1 && (planned == 1 || layers == 1) && (drafter == 0 || planned == 0));
    static_assert(accept_words <= argmax_at && argmax_at + S <= positions_at && positions_at + S <= status_at);
    static_assert(status_at + 4 * G <= drafts_at && 1 + D <= out_words);
    static_assert(drafts_at + G * draft_words <= result_words && drafts_at % 16 == 0 && result_words % 16 == 0);
    constexpr uint32_t kTile = 32;  // DFlash's token, block, context and ring-row sections
    constexpr uint32_t dquery = kTile, dblock_at = dquery + G * dblock * dheads, dctx = dblock_at + kTile;
    constexpr uint32_t drows = dctx + kTile;
    static_assert(drafter <= 1 && (drafter == 0 || (dwords == drows + kTile && dblock >= 2 && G * dblock <= kTile)));
    static_assert(drafter == 0 || (layers == 1 && B <= kTile));
    constexpr uint32_t result_bytes = result_words * 4;
    constexpr uint32_t mtp_bytes = drafter == 1 ? dwords * 4 : mtp_rows * B * 4;  // the draft's upload
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
        // The idle upload: pack_mtp's (tokens 0, cache positions -1, RoPE indices 0; pack_mtp_layers': zeros), or
        // DFlash's (no context row, every group's block from position 0).
        if constexpr (drafter == 1) {
            for (uint32_t i = 0; i < dwords; ++i) {
                mtp[i] = 0;
            }
            for (uint32_t j = 0; j < kTile; ++j) {
                mtp[drows + j] = static_cast<int32_t>(G * dring);
            }
        } else {
            for (uint32_t i = 0; i < mtp_rows * B; ++i) {
                mtp[i] = (planned == 0 && i >= B && i < 4 * B) ? -1 : 0;
            }
        }
        int32_t starts[G];
        for (uint32_t g = 0; g < G; ++g) {
            starts[g] = 0;
            header[status_at + 4 * g + 1] = kNone;
            const int32_t* blk = chain + g * cgw;
            const uint32_t m = accept[g * out_words];
            const uint32_t columns = static_cast<uint32_t>(blk[1]);
            const uint32_t width = static_cast<uint32_t>(blk[0]);  // the group's slots (0: no source)
            if (width == 0 || m < 1 || m > columns || columns > width || width > GS) {
                continue;
            }
            const uint32_t* tokens = accept + g * out_words + 1;
            const uint32_t last = static_cast<uint32_t>(blk[3 + m - 1]);
            const int32_t bonus = pos[last] + 1;
            if (bonus >= blk[2]) {
                continue;
            }
            uint32_t anchor = last;
            if constexpr (drafter == 1) {
                starts[g] = bonus;
                mtp[g * dblock] = static_cast<int32_t>(tokens[m - 1]);
                mtp[g * dblock + 1] = static_cast<int32_t>(last + 1);
                for (uint32_t c = 0; c < m; ++c) {
                    const uint32_t s = static_cast<uint32_t>(blk[3 + c]);
                    const int32_t t = pos[s];
                    if (s < kTile && t >= 0) {
                        mtp[dctx + s] = t < dlast ? t : dlast;
                        mtp[drows + s] = static_cast<int32_t>(g * dring) + t % static_cast<int32_t>(dring);
                    }
                }
            } else if constexpr (planned == 1) {
                // The description: drafting, m, A = P[slots[0]], the slots (-1 past m), the committed tokens.
                volatile tt_l1_ptr int32_t* desc = mtp + g * B;
                desc[0] = static_cast<int32_t>(width);
                desc[1] = static_cast<int32_t>(m);
                desc[2] = pos[blk[3]];
                for (uint32_t c = 0; c < GS; ++c) {
                    desc[3 + c] = c < m ? blk[3 + c] : -1;
                    desc[3 + GS + c] = c < m ? static_cast<int32_t>(tokens[c]) : 0;
                }
                anchor = (g + 1) * width - 1;
            } else {
                auto cache = [](int32_t t) -> int32_t { return t >= 0 ? static_cast<int32_t>(ring + t % ring) : -1; };
                for (uint32_t c = 0; c < m; ++c) {
                    const uint32_t s = static_cast<uint32_t>(blk[3 + c]);
                    const int32_t t = pos[s];
                    mtp[s] = static_cast<int32_t>(tokens[c]);
                    mtp[B + s] = cache(t);
                    mtp[4 * B + s] = t > 0 ? t : 0;
                }
                mtp[2 * B + last] = cache(bonus);
                mtp[3 * B + last] = cache(bonus + 1);
                mtp[5 * B + last] = bonus;
                mtp[6 * B + last] = bonus + 1;
            }
            header[status_at + 4 * g] = 1;
            header[status_at + 4 * g + 1] = anchor;
            header[status_at + 4 * g + 2] = static_cast<uint32_t>(bonus);
        }
        if constexpr (drafter == 1) {
            // Block row j of group g = min(j / dblock, G - 1) at start_g + min(j - g * dblock, dblock - 1); a query
            // (group g, row r, head h) at its block row.
            for (uint32_t j = 0; j < kTile; ++j) {
                const uint32_t g = j / dblock < G ? j / dblock : G - 1;
                const uint32_t r = j - g * dblock;
                const int32_t p = starts[g] + static_cast<int32_t>(r < dblock - 1 ? r : dblock - 1);
                mtp[dblock_at + j] = p < dlast ? p : dlast;
            }
            for (uint32_t g = 0; g < G; ++g) {
                for (uint32_t r = 0; r < dblock; ++r) {
                    for (uint32_t h = 0; h < dheads; ++h) {
                        mtp[dquery + (g * dblock + r) * dheads + h] = mtp[dblock_at + g * dblock + r];
                    }
                }
            }
        }
        noc_async_write(mtp_l1, dram_addr(mtp_addr, mtp_bytes), mtp_bytes);
        noc_async_write(header_l1, dram_addr(result_addr, result_bytes), header_bytes);
        noc_async_write_barrier();
    } else if constexpr (mode == 1) {
        const uint32_t mtp_addr = get_arg_val<uint32_t>(0);
        const uint32_t result_addr = get_arg_val<uint32_t>(1);
        constexpr uint32_t drafts_bytes = (result_words - drafts_at) * 4;
        constexpr uint32_t heads = drafter == 1 ? 1 : D;  // DFlash: one GreedyHead row
        const uint32_t mtp_l1 = base;
        const uint32_t heads_l1 = mtp_l1 + align64(mtp_bytes);
        const uint32_t out_l1 = heads_l1 + D * align64(row_bytes);
        noc_async_read(dram_addr(mtp_addr, mtp_bytes), mtp_l1, mtp_bytes);
        for (uint32_t k = 0; k < heads; ++k) {
            const uint32_t head_addr = get_arg_val<uint32_t>(2 + k);
            noc_async_read(dram_addr(head_addr, row_bytes), heads_l1 + k * align64(row_bytes), row_bytes);
        }
        noc_async_read_barrier();

        const int32_t* upload = reinterpret_cast<const int32_t*>(mtp_l1);
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
        for (uint32_t i = 0; i < result_words - drafts_at; ++i) {
            out[i] = 0;
        }
        for (uint32_t g = 0; g < G; ++g) {
            uint32_t anchor = kNone;
            if constexpr (drafter == 1) {
                const uint32_t flag = static_cast<uint32_t>(upload[g * dblock + 1]);
                anchor = flag != 0 ? flag - 1 : kNone;
            } else if constexpr (planned == 1) {
                const uint32_t width = static_cast<uint32_t>(upload[g * B]);
                anchor = width != 0 ? (g + 1) * width - 1 : kNone;
            } else {
                const int32_t* step2 = upload + 2 * B;
                for (uint32_t s = 0; s < S; ++s) {
                    if (step2[s] >= 0) {
                        anchor = s;
                        break;
                    }
                }
            }
            volatile tt_l1_ptr uint32_t* section = out + g * draft_words;
            section[0] = anchor;
            for (uint32_t k = 0; k < D; ++k) {
                // DFlash: draft k of group g in row g * D + k of its one head; MTP: pass k's head at the anchor (one
                // head when D == 1, so the drafter, not the head count, picks the layout).
                const uint32_t* head =
                    reinterpret_cast<const uint32_t*>(heads_l1 + (drafter == 1 ? 0 : k * align64(row_bytes)));
                section[1 + k] = anchor == kNone ? kNone : head[drafter == 1 ? g * D + k : anchor];
            }
        }
        noc_async_write(out_l1, dram_addr(result_addr, result_bytes, drafts_at * 4), drafts_bytes);
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
        const uint32_t* page = reinterpret_cast<const uint32_t*>(result_l1);
        volatile tt_l1_ptr uint32_t* ctrl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_l1);
        bool relayed = false;
        for (uint32_t g = 0; g < G; ++g) {
            volatile tt_l1_ptr int32_t* blk = tok + chain_row * B + g * cgw;
            if (blk[chain_relay] == 0) {
                continue;
            }
            relayed = true;
            const uint32_t* status = page + status_at + 4 * g;
            const uint32_t* drafts = page + drafts_at + g * draft_words;
            const uint32_t* accept = page + g * out_words;  // the accept page starts the result page
            const uint32_t columns = static_cast<uint32_t>(blk[1]);
            const uint32_t m = accept[0];
            const bool ok =
                status[0] == 1 && drafts[0] == status[1] && columns >= 1 && columns <= 1 + D && m >= 1 && m <= 1 + D;
            const int32_t bonus = static_cast<int32_t>(status[2]);
            const uint32_t n = columns < GS ? columns : GS;
            for (uint32_t c = 0; c < n; ++c) {
                const uint32_t s = static_cast<uint32_t>(blk[3 + c]);
                if (s >= S) {
                    continue;  // not a slot of the pass (the host lays the source's columns into slots 0 .. S - 1)
                }
                const uint32_t token = ok ? (c == 0 ? accept[m] : drafts[c]) : 0;
                tok[s] = static_cast<int32_t>(token);
                pos[s] = ok ? bonus + static_cast<int32_t>(c) : -1;
                if (c > 0) {
                    ctrl[g * group_words + ctrl_drafts + c - 1] = ok ? drafts[c] : 0;
                }
            }
            if (!ok) {
                blk[0] = 0;
            }
        }
        if (relayed) {
            noc_async_write(verify_l1, dram_addr(verify_addr, verify_bytes), verify_bytes);
            noc_async_write(ctrl_l1, dram_addr(ctrl_addr, ctrl_bytes), ctrl_bytes);
            noc_async_write_barrier();
        }
    } else if constexpr (mode == 3) {
        // Runtime args: mtp, stash, then the outputs: positions, RoPE indices, hidden rows, layer-0 tokens, SR serial.
        const uint32_t mtp_addr = get_arg_val<uint32_t>(0);
        const uint32_t stash_addr = get_arg_val<uint32_t>(1);
        constexpr uint32_t row_l1 = align64(row_bytes);
        constexpr uint32_t outs = 4 + SR;
        const uint32_t desc_l1 = base;  // G description rows
        const uint32_t stash_l1 = desc_l1 + G * row_l1;
        const uint32_t out_l1 = stash_l1 + row_l1;  // outs rows, then the new stash page
        for (uint32_t g = 0; g < G; ++g) {
            noc_async_read(dram_addr(mtp_addr, mtp_bytes, g * row_bytes), desc_l1 + g * row_l1, row_bytes);
        }
        noc_async_read(dram_addr(stash_addr, row_bytes), stash_l1, row_bytes);
        noc_async_read_barrier();

        const int32_t* stash = reinterpret_cast<const int32_t*>(stash_l1);
        volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1);
        volatile tt_l1_ptr int32_t* rope = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + row_l1);
        volatile tt_l1_ptr int32_t* hidden = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + 2 * row_l1);
        volatile tt_l1_ptr int32_t* tokens0 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + 3 * row_l1);
        volatile tt_l1_ptr int32_t* fresh = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + outs * row_l1);
        auto serial = [&](uint32_t j) {
            return reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + (4 + j) * row_l1);
        };
        for (uint32_t i = 0; i < B; ++i) {
            pos[i] = -1;
            rope[i] = 0;
            hidden[i] = static_cast<int32_t>(table_rows + i);
            tokens0[i] = 0;
            fresh[i] = stash[i];
            for (uint32_t k = 0; k < SR; ++k) {
                serial(k)[i] = -1;
            }
        }
        bool any = false;
        for (uint32_t g = 0; g < G; ++g) {
            const int32_t* desc = reinterpret_cast<const int32_t*>(desc_l1 + g * row_l1);
            const int32_t m = desc[1];
            const int32_t width = desc[0];  // the group's slots (0: idle)
            if (width == 0 || m < 1 || m > width || width > static_cast<int32_t>(GS) ||
                (g + 1) * static_cast<uint32_t>(width) > S) {
                continue;
            }
            any = true;
            const int32_t a = desc[2];
            const uint32_t w = static_cast<uint32_t>(width);
            const uint32_t lo = g * w;
            // A group of 2 slots drafts one token (layer 0's): it re-runs no row, whatever the later passes compute.
            const int32_t drafting =
                static_cast<int32_t>(layers) < width - 1 ? static_cast<int32_t>(layers) : width - 1;
            // found[k]: the group's stash slot holding position a - 1 - k (consecutive from a - 1 down).
            uint32_t found[GS];
            const int32_t want = drafting - m;
            uint32_t r = 0;
            while (static_cast<int32_t>(r) < want) {
                const int32_t q = a - 1 - static_cast<int32_t>(r);
                uint32_t hit = kNone;
                for (uint32_t k = lo; q >= 0 && k < lo + w; ++k) {
                    if (stash[k] == q) {
                        hit = k;
                        break;
                    }
                }
                if (hit == kNone) {
                    break;
                }
                found[r++] = hit;
            }
            const uint32_t rows = r + static_cast<uint32_t>(m);
            const uint32_t first = lo + w - rows;
            for (uint32_t j = lo; j < lo + w; ++j) {
                fresh[j] = -1;
                fresh[S + j] = 0;
            }
            for (uint32_t t = 0; t < rows; ++t) {
                const uint32_t j = first + t;
                const int32_t q = a - static_cast<int32_t>(r) + static_cast<int32_t>(t);
                if (t < r) {
                    const uint32_t k = found[r - 1 - t];
                    hidden[j] = static_cast<int32_t>(table_rows + k);
                    tokens0[j] = stash[S + k];
                } else {
                    hidden[j] = desc[3 + t - r];
                    tokens0[j] = desc[3 + GS + t - r];
                }
                const int32_t cached = q >= 0 ? static_cast<int32_t>(ring + q % ring) : -1;
                pos[j] = cached;
                serial(j % SR)[j] = cached;
                rope[j] = q > 0 ? q : 0;
                fresh[j] = q;
                fresh[S + j] = tokens0[j];
            }
        }
        // Outputs in runtime-arg order (2 ..): positions, RoPE, hidden rows, tokens, serial 0 .. SR - 1.
        for (uint32_t k = 0; k < outs; ++k) {
            noc_async_write(out_l1 + k * row_l1, dram_addr(get_arg_val<uint32_t>(2 + k), row_bytes), row_bytes);
        }
        if (any) {
            noc_async_write(out_l1 + outs * row_l1, dram_addr(stash_addr, row_bytes), row_bytes);
        }
        noc_async_write_barrier();
    } else {
        // Runtime args: the layer-0 tokens, the output, then the heads of passes 0 .. layer - 1.
        constexpr uint32_t row_l1 = align64(row_bytes);
        const uint32_t tokens_l1 = base;
        const uint32_t heads_l1 = tokens_l1 + row_l1;
        const uint32_t out_l1 = heads_l1 + layer * row_l1;
        const uint32_t desc_l1 = out_l1 + row_l1;  // the descriptions' drafting words (each group's width)
        noc_async_read(dram_addr(get_arg_val<uint32_t>(0), row_bytes), tokens_l1, row_bytes);
        for (uint32_t k = 0; k < layer; ++k) {
            noc_async_read(dram_addr(get_arg_val<uint32_t>(2 + k), row_bytes), heads_l1 + k * row_l1, row_bytes);
        }
        const uint32_t mtp_addr = get_arg_val<uint32_t>(2 + layer);
        for (uint32_t g = 0; g < G; ++g) {
            noc_async_read(dram_addr(mtp_addr, mtp_bytes, g * row_bytes), desc_l1 + g * row_l1, row_bytes);
        }
        noc_async_read_barrier();
        const uint32_t* tokens0 = reinterpret_cast<const uint32_t*>(tokens_l1);
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
        for (uint32_t i = 0; i < B; ++i) {
            out[i] = 0;
        }
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t drafting = reinterpret_cast<const uint32_t*>(desc_l1 + g * row_l1)[0];
            const uint32_t w = drafting != 0 && drafting <= GS ? drafting : GS;  // an idle group: GS
            const uint32_t lo = g * w;
            for (uint32_t j = 0; j < w; ++j) {
                const uint32_t x = j + layer;
                out[lo + j] = x < w ? tokens0[lo + x]
                                    : reinterpret_cast<const uint32_t*>(heads_l1 + (x - w) * row_l1)[lo + w - 1];
            }
        }
        noc_async_write(out_l1, dram_addr(get_arg_val<uint32_t>(1), row_bytes), row_bytes);
        noc_async_write_barrier();
    }
}
