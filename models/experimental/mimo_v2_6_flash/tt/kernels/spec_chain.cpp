// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The MTP draft chained behind a verify pass on device (tt/spec_chain.py; host mirror reference/spec_chain.py), one
// core, five modes. Every buffer is a one-page interleaved DRAM tensor, read and written whole (the result page in
// 64-byte sections, one per writing mode).
//
// Mode 0, the chain step (end of the accept trace, after SampleVerify):
// - verify: the verify's packed upload, int32 [verify_rows * B]: tokens, positions P, and row chain_row, the chain
//   row: source k's words at k * chain_next (k < sources): + 0 source, + 1 columns, + 3 .. + 3 + S its slots, + 3 + S
//   relay (mode 2), + chain_ring its MTP ring, + chain_group its page group (mode 2); word 2 the limit, word
//   chain_pair the pair draft, and the MTP history's slots after source 0's relay word (kernels/mtp_history.cpp).
// - accept: SampleVerify's output page, uint32 [accept_words]: group k's committed count m and tokens at k * out_words
//   (source k is the pass's request k).
// - argmax: the verify's greedy tokens, uint32 [B].
// Writes the draft's packed upload (mtp, int32 [mtp_rows * B]), idle except for each source with 1 <= m <= columns
// and the bonus P[slots[m - 1]] + 1 below the limit; cache positions in the source's ring r: 2 r R + R + t % R. The
// pair draft (chain_pair; each source, from its group): reference.spec_inputs pack_mtp_pair's rows: slot slots[c]
// (c < m) takes committed token c with its fill at P[slots[c]] and that RoPE index, and word k of row pair_anchors the
// anchor slots[m - 1]. Else source 0 alone: one layer (layers == 1): pack_mtp's rows (tokens; fill, step2, step3 ring
// cache positions; fill, step2, step3 RoPE indices) as pack_mtp(plan_mtp(...)): the fill as above, the anchor's steps
// at the bonus and the bonus + 1; every layer: pack_mtp_layers' description of the commit (drafting, m, A =
// P[slots[0]], the slots, the committed tokens, the ring), the anchor the last slot. Writes the result header: the
// accept page, the argmax and the positions of slots 0 .. S - 1, and each source's status (drafting, anchor, bonus,
// pair).
//
// Mode 1, the drafts step (end of an MTP draft trace): reads mtp (written by mode 0 or copied by the host) and each
// MTP pass's greedy tokens (heads, uint32 [B]). Writes the result's drafts section, per source k its anchor
// (0xFFFFFFFF: none) and the anchor's token of each pass (0xFFFFFFFF past the drafts made). The pair draft (compile
// arg pair): source k's anchor is word k of row pair_anchors, with pair_drafts drafts (head 0's); else source 0's
// anchor is the slot with a step2 position (every layer: the last slot when the description drafts).
//
// Mode 2, the relay step (first in the verify trace, a chained verify): reads verify as the host copied it, the
// result page the lane's last pass left, and the accept step's control page (ctrl, uint32 [ctrl_words], group k at
// k * group_words, source k's: drafts at word ctrl_drafts). Unless a source's relay word is set it writes nothing.
// For each relayed source k, from its page group g: when g's chain drafted (status drafting 1 and the drafts step's
// anchor the chain's), its count m is in range and the source verifies no more drafts than g holds (pair_drafts for a
// pair draft's, D otherwise), column c of the source (slot slots[c]) takes g's last committed token (c = 0, accept
// token m - 1) or g's draft c at position g's bonus + c, and ctrl group k's draft c - 1 the same draft; else its
// columns go idle (token 0, position -1, draft 0) and its source word is cleared (no draft behind it). Writes verify
// and ctrl back whole. The page reads the same whichever drafter wrote it.
//
// Mode 3, the plan step (start of the MTP draft trace, every layer): plan_mtp_layers from the description (mtp row 0)
// and the stash page (each slot's position and layer-0 token in the last draft's pass). The rows run are the m
// committed ones (A .. A + m - 1) after up to max(0, layers - m) stash rows (A - 1, A - 2, .. while the stash holds
// them), right-aligned in the S slots. Writes the passes' inputs: cache positions in the description's ring (int32
// [B]), S serial positions (int32 [B], slot j's alone), RoPE indices, hidden table rows (step slot s: s; stash slot k:
// table_rows + k; idle slot j: table_rows + j, its own stash row) and layer-0 tokens (uint32 [1, B]); and, when
// drafting, the new stash page (the pass's slots).
//
// Mode 4, the tokens step (before MTP pass `layer`, every layer): that pass's tokens, slot j taking the layer-0 token
// of slot j + layer, or past the last slot the anchor's greedy token of an earlier pass (a draft).

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
    constexpr uint32_t ctrl_drafts = get_compile_time_arg_val(17);  // a control group's first draft word
    constexpr uint32_t layers = get_compile_time_arg_val(18);       // MTP layers drafting (1, or one per draft)
    constexpr uint32_t table_rows = get_compile_time_arg_val(19);   // the hidden table's step rows (mode 3)
    constexpr uint32_t layer = get_compile_time_arg_val(20);        // the pass whose tokens mode 4 writes
    constexpr bool pair = get_compile_time_arg_val(21) != 0;        // mode 1: the pair draft's drafts step
    constexpr uint32_t sources = get_compile_time_arg_val(22);      // sources of a pass (a pair's two)
    constexpr uint32_t chain_next = get_compile_time_arg_val(23);   // source k's chain words at k * chain_next
    constexpr uint32_t chain_pair = get_compile_time_arg_val(24);
    constexpr uint32_t chain_ring = get_compile_time_arg_val(25);
    constexpr uint32_t chain_group = get_compile_time_arg_val(26);
    constexpr uint32_t desc_ring = get_compile_time_arg_val(27);     // the description's ring word
    constexpr uint32_t status_words = get_compile_time_arg_val(28);  // per source
    constexpr uint32_t draft_words = get_compile_time_arg_val(29);   // per source
    constexpr uint32_t out_words = get_compile_time_arg_val(30);     // an accept page group
    constexpr uint32_t group_words = get_compile_time_arg_val(31);   // a control page group
    constexpr uint32_t pair_drafts = get_compile_time_arg_val(32);   // drafts per pair source
    constexpr uint32_t pair_anchors = get_compile_time_arg_val(33);  // the pair upload's anchors row
    static_assert(S <= B && B * 4 <= 1024 && 3 + S <= B && mtp_rows == 7 && chain_row < verify_rows);
    static_assert(chain_relay == 3 + S && ctrl_drafts + D <= group_words && sources * group_words <= ctrl_words);
    static_assert(3 + 2 * S <= B && 2 * S <= B && B <= table_rows && layer < layers + (layers == 1));
    static_assert(accept_words <= argmax_at && argmax_at + S <= positions_at && positions_at + S <= status_at);
    static_assert(status_at + sources * status_words <= drafts_at && status_words >= 4 && 1 + D <= out_words);
    static_assert(drafts_at + sources * draft_words <= result_words && draft_words >= 1 + D);
    static_assert(drafts_at % 16 == 0 && result_words % 16 == 0 && sources * out_words <= accept_words);
    static_assert(chain_relay < chain_pair && chain_group < chain_next && (sources - 1) * chain_next + chain_group < B);
    static_assert(desc_ring < B && sources <= B && pair_anchors < mtp_rows && pair_drafts <= D && sources >= 1);
    constexpr uint32_t result_bytes = result_words * 4;
    constexpr uint32_t mtp_bytes = mtp_rows * B * 4;
    constexpr uint32_t row_bytes = B * 4;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t base = get_write_ptr(cb_scratch);

    // Ring r's cache position of true position t (reference.spec_decode.mtp_cache_position).
    auto cache = [](int32_t t, int32_t r) -> int32_t {
        return t >= 0 ? static_cast<int32_t>(2 * ring * static_cast<uint32_t>(r) + ring + t % ring) : -1;
    };

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
        const bool pairs = chain[chain_pair] != 0;

        // pack_mtp's idle draft: tokens 0, cache positions -1, RoPE indices 0 (pack_mtp_layers': zeros; the pair's:
        // pack_mtp's, its anchor words -1 too).
        const bool idle_minus = layers == 1 || pairs;
        for (uint32_t i = 0; i < mtp_rows * B; ++i) {
            mtp[i] = (idle_minus && i >= B && i < 4 * B) ? -1 : 0;
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
        for (uint32_t k = 0; k < sources; ++k) {
            header[status_at + k * status_words + 1] = kNone;
        }

        const int32_t limit = chain[2];
        const uint32_t drafting_sources = pairs ? sources : 1;
        for (uint32_t k = 0; k < drafting_sources; ++k) {
            const int32_t* src = chain + k * chain_next;
            const uint32_t* group = accept + k * out_words;
            const uint32_t m = group[0];
            const uint32_t columns = static_cast<uint32_t>(src[1]);
            const int32_t r = src[chain_ring];
            if (src[0] == 0 || m < 1 || m > columns || columns > S) {
                continue;
            }
            const uint32_t anchor = static_cast<uint32_t>(src[3 + m - 1]);
            const int32_t bonus = pos[anchor] + 1;
            if (bonus >= limit) {
                continue;
            }
            volatile tt_l1_ptr uint32_t* status = header + status_at + k * status_words;
            if (pairs) {
                // pack_mtp_pair: the source's committed slots fill its ring; its anchor in word k of the anchors row.
                for (uint32_t c = 0; c < m; ++c) {
                    const uint32_t s = static_cast<uint32_t>(src[3 + c]);
                    const int32_t t = pos[s];
                    mtp[s] = static_cast<int32_t>(group[1 + c]);
                    mtp[B + s] = cache(t, r);
                    mtp[4 * B + s] = t > 0 ? t : 0;
                }
                mtp[pair_anchors * B + k] = static_cast<int32_t>(anchor);
                status[0] = 1;
                status[1] = anchor;
                status[2] = static_cast<uint32_t>(bonus);
                status[3] = 1;
            } else if (layers > 1) {
                // The description: drafting, m, A = P[slots[0]], the slots (-1 past m), the committed tokens, the ring.
                mtp[0] = 1;
                mtp[1] = static_cast<int32_t>(m);
                mtp[2] = pos[src[3]];
                for (uint32_t c = 0; c < S; ++c) {
                    mtp[3 + c] = c < m ? src[3 + c] : -1;
                    mtp[3 + S + c] = c < m ? static_cast<int32_t>(group[1 + c]) : 0;
                }
                mtp[desc_ring] = r;
                status[0] = 1;
                status[1] = S - 1;
                status[2] = static_cast<uint32_t>(bonus);
            } else {
                for (uint32_t c = 0; c < m; ++c) {
                    const uint32_t s = static_cast<uint32_t>(src[3 + c]);
                    const int32_t t = pos[s];
                    mtp[s] = static_cast<int32_t>(group[1 + c]);
                    mtp[B + s] = cache(t, r);
                    mtp[4 * B + s] = t > 0 ? t : 0;
                }
                mtp[2 * B + anchor] = cache(bonus, r);
                mtp[3 * B + anchor] = cache(bonus + 1, r);
                mtp[5 * B + anchor] = bonus;
                mtp[6 * B + anchor] = bonus + 1;
                status[0] = 1;
                status[1] = anchor;
                status[2] = static_cast<uint32_t>(bonus);
            }
        }
        noc_async_write(mtp_l1, dram_addr(mtp_addr, mtp_bytes), mtp_bytes);
        noc_async_write(header_l1, dram_addr(result_addr, result_bytes), header_bytes);
        noc_async_write_barrier();
    } else if constexpr (mode == 1) {
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

        const int32_t* upload = reinterpret_cast<const int32_t*>(mtp_l1);
        auto head = [&](uint32_t d) { return reinterpret_cast<const uint32_t*>(heads_l1 + d * align64(row_bytes)); };
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
        uint32_t anchors[sources];
        for (uint32_t k = 0; k < sources; ++k) {
            anchors[k] = kNone;
        }
        if constexpr (pair) {
            for (uint32_t k = 0; k < sources; ++k) {
                const int32_t a = upload[pair_anchors * B + k];
                anchors[k] = a >= 0 && a < static_cast<int32_t>(S) ? static_cast<uint32_t>(a) : kNone;
            }
        } else if constexpr (layers > 1) {
            anchors[0] = upload[0] != 0 ? S - 1 : kNone;
        } else {
            for (uint32_t s = 0; s < S; ++s) {
                if (upload[2 * B + s] >= 0) {
                    anchors[0] = s;
                    break;
                }
            }
        }
        constexpr uint32_t drafted = pair ? pair_drafts : D;
        for (uint32_t i = 0; i < result_words - drafts_at; ++i) {
            out[i] = 0;
        }
        for (uint32_t k = 0; k < sources; ++k) {
            out[k * draft_words] = anchors[k];
            for (uint32_t d = 0; d < D; ++d) {
                out[k * draft_words + 1 + d] = anchors[k] == kNone || d >= drafted ? kNone : head(d)[anchors[k]];
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
        volatile tt_l1_ptr int32_t* chain = tok + chain_row * B;
        const uint32_t* page = reinterpret_cast<const uint32_t*>(result_l1);
        volatile tt_l1_ptr uint32_t* ctrl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_l1);
        bool relayed = false;
        for (uint32_t k = 0; k < sources; ++k) {
            volatile tt_l1_ptr int32_t* src = chain + k * chain_next;
            if (src[chain_relay] == 0) {
                continue;
            }
            relayed = true;
            const uint32_t columns = static_cast<uint32_t>(src[1]);
            const uint32_t g = static_cast<uint32_t>(src[chain_group]);
            const bool in_page = g < sources;
            const uint32_t gs = in_page ? g : 0;  // the page group read (any, when out of range: not relayed)
            const uint32_t st = status_at + gs * status_words;
            const uint32_t dr = drafts_at + gs * draft_words;
            const uint32_t m = page[gs * out_words];  // group g's committed count (the accept page starts the page)
            const uint32_t drafted = page[st + 3] != 0 ? pair_drafts : D;
            const bool ok = in_page && page[st] == 1 && page[dr] == page[st + 1] && columns >= 1 &&
                            columns <= 1 + drafted && m >= 1 && m <= 1 + D;
            const int32_t bonus = static_cast<int32_t>(page[st + 2]);
            const uint32_t n = columns < S ? columns : S;
            for (uint32_t c = 0; c < n; ++c) {
                const uint32_t s = static_cast<uint32_t>(src[3 + c]);
                if (s >= S) {
                    continue;  // not a slot of the pass (the host lays the source's columns into slots 0 .. S - 1)
                }
                const uint32_t token = ok ? (c == 0 ? page[gs * out_words + m] : page[dr + c]) : 0;
                tok[s] = static_cast<int32_t>(token);
                pos[s] = ok ? bonus + static_cast<int32_t>(c) : -1;
                if (c > 0) {
                    ctrl[k * group_words + ctrl_drafts + c - 1] = ok ? page[dr + c] : 0;
                }
            }
            if (!ok) {
                src[0] = 0;
            }
        }
        if (relayed) {
            noc_async_write(verify_l1, dram_addr(verify_addr, verify_bytes), verify_bytes);
            noc_async_write(ctrl_l1, dram_addr(ctrl_addr, ctrl_bytes), ctrl_bytes);
            noc_async_write_barrier();
        }
    } else if constexpr (mode == 3) {
        // Runtime args: mtp, stash, then the outputs: positions, RoPE indices, hidden rows, layer-0 tokens, S serial.
        const uint32_t mtp_addr = get_arg_val<uint32_t>(0);
        const uint32_t stash_addr = get_arg_val<uint32_t>(1);
        constexpr uint32_t row_l1 = align64(row_bytes);
        constexpr uint32_t outs = 4 + S;
        const uint32_t desc_l1 = base;
        const uint32_t stash_l1 = desc_l1 + row_l1;
        const uint32_t out_l1 = stash_l1 + row_l1;  // outs rows, then the new stash page
        noc_async_read(dram_addr(mtp_addr, mtp_bytes), desc_l1, row_bytes);
        noc_async_read(dram_addr(stash_addr, row_bytes), stash_l1, row_bytes);
        noc_async_read_barrier();

        const int32_t* desc = reinterpret_cast<const int32_t*>(desc_l1);
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
            for (uint32_t j = 0; j < S; ++j) {
                serial(j)[i] = -1;
            }
        }
        const int32_t m = desc[1];
        const bool drafting = desc[0] != 0 && m >= 1 && m <= static_cast<int32_t>(S);
        if (drafting) {
            const int32_t a = desc[2];
            const int32_t r_ring = desc[desc_ring];
            // found[k]: the stash slot holding position a - 1 - k (consecutive from a - 1 down).
            uint32_t found[S];
            const int32_t want = static_cast<int32_t>(layers) - m;
            uint32_t r = 0;
            while (static_cast<int32_t>(r) < want) {
                const int32_t q = a - 1 - static_cast<int32_t>(r);
                uint32_t hit = kNone;
                for (uint32_t k = 0; q >= 0 && k < S; ++k) {
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
            const uint32_t first = S - rows;
            for (uint32_t j = 0; j < S; ++j) {
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
                    tokens0[j] = desc[3 + S + t - r];
                }
                const int32_t cached = cache(q, r_ring);
                pos[j] = cached;
                serial(j)[j] = cached;
                rope[j] = q > 0 ? q : 0;
                fresh[j] = q;
                fresh[S + j] = tokens0[j];
            }
        }
        // Outputs in runtime-arg order (2 ..): positions, RoPE, hidden rows, tokens, serial 0 .. S - 1.
        for (uint32_t k = 0; k < outs; ++k) {
            noc_async_write(out_l1 + k * row_l1, dram_addr(get_arg_val<uint32_t>(2 + k), row_bytes), row_bytes);
        }
        if (drafting) {
            noc_async_write(out_l1 + outs * row_l1, dram_addr(stash_addr, row_bytes), row_bytes);
        }
        noc_async_write_barrier();
    } else {
        // Runtime args: the layer-0 tokens, the output, then the heads of passes 0 .. layer - 1.
        constexpr uint32_t row_l1 = align64(row_bytes);
        const uint32_t tokens_l1 = base;
        const uint32_t heads_l1 = tokens_l1 + row_l1;
        const uint32_t out_l1 = heads_l1 + layer * row_l1;
        noc_async_read(dram_addr(get_arg_val<uint32_t>(0), row_bytes), tokens_l1, row_bytes);
        for (uint32_t k = 0; k < layer; ++k) {
            noc_async_read(dram_addr(get_arg_val<uint32_t>(2 + k), row_bytes), heads_l1 + k * row_l1, row_bytes);
        }
        noc_async_read_barrier();
        const uint32_t* tokens0 = reinterpret_cast<const uint32_t*>(tokens_l1);
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
        for (uint32_t i = 0; i < B; ++i) {
            out[i] = 0;
        }
        for (uint32_t j = 0; j < S; ++j) {
            const uint32_t x = j + layer;
            out[j] = x < S ? tokens0[x] : reinterpret_cast<const uint32_t*>(heads_l1 + (x - S) * row_l1)[S - 1];
        }
        noc_async_write(out_l1, dram_addr(get_arg_val<uint32_t>(1), row_bytes), row_bytes);
        noc_async_write_barrier();
    }
}
