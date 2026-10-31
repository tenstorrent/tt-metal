// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Speculative acceptance of one verify pass (tt/verify_sampler.py; host mirror reference/sample_verify.py): each
// request of the pass ("group") holds its last committed token and its n drafts in consecutive slots s0 .. s0 + n,
// slot s0 + j's logits the target's distribution after column j. Steps 1-2 of the decode sampler (SampleScan,
// SampleChipMerge, the row all-gather) gave every slot its chips' sorted top-K lists and normalizers.
//
// Inputs (interleaved):
// - cands: the row all-gather of the chip candidates (sample_chip_merge.cpp), as for sample_merge.cpp: page ch holds
//   chip ch's sorted K (key, local index) pairs of slot u at byte offset u * K * 8 and its normalizer (uint64 Q30) at
//   byte offset 2048 + u * 16.
// - params: uint32 row-major [1, max_batch * 8], per slot as sample_merge.cpp reads them: log2(e) / T as fp32 bits,
//   top_p in Q24, top_k (0: none), greedy, 1 / T as fp32 bits, three unused words.
// - argmax: uint32 row-major [1, max_batch], each slot's greedy token (GreedyHead's: the lowest id among tied
//   maxima), the token the argmax_ids walk compares with.
// - ctrl: uint32 row-major [1, groups * group_words], per group: first slot (0xFFFFFFFF: unused), drafts n, greedy,
// unused,
//   drafts [D], then the uniforms in Q24: bonus, accept [D], correction; zero padding.
// Per group: a greedy group accepts draft j while it equals argmax[s0 + j] and commits the argmax chain. A sampled
// group accepts draft j when its probability under slot s0 + j's distribution (as sample_merge.cpp builds it:
// softmax(logits / T) over the vocabulary or over the top_k candidates, the nucleus keeping a candidate while the mass
// ranked above it is below top_p) is positive and at least u_accept[j]; at the first rejection it commits a token drawn
// from that distribution without the draft (u_corr), else the bonus from slot s0 + n's (u_bonus). Draws are by inverse
// CDF in (logit descending, id ascending) order for a row with top_k and in id order for a nucleus row without, the
// orders the plugin's host walk uses (vllm_tt_plugin.spec_accept). Fixed point throughout (sample_fixed.hpp).
// A slot whose walk took a candidate after a chip's list ran out (so possibly not the row's next largest), or ran out
// of candidates with the nucleus still open, marks its group truncated; the group is still accepted on its
// candidates, as the decode sampler draws from the nucleus's tokens among them.
// Output: uint32 row-major [1, groups * out_words], per group: committed count (0: unused group), the committed tokens
// [D + 1] (0xFFFFFFFF past the count), truncated, unused.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sample_fixed.hpp"

namespace {

template <bool dram>
FORCE_INLINE uint64_t page_addr(uint32_t base, uint32_t page, uint32_t page_size, uint32_t offset = 0) {
    const InterleavedAddrGen<dram> g = {.bank_base_address = base, .page_size = page_size};
    return get_noc_addr(page, g, offset);
}

FORCE_INLINE uint64_t page_addr(bool dram, uint32_t base, uint32_t page, uint32_t page_size, uint32_t offset = 0) {
    return dram ? page_addr<true>(base, page, page_size, offset) : page_addr<false>(base, page, page_size, offset);
}

constexpr uint32_t kNone = 0xFFFFFFFFu;

// Scratch segments start on 64-byte boundaries (DRAM reads land there).
constexpr uint32_t align64(uint32_t bytes) { return (bytes + 63) & ~63u; }

}  // namespace

void kernel_main() {
    const uint32_t cands_addr = get_arg_val<uint32_t>(0);
    const uint32_t cands_is_dram = get_arg_val<uint32_t>(1);
    const uint32_t params_addr = get_arg_val<uint32_t>(2);
    const uint32_t argmax_addr = get_arg_val<uint32_t>(3);
    const uint32_t argmax_is_dram = get_arg_val<uint32_t>(4);
    const uint32_t ctrl_addr = get_arg_val<uint32_t>(5);
    const uint32_t out_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t tp = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t vocab_local = get_compile_time_arg_val(3);
    constexpr uint32_t max_batch = get_compile_time_arg_val(4);
    constexpr uint32_t groups = get_compile_time_arg_val(5);
    constexpr uint32_t D = get_compile_time_arg_val(6);  // drafts per group at most
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(7);
    constexpr uint32_t group_words = get_compile_time_arg_val(8);
    constexpr uint32_t out_words = get_compile_time_arg_val(9);
    static_assert(users >= 1 && users <= 16 && tp <= 32 && max_batch * 4 <= 1024 && users * K * 8 <= 4096);
    static_assert(groups <= users && D + 1 <= users && 6 + 2 * D <= group_words && 3 + D <= out_words);
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t cand_bytes = users * slot_bytes;     // every slot's list of a chip page
    constexpr uint32_t norm_bytes =
        users <= 4 ? 64 : (users <= 8 ? 128 : 256);  // every slot's normalizer of a chip page
    constexpr uint32_t params_bytes = max_batch * 32;
    constexpr uint32_t argmax_bytes = max_batch * 4;
    constexpr uint32_t ctrl_bytes = groups * group_words * 4;
    constexpr uint32_t out_bytes = groups * out_words * 4;
    constexpr uint32_t max_cands = tp * K;
    // A chip's candidates: its users' lists then (at byte 2048) their normalizers in one page, or with more than 8
    // users the lists in one page and the normalizers in the next (sample_chip_merge.cpp).
    constexpr uint32_t cand_pages = cand_bytes > 2048 ? 2 : 1;
    constexpr uint32_t norm_at = cand_pages == 1 ? 2048 : 0;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t cands_l1 = get_write_ptr(cb_scratch);
    const uint32_t norms_l1 = cands_l1 + align64(tp * cand_bytes);
    const uint32_t params_l1 = norms_l1 + align64(tp * norm_bytes);
    const uint32_t argmax_l1 = params_l1 + align64(params_bytes);
    const uint32_t ctrl_l1 = argmax_l1 + align64(argmax_bytes);
    const uint32_t out_l1 = ctrl_l1 + align64(ctrl_bytes);
    const uint32_t probs_l1 = out_l1 + align64(out_bytes);
    const uint32_t ids_l1 = probs_l1 + max_cands * 4;
    const uint32_t order_l1 = ids_l1 + max_cands * 4;

    for (uint32_t ch = 0; ch < tp; ++ch) {
        noc_async_read(
            page_addr(cands_is_dram, cands_addr, ch * cand_pages, 4096), cands_l1 + ch * cand_bytes, cand_bytes);
        noc_async_read(
            page_addr(cands_is_dram, cands_addr, ch * cand_pages + cand_pages - 1, 4096, norm_at),
            norms_l1 + ch * norm_bytes,
            norm_bytes);
    }
    noc_async_read(page_addr<true>(params_addr, 0, params_bytes), params_l1, params_bytes);
    noc_async_read(page_addr(argmax_is_dram, argmax_addr, 0, argmax_bytes), argmax_l1, argmax_bytes);
    noc_async_read(page_addr<true>(ctrl_addr, 0, ctrl_bytes), ctrl_l1, ctrl_bytes);
    noc_async_read_barrier();

    // Read only after the barrier, and written by nothing else: plain pointers.
    const uint32_t* cands = reinterpret_cast<const uint32_t*>(cands_l1);
    const uint64_t* norms = reinterpret_cast<const uint64_t*>(norms_l1);
    const uint32_t* params = reinterpret_cast<const uint32_t*>(params_l1);
    const uint32_t* argmax = reinterpret_cast<const uint32_t*>(argmax_l1);
    const uint32_t* ctrl = reinterpret_cast<const uint32_t*>(ctrl_l1);
    uint32_t* probs = reinterpret_cast<uint32_t*>(probs_l1);
    uint32_t* ids = reinterpret_cast<uint32_t*>(ids_l1);
    uint32_t* order = reinterpret_cast<uint32_t*>(order_l1);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);

    // Slot u's kept candidates into probs / ids (Q30 relative to its largest logit, in (logit descending, global id
    // ascending) order), as sample_merge.cpp builds them. Returns how many are kept; sets their mass, whether draws
    // walk them in id order (no top_k), and whether the walk left the certified candidates.
    auto build = [&](uint32_t u, uint64_t& kept_total, bool& by_id, bool& truncated) -> uint32_t {
        uint32_t head[tp], head_key[tp], head_id[tp];
        bool spent = false;  // a chip's list ran out: the merge's later candidates may not be the slot's next largest
        auto load = [&](uint32_t ch) {
            const uint32_t e = (ch * users + u) * 2 * K + 2 * head[ch];
            head_key[ch] = head[ch] < K ? cands[e] : 0;
            head_id[ch] = head[ch] < K ? ch * vocab_local + cands[e + 1] : kNone;
        };
        for (uint32_t ch = 0; ch < tp; ++ch) {
            head[ch] = 0;
            load(ch);
        }
        auto next = [&](uint32_t& key, uint32_t& id) {
            uint32_t best = 0;
            for (uint32_t ch = 1; ch < tp; ++ch) {
                if (sample::better(head_key[ch], head_id[ch], head_key[best], head_id[best])) {
                    best = ch;
                }
            }
            if (head[best] == K) {
                return false;
            }
            key = head_key[best];
            id = head_id[best];
            ++head[best];
            spent = spent || head[best] == K;
            load(best);
            return true;
        };

        uint32_t key0 = 0, id0 = 0;
        next(key0, id0);
        const uint32_t* p = params + u * 8;
        const uint32_t top_p = p[1];
        const uint32_t top_k = p[2];
        const sample::Rel rel(p[0], sample::key_bits(key0));
        uint64_t total = 0;
        for (uint32_t ch = 0; ch < tp; ++ch) {
            const uint32_t chip_max = cands[(ch * users + u) * 2 * K];
            total += sample::mul_q30(norms[ch * (norm_bytes / 8) + u * 2], rel(sample::key_bits(chip_max)));
        }
        const bool has_k = top_k >= 1;
        const uint32_t n_max = has_k && top_k < max_cands ? top_k : max_cands;
        const uint64_t limit = sample::mul_q24(total, top_p);
        probs[0] = sample::kOne30;
        ids[0] = id0;
        uint64_t cum = sample::kOne30;
        uint32_t n = 1;
        truncated = false;
        while (n < n_max) {
            if (!has_k && cum >= limit) {
                break;
            }
            truncated = truncated || spent;
            uint32_t key = 0, id = 0;
            if (!next(key, id)) {
                truncated = true;
                break;
            }
            const uint64_t q = rel(sample::key_bits(key));
            probs[n] = static_cast<uint32_t>(q);
            ids[n] = id;
            cum += q;
            ++n;
        }
        if (!has_k && cum < limit) {
            truncated = true;  // every candidate taken and the nucleus still open
        }
        uint32_t kept = n;
        if (has_k) {
            const uint64_t k_limit = sample::mul_q24(cum, top_p);
            uint64_t above = 0;
            kept = 0;
            while (kept < n && (kept == 0 || above < k_limit)) {
                above += probs[kept];
                ++kept;
            }
        }
        kept_total = 0;
        for (uint32_t i = 0; i < kept; ++i) {
            kept_total += probs[i];
        }
        by_id = !has_k;
        return kept;
    };

    // Inverse CDF over the kept candidates without `exclude`: the first in draw order whose running mass exceeds
    // u * (the kept mass without it); past every one (rounding), the last positive one.
    auto draw = [&](uint32_t kept, uint64_t kept_total, bool by_id, uint32_t u_q24, uint32_t exclude) -> uint32_t {
        for (uint32_t i = 0; i < kept; ++i) {
            order[i] = i;
        }
        if (by_id) {
            for (uint32_t i = 1; i < kept; ++i) {
                const uint32_t v = order[i];
                uint32_t j = i;
                while (j > 0 && ids[order[j - 1]] > ids[v]) {
                    order[j] = order[j - 1];
                    --j;
                }
                order[j] = v;
            }
        }
        uint64_t rest = kept_total;
        for (uint32_t i = 0; i < kept; ++i) {
            if (ids[i] == exclude) {
                rest -= probs[i];
            }
        }
        const uint64_t threshold = sample::mul_q24(rest, u_q24);
        uint64_t run = 0;
        uint32_t last = kNone;
        for (uint32_t i = 0; i < kept; ++i) {
            const uint32_t c = order[i];
            if (ids[c] == exclude || probs[c] == 0) {
                continue;
            }
            last = ids[c];
            run += probs[c];
            if (run > threshold) {
                return ids[c];
            }
        }
        return last != kNone ? last : ids[0];
    };

    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t* c = ctrl + g * group_words;
        volatile tt_l1_ptr uint32_t* o = out + g * out_words;
        o[0] = 0;
        for (uint32_t j = 0; j <= D; ++j) {
            o[1 + j] = kNone;
        }
        for (uint32_t j = 2 + D; j < out_words; ++j) {
            o[j] = 0;
        }
        const uint32_t first = c[0];
        if (first == kNone) {
            continue;
        }
        const uint32_t n = c[1];
        const uint32_t* drafts = c + 4;
        const uint32_t u_bonus = c[4 + D];
        const uint32_t* u_accept = c + 5 + D;
        const uint32_t u_corr = c[5 + 2 * D];
        if (c[2] != 0) {
            uint32_t m = n;
            for (uint32_t j = 0; j < n; ++j) {
                o[1 + j] = argmax[first + j];
                if (argmax[first + j] != drafts[j]) {
                    m = j;
                    break;
                }
            }
            if (m == n) {
                o[1 + n] = argmax[first + n];
            }
            o[0] = m + 1;
            continue;
        }
        bool any_truncated = false;
        uint32_t m = n;
        for (uint32_t j = 0; j < n; ++j) {
            uint64_t kept_total = 0;
            bool by_id = false, truncated = false;
            const uint32_t kept = build(first + j, kept_total, by_id, truncated);
            any_truncated = any_truncated || truncated;
            uint64_t q = 0;
            for (uint32_t i = 0; i < kept; ++i) {
                if (ids[i] == drafts[j]) {
                    q = probs[i];
                    break;
                }
            }
            if (q > 0 && q >= sample::mul_q24(kept_total, u_accept[j])) {
                o[1 + j] = drafts[j];
                continue;
            }
            o[1 + j] = draw(kept, kept_total, by_id, u_corr, drafts[j]);
            m = j;
            break;
        }
        if (m == n) {
            uint64_t kept_total = 0;
            bool by_id = false, truncated = false;
            const uint32_t kept = build(first + n, kept_total, by_id, truncated);
            any_truncated = any_truncated || truncated;
            o[1 + n] = draw(kept, kept_total, by_id, u_bonus, kNone);
        }
        o[0] = m + 1;
        o[2 + D] = any_truncated ? 1 : 0;
    }

    noc_async_write(out_l1, page_addr<true>(out_addr, 0, out_bytes), out_bytes);
    noc_async_write_barrier();
}
