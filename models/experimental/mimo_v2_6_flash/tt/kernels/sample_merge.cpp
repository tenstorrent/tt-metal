// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 3 (tt/device_sampler.py): each user's token from every chip's sorted top-K list.
//
// Inputs (interleaved):
// - cands: the row all-gather of the chip candidates (sample_chip_merge.cpp), uint32 TILE with tp pages; page ch
//   holds chip ch's sorted K (key, local index) pairs of user u at byte offset u * K * 8 and its normalizer (the sum
//   of exp((logit - chip ch's largest) / T), uint64 Q30) at byte offset 2048 + u * 16 (more than 8 users: two
//   pages per chip, the lists in page 2 ch and the normalizers at u * 16 of page 2 ch + 1).
// - params: uint32 row-major [1, max_batch * 8]: per row log2(e) / T as fp32 bits, top_p in Q24, top_k (0: none),
//   greedy, 1 / T as fp32 bits, three unused words.
// - rng: uint32 row-major [1, max_batch * 2]: per row a 64-bit splitmix64 state (low word first), seeded by the host;
//   each sampled row's draw advances it, and the kernel writes it back, so every replay draws anew.
// The candidates of a user are walked in descending logit order (lowest global id first among equal logits) by
// merging the chips' sorted lists. A greedy row takes the first. Otherwise, as vLLM's sampler: probabilities are
// softmax(logits / T) over the vocabulary (the chips' normalizers), or over the first top_k candidates when top_k is
// set; the nucleus keeps a candidate while the probability ranked above it is below top_p (the first always); the
// token is the first kept candidate whose cumulative probability exceeds u * the kept mass, u the row's next uniform
// in [0, 1) (the top 24 bits of its splitmix64 output). Fixed point
// throughout (sample_fixed.hpp).
// Output: global token ids (chip * vocab_local + local index) of users 0..users-1 and zeros for the other slots,
// one uint32 [1, max_batch] row-major page written to each of up to two tensors (as GreedyMerge).

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

}  // namespace

void kernel_main() {
    const uint32_t cands_addr = get_arg_val<uint32_t>(0);
    const uint32_t cands_is_dram = get_arg_val<uint32_t>(1);
    const uint32_t params_addr = get_arg_val<uint32_t>(2);
    const uint32_t rng_addr = get_arg_val<uint32_t>(3);
    const uint32_t out0_addr = get_arg_val<uint32_t>(4);
    const uint32_t out0_is_dram = get_arg_val<uint32_t>(5);
    const uint32_t out1_addr = get_arg_val<uint32_t>(6);  // 0: none
    const uint32_t out1_is_dram = get_arg_val<uint32_t>(7);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t tp = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t vocab_local = get_compile_time_arg_val(3);
    constexpr uint32_t max_batch = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(5);
    static_assert(users >= 1 && users <= 16 && tp <= 32 && max_batch * 4 <= 1024 && users * K * 8 <= 4096);
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t cand_bytes = users * slot_bytes;     // every user's list of a chip page
    constexpr uint32_t norm_bytes =
        users <= 4 ? 64 : (users <= 8 ? 128 : 256);  // every user's normalizer of a chip page
    constexpr uint32_t params_bytes = max_batch * 32;
    constexpr uint32_t rng_bytes = max_batch * 8;
    constexpr uint32_t max_cands = tp * K;
    // A chip's candidates: its users' lists then (at byte 2048) their normalizers in one page, or with more than 8
    // users the lists in one page and the normalizers in the next (sample_chip_merge.cpp).
    constexpr uint32_t cand_pages = cand_bytes > 2048 ? 2 : 1;
    constexpr uint32_t norm_at = cand_pages == 1 ? 2048 : 0;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t cands_l1 = get_write_ptr(cb_scratch);
    const uint32_t norms_l1 = cands_l1 + tp * cand_bytes;
    const uint32_t params_l1 = norms_l1 + tp * norm_bytes;
    const uint32_t rng_l1 = params_l1 + params_bytes;
    const uint32_t probs_l1 = rng_l1 + rng_bytes;
    const uint32_t ids_l1 = probs_l1 + max_cands * 4;
    const uint32_t tokens_l1 = ids_l1 + max_cands * 4;

    for (uint32_t ch = 0; ch < tp; ++ch) {
        noc_async_read(
            page_addr(cands_is_dram, cands_addr, ch * cand_pages, 4096), cands_l1 + ch * cand_bytes, cand_bytes);
        noc_async_read(
            page_addr(cands_is_dram, cands_addr, ch * cand_pages + cand_pages - 1, 4096, norm_at),
            norms_l1 + ch * norm_bytes,
            norm_bytes);
    }
    noc_async_read(page_addr<true>(params_addr, 0, params_bytes), params_l1, params_bytes);
    noc_async_read(page_addr<true>(rng_addr, 0, rng_bytes), rng_l1, rng_bytes);
    noc_async_read_barrier();

    // Read only after the barrier, and written by nothing else: plain pointers.
    const uint32_t* cands = reinterpret_cast<const uint32_t*>(cands_l1);
    const uint64_t* norms = reinterpret_cast<const uint64_t*>(norms_l1);
    const uint32_t* params = reinterpret_cast<const uint32_t*>(params_l1);
    uint32_t* rng = reinterpret_cast<uint32_t*>(rng_l1);
    uint32_t* probs = reinterpret_cast<uint32_t*>(probs_l1);
    uint32_t* ids = reinterpret_cast<uint32_t*>(ids_l1);
    volatile tt_l1_ptr uint32_t* tokens = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tokens_l1);

    for (uint32_t r = 0; r < max_batch; ++r) {
        tokens[r] = 0;
    }
    for (uint32_t u = 0; u < users; ++u) {
        // Chip ch's pair j of user u: cands[(ch * users + u) * 2K + 2j], key then local index. Each list's next
        // pair is cached (key 0 once spent: below every real key).
        uint32_t head[tp], head_key[tp], head_id[tp];
        auto load = [&](uint32_t ch) {
            const uint32_t e = (ch * users + u) * 2 * K + 2 * head[ch];
            head_key[ch] = head[ch] < K ? cands[e] : 0;
            head_id[ch] = head[ch] < K ? ch * vocab_local + cands[e + 1] : 0xFFFFFFFFu;
        };
        for (uint32_t ch = 0; ch < tp; ++ch) {
            head[ch] = 0;
            load(ch);
        }
        // The next candidate in (logit descending, global id ascending) order; false when every list is spent.
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
            load(best);
            return true;
        };

        uint32_t key0 = 0, id0 = 0;
        next(key0, id0);
        const uint32_t* p = params + u * 8;
        if (p[3] != 0) {
            tokens[u] = id0;
            continue;
        }
        const uint32_t top_p = p[1];
        const uint32_t top_k = p[2];
        const sample::Rel rel(p[0], sample::key_bits(key0));
        // The vocabulary's normalizer relative to the largest logit: each chip's is relative to its own largest.
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
        while (n < n_max) {
            // Without top_k the next candidate is in the nucleus only while the mass ranked above it is below top_p.
            if (!has_k && cum >= limit) {
                break;
            }
            uint32_t key = 0, id = 0;
            if (!next(key, id)) {
                break;
            }
            const uint64_t q = rel(sample::key_bits(key));
            probs[n] = static_cast<uint32_t>(q);
            ids[n] = id;
            cum += q;
            ++n;
        }
        uint32_t kept = n;
        if (has_k) {
            // Normalized over the top_k candidates: the nucleus of those.
            const uint64_t k_limit = sample::mul_q24(cum, top_p);
            uint64_t above = 0;
            kept = 0;
            while (kept < n && (kept == 0 || above < k_limit)) {
                above += probs[kept];
                ++kept;
            }
        }
        uint64_t kept_total = 0;
        for (uint32_t i = 0; i < kept; ++i) {
            kept_total += probs[i];
        }
        const uint64_t threshold = sample::mul_q24(kept_total, sample::next_uniform_q24(rng + 2 * u));
        uint32_t pick = kept - 1;
        uint64_t run = 0;
        for (uint32_t i = 0; i < kept; ++i) {
            run += probs[i];
            if (run > threshold) {
                pick = i;
                break;
            }
        }
        tokens[u] = ids[pick];
    }

    noc_async_write(rng_l1, page_addr<true>(rng_addr, 0, rng_bytes), rng_bytes);
    constexpr uint32_t bytes = max_batch * 4;
    noc_async_write(tokens_l1, page_addr(out0_is_dram, out0_addr, 0, bytes), bytes);
    if (out1_addr != 0) {
        noc_async_write(tokens_l1, page_addr(out1_is_dram, out1_addr, 0, bytes), bytes);
    }
    noc_async_write_barrier();
}
