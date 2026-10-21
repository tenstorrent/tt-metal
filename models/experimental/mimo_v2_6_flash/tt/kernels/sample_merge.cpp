// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 3 (tt/device_sampler.py): each user's token from every chip's sorted top-K list.
//
// Inputs (all interleaved):
// - cands: the row all-gather of the chip candidates, uint32 TILE with tp pages; page ch holds chip ch's lists, user
//   u's K (key, local index) pairs sorted (sample_chip_merge.cpp) at byte offset u * K * 8.
// - sums: the row all-gather of each chip's sum of exp((logit - chip max) / T) over its shard, fp32 TILE with tp
//   pages; page ch, row u, column 0 (face 0, byte offset u * 64).
// - params: uint32 row-major [1, max_batch * 4]: per row (log2(e) / T as fp32 bits, top_p in Q24, top_k (0: none),
//   greedy).
// - uniforms: uint32 row-major [1, max_batch]: one draw in [0, 1) per row, in Q24.
// The candidates of a user are walked in descending logit order (lowest global id first among equal logits) by
// merging the chips' sorted lists. A greedy row takes the first. Otherwise, as vLLM's sampler: probabilities are
// softmax(logits / T) over the vocabulary (the chips' sums), or over the first top_k candidates when top_k is set;
// the nucleus keeps a candidate while the probability ranked above it is below top_p (the first always); the token is
// the first kept candidate whose cumulative probability exceeds uniform * the kept mass.
// The data movement cores have no FPU: everything is fixed point. Logits are Q24, a candidate's probability relative
// to the largest, 2^((logit - largest) * log2(e) / T), is Q30 (absolute error ~1e-9), sums of those are uint64.
// Output: global token ids (chip * vocab_local + local index) of users 0..users-1 and zeros for the other slots,
// one uint32 [1, max_batch] row-major page written to each of up to two tensors (as GreedyMerge).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kOne30 = 1u << 30;

// 2^(j / 64) for j = 0..63, Q30.
constexpr uint32_t kExp2Table[64] = {
    1073741824, 1085434106, 1097253708, 1109202018, 1121280436, 1133490379, 1145833280, 1158310587,
    1170923762, 1183674286, 1196563654, 1209593378, 1222764986, 1236080024, 1249540052, 1263146652,
    1276901417, 1290805962, 1304861917, 1319070932, 1333434672, 1347954824, 1362633090, 1377471191,
    1392470869, 1407633882, 1422962010, 1438457051, 1454120821, 1469955159, 1485961921, 1502142985,
    1518500250, 1535035634, 1551751076, 1568648537, 1585730000, 1602997467, 1620452965, 1638098541,
    1655936265, 1673968228, 1692196547, 1710623359, 1729250827, 1748081133, 1767116489, 1786359126,
    1805811301, 1825475297, 1845353420, 1865448001, 1885761398, 1906295993, 1927054196, 1948038440,
    1969251188, 1990694927, 2012372174, 2034285470, 2056437387, 2078830522, 2101467502, 2124350982};
constexpr uint64_t kLn2Q30 = 744261118;  // ln(2) in Q30

FORCE_INLINE bool better(uint32_t key, uint32_t id, uint32_t other_key, uint32_t other_id) {
    return key > other_key || (key == other_key && id < other_id);
}

// fp32 bits of a sort key (inverse of sample_scan.cpp's sort_key).
FORCE_INLINE uint32_t key_bits(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key; }

// An fp32 (bits) times 2^frac, toward zero, as int64; magnitudes past 2^62 saturate.
int64_t to_fixed(uint32_t bits, int frac) {
    const int e = static_cast<int>((bits >> 23) & 0xFF);
    if (e == 0) {
        return 0;
    }
    const int64_t m = static_cast<int64_t>((bits & 0x7FFFFF) | 0x800000);
    const int sh = e - 150 + frac;
    int64_t v;
    if (sh >= 0) {
        v = sh >= 39 ? (int64_t{1} << 62) : (m << sh);
    } else {
        v = sh <= -32 ? 0 : (m >> -sh);
    }
    return (bits & 0x80000000u) ? -v : v;
}

// 2^t for t <= 0 given in Q24, as Q30 (0 below 2^-31).
uint64_t exp2_q30(int64_t t) {
    const int64_t n = t >> 24;  // floor
    if (n < -31) {
        return 0;
    }
    const uint32_t f = static_cast<uint32_t>(t & 0xFFFFFF);  // [0, 1) in Q24
    const uint32_t j = f >> 18;                              // 2^(j / 64) from the table
    const uint64_t g = f & 0x3FFFF;                          // the rest, < 1/64, Q24
    // e^h with h = g * ln2 < 0.011: 1 + h + h^2 / 2 + h^3 / 6 (error < 1e-9), Q30.
    const uint64_t h = (g * kLn2Q30) >> 24;
    const uint64_t h2 = (h * h) >> 30;
    const uint64_t h3 = (h2 * h) >> 30;
    const uint64_t eh = kOne30 + h + (h2 >> 1) + static_cast<uint32_t>(h3) / 6u;
    const uint64_t p = (static_cast<uint64_t>(kExp2Table[j]) * eh) >> 30;  // 2^f in Q30, [2^30, 2^31)
    return p >> static_cast<uint32_t>(-n);
}

// a * b / 2^24 for b <= 2^24 (a Q24 fraction) without overflow for a < 2^63.
FORCE_INLINE uint64_t mul_q24(uint64_t a, uint32_t b) { return (a >> 24) * b + (((a & 0xFFFFFF) * b) >> 24); }

template <bool dram>
FORCE_INLINE uint64_t page_addr(uint32_t base, uint32_t page, uint32_t page_size) {
    const InterleavedAddrGen<dram> g = {.bank_base_address = base, .page_size = page_size};
    return get_noc_addr(page, g);
}

FORCE_INLINE uint64_t page_addr(bool dram, uint32_t base, uint32_t page, uint32_t page_size) {
    return dram ? page_addr<true>(base, page, page_size) : page_addr<false>(base, page, page_size);
}

}  // namespace

void kernel_main() {
    const uint32_t cands_addr = get_arg_val<uint32_t>(0);
    const uint32_t cands_is_dram = get_arg_val<uint32_t>(1);
    const uint32_t sums_addr = get_arg_val<uint32_t>(2);
    const uint32_t sums_is_dram = get_arg_val<uint32_t>(3);
    const uint32_t params_addr = get_arg_val<uint32_t>(4);
    const uint32_t uniforms_addr = get_arg_val<uint32_t>(5);
    const uint32_t out0_addr = get_arg_val<uint32_t>(6);
    const uint32_t out0_is_dram = get_arg_val<uint32_t>(7);
    const uint32_t out1_addr = get_arg_val<uint32_t>(8);  // 0: none
    const uint32_t out1_is_dram = get_arg_val<uint32_t>(9);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t tp = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t vocab_local = get_compile_time_arg_val(3);
    constexpr uint32_t max_batch = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(5);
    static_assert(users >= 1 && users <= 4 && tp <= 32 && max_batch * 4 <= 1024);
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t cand_bytes = users * slot_bytes;  // every user's slot of a chip page
    constexpr uint32_t sums_bytes = 256;                 // rows 0-3 of face 0
    constexpr uint32_t params_bytes = max_batch * 16;
    constexpr uint32_t uniforms_bytes = max_batch * 4;
    constexpr uint32_t max_cands = tp * K;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t cands_l1 = get_write_ptr(cb_scratch);
    const uint32_t sums_l1 = cands_l1 + tp * cand_bytes;
    const uint32_t params_l1 = sums_l1 + tp * sums_bytes;
    const uint32_t uniforms_l1 = params_l1 + params_bytes;
    const uint32_t probs_l1 = uniforms_l1 + 1024;
    const uint32_t ids_l1 = probs_l1 + max_cands * 4;
    const uint32_t tokens_l1 = ids_l1 + max_cands * 4;

    for (uint32_t ch = 0; ch < tp; ++ch) {
        noc_async_read(page_addr(cands_is_dram, cands_addr, ch, 4096), cands_l1 + ch * cand_bytes, cand_bytes);
        noc_async_read(page_addr(sums_is_dram, sums_addr, ch, 4096), sums_l1 + ch * sums_bytes, sums_bytes);
    }
    noc_async_read(page_addr<true>(params_addr, 0, params_bytes), params_l1, params_bytes);
    noc_async_read(page_addr<true>(uniforms_addr, 0, uniforms_bytes), uniforms_l1, uniforms_bytes);
    noc_async_read_barrier();

    // Read only after the barrier, and written by nothing else: plain pointers.
    const uint32_t* cands = reinterpret_cast<const uint32_t*>(cands_l1);
    const uint32_t* sums = reinterpret_cast<const uint32_t*>(sums_l1);
    const uint32_t* params = reinterpret_cast<const uint32_t*>(params_l1);
    const uint32_t* uniforms = reinterpret_cast<const uint32_t*>(uniforms_l1);
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
                if (better(head_key[ch], head_id[ch], head_key[best], head_id[best])) {
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
        if (params[u * 4 + 3] != 0) {
            tokens[u] = id0;
            continue;
        }
        // log2(e) / T = scale_m * 2^scale_e (scale_m: the 24-bit mantissa).
        const uint32_t scale_bits = params[u * 4];
        const uint64_t scale_m = (scale_bits & 0x7FFFFF) | 0x800000;
        const int scale_e = static_cast<int>((scale_bits >> 23) & 0xFF) - 150;
        const uint32_t top_p = params[u * 4 + 1];
        const uint32_t top_k = params[u * 4 + 2];
        const int64_t v0 = to_fixed(key_bits(key0), 24);
        // A logit's probability relative to the largest, 2^((v - v0) * log2(e) / T), Q30.
        auto rel = [&](uint32_t key) -> uint64_t {
            const int64_t d = to_fixed(key_bits(key), 24) - v0;  // <= 0, Q24
            if (d <= -(int64_t{1} << 31)) {                      // 128 below the largest: Q30 zero unless T > 4
                return d < -(int64_t{1} << 34) ? 0
                                               : exp2_q30(((d >> 8) * static_cast<int64_t>(scale_m)) >> (-scale_e - 8));
            }
            const int64_t prod = static_cast<int64_t>(static_cast<int32_t>(d)) * static_cast<int32_t>(scale_m);
            const int64_t t = scale_e >= 0 ? (prod << scale_e) : (prod >> -scale_e);
            return exp2_q30(t);
        };
        // The vocabulary's normalizer relative to the largest: each chip's sum is relative to its own largest logit.
        uint64_t total = 0;
        for (uint32_t ch = 0; ch < tp; ++ch) {
            const uint32_t s = sums[ch * (sums_bytes / 4) + u * 16];
            const uint64_t sm = (s & 0x7FFFFF) | 0x800000;
            const int se = static_cast<int>((s >> 23) & 0xFF) - 150;          // S >= 1: se >= -23; S < 2^15: se < -8
            const uint64_t prod = sm * rel(cands[(ch * users + u) * 2 * K]);  // < 2^54
            total += se >= 0 ? (prod << se) : (prod >> -se);
        }
        const bool has_k = top_k >= 1;
        const uint32_t n_max = has_k && top_k < max_cands ? top_k : max_cands;
        const uint64_t limit = mul_q24(total, top_p);
        probs[0] = kOne30;
        ids[0] = id0;
        uint64_t cum = kOne30;
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
            const uint64_t p = rel(key);
            probs[n] = static_cast<uint32_t>(p);
            ids[n] = id;
            cum += p;
            ++n;
        }
        uint32_t kept = n;
        if (has_k) {
            // Normalized over the top_k candidates: the nucleus of those.
            const uint64_t k_limit = mul_q24(cum, top_p);
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
        const uint64_t threshold = mul_q24(kept_total, uniforms[u]);
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

    constexpr uint32_t bytes = max_batch * 4;
    noc_async_write(tokens_l1, page_addr(out0_is_dram, out0_addr, 0, bytes), bytes);
    if (out1_addr != 0) {
        noc_async_write(tokens_l1, page_addr(out1_is_dram, out1_addr, 0, bytes), bytes);
    }
    noc_async_write_barrier();
}
