// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The KV pair split's decode merge (tt/pair_split.py PairSplit.merge_heads) in one launch: from this chip's partial
// attention o [1, B, heads, D] (bf16 tiles, interleaved; user b's head h is row h % 32 of its tile row h / 32), its LSE
// [1, B, heads, 1] (fp32 tiles: head h in row h, column 0) and the partner's LSEs (sent in the same launch by the
// partner's exchange core, kernels/attn/pair_lse_send.cpp, into slot (n & 1) of this chip's exchange buffer: 64 bytes
// per user, head h at word h) to the concatenated heads of alpha * o [1, 1, 32, heads * vd] (bf16 tiles, L1
// interleaved; rows >= B zeros), alpha = 1 / (1 + exp(clamp(lse_p - lse, -c, c))) (reference/kv_pair_split.py
// merge_weights_device). Row b of output tile column h * v_tiles + dt is row h of user b's tile column dt times alpha.
// n is this kernel's call count (a persistent global semaphore per core and RISC, never reset): the partner's packet of
// call n is the exchange core's arrival n + 1, polled over the NoC.
// The data movement RISCs have no FPU: alpha is computed in fixed point (the LSE difference exact in Q32.32, exp by a
// Taylor polynomial and squarings, 1 / x by Newton steps) to ~1e-8, kept with 24 significant bits (as fp32), and every
// bf16 product is formed exactly in integers and rounded once to nearest even (0 stays exactly 0).
// Two kernels per core (BRISC: column face 0, NCRISC: column face 1) each do output tile columns [first, first +
// tiles): their faces `face` (rows 0-15) and face + 2 (rows 16-31).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kLog2e = uint32_t(1.4426950408889634 * 2147483648.0 + 0.5);  // log2(e), Q1.31
constexpr uint32_t kLn2 = uint32_t(0.6931471805599453 * 4294967296.0 + 0.5);    // ln(2), Q0.32
constexpr uint32_t kRecip0 = uint32_t(24.0 / 17.0 * 2147483648.0 + 0.5);        // 24 / 17, Q1.31
constexpr uint32_t kRecip1 = uint32_t(16.0 / 17.0 * 4294967296.0 + 0.5);        // 16 / 17, Q0.32

FORCE_INLINE uint32_t mulhi(uint32_t a, uint32_t b) { return uint32_t((uint64_t(a) * b) >> 32); }

// An fp32 LSE (bits) in Q32.32, saturated at +-2^24 (NO_KEY_LSE saturates; the clamp of the difference then gives
// what fp32 gives), values below 2^-32 as 0.
FORCE_INLINE int64_t lse_fixed(uint32_t bits) {
    const uint32_t e = (bits >> 23) & 0xff;
    int64_t v = 0;
    if (e >= 151) {
        v = int64_t(1) << 56;
    } else if (e >= 95) {
        const uint64_t m = (bits & 0x7fffffu) | 0x800000u;  // value = m 2^(e - 150)
        const int32_t sh = int32_t(e) - 118;
        v = int64_t(sh >= 0 ? m << sh : m >> -sh);
    }
    return (bits >> 31) ? -v : v;
}

// 1 / s for s in [1, 2] (Q2.30), in Q1.31: a linear first guess (within 1/17) and three Newton steps.
FORCE_INLINE uint32_t reciprocal(uint32_t s) {
    if (s == 0x40000000u) {
        return 0x80000000u;
    }
    uint32_t x = kRecip0 - mulhi(s, kRecip1);  // 24/17 - 8/17 s
    for (uint32_t i = 0; i < 3; i++) {
        const uint32_t e = uint32_t(((uint64_t(1) << 62) - uint64_t(s) * x) >> 30);  // 2 - s x, Q1.31
        x = uint32_t((uint64_t(x) * e) >> 31);
    }
    return x;
}

// alpha = 1 / (1 + exp(clamp(lse_p - lse, -clamp, clamp))) = m 2^-x, m in [2^23, 2^24).
FORCE_INLINE void merge_weight(uint32_t lse_bits, uint32_t lse_p_bits, int64_t clamp, uint32_t& m24, int32_t& x) {
    int64_t d = lse_fixed(lse_p_bits) - lse_fixed(lse_bits);
    if (d > clamp) {
        d = clamp;
    } else if (d < -clamp) {
        d = -clamp;
    }
    const bool pos = d > 0;
    const uint64_t a = uint64_t(pos ? d : -d);
    // |d| log2(e) in Q32.32: exp(-|d|) = 2^-k 2^-f
    const uint64_t y = ((uint64_t(uint32_t(a >> 32)) * kLog2e) << 1) + ((uint64_t(uint32_t(a)) * kLog2e) >> 31);
    const uint32_t k = uint32_t(y >> 32);
    const uint32_t f = uint32_t(y);
    // 2^-f = exp(-w)^16, w = f ln2 / 16 < 0.0434: Taylor to w^4 (Q0.32), then four squarings (Q1.31)
    const uint32_t w = mulhi(f, kLn2) >> 4;
    const uint32_t w2 = mulhi(w, w);
    const uint32_t w3 = mulhi(w2, w);
    const uint32_t w4 = mulhi(w3, w);
    uint32_t p = 0x80000000u - ((w - (w2 >> 1) + w3 / 6 - w4 / 24) >> 1);
    for (uint32_t i = 0; i < 4; i++) {
        p = uint32_t((uint64_t(p) * p) >> 31);
    }
    // t = exp(-|d|) = p 2^-31 2^-k; r = 1 / (1 + t); alpha = r (d <= 0) or t r (d > 0)
    const uint32_t s = 0x40000000u + (k < 31 ? p >> (k + 1) : 0);
    const uint32_t r = reciprocal(s);
    uint32_t m;
    int32_t e;
    if (pos) {
        m = uint32_t((uint64_t(p) * r) >> 30);
        e = 32 + int32_t(k);
    } else {
        m = r;
        e = 31;
    }
    for (uint32_t i = 0; i < 4 && m < 0x80000000u; i++) {
        m <<= 1;
        e++;
    }
    m24 = m >> 8;
    x = e - 8;
}

// bf16(v * m 2^-x), the exact product rounded once to nearest even; 0 (and subnormals) give 0.
FORCE_INLINE uint32_t scale_bf16(uint32_t v, uint32_t m, int32_t x) {
    const uint32_t eo = (v >> 7) & 0xff;
    if (eo == 0) {
        return 0;
    }
    if (eo == 0xff) {
        return v;
    }
    const uint32_t prod = ((v & 0x7f) | 0x80) * m;  // [2^30, 2^32)
    const uint32_t n = (prod >> 31) ? 24 : 23;
    uint32_t mant = prod >> n;
    const uint32_t rem = prod - (mant << n);
    const uint32_t half = 1u << (n - 1);
    int32_t e = int32_t(eo + n) - x;
    if (rem > half || (rem == half && (mant & 1))) {
        if (++mant == 256) {
            mant = 128;
            ++e;
        }
    }
    if (e <= 0) {
        return 0;
    }
    if (e >= 255) {
        return (v & 0x8000) | 0x7f80;
    }
    return (v & 0x8000) | (uint32_t(e) << 7) | (mant & 0x7f);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t face = get_compile_time_arg_val(1);          // output column face: 0 (columns 0-15) or 1 (16-31)
    constexpr uint32_t tiles = get_compile_time_arg_val(2);         // output tile columns per core
    constexpr uint32_t users = get_compile_time_arg_val(3);         // B (<= 32)
    constexpr uint32_t o_user_pages = get_compile_time_arg_val(4);  // o tiles per user
    constexpr uint32_t d_tiles = get_compile_time_arg_val(5);       // o tile columns (padded D / 32)
    constexpr uint32_t v_tiles = get_compile_time_arg_val(6);       // kept tile columns per head (vd / 32)
    constexpr uint32_t lse_user_pages = get_compile_time_arg_val(7);  // LSE tiles per user
    constexpr bool o_dram = get_compile_time_arg_val(8) == 1;
    constexpr bool lse_dram = get_compile_time_arg_val(9) == 1;
    constexpr int64_t clamp = int64_t(get_compile_time_arg_val(10)) << 32;
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(11);  // one slot of the exchange buffer
    constexpr uint32_t bf16_tile = 2048;
    constexpr uint32_t fp32_tile = 4096;
    constexpr uint32_t face_bytes = 512;  // bf16 face
    constexpr uint32_t row_bytes = 32;    // one bf16 face row
    constexpr uint32_t lse_row = 64;      // one fp32 face row
    constexpr uint32_t packed_row = 64;   // a user's 16 partner LSEs in the exchange slot
    static_assert(users >= 1 && users <= 32 && face < 2 && users * packed_row <= slot_bytes);

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t lse_addr = get_arg_val<uint32_t>(1);
    const uint32_t slots_addr = get_arg_val<uint32_t>(2);
    const uint32_t arrive_addr = get_arg_val<uint32_t>(3);
    const uint32_t count_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);
    const uint32_t first = get_arg_val<uint32_t>(6);
    const uint32_t ex_x = get_arg_val<uint32_t>(7);  // the exchange core
    const uint32_t ex_y = get_arg_val<uint32_t>(8);

    // Scratch (64-byte aligned): the polled arrival count, the LSE rows, the partner's packed LSEs, o's rows, two
    // output faces.
    cb_reserve_back(cb_scratch, 1);
    const uint32_t base = (get_write_ptr(cb_scratch) + 63) & ~63u;
    const uint32_t poll_l1 = base;
    const uint32_t lse_l1 = base + 64;
    const uint32_t lse_p_l1 = lse_l1 + users * lse_row;
    const uint32_t in_l1 = lse_p_l1 + users * packed_row;
    const uint32_t out_l1 = in_l1 + users * row_bytes;

    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(count_addr);
    const uint32_t calls = count[0];
    const uint64_t arrive_noc = get_noc_addr(ex_x, ex_y, arrive_addr);
    const uint64_t slot_noc = get_noc_addr(ex_x, ex_y, slots_addr + (calls & 1) * slot_bytes);
    volatile tt_l1_ptr uint32_t* poll = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll_l1);

    const InterleavedAddrGen<o_dram> o_gen = {.bank_base_address = o_addr, .page_size = bf16_tile};
    const InterleavedAddrGen<lse_dram> lse_gen = {.bank_base_address = lse_addr, .page_size = fp32_tile};
    const InterleavedAddrGen<false> out_gen = {.bank_base_address = out_addr, .page_size = bf16_tile};

    volatile tt_l1_ptr uint32_t* lse = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lse_l1);
    volatile tt_l1_ptr uint32_t* lse_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lse_p_l1);
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t tc = first + t;
        const uint32_t h = tc / v_tiles;
        const uint32_t dt = tc - h * v_tiles;
        const uint32_t o_page0 = (h >> 5) * d_tiles + dt;
        const uint32_t o_off = ((h >> 4) & 1) * 2 * face_bytes + face * face_bytes + (h & 15) * row_bytes;
        const uint32_t lse_page0 = h >> 5;
        const uint32_t lse_off = ((h >> 4) & 1) * 2048 + (h & 15) * lse_row;  // fp32 faces are 1 KB
        for (uint32_t b = 0; b < users; b++) {
            noc_async_read(o_gen.get_noc_addr(b * o_user_pages + o_page0, o_off), in_l1 + b * row_bytes, row_bytes);
            noc_async_read(
                lse_gen.get_noc_addr(b * lse_user_pages + lse_page0, lse_off), lse_l1 + b * lse_row, lse_row);
        }
        if (t == 0) {
            // The partner's packet of this call: arrival calls + 1 on the exchange core.
            for (;;) {
                noc_async_read(arrive_noc, poll_l1, 4);
                noc_async_read_barrier();
                invalidate_l1_cache();
                if (int32_t(poll[0] - (calls + 1)) >= 0) {
                    break;
                }
            }
            noc_async_read(slot_noc, lse_p_l1, users * packed_row);
        }
        noc_async_read_barrier();

        for (uint32_t b = 0; b < users; b++) {
            uint32_t m;
            int32_t x;
            merge_weight(lse[b * (lse_row / 4)], lse_p[b * (packed_row / 4) + h], clamp, m, x);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_l1 + b * row_bytes);
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1 + (b >> 4) * face_bytes + (b & 15) * row_bytes);
            for (uint32_t w = 0; w < row_bytes / 4; w++) {
                const uint32_t v = src[w];
                dst[w] = scale_bf16(v & 0xffff, m, x) | (scale_bf16(v >> 16, m, x) << 16);
            }
        }
        for (uint32_t b = users; b < 32; b++) {
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1 + (b >> 4) * face_bytes + (b & 15) * row_bytes);
            for (uint32_t w = 0; w < row_bytes / 4; w++) {
                dst[w] = 0;
            }
        }
        noc_async_write(out_l1, out_gen.get_noc_addr(tc, face * face_bytes), face_bytes);
        noc_async_write(out_l1 + face_bytes, out_gen.get_noc_addr(tc, (face + 2) * face_bytes), face_bytes);
        noc_async_write_barrier();
    }
    count[0] = calls + 1;
}
