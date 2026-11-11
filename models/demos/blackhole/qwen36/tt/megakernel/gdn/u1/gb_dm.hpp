// G-b (E1_GB): the type-10 GDN prefix INSIDE the engine GDN role, data-movement side (reader / writer hooks).
// Input: G-a's in-proj tiles in the GDN cores' staging CB 41 (tile t of the layer's [q k v z | a b] row -> a core / slot
// from G-a's layout; the host table below says which local slots go where). Per layer on every GDN core:
//   reader (NCRISC, inside gdn_in_wait()): ctrl + window + tap reads (before the in-proj lands), wait in_ready, PUSH its
//     landed tiles to the consumer cores' QIN pages (+1 on their QIN counter per tile), wait its own QIN tiles, stage the
//     conv operands (gpx window-out rows + shifted operands, the qkv tile stands in for SH_{K-1} at B*T = 32), write
//     win[1 - par] (live users: rows 0..Lw_w-1; HOLD users: the whole win[par] tile), then wait the prefix-done count.
//   compute (gb_compute.hpp): conv + SiLU + one-hot placement (tp's op sequence), silu(z), the SFPU gate pair (tg's).
//   writer (BRISC, inside gdn_in_wait()): the outputs to `pre` (op layout: kn | qn | vc | zs | 16 gate tiles per head;
//     q / k RAW: the step normalizes them itself, GDN_RAWKQ = the L2-root fallback), the gate head's coefficient tiles
//     (tg_writer's float ops, same order), then +1 prefix-done on every GDN core and wait for 24 (L + 1).
// Payload block (argblock words from OGB, same offsets on every core): see the k* constants.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "gpx_dm.hpp"

#ifndef GB_PULL_NOC
#define GB_PULL_NOC noc_index
#endif
#ifndef GB_GATE_DIRECT
#define GB_GATE_DIRECT 0
#endif
#ifndef GB_COEF_REP
#define GB_COEF_REP 9
#endif
#ifndef GB_KQPUSH
#define GB_KQPUSH 0
#endif
#ifndef GB_MCAST
#define GB_MCAST 0
#endif
#ifndef GBZ
#if defined(GB_ZONES) && GB_ZONES
#define GBZ(n) DeviceZoneScopedN(n)
#else
#define GBZ(n)
#endif
#endif
namespace gb {
inline uint32_t g_gb_pb;  // L1 address of the G-b payload block (same on every GDN core)
inline uint32_t g_gb_coef;  // L1 base of the step writer's coef DFB (captured at kernel start)

constexpr uint32_t kRest = 0;    // arrivals of vc (1) + zs (Vt) + the head's gate rows (VB) per layer (+1 each)
constexpr uint32_t kQinCnt = 1;  // QIN counter (+1 per tile pushed into this core's QIN)
constexpr uint32_t kNin = 2;     // tiles this core receives per layer
constexpr uint32_t kNconv = 3;   // conv jobs (<= 2): QIN pages 0, 1
// an output's destination (kDstW words): [nrect, (x0 | y0 << 16, x1 | y1 << 16, n) x 2, cb id | slot << 16]: the tile goes to
// slot `slot` of the consumers' DFB `cb id` (its base: every layer fills the whole DFB, so the slots never move)
constexpr uint32_t kDstW = 8;
constexpr uint32_t kConv = 4;    // per conv job: [col, dst (kDstW)]
constexpr uint32_t kConvW = 1 + kDstW;
constexpr uint32_t kSilu = kConv + 2 * kConvW;  // [has, dst (kDstW)] (QIN page 2)
constexpr uint32_t kGate = kSilu + 1 + kDstW;   // gate head (QIN page 3: the a|b tile; every core builds its users' rows)
constexpr uint32_t kGdst = kGate + 1;           // first `pre` page of the head's 16 gate tiles
constexpr uint32_t kGcore = kGate + 2;          // this core's block index s within the head (its users: [s NU, (s + 1) NU))
constexpr uint32_t kNpush = kGate + 3;          // pushes of this core's landed tiles
constexpr uint32_t kPush = kGate + 4;           // per push: [slot, dst x | y << 16, page]
constexpr uint32_t kPushMax = 32;
constexpr uint32_t kPerLayer = kPush + 3 * kPushMax + 3 + kDstW + 6;  // per layer: [dtb bits, nea bits] of the gate head
constexpr uint32_t kKqCnt = kPush + 3 * kPushMax;   // arrivals of the normalized k / q tiles into knq (2 Kt per layer)
constexpr uint32_t kRawCnt = kKqCnt + 1;            // L2 roots: arrivals of their 4 raw tiles (Kt per layer)
constexpr uint32_t kRootPost = kKqCnt + 2;          // L2 root: post-scale bits (1.0 for k, SCALE for q); 0 = not a root
constexpr uint32_t kRootDst = kKqCnt + 3;           // L2 root: its consumers (rects; the ready +1 goes to their kKqCnt)
constexpr uint32_t kRootSrc = kRootDst + kDstW;     // consumer: NoC xy of its key head's k root, q root (pull model)
constexpr uint32_t kRootDone = kRootSrc + 2;        // L2 root: consumers' "read" +1 (rf VB per layer) before OUT pops
constexpr uint32_t kOutBase = kRootDone + 1;        // L1 base of OUT (stored by the writer at init: the reader's view)
constexpr uint32_t kDbgR = kOutBase + 1, kDbgW = kOutBase + 2;  // hang triage: last wait (L << 8 | step)
static_assert(kPerLayer == 148, "gb_compute.hpp / e1_shim mirror this layout");
constexpr uint32_t kNone = 0xFFFFFFFFu;

// ---------------- reader ----------------
// CBs (4 ids): QPX = [QIN 0..3 | P 4..5 | raw scratch] (reader pushes / compute pops all of it per layer: fixed slots),
// SHT = per conv job [SH_0 SH_1 SH_2 | tap] (4 bf16 pages), OUT (fp32: gate pair, silu, convs, pad to even), KQ (raw k | q).
// Scratch (QPX tail): win_l1: 2 x B bf16 tile slots (one window set per conv job); wo_l1: B x 1 KB window-out slots (row Lw
// zeroed once); ctrl_l1: the ctrl page.
// L1 diet (in-model CB floor): compact window / window-out slots -- 512 B per (job, user) = [face-0 rows 0..7 | face-1 rows
// 0..7] (rows of face hh at hh * 256 + r * 32) instead of whole tiles; HOLD users' whole-tile copy win[par] -> win[1-par]
// goes through one 2 KB bounce, before in_ready. SHT holds one job (4 pages): job 1's tap waits in tap1_l1.
constexpr uint32_t kCs = 512;
template <uint32_t B, uint32_t T, uint32_t K, uint32_t CT, uint32_t M, uint32_t NV, uint32_t NC, uint32_t KT_, typename WAcc,
          typename TAcc, typename CAcc, typename DAcc>
inline void reader_layer(volatile tt_l1_ptr uint32_t* pb, uint32_t L, const WAcc& wa, const WAcc& wb, const TAcc& taps,
                         const CAcc& ctrl, uint32_t ctrl_bytes, uint32_t gin_l1, uint32_t qin_l1, uint32_t win_l1,
                         uint32_t wo_l1, uint32_t ctrl_l1, uint32_t cb_qpx, uint32_t npx, uint32_t cb_sht, uint32_t cb_kq,
                         volatile tt_l1_ptr uint32_t* in_ready, uint32_t in_target, const DAcc& dtb, const DAcc& nea,
                         uint32_t bounce_l1, uint32_t tap1_l1) {
    static_assert(B * T == 32 && K - 1 + T <= 8, "G-b: B * T == 32; the window rows fit face rows 0..7");
    constexpr uint32_t Lw = K - 1 + T, Lw_w = (Lw + 1) & ~1u;
    const uint32_t nconv = pb[kNconv];
    gpx::Ctrl<B, T, M, NV> c;
    {
        GBZ("GB-RPRE");
        noc_async_read(ctrl.get_noc_addr(0), ctrl_l1, ctrl_bytes);
        // this device's dt_bias / -exp(A_log) of the gate head (fp32 [1, 1, Nv] tiles: row 0 of face 0 = heads 0..15) into the
        // per-layer payload words the compute's gate pair reads (per device on the mesh: a replicated host value would be wrong)
        noc_async_read(dtb.get_noc_addr(0), ctrl_l1 + 512, 64);
        noc_async_read(nea.get_noc_addr(0), ctrl_l1 + 576, 64);
        noc_async_read_barrier();
        c.parse(ctrl_l1);
        {
            const uint32_t hh = pb[kGate];
            pb[kPerLayer + 2 * L] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_l1 + 512)[hh];
            pb[kPerLayer + 2 * L + 1] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_l1 + 576)[hh];
        }
        cb_reserve_back(cb_sht, 4);  // job 0's [SH | tap]; job 1's tap waits in tap1_l1
        for (uint32_t j = 0; j < nconv; ++j) {
            const uint32_t col = pb[kConv + j * kConvW];
            for (uint32_t u = 0; u < B; ++u) {
                const uint64_t src = c.par ? wb.get_noc_addr(u * CT + col) : wa.get_noc_addr(u * CT + col);
                const uint32_t d = win_l1 + (j * B + u) * kCs;
                noc_async_read(src, d, 256);
                noc_async_read(src + 512, d + 256, 256);
            }
            noc_async_read(taps.get_noc_addr(col), j ? tap1_l1 : get_write_ptr(cb_sht) + 3 * gpx::kTB, gpx::kTB);
        }
        noc_async_read_barrier();
        // HOLD users: the op copies their window tile through unchanged (win[par] -> win[1-par]); off the in-proj's path
        for (uint32_t j = 0; j < nconv; ++j) {
            const uint32_t col = pb[kConv + j * kConvW];
            for (uint32_t u = 0; u < B; ++u) {
                if (!c.hold[u]) continue;
                noc_async_read(c.par ? wb.get_noc_addr(u * CT + col) : wa.get_noc_addr(u * CT + col), bounce_l1, gpx::kTB);
                noc_async_read_barrier();
                noc_async_write(bounce_l1, c.par ? wa.get_noc_addr(u * CT + col) : wb.get_noc_addr(u * CT + col), gpx::kTB);
                noc_async_write_barrier();
            }
        }
        // window-out rows 0..K-2 (window rows mi+1..) of every job, before the in-proj lands
        for (uint32_t j = 0; j < nconv; ++j) {
            for (uint32_t u = 0; u < B; ++u) {
                const uint32_t w = win_l1 + (j * B + u) * kCs, o = wo_l1 + (j * B + u) * kCs;
                for (uint32_t hh = 0; hh < 2; ++hh) gpx::l1cp(o + hh * 256, w + hh * 256 + (c.mi[u] + 1) * 32, (K - 1) * 32);
            }
        }
    }
    {
        GBZ("GB-GIN");
        pb[kDbgR] = (L << 8) | 1;
        if (pb[kNpush]) noc_semaphore_wait_min(in_ready, in_target);  // no landed tiles (AR corners): nothing to wait for
    }
    {
        GBZ("GB-PUSH");
        const uint32_t np = pb[kNpush];
        for (uint32_t i = 0; i < np; ++i) {
            const uint32_t* e = const_cast<const uint32_t*>(pb + kPush + 3 * i);
            const uint32_t xy = e[1];
            noc_async_write(gin_l1 + e[0] * gpx::kTB, get_noc_addr(xy & 0xFFFFu, xy >> 16, qin_l1 + e[2] * gpx::kTB),
                            gpx::kTB);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < np; ++i) {
            const uint32_t xy = pb[kPush + 3 * i + 1];
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFFu, xy >> 16, reinterpret_cast<uint32_t>(pb + kQinCnt)), 1);
        }
    }
    noc_async_read_barrier();
    {
        GBZ("GB-QIN");
        pb[kDbgR] = (L << 8) | 2;
        noc_semaphore_wait_min(pb + kQinCnt, pb[kNin] * (L + 1));
    }
    cb_reserve_back(cb_qpx, npx);
    cb_push_back(cb_qpx, npx);
    {
        GBZ("GB-STG");
        for (uint32_t j = 0; j < nconv; ++j) {  // rows K-1.. of the window-out = the qkv rows, then the shifted operands
            const uint32_t q = qin_l1 + j * gpx::kTB;
            if (j) {  // job 1: its SHT slot after the compute popped job 0; its tap from tap1_l1
                cb_reserve_back(cb_sht, 4);
                gpx::l1cp(get_write_ptr(cb_sht) + 3 * gpx::kTB, tap1_l1, gpx::kTB);
            }
            const uint32_t sh = get_write_ptr(cb_sht);
            for (uint32_t u = 0; u < B; ++u) {
                const uint32_t o = wo_l1 + (j * B + u) * kCs;
                for (uint32_t hh = 0; hh < 2; ++hh) gpx::l1cp(o + hh * 256 + (K - 1) * 32, q + gpx::row_off(u * T, hh), T * 32);
            }
            noc_async_read_barrier();
            for (uint32_t k = 0; k < K - 1; ++k) {
                for (uint32_t u = 0; u < B; ++u) {
                    const uint32_t o = wo_l1 + (j * B + u) * kCs;
                    for (uint32_t hh = 0; hh < 2; ++hh)
                        gpx::l1cp(sh + k * gpx::kTB + gpx::row_off(u * T, hh), o + hh * 256 + k * 32, T * 32);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_sht, 4);
        }
        for (uint32_t j = 0; j < nconv; ++j) {  // win[1 - par] rows 0..Lw_w-1 of the live users (off the convs' path)
            const uint32_t col = pb[kConv + j * kConvW];
            for (uint32_t u = 0; u < B; ++u) {
                if (c.hold[u]) continue;
                const uint64_t dst = c.par ? wa.get_noc_addr(u * CT + col) : wb.get_noc_addr(u * CT + col);
                const uint32_t o = wo_l1 + (j * B + u) * kCs;
                noc_async_write(o, dst, Lw_w * 32);
                noc_async_write(o + 256, dst + 512, Lw_w * 32);
            }
        }
        noc_async_write_barrier();
    }
    if (pb[kRootPost]) {  // L2 root: its Kt raw tiles (from the q / k conv cores) -> the compute's l2
        GBZ("GB-RAWW");
        pb[kDbgR] = (L << 8) | 3;
        noc_semaphore_wait_min(pb + kRawCnt, KT_ * (L + 1));
        cb_reserve_back(cb_kq, KT_);  // the whole DFB (Kt pages): the raw slots never move
        cb_push_back(cb_kq, KT_);
    }
    noc_async_atomic_barrier();
}

// one-time reader init: zero the window-out slots (row Lw stays 0); P page 0 = the placement one-hot (1.0 at (r, r)),
// page 1 = the L2 mask (1.0 at (r, 0)), r < 32 (bf16; resident, never popped)
inline void reader_init(uint32_t wo_l1, uint32_t wo_bytes, uint32_t pp) {
    gpx::zero_l1(wo_l1, wo_bytes);
    gpx::zero_l1(pp, 2 * gpx::kTB);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pp);
    for (uint32_t r = 0; r < 32; ++r) {
        p[((r >> 4) * 2 + (r >> 4)) * 256 + (r & 15) * 16 + (r & 15)] = 0x3F80;
        p[1024 + (r >> 4) * 512 + (r & 15) * 16] = 0x3F80;
    }
}

// consumer (reader, NOC1): once both roots of its key head signalled, pull their Kt normalized tiles (root OUT pages 4..7,
// same L1 address everywhere) into knq [k | q], then +1 "read" on each root
template <uint32_t KT_>
inline void pull_knq(volatile tt_l1_ptr uint32_t* pb, uint32_t L, uint32_t knq_l1) {
    const uint32_t out_l1 = pb[kOutBase];
    {
        GBZ("GB-KQW");
        pb[kDbgR] = (L << 8) | 4;
        noc_semaphore_wait_min(pb + kKqCnt, 2 * (L + 1));
    }
#if GB_KQPUSH == 1
    (void)knq_l1;
    return;
#endif
    GBZ("GB-PULL");
    const uint32_t rk = pb[kRootSrc], rq = pb[kRootSrc + 1];
    for (uint32_t j = 0; j < KT_; ++j) {
        noc_async_read(get_noc_addr(rk & 0xFFFFu, rk >> 16, out_l1 + (4 + j) * 4096, GB_PULL_NOC), knq_l1 + j * 4096, 4096, GB_PULL_NOC);
        noc_async_read(get_noc_addr(rq & 0xFFFFu, rq >> 16, out_l1 + (4 + j) * 4096, GB_PULL_NOC), knq_l1 + (KT_ + j) * 4096, 4096, GB_PULL_NOC);
    }
    noc_async_read_barrier(GB_PULL_NOC);
    noc_semaphore_inc(get_noc_addr(rk & 0xFFFFu, rk >> 16, reinterpret_cast<uint32_t>(pb + kRootDone)), 1);
    noc_semaphore_inc(get_noc_addr(rq & 0xFFFFu, rq >> 16, reinterpret_cast<uint32_t>(pb + kRootDone)), 1);
    pb[kDbgR] = (L << 8) | 5;
}

// ---------------- writer ----------------
inline uint32_t tg_elem(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}
inline float f_from_bits(uint32_t b) {
    union { uint32_t u; float f; } x;
    x.u = b;
    return x.f;
}
inline uint32_t bits_from_f(float f) {
    union { uint32_t u; float f; } x;
    x.f = f;
    return x.u;
}

// +1 on payload word w of every core in the descriptor's rects
inline void inc_all(const uint32_t* d, uint32_t w, uint32_t v = 1);

// one fp32 tile to its consumers (multicast per rect; unicast for 1 core, the sender included)
inline void send_out(uint32_t src, const uint32_t* d, uint32_t slot_add = 0, bool fin = true) {
    const bool mc = (d[7] >> 12) & 1u;  // descriptor flag: multicast (few senders, e.g. the L2 roots)
    const uint32_t dst = get_write_ptr(d[7] & 0xFFu) + ((d[7] >> 16) + slot_add) * 4096;
    for (uint32_t r = 0; r < d[0]; ++r) {
        const uint32_t a = d[1 + 3 * r], b = d[2 + 3 * r], n = d[3 + 3 * r];
        const uint32_t x0 = a & 0xFFFFu, y0 = a >> 16, x1 = b & 0xFFFFu, y1 = b >> 16;
        if (mc) {
            if (my_x[0] >= x0 && my_x[0] <= x1 && my_y[0] >= y0 && my_y[0] <= y1) {
                noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4096, n, false);
            } else {
                noc_async_write_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4096, n, false);
            }
            continue;
        }
#if GB_MCAST
        if (n == 1) {
            noc_async_write(src, get_noc_addr(x0, y0, dst), 4096);
        } else if (my_x[0] >= x0 && my_x[0] <= x1 && my_y[0] >= y0 && my_y[0] <= y1) {
            noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4096, n, false);
        } else {
            noc_async_write_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4096, n, false);
        }
#else
        // unicasts: 24 cores multicasting into overlapping row rects at once gave a 25-45 us tail (NoC multicast contention)
        (void)n;
        for (uint32_t y = y0; y <= y1; ++y) {
            for (uint32_t x = x0; x <= x1; ++x) noc_async_write(src, get_noc_addr(x, y, dst), 4096);
        }
#endif
    }
    // arrival: +1 on each consumer's counter once the tile has landed (kq or rest by the descriptor's flag)
    if (!fin) return;
    noc_async_write_barrier();
    const uint32_t cw = (d[7] >> 8) & 0x3u;
    inc_all(d, cw == 1 ? kKqCnt : (cw == 2 ? kRawCnt : kRest));
}

inline void inc_all(const uint32_t* d, uint32_t w, uint32_t v) {
    for (uint32_t r = 0; r < d[0]; ++r) {
        const uint32_t a = d[1 + 3 * r], b = d[2 + 3 * r];
        for (uint32_t y = a >> 16; y <= (b >> 16); ++y) {
            for (uint32_t x = a & 0xFFFFu; x <= (b & 0xFFFFu); ++x) noc_semaphore_inc(get_noc_addr(x, y, g_gb_pb + 4 * w), v);
        }
    }
}

// tg_writer (u1 producer) for value head h, users [u0, u0 + NU) only (the head's VB cores split the users: each writes its
// users' rows), the tiles the early-replay step reads from `pre`: NGB G E BETA NBE EYE (0..5) and GD (15), from the gate
// pair gt (dec over a's tile, beta over b's tile), same float ops in the same order. W / C_u (6..14) are not written:
// under GDN_EARLY_REPLAY the step writer builds them itself from the stash (wr_rcoef). Per tile, the users' NU*T rows
// = two 16-column face runs (sc_l1: 7 x [face A rows | face B rows]).
template <uint32_t T, uint32_t NV, uint32_t NU>
inline void gate_rows_build(uint32_t h, uint32_t u0, uint32_t gt_l1, uint32_t sc_l1) {
    constexpr uint32_t R = NU * T, CH = R * 64;  // rows per core, bytes per face run
    static_assert(16 % R == 0, "a core's rows inside one face row");
    constexpr uint32_t kSign = 0x80000000u;
    volatile uint32_t* q32 = reinterpret_cast<volatile uint32_t*>(gt_l1);
    float beta[NU][T], gg[NU][T][T], ecum[NU][T];
    uint32_t beta_all[NU][T], dec_all[NU][T];
    for (uint32_t v = 0; v < NU; ++v) {
        const uint32_t r0 = (u0 + v) * T;
        float dec[T];
        for (uint32_t t = 0; t < T; ++t) {
            dec_all[v][t] = q32[tg_elem(r0 + t, h & 31u)];
            beta_all[v][t] = q32[1024 + tg_elem(r0 + t, (NV + h) & 31u)];
            beta[v][t] = f_from_bits(beta_all[v][t]);
            dec[t] = f_from_bits(dec_all[v][t]);
        }
        float e_cum = 1.0f;
        for (uint32_t t = 0; t < T; ++t) {
            e_cum = e_cum * dec[t];
            ecum[v][t] = e_cum;
            float g = 1.0f;
            for (uint32_t i = t; i-- > 0;) {
                g = g * dec[i + 1];
                gg[v][t][i] = g;
            }
        }
    }
    const uint32_t R0 = u0 * T;
    GBZ("GB-GROW");
    {
        const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t o = 0; o < 7 * 2 * CH; o += MEM_ZEROS_SIZE) noc_async_read(z, sc_l1 + o, MEM_ZEROS_SIZE);
        noc_async_read_barrier();
    }
    static constexpr uint8_t kTiles[7] = {0, 1, 2, 3, 4, 5, 15};
    for (uint32_t v = 0; v < NU; ++v) {
        const uint32_t r0 = (u0 + v) * T;
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t row = r0 + t, rr = (row - R0) * 16;
            // tile n's run A (cols 0..15) / run B (cols 16..31) of this row
            auto ra = [&](uint32_t n) { return reinterpret_cast<volatile uint32_t*>(sc_l1 + n * 2 * CH) + rr; };
            const uint32_t e = bits_from_f(ecum[v][t]), bb = beta_all[v][t], nb = bits_from_f(beta[v][t] * ecum[v][t]) ^ kSign;
            volatile uint32_t* pe = ra(2);
            volatile uint32_t* pbt = ra(3);
            volatile uint32_t* pn = ra(4);
            constexpr uint32_t W = CH / 4;
            for (uint32_t cc = 0; cc < 16; ++cc) {
                pe[cc] = e;
                pe[W + cc] = e;
                pbt[cc] = bb;
                pbt[W + cc] = bb;
                pn[cc] = nb;
                pn[W + cc] = nb;
            }
            const uint32_t cdiag = (row & 16u) ? W + (row & 15u) : (row & 15u);
            ra(1)[cdiag] = 0x3F800000u;
            ra(5)[cdiag] = 0x3F800000u;
            for (uint32_t i = 0; i < t; ++i) {
                const uint32_t c = r0 + i, ci = (c & 16u) ? W + (c & 15u) : (c & 15u);
                ra(0)[ci] = bits_from_f(beta[v][t] * gg[v][t][i]) ^ kSign;
                ra(1)[ci] = bits_from_f(gg[v][t][i]);
            }
            ra(6)[0] = dec_all[v][t];
            ra(6)[1] = beta_all[v][t];
        }
    }
}

template <uint32_t T, uint32_t NU, typename PAcc>
inline void gate_rows_send(const PAcc& pre, uint32_t u0, uint32_t pg0, uint32_t sc_l1, const uint32_t* dz, uint32_t coef_l1,
                           uint32_t gd_l1) {
    constexpr uint32_t R = NU * T, CH = R * 64;
    static constexpr uint8_t kTiles[7] = {0, 1, 2, 3, 4, 5, 15};
    const uint32_t R0 = u0 * T;
    const uint32_t off = (R0 >> 4) * 2048 + (R0 & 15) * 64;  // face (R0 >> 4) * 2 (cols 0..15), + 1024 (cols 16..31)
#if GB_GATE_DIRECT
    // straight into the head's cores (the zs consumers' rects dz): NGB..EYE -> their coef slots kCoefRep + n (the early-replay
    // writer reserves / pushes them without reading), GD -> their scratch gd page (copied into gprev by wr_pgates); L1 only
    (void)pre;
    (void)pg0;
    for (uint32_t r = 0; r < dz[0]; ++r) {
        const uint32_t a = dz[1 + 3 * r], b = dz[2 + 3 * r];
        for (uint32_t x = a & 0xFFFFu; x <= (b & 0xFFFFu); ++x) {
            const uint32_t y = a >> 16;
            for (uint32_t n = 0; n < 7; ++n) {
                const uint32_t base = n < 6 ? coef_l1 + (GB_COEF_REP + n) * 4096 : gd_l1;
                const uint64_t d = get_noc_addr(x, y, base + off);
                noc_async_write(sc_l1 + n * 2 * CH, d, CH);
                noc_async_write(sc_l1 + n * 2 * CH + CH, d + 1024, CH);
            }
        }
    }
#else
    (void)dz;
    (void)coef_l1;
    (void)gd_l1;
    for (uint32_t n = 0; n < 7; ++n) {
        const uint64_t d = pre.get_noc_addr(pg0 + kTiles[n]) + off;
        noc_async_write(sc_l1 + n * 2 * CH, d, CH);
        noc_async_write(sc_l1 + n * 2 * CH + CH, d + 1024, CH);
    }
#endif
    noc_async_write_barrier();
}

template <uint32_t B, uint32_t T, uint32_t NV, uint32_t VB, uint32_t RN, uint32_t KT_, uint32_t RD_, typename PAcc>
inline void writer_layer(volatile tt_l1_ptr uint32_t* pb, uint32_t L, const PAcc& pre, uint32_t cb_out, uint32_t sc_l1) {
    // OUT order (the compute's): gate pair, silu, conv 0, conv 1, [L2 root: Kt normalized k / q], pad to 4 per layer
    const uint32_t nconv = pb[kNconv];
    const bool root = pb[kRootPost] != 0;
    g_gb_pb = reinterpret_cast<uint32_t>(pb);
    auto out1 = [&](const uint32_t* d) {
        cb_wait_front(cb_out, 1);
        send_out(get_read_ptr(cb_out), d);
        cb_pop_front(cb_out, 1);
    };
    const uint32_t* dz = const_cast<const uint32_t*>(pb + kSilu + 1);
    // the first 4 OUT pages of a layer = [gate pair | silu | conv 0] (the layer starts at the base: 4 pages, multiple of 4 per
    // layer): conv 0 (raw q / k -> the L2 roots: the long pole) and silu go out before the gate rows are built
    constexpr uint32_t NU = B / VB;
    pb[kDbgW] = (L << 8) | 1;
    cb_wait_front(cb_out, 4);
    {
        GBZ("GB-WOUT");
        // conv 0 first: on the q / k cores it feeds the L2 roots (the long pole); the gate rows are built after
        send_out(get_read_ptr(cb_out) + 3 * 4096, const_cast<const uint32_t*>(pb + kConv + 1));
    }
    if (root) {  // the root's Kt normalized tiles (OUT pages 4..7 of its 8-page layer) before its gate rows
        GBZ("GB-ROOTW");
        cb_wait_front(cb_out, 4 + KT_);  // the Kt normalized tiles sit in OUT pages 4..7 (the consumers pull them)
#if GB_KQPUSH
        const uint32_t* dr = const_cast<const uint32_t*>(pb + kRootDst);
        for (uint32_t j = 0; j < KT_; ++j) send_out(get_read_ptr(cb_out) + (4 + j) * 4096, dr, j, false);
        noc_async_write_barrier();
        inc_all(dr, kKqCnt, 1);
#else
        inc_all(const_cast<const uint32_t*>(pb + kRootDst), kKqCnt, 1);
#endif
    }
    // (L2 roots: the ready signal above went out first -- the pulls of 12 consumers wait on it)
    send_out(get_read_ptr(cb_out) + 2 * 4096, dz);
    gate_rows_build<T, NV, NU>(pb[kGate], pb[kGcore] * NU, get_read_ptr(cb_out), sc_l1);
    {
        GBZ("GB-GATE");
        gate_rows_send<T, NU>(pre, pb[kGcore] * NU, pb[kGdst], sc_l1, dz, g_gb_coef, sc_l1 + 8192);
        inc_all(dz, kRest);  // the head's cores (= zs's consumers) read these rows
    }
    if (root) {
        GBZ("GB-ROOTR");
#if GB_KQPUSH != 1
        pb[kDbgW] = (L << 8) | 2;
        noc_semaphore_wait_min(pb + kRootDone, RD_ * (L + 1));
#endif  // every consumer has pulled the normalized tiles
    }
    cb_pop_front(cb_out, root ? 4 + KT_ : 4);
    uint32_t n = root ? 4 + KT_ : 4;
    if (nconv == 2) {
        out1(const_cast<const uint32_t*>(pb + kConv + kConvW + 1));
        ++n;
    }
    for (; n & 3u; ++n) {  // the compute's pads
        cb_wait_front(cb_out, 1);
        cb_pop_front(cb_out, 1);
    }
    {
        GBZ("GB-RESTW");
        pb[kDbgW] = (L << 8) | 3;
        noc_semaphore_wait_min(pb + kRest, RN * (L + 1));  // this core's gate rows / vc / zs (wr_pgates reads the rows)
        pb[kDbgW] = (L << 8) | 4;
    }
}

}  // namespace gb
