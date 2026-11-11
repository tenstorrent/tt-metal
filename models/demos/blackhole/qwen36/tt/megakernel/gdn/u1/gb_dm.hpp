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

#ifndef GB_MCAST
#define GB_MCAST 0
#endif
namespace gb {
inline uint32_t g_gb_pb;  // L1 address of the G-b payload block (same on every GDN core)

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
constexpr uint32_t kPerLayer = kPush + 3 * kPushMax + 1;  // per layer: [dtb bits, nea bits] of the gate head
constexpr uint32_t kKqCnt = kPush + 3 * kPushMax;   // arrivals of the raw k / q tiles (2 Kt per layer)
static_assert(kPerLayer == 132, "gb_compute.hpp / e1_shim mirror this layout");
constexpr uint32_t kNone = 0xFFFFFFFFu;

// ---------------- reader ----------------
// CBs (4 ids): QPX = [QIN 0..3 | P 4..5 | raw scratch] (reader pushes / compute pops all of it per layer: fixed slots),
// SHT = per conv job [SH_0 SH_1 SH_2 | tap] (4 bf16 pages), OUT (fp32: gate pair, silu, convs, pad to even), KQ (raw k | q).
// Scratch (QPX tail): win_l1: 2 x B bf16 tile slots (one window set per conv job); wo_l1: B x 1 KB window-out slots (row Lw
// zeroed once); ctrl_l1: the ctrl page.
template <uint32_t B, uint32_t T, uint32_t K, uint32_t CT, uint32_t M, uint32_t NV, uint32_t NC, typename WAcc,
          typename TAcc, typename CAcc>
inline void reader_layer(volatile tt_l1_ptr uint32_t* pb, uint32_t L, const WAcc& wa, const WAcc& wb, const TAcc& taps,
                         const CAcc& ctrl, uint32_t ctrl_bytes, uint32_t gin_l1, uint32_t qin_l1, uint32_t win_l1,
                         uint32_t wo_l1, uint32_t ctrl_l1, uint32_t cb_qpx, uint32_t npx, uint32_t cb_sht,
                         volatile tt_l1_ptr uint32_t* in_ready, uint32_t in_target) {
    static_assert(B * T == 32, "G-b: B * T == 32 only (the qkv tile is SH_{K-1})");
    const uint32_t nconv = pb[kNconv];
    gpx::Ctrl<B, T, M, NV> c;
    {
        DeviceZoneScopedN("GB-RPRE");
        noc_async_read(ctrl.get_noc_addr(0), ctrl_l1, ctrl_bytes);
        noc_async_read_barrier();
        c.parse(ctrl_l1);
        cb_reserve_back(cb_sht, 4 * nconv);
        for (uint32_t j = 0; j < nconv; ++j) {
            const uint32_t col = pb[kConv + j * kConvW];
            gpx::read_window<B, CT>(wa, wb, c, col, win_l1 + j * B * gpx::kTB);
            noc_async_read(taps.get_noc_addr(col), get_write_ptr(cb_sht) + (4 * j + 3) * gpx::kTB, gpx::kTB);
        }
    }
    {
        DeviceZoneScopedN("GB-GIN");
        if (pb[kNpush]) noc_semaphore_wait_min(in_ready, in_target);  // no landed tiles (AR corners): nothing to wait for
    }
    {
        DeviceZoneScopedN("GB-PUSH");
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
    noc_async_read_barrier();  // window + taps
    {
        DeviceZoneScopedN("GB-QIN");
        noc_semaphore_wait_min(pb + kQinCnt, pb[kNin] * (L + 1));
    }
    cb_reserve_back(cb_qpx, npx);
    cb_push_back(cb_qpx, npx);
    {
        DeviceZoneScopedN("GB-STG");
        for (uint32_t j = 0; j < nconv; ++j) {
            const uint32_t col = pb[kConv + j * kConvW];
            const uint32_t wj = win_l1 + j * B * gpx::kTB;
            if (j) noc_async_write_barrier();  // the previous job's window write left wo_l1
            gpx::stage_window_out<B, T, K>(c, wj, qin_l1 + j * gpx::kTB, wo_l1);
            noc_async_read_barrier();
            gpx::stage_shift<B, T, K>(wo_l1, get_write_ptr(cb_sht));
            noc_async_read_barrier();
            cb_push_back(cb_sht, 4);
            gpx::write_window<B, T, K, CT>(wa, wb, c, col, wo_l1, wj);
        }
        noc_async_write_barrier();
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
inline void inc_all(const uint32_t* d, uint32_t w);

// one fp32 tile to its consumers (multicast per rect; unicast for 1 core, the sender included)
inline void send_out(uint32_t src, const uint32_t* d) {
    const uint32_t dst = get_write_ptr(d[7] & 0xFFu) + (d[7] >> 16) * 4096;
    for (uint32_t r = 0; r < d[0]; ++r) {
        const uint32_t a = d[1 + 3 * r], b = d[2 + 3 * r], n = d[3 + 3 * r];
        const uint32_t x0 = a & 0xFFFFu, y0 = a >> 16, x1 = b & 0xFFFFu, y1 = b >> 16;
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
    noc_async_write_barrier();
    inc_all(d, (d[7] & 0x100u) ? kKqCnt : kRest);
}

inline void inc_all(const uint32_t* d, uint32_t w) {
    for (uint32_t r = 0; r < d[0]; ++r) {
        const uint32_t a = d[1 + 3 * r], b = d[2 + 3 * r];
        for (uint32_t y = a >> 16; y <= (b >> 16); ++y) {
            for (uint32_t x = a & 0xFFFFu; x <= (b & 0xFFFFu); ++x) noc_semaphore_inc(get_noc_addr(x, y, g_gb_pb + 4 * w), 1);
        }
    }
}

// tg_writer (u1 producer) for value head h, users [u0, u0 + NU) only (the head's VB cores split the users: each writes its
// users' rows), the tiles the early-replay step reads from `pre`: NGB G E BETA NBE EYE (0..5) and GD (15), from the gate
// pair gt (dec over a's tile, beta over b's tile), same float ops in the same order. W / C_u (6..14) are not written:
// under GDN_EARLY_REPLAY the step writer builds them itself from the stash (wr_rcoef). Per tile, the users' NU*T rows
// = two 16-column face runs (sc_l1: 7 x [face A rows | face B rows]).
template <uint32_t T, uint32_t NV, uint32_t NU, typename PAcc>
inline void gate_rows(const PAcc& pre, uint32_t h, uint32_t u0, uint32_t pg0, uint32_t gt_l1, uint32_t sc_l1) {
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
    DeviceZoneScopedN("GB-GROW");
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
    const uint32_t off = (R0 >> 4) * 2048 + (R0 & 15) * 64;  // face (R0 >> 4) * 2 (cols 0..15), + 1024 (cols 16..31)
    for (uint32_t n = 0; n < 7; ++n) {
        const uint64_t d = pre.get_noc_addr(pg0 + kTiles[n]) + off;
        noc_async_write(sc_l1 + n * 2 * CH, d, CH);
        noc_async_write(sc_l1 + n * 2 * CH + CH, d + 1024, CH);
    }
    noc_async_write_barrier();
}

template <uint32_t B, uint32_t T, uint32_t NV, uint32_t VB, uint32_t RN, typename PAcc>
inline void writer_layer(volatile tt_l1_ptr uint32_t* pb, uint32_t L, const PAcc& pre, uint32_t cb_out, uint32_t sc_l1) {
    // OUT order (the compute's): silu, conv 0 (q / k on the q / k cores: their consumers' l2 is the long pole), gate pair,
    // conv 1, pad (even per-layer count: the gate pair stays contiguous)
    const uint32_t nconv = pb[kNconv];
    g_gb_pb = reinterpret_cast<uint32_t>(pb);
    auto out1 = [&](const uint32_t* d) {
        cb_wait_front(cb_out, 1);
        send_out(get_read_ptr(cb_out), d);
        cb_pop_front(cb_out, 1);
    };
    const uint32_t* dz = const_cast<const uint32_t*>(pb + kSilu + 1);
    {
        DeviceZoneScopedN("GB-WOUT");
        out1(dz);
        out1(const_cast<const uint32_t*>(pb + kConv + 1));
    }
    {
        DeviceZoneScopedN("GB-GATE");
        constexpr uint32_t NU = B / VB;
        cb_wait_front(cb_out, 2);
        gate_rows<T, NV, NU>(pre, pb[kGate], pb[kGcore] * NU, pb[kGdst], get_read_ptr(cb_out), sc_l1);
        cb_pop_front(cb_out, 2);
        inc_all(dz, kRest);  // the head's cores (= zs's consumers) read these rows
    }
    if (nconv == 2) {
        DeviceZoneScopedN("GB-WOUT1");
        out1(const_cast<const uint32_t*>(pb + kConv + kConvW + 1));
        cb_wait_front(cb_out, 1);
        cb_pop_front(cb_out, 1);
    }
    {
        DeviceZoneScopedN("GB-RESTW");
        noc_semaphore_wait_min(pb + kRest, RN * (L + 1));  // this core's gate rows / vc / zs (wr_pgates reads the rows)
    }
}

}  // namespace gb
