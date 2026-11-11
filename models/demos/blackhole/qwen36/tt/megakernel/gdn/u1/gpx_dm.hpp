// GDN_PREFIX (megakernel phase type 10, B >= 2): data-movement phase functions, includable into the GDN role's reader /
// writer (DESIGN v0.9.16 contract, layout of gdn_spec_step). Validated on silicon via gpc_reader / gpc_writer
// (bench_gdn_prefix_c.py: win[1-par] bit-identical to the op incl. HOLD).
//   reader side: gpx_ctrl (parity, mi with FRESH -> 0, HOLD per user), gpx_read_window (issue only; the engine issues it at
//   layer start, off the in-proj's path), gpx_stage (local NoC row-block copies: window-out rows + the K shifted operands)
//   writer side: gpx_write_window (win[1-par] rows 0..Lw_w-1 for live users, the whole win[par] tile for HOLD users)
// Buffers are plain L1 addresses (the engine's CB slots); no CB calls inside.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace gpx {

constexpr uint32_t kTB = 2048;     // bf16 tile
constexpr uint32_t kWoSlot = 1024;  // window-out slot: face 0 rows 0..7 at +0, face 1 rows 0..7 at +512
constexpr uint32_t kHold = 0xFFFFFFFFu;

// byte offset of (row, 16-col half hh) in a bf16 32x32 tile
inline uint32_t row_off(uint32_t row, uint32_t hh) { return (((row >> 4) * 2 + hh) * 256 + (row & 15) * 16) * 2; }

template <uint32_t B, uint32_t T, uint32_t M, uint32_t NV>
struct Ctrl {
    uint32_t par;
    uint32_t mi[B];    // window mi (FRESH -> 0)
    uint32_t hold[B];  // 1: HOLD (the caller holds all Nv heads of a user: word of (u, h = 0))
    void parse(uint32_t l1) {
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(l1);
        par = p[0] & 1u;
        for (uint32_t u = 0; u < B; ++u) {
            mi[u] = p[1 + u] >= T ? 0u : p[1 + u];
            hold[u] = p[1 + M + u * NV] == kHold ? 1u : 0u;
        }
    }
};

// window rows of column `col` for every user into win_l1 (B tile slots): faces 0/1 rows 0..7 (live), whole tile (HOLD)
template <uint32_t B, uint32_t CT, typename Acc, typename CtrlT>
inline void read_window(const Acc& wa, const Acc& wb, const CtrlT& c, uint32_t col, uint32_t win_l1) {
    for (uint32_t u = 0; u < B; ++u) {
        const uint64_t src = c.par ? wb.get_noc_addr(u * CT + col) : wa.get_noc_addr(u * CT + col);
        if (c.hold[u]) {
            noc_async_read(src, win_l1 + u * kTB, kTB);
        } else {
            noc_async_read(src, win_l1 + u * kTB, 8 * 32);
            noc_async_read(src + 512, win_l1 + u * kTB + 512, 8 * 32);
        }
    }
}

inline void l1cp(uint32_t dst, uint32_t src, uint32_t bytes) { noc_async_read(get_noc_addr(src), dst, bytes); }

// stg_u (wo_l1 slot u) rows 0..Lw-1 = [win rows mi+1..mi+K-1 | qkv rows u*T..u*T+T-1]; row Lw (Lw odd) must be zero
// already (the slots are zeroed once: rows >= Lw are never written). Reader side; caller read-barriers.
template <uint32_t B, uint32_t T, uint32_t K, typename CtrlT>
inline void stage_window_out(const CtrlT& c, uint32_t win_l1, uint32_t q_l1, uint32_t wo_l1) {
    constexpr uint32_t Lw = K - 1 + T;
    static_assert(16 % T == 0 && Lw <= 8 && B * T <= 32, "geometry");
    for (uint32_t u = 0; u < B; ++u) {
        for (uint32_t hh = 0; hh < 2; ++hh) {
            const uint32_t d = wo_l1 + u * kWoSlot + hh * 512;
            l1cp(d, win_l1 + u * kTB + row_off(c.mi[u] + 1, hh), (K - 1) * 32);
            l1cp(d + (K - 1) * 32, q_l1 + row_off(u * T, hh), T * 32);
        }
    }
}

// at B*T = 32 the last shifted operand SH_{K-1} (rows u*T+t = stg_u rows t+K-1 = qkv rows u*T+t) IS the qkv tile
template <uint32_t B, uint32_t T>
constexpr bool q_direct() { return B * T == 32; }

// SH_k (sh_l1 tile k) rows u*T+t = stg_u rows t+k, for k < K-1 (k < K when the qkv tile cannot stand in for SH_{K-1}).
// Rows of absent users (B*T < 32) must be zero in sh_l1 already. Writer side (BRISC: splits the staging with the
// reader); caller read-barriers.
template <uint32_t B, uint32_t T, uint32_t K>
inline void stage_shift(uint32_t wo_l1, uint32_t sh_l1) {
    constexpr uint32_t NK = q_direct<B, T>() ? K - 1 : K;
    for (uint32_t k = 0; k < NK; ++k)
        for (uint32_t u = 0; u < B; ++u)
            for (uint32_t hh = 0; hh < 2; ++hh)
                l1cp(sh_l1 + k * kTB + row_off(u * T, hh), wo_l1 + u * kWoSlot + hh * 512 + k * 32, T * 32);
}

// zero `bytes` of L1 from the hardware zero region (one-time init of the window-out / shift slots)
inline void zero_l1(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) noc_async_read(z, addr + o, MEM_ZEROS_SIZE);
}

// the op's wr_win for column `col` (caller write-barriers)
template <uint32_t B, uint32_t T, uint32_t K, uint32_t CT, typename Acc, typename CtrlT>
inline void write_window(const Acc& wa, const Acc& wb, const CtrlT& c, uint32_t col, uint32_t wo_l1, uint32_t win_l1) {
    constexpr uint32_t Lw = K - 1 + T, Lw_w = (Lw + 1) & ~1u;
    for (uint32_t u = 0; u < B; ++u) {
        const uint64_t dst = c.par ? wa.get_noc_addr(u * CT + col) : wb.get_noc_addr(u * CT + col);
        if (c.hold[u]) {
            noc_async_write(win_l1 + u * kTB, dst, kTB);
        } else {
            noc_async_write(wo_l1 + u * kWoSlot, dst, Lw_w * 32);
            noc_async_write(wo_l1 + u * kWoSlot + 512, dst + 512, Lw_w * 32);
        }
    }
}

}  // namespace gpx
