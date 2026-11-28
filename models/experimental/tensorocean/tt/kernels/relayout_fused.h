// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused input and output rearranging for v27 (kernels/{reader,writer,compute}.cpp with FUSED_IN / FUSED_OUT).
//
// Input (before v27's main loop). The per-step inputs are natural row-major arrays in DRAM: cell [L, M, M],
// f2, mask2 [L, N, N + 1], f1, mask1 [L, N + 1, 2N + 1]. v27 wants, in the L1 of core (strip x, level row y):
//   tracer planes: CB_PLANES, level li at li * LVL_PITCH, plane p (natural rows p, p + 2, ...) at p * CELL_LEN,
//                  column by column (H floats per column, column c = natural column BAND0[x] + c)
//   f and mask:    CB_SFMK [6 groups][LCM levels][f|mask][F_LEN], same column-by-column order
// A plane unit = (level, k): kind k / 2 (0 cell, 1 f2, 2 mask2, 3 f1, 4 mask1), plane p = k % 2. The units of a
// row's levels are shared by that row's cores (so the traffic stays within the row):
//   reader  (hw_read):  the unit's natural rows -> CB_RIN, row r at r * ROWB; rows [UNIT_ROWS[k], H) zeroed
//   compute (hw_unit):  tilize, transpose, untilize -> CB_TOUT, where natural column c is row c (TR * 32 floats)
//   writer  (hw_write): sends the first H floats of each row a strip needs to the owning core, then zeroes the rest
//                       of that block (columns past the strip) from a zero buffer
// Then a barrier over all cores (barrier): every RISC increments ARRIVE on core (0, 0) after its writes have been
// acknowledged; the reader of core (0, 0) waits for all of them and multicasts GO = 1; every RISC waits for GO.
#pragma once

namespace rf {
constexpr uint32_t C1 = 2 * R::N + 1, C2 = R::N + 1;  // row lengths of family 1 and 2 arrays

// Units of my row's levels [l0, l0 + nl): unit i = (level l0 + i / 10, k = i % 10); core x takes i = x, x + NBX, ...
template <typename AC, typename AF1, typename AF2, typename AM1, typename AM2>
inline void hw_read(
    const AC& Cn, const AF1& F1, const AF2& F2, const AM1& M1, const AM2& M2, uint32_t x, uint32_t l0, uint32_t nl) {
    for (uint32_t i = x; i < 10 * nl; i += R::NBX) {
        {
            const uint32_t l = l0 + i / 10, k = i % 10;
            const uint32_t p = k & 1, kind = k >> 1, nr = R::UNIT_ROWS[k];
            cb_reserve_back(CB_RIN, R::UNIT_PAGES);
            const uint32_t base = get_write_ptr(CB_RIN);
            for (uint32_t r = 0; r < nr; ++r) {
                const uint32_t dst = base + r * R::ROWB;
                switch (kind) {
                    case 0: noc_async_read(Cn.get_noc_addr(l * R::M + 2 * r + p), dst, R::M * 4); break;
                    case 1: noc_async_read(F2.get_noc_addr(l * R::N + 2 * r + p), dst, C2 * 4); break;
                    case 2: noc_async_read(M2.get_noc_addr(l * R::N + 2 * r + p), dst, C2 * 4); break;
                    case 3: noc_async_read(F1.get_noc_addr(l * (R::N + 1) + 2 * r + p), dst, C1 * 4); break;
                    default: noc_async_read(M1.get_noc_addr(l * (R::N + 1) + 2 * r + p), dst, C1 * 4); break;
                }
            }
            for (uint32_t r = nr; r < R::H; ++r) {  // edge rows past the mesh: v27 expects zeros
                uint32_t* z = (uint32_t*)(base + r * R::ROWB);
                for (uint32_t c = 0; c < R::UNIT_COLS[k]; ++c) {
                    z[c] = 0;
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_RIN, R::UNIT_PAGES);
        }
    }
}
inline void hw_write(uint32_t x0, uint32_t l0, uint32_t nl, uint32_t st_cell, uint32_t st_fmk, uint32_t zb) {
    constexpr uint32_t TROWB = R::TR * 128;  // bytes per transposed row in CB_TOUT
    constexpr uint32_t CB_ = R::H * 4;       // bytes per v27 column
    for (uint32_t i = x0; i < 10 * nl; i += R::NBX) {
        const uint32_t l = l0 + i / 10, k = i % 10;
        const uint32_t li = R::LEV_LI[l];
        {
            const uint32_t p = k & 1, kind = k >> 1;
            const uint32_t base = get_read_ptr(CB_TOUT);
            uint32_t avail = 0;  // transposed rows available so far
            auto row = [&](uint32_t c) {
                if (c >= avail) {
                    avail = (c / 32 + 1) * 32;
                    cb_wait_front(CB_TOUT, (avail / 32) * R::TR);
                }
                return base + c * TROWB;
            };
            // ncol columns starting at natural column c0 (stride cs) -> block at dst (len words), zero tail
            auto block = [&](uint32_t x, uint32_t c0, uint32_t cs, uint32_t ncol, uint32_t dst, uint32_t len) {
                const uint32_t t = R::LEV_Y[l] * R::NBX + x;
                const uint64_t tn = get_noc_addr(R::NOCX[t], R::NOCY[t], 0);
                noc_async_write_one_packet_set_state(tn, CB_);  // fixed target core and size
                for (uint32_t c = 0; c < ncol; ++c) {
                    const uint32_t src = row(c0 + c * cs);
                    noc_async_write_one_packet_with_state(src, dst + c * CB_);
                }
                if (ncol * R::H < len) {
                    noc_async_write(zb, tn | (dst + ncol * CB_), (len - ncol * R::H) * 4);
                }
            };
            for (uint32_t x = 0; x < R::NBX; ++x) {
                const uint32_t oc0 = R::BAND0[x], w = R::BANDW[x];
                if (kind == 0) {
                    const uint32_t c1 = oc0 + w + 4 < R::M ? oc0 + w + 4 : R::M;
                    block(x, oc0, 1, c1 - oc0, st_cell + li * R::LVL_PITCH + p * R::CELL_LEN * 4, R::CELL_LEN);
                } else {
                    const uint32_t fam = kind <= 2 ? 2 : 1, arr = (kind - 1) & 1;
                    const uint32_t g0 = fam == 1 ? 2 * p : 4 + p, ngr = fam == 1 ? 2 : 1;
                    const uint32_t cols = R::G_COLS[g0];
                    const int32_t fc = (int32_t)(oc0 + w + 1 < cols ? oc0 + w + 1 : cols) - (int32_t)oc0;
                    const uint32_t fcols = fc > 0 ? (uint32_t)fc : 0;
                    for (uint32_t gs = 0; gs < ngr; ++gs) {
                        block(
                            x,
                            fam == 1 ? 2 * oc0 + gs : oc0,
                            fam == 1 ? 2 : 1,
                            fcols,
                            st_fmk + (((g0 + gs) * R::LCM + li) * 2 + arr) * R::F_LEN * 4,
                            R::F_LEN);
                    }
                }
            }
            noc_async_writes_flushed();             // CB_TOUT may be overwritten after the pop
            cb_wait_front(CB_TOUT, R::UNIT_PAGES);  // the whole (padded) unit slot
            cb_pop_front(CB_TOUT, R::UNIT_PAGES);
        }
    }
    noc_async_write_barrier();  // every column I sent has landed
}

#ifdef FUSED_OUT
// ---- fused output rearranging ----
// Output unit a = (level l0 + a / 2, part a % 2) of a row is assembled by core x = a % NBX of the same row, in its
// CB_RIN slot (n_pro(x) + a / NBX) % NRIN (n_pro(x) = that core's input units: CB_RIN's ring position afterwards).
// Slot layout: row c = natural column c (TR * 32 floats, first H = v27's column), rows up to TRO * 32.
// Every core of the row sends its columns there (out_send), then increments SEM_OUT0 + a / NBX on the assembler
// (out_done); the assembler's reader waits for NBX increments (out_arrive), compute transposes, and its writer
// writes the N / 2 natural rows to DRAM (out_write).
inline uint32_t n_pro(uint32_t x, uint32_t nl) { return 10 * nl > x ? (10 * nl - x + R::NBX - 1) / R::NBX : 0; }
inline uint32_t out_slot_addr(uint32_t xa, uint32_t ja, uint32_t nl) {
    return get_write_ptr(CB_RIN) + ((n_pro(xa, nl) + ja) % R::NRIN) * R::UNIT_PAGES * 4096;
}
// output block ob (items [ob * 128, ob * 128 + 128) of my strip, column-major with pitch H) of level li, part
inline void out_send(uint32_t x, uint32_t l0, uint32_t nl, uint32_t li, uint32_t part, uint32_t ob, uint32_t src) {
    const uint32_t a = li * 2 + part, xa = a % R::NBX, ja = a / R::NBX;
    const uint32_t t = R::LEV_Y[l0] * R::NBX + xa;
    const uint64_t tn = get_noc_addr(R::NOCX[t], R::NOCY[t], 0);
    const uint32_t slot = out_slot_addr(xa, ja, nl), oc0 = R::BAND0[x];
    const uint32_t s0 = ob * 128, end0 = s0 + 128, lim = R::BANDW[x] * R::H;
    const uint32_t end = end0 < lim ? end0 : lim;
    for (uint32_t it = s0; it < end;) {
        const uint32_t c = it / R::H, r0 = it - c * R::H;
        const uint32_t n = (R::H - r0) < (end - it) ? (R::H - r0) : (end - it);
        noc_async_write(src + (it - s0) * 4, tn | (slot + (oc0 + c) * R::TR * 128 + r0 * 4), n * 4);
        it += n;
    }
}
inline void out_done(uint32_t l0, uint32_t nl) {
    for (uint32_t a = 0; a < 2 * nl; ++a) {
        const uint32_t t = R::LEV_Y[l0] * R::NBX + a % R::NBX;
        noc_semaphore_inc(get_noc_addr(R::NOCX[t], R::NOCY[t], get_semaphore(SEM_OUT0 + a / R::NBX)), 1);
    }
    noc_async_atomic_barrier();
}
inline void out_arrive(uint32_t x, uint32_t nl) {
    for (uint32_t a = x, j = 0; a < 2 * nl; a += R::NBX, ++j) {
        volatile tt_l1_ptr uint32_t* sem = (volatile tt_l1_ptr uint32_t*)get_semaphore(SEM_OUT0 + j);
        while (*sem < R::NBX) {
            invalidate_l1_cache();
        }
        cb_reserve_back(CB_RIN, R::UNIT_PAGES);  // the slot this reserve returns is out_slot_addr(x, j)
        cb_push_back(CB_RIN, R::UNIT_PAGES);
    }
}
template <typename AE, typename AO>
inline void out_write(const AE& EV, const AO& OD, uint32_t x, uint32_t l0, uint32_t nl) {
    for (uint32_t a = x; a < 2 * nl; a += R::NBX) {
        const uint32_t l = l0 + a / 2, part = a % 2;
        cb_wait_front(CB_TOUT, R::UNIT_PAGES);
        const uint32_t base = get_read_ptr(CB_TOUT);
        for (uint32_t r = 0; r < R::N / 2; ++r) {
            const uint32_t src = base + r * R::TRO * 128;
            if (part) {
                noc_async_write(src, OD.get_noc_addr(l * (R::N / 2) + r), R::N * 4);
            } else {
                noc_async_write(src, EV.get_noc_addr(l * (R::N / 2) + r), R::N * 4);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(CB_TOUT, R::UNIT_PAGES);
    }
    noc_async_write_barrier();
}
#endif

// all RISCs of all cores: arrive on core (0, 0); its reader multicasts GO once everyone has arrived
inline void barrier(uint32_t arrive_sem, uint32_t go_sem, bool master, uint32_t go_word) {
    noc_semaphore_inc(get_noc_addr(R::NOCX[0], R::NOCY[0], arrive_sem), 1);
    noc_async_atomic_barrier();
    if (master) {
        volatile tt_l1_ptr uint32_t* a = (volatile tt_l1_ptr uint32_t*)arrive_sem;
        while (*a < 2 * R::NCORES) {
            invalidate_l1_cache();
        }
        *(volatile tt_l1_ptr uint32_t*)go_word = 1;
        const uint64_t mc = noc_index == 0 ? get_noc_multicast_addr(R::MCX0, R::MCY0, R::MCX1, R::MCY1, go_sem)
                                           : get_noc_multicast_addr(R::MCX1, R::MCY1, R::MCX0, R::MCY0, go_sem);
        noc_semaphore_set_multicast_loopback_src(go_word, mc, R::NCORES);
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* go = (volatile tt_l1_ptr uint32_t*)go_sem;
    while (*go != 1) {
        invalidate_l1_cache();
    }
}
}  // namespace rf
