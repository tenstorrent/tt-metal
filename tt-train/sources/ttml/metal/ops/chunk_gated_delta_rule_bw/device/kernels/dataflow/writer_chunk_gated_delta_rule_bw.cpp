// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gated_delta_rule_bw writer.  Owns every L1 -> DRAM store and the two semaphore INCREMENTS
// that release the stage barriers:
//   Stage P : drains the derived prep blocks to the f32 scratch, then sends one sem_prep increment
//             per item to that item's scan owner (Barrier 1 fan-in, NC increments per group).
//   Stage S : drains S_i / v_new / dS / dv_new / dh0, then sends one sem_scan increment to every
//             core holding stage-G work in the group it just scanned (Barrier 2 fan-out).
//   Stage G : the five gradients.
// ALL prep increments are issued before ANY wait anywhere in the program, so the rendezvous cannot
// deadlock.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_bw/device/kernels/chunk_gated_delta_rule_bw_common.hpp"

#pragma GCC optimize("Os")

constexpr uint32_t f32_tile = 32 * 32 * 4;
constexpr uint32_t in_tile = 32 * 32 * ESZ;

template <typename ACC>
static __attribute__((noipa)) void write_sub(
    const ACC& acc,
    uint32_t src,
    uint32_t base_tile,
    uint32_t rows,
    uint32_t row_stride,
    uint32_t col0,
    uint32_t cols,
    uint32_t tb) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write(src + (r * cols + c) * tb, acc.get_noc_addr(base_tile + r * row_stride + col0 + c), tb);
        }
    }
}

// Pop one MAXBLK-sized egress block and write it to a contiguous scratch range.
template <typename ACC>
static __attribute__((noipa)) void drain_flat(const ACC& acc, uint32_t base_tile, uint32_t ntiles) {
    cb_wait_front(cb_egr, MAXBLK);
    const uint32_t src = get_read_ptr(cb_egr);
    for (uint32_t i = 0; i < ntiles; ++i) {
        noc_async_write(src + i * f32_tile, acc.get_noc_addr(base_tile + i), f32_tile);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_egr, MAXBLK);
}

template <typename ACC>
static __attribute__((noipa)) void drain_strided(
    const ACC& acc, uint32_t base_tile, uint32_t rows, uint32_t row_stride, uint32_t col0, uint32_t cols) {
    cb_wait_front(cb_egr, MAXBLK);
    write_sub(acc, get_read_ptr(cb_egr), base_tile, rows, row_stride, col0, cols, f32_tile);
    noc_async_write_barrier();
    cb_pop_front(cb_egr, MAXBLK);
}

// A [Ct, dn] gradient block of head h into a token-major [N,1,T,Hx*D] output, as Ct*dn whole pages.
template <typename ACC>
static __attribute__((noipa)) void write_flat_block(
    const ACC& acc,
    uint32_t src,
    uint32_t n,
    uint32_t tt0,
    uint32_t hx_dt,
    uint32_t h,
    uint32_t dt,
    uint32_t d0,
    uint32_t dn) {
    for (uint32_t r = 0; r < Ct; ++r) {
        for (uint32_t j = 0; j < dn; ++j) {
            noc_async_write(
                src + (r * dn + j) * in_tile, acc.get_noc_addr(flat_page(n, tt0 + r, hx_dt, h, dt, d0 + j)), in_tile);
        }
    }
}

// Scalar scatter of a [C, 1] gradient column into column h of a [B,1,T,HV] output.  A DRAM write
// needs (l1_src & 15) == (dram_dst & 15), and the element offset of column h is not 16-byte
// aligned, so each value is first staged at a matching offset inside cb_stage.
template <typename ACC>
static __attribute__((noipa)) void scatter_scalars(
    const ACC& acc, uint32_t src_base, uint32_t wstage, uint32_t b, uint32_t t0, uint32_t h) {
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        const uint32_t tbase = t0 + ct * 32;
        if (tbase >= gT) {
            break;
        }
        const uint32_t tt = tbase >> 5;
        const uint32_t tile_addr = src_base + ct * in_tile;
        uint32_t issued = 0;
        for (uint32_t r = 0; r < 32; ++r) {
            if (tbase + r >= gT) {
                break;
            }
            const uint32_t doff = tile_elem_off(r, h) * ESZ;
            const uint32_t slot = wstage + r * 16u + (doff & 15u);
            if constexpr (ESZ == 4) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
                d[0] = s[tile_elem_off(r, 0)];
            } else {
                volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
                volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(slot);
                d[0] = s[tile_elem_off(r, 0)];
            }
            noc_async_write(slot, acc.get_noc_addr(b * Tt + tt, doff), ESZ);
            ++issued;
        }
        if (issued) {
            noc_async_write_barrier();
        }
    }
}

// Columns gH..31 of a [B,1,T,HV] output are tile padding that no head writes; the item of head
// gH-1 zeroes them for its rows.  `zbuf` holds 80 zero bytes, enough for a 16-element face row at
// any 16-byte phase.
template <typename ACC>
static __attribute__((noipa)) void zero_pad_columns(const ACC& acc, uint32_t zbuf, uint32_t b, uint32_t t0) {
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        const uint32_t tbase = t0 + ct * 32;
        if (tbase >= gT) {
            break;
        }
        const uint32_t tt = tbase >> 5;
        for (uint32_t r = 0; r < 32; ++r) {
            if (tbase + r >= gT) {
                break;
            }
            for (uint32_t c0 = gH; c0 < 32;) {
                const uint32_t c1 = (c0 < 16) ? 16 : 32;
                const uint32_t doff = tile_elem_off(r, c0) * ESZ;
                noc_async_write(zbuf + (doff & 15u), acc.get_noc_addr(b * Tt + tt, doff), (c1 - c0) * ESZ);
                c0 = c1;
            }
        }
    }
    noc_async_write_barrier();
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t sc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dq_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dk_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dv_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dg_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dbeta_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dh0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t item_start = get_arg_val<uint32_t>(arg++);
    const uint32_t num_items = get_arg_val<uint32_t>(arg++);
    const uint32_t num_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t group_base = arg;
    const uint32_t owned_cnt_idx = group_base + 3 * num_groups;
    const uint32_t num_owned = get_arg_val<uint32_t>(owned_cnt_idx);
    const uint32_t owned_base = owned_cnt_idx + 1;
    const uint32_t release_base = owned_base + num_owned;

    constexpr auto sca = TensorAccessorArgs<CT_ACC_BASE>();
    constexpr auto dqa = TensorAccessorArgs<sca.next_compile_time_args_offset()>();
    constexpr auto dka = TensorAccessorArgs<dqa.next_compile_time_args_offset()>();
    constexpr auto dva = TensorAccessorArgs<dka.next_compile_time_args_offset()>();
    constexpr auto dga = TensorAccessorArgs<dva.next_compile_time_args_offset()>();
    constexpr auto dba = TensorAccessorArgs<dga.next_compile_time_args_offset()>();
    constexpr auto dh0a = TensorAccessorArgs<dba.next_compile_time_args_offset()>();

    const auto sc_acc = TensorAccessor(sca, sc_addr, f32_tile);
    const auto dq_acc = TensorAccessor(dqa, dq_addr, in_tile);
    const auto dk_acc = TensorAccessor(dka, dk_addr, in_tile);
    const auto dv_acc = TensorAccessor(dva, dv_addr, in_tile);
    const auto dg_acc = TensorAccessor(dga, dg_addr, in_tile);
    const auto db_acc = TensorAccessor(dba, dbeta_addr, in_tile);
    [[maybe_unused]] const auto dh0_acc = TensorAccessor(dh0a, dh0_addr, in_tile);

    const uint32_t sem_prep_addr = get_semaphore(SEM_PREP);
    const uint32_t sem_scan_addr = get_semaphore(SEM_SCAN);

    // ------------------------------------------------------------------
    // Stage P: drain the prep blocks, then signal each item's scan owner.
    // ------------------------------------------------------------------
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;

        if constexpr (!HAS_SAVED) {
            drain_flat(sc_acc, SC_ATTN + wi * ST_ATTN, CtCt);
        }
        drain_flat(sc_acc, SC_KCD + wi * ST_KCD, CtKt);
        drain_flat(sc_acc, SC_P + wi * ST_P, CtKt);
        drain_flat(sc_acc, SC_VEC + wi * ST_VEC, 4 * Ct);

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            if constexpr (!HAS_SAVED) {
                drain_strided(sc_acc, SC_VCORR + wi * ST_VCORR, Ct, Vt, vb * Vb, Vb);
            }
            drain_strided(sc_acc, SC_U + wi * ST_U, Ct, Vt, vb * Vb, Vb);
            drain_strided(sc_acc, SC_C + wi * ST_C, Kt, Vt, vb * Vb, Vb);
        }

        // Barrier-1 fan-in: +1 on the owner of this item's group.
        const uint32_t bh = wi / NC;
        for (uint32_t gidx = 0; gidx < num_groups; ++gidx) {
            if (get_arg_val<uint32_t>(group_base + 3 * gidx) == bh) {
                const uint32_t ox = get_arg_val<uint32_t>(group_base + 3 * gidx + 1);
                const uint32_t oy = get_arg_val<uint32_t>(group_base + 3 * gidx + 2);
                noc_semaphore_inc(get_noc_addr(ox, oy, sem_prep_addr), 1);
                break;
            }
        }
    }
    noc_async_atomic_barrier();

    // ------------------------------------------------------------------
    // Stage S: drain the scan, then release the group.
    // ------------------------------------------------------------------
    {
        uint32_t rcursor = release_base;
        for (uint32_t oi = 0; oi < num_owned; ++oi) {
            const uint32_t bh = get_arg_val<uint32_t>(owned_base + oi);
            const uint32_t b = bh / gH;
            const uint32_t h = bh % gH;

            for (uint32_t vb = 0; vb < NVB; ++vb) {
                for (uint32_t i = 0; i < (HAS_SAVED ? 0 : NC); ++i) {
                    const uint32_t wi = bh * NC + i;
                    drain_strided(sc_acc, SC_S + wi * ST_S, Kt, Vt, vb * Vb, Vb);
                    drain_strided(sc_acc, SC_VNEW + wi * ST_VNEW, Ct, Vt, vb * Vb, Vb);
                }
                for (int32_t i = (int32_t)NC - 1; i >= 0; --i) {
                    const uint32_t wi = bh * NC + (uint32_t)i;
                    drain_strided(sc_acc, SC_DS + wi * ST_DS, Kt, Vt, vb * Vb, Vb);
                    drain_strided(sc_acc, SC_DVNEW + wi * ST_DVNEW, Ct, Vt, vb * Vb, Vb);
                }
                if constexpr (HAS_H0) {
                    cb_wait_front(cb_gegr, MAXBLK_G);
                    const uint32_t src = get_read_ptr(cb_gegr);
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        for (uint32_t j = 0; j < Vb; ++j) {
                            noc_async_write(
                                src + (kt * Vb + j) * in_tile,
                                dh0_acc.get_noc_addr(((b * gH + h) * Kt + kt) * Vt + vb * Vb + j),
                                in_tile);
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_gegr, MAXBLK_G);
                }
            }

            // Barrier-2 fan-out: +1 to every core with stage-G work in this group.
            const uint32_t ntargets = get_arg_val<uint32_t>(rcursor);
            ++rcursor;
            for (uint32_t t = 0; t < ntargets; ++t) {
                const uint32_t tx = get_arg_val<uint32_t>(rcursor + 2 * t);
                const uint32_t ty = get_arg_val<uint32_t>(rcursor + 2 * t + 1);
                noc_semaphore_inc(get_noc_addr(tx, ty, sem_scan_addr), 1);
            }
            rcursor += 2 * ntargets;
        }
        noc_async_atomic_barrier();
    }

    // ------------------------------------------------------------------
    // Stage G: the five gradients.
    // ------------------------------------------------------------------
    const uint32_t wstage = (get_write_ptr(cb_stage) + 63u) & ~63u;
    // The scatter slots occupy wstage[0, 512); the zero source sits right after them.
    const uint32_t zbuf = wstage + 32u * 16u;
    if constexpr (gH < 32) {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbuf);
        for (uint32_t i = 0; i < 20; ++i) {
            z[i] = 0;
        }
    }
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;
        const uint32_t bh = wi / NC;
        const uint32_t ci = wi % NC;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t t0 = ci * CHUNK;
        const uint32_t tt0 = ci * Ct;
        // dq/dk of value head h land in plane h % G at key head h / G.
        const uint32_t plane = (h % gG) * gB + b;
        const uint32_t hk = h / gG;

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            cb_wait_front(cb_gegr, MAXBLK_G);
            write_flat_block(dv_acc, get_read_ptr(cb_gegr), b, tt0, gH * Vt, h, Vt, vb * Vb, Vb);
            noc_async_write_barrier();
            cb_pop_front(cb_gegr, MAXBLK_G);
        }

        // Drain order mirrors the compute kernel's push order: dv (per V block), dq, dbeta, dk, dg.
        cb_wait_front(cb_gegr, MAXBLK_G);
        write_flat_block(dq_acc, get_read_ptr(cb_gegr), plane, tt0, gHK * Kt, hk, Kt, 0, Kt);
        noc_async_write_barrier();
        cb_pop_front(cb_gegr, MAXBLK_G);

        cb_wait_front(cb_gegr, MAXBLK_G);
        scatter_scalars(db_acc, get_read_ptr(cb_gegr), wstage, b, t0, h);
        cb_pop_front(cb_gegr, MAXBLK_G);

        cb_wait_front(cb_gegr, MAXBLK_G);
        write_flat_block(dk_acc, get_read_ptr(cb_gegr), plane, tt0, gHK * Kt, hk, Kt, 0, Kt);
        noc_async_write_barrier();
        cb_pop_front(cb_gegr, MAXBLK_G);

        cb_wait_front(cb_gegr, MAXBLK_G);
        scatter_scalars(dg_acc, get_read_ptr(cb_gegr), wstage, b, t0, h);
        cb_pop_front(cb_gegr, MAXBLK_G);

        if constexpr (gH < 32) {
            if (h == gH - 1) {
                zero_pad_columns(db_acc, zbuf, b, t0);
                zero_pad_columns(dg_acc, zbuf, b, t0);
            }
        }
    }
}
