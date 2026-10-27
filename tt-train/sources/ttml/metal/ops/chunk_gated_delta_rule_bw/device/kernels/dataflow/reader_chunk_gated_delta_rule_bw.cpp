// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// chunk_gated_delta_rule_bw reader.  Owns every DRAM -> L1 load and the two semaphore WAITS that
// separate the three stages:
//   Stage P : q/k/v/do blocks of each (bh, chunk) item, and the g/beta columns.
//   Stage S : per-chunk loads of the prep scratch for the sequential scan (owner cores only),
//             after the Barrier-1 wait.
//   Stage G : the inputs again plus the derived per-chunk / per-V-block scratch, after the
//             Barrier-2 wait.
// q/k are read from key head h / G, so GVA needs no host-side repeat.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_bw/device/kernels/chunk_gated_delta_rule_bw_common.hpp"

// The three kernels share one kernel-config ring buffer and the compute binaries dominate it; these
// loops are NoC-issue bound, not instruction bound.
#pragma GCC optimize("Os")

static __attribute__((noipa)) void store_f32(uint32_t addr, uint32_t elem, float value) {
    volatile tt_l1_ptr float* p = reinterpret_cast<volatile tt_l1_ptr float*>(addr);
    p[elem] = value;
}

// L1 address of the resident all-zeros constant tile (set once at boot).
static uint32_t g_zero_tile_addr = 0;
static constexpr uint32_t ZTILE_BYTES = 32 * 32 * 4;

// Zero an L1 range by NoC-copying the resident zero tile over it, so the RISC stays free.
static __attribute__((noipa)) void zero_l1(uint32_t dst, uint32_t nbytes) {
    while (nbytes) {
        const uint32_t chunk = nbytes > ZTILE_BYTES ? ZTILE_BYTES : nbytes;
        noc_async_write(g_zero_tile_addr, get_noc_addr(my_x[noc_index], my_y[noc_index], dst), chunk);
        dst += chunk;
        nbytes -= chunk;
    }
    noc_async_write_barrier();
}

// Six [C,C] mask blocks, a scale tile, a zero tile and Ct row-of-ones tiles, built with raw L1
// stores once per core.  Zero-filled over the NoC first; only the non-zero lanes are then written.
void build_constant_tiles(uint32_t scale_bits) {
    constexpr uint32_t f32_tile = 32 * 32 * 4;
    constexpr uint32_t nconst = NCONST;

    cb_reserve_back(cb_const, nconst);
    const uint32_t cbase = get_write_ptr(cb_const);
    // Bootstrap: the zero tile itself is the only region written by the RISC.
    g_zero_tile_addr = cbase + CST_ZERO * f32_tile;
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_zero_tile_addr);
        for (uint32_t i = 0; i < 32 * 32; ++i) {
            z[i] = 0;
        }
    }
    zero_l1(cbase, CST_ZERO * f32_tile);
    zero_l1(cbase + (CST_ZERO + 1) * f32_tile, (nconst - CST_ZERO - 1) * f32_tile);

    for (uint32_t ti = 0; ti < Ct; ++ti) {
        for (uint32_t si = 0; si < Ct; ++si) {
            const uint32_t t = ti * Ct + si;
            const uint32_t lt = cbase + (CST_LT + t) * f32_tile;
            const uint32_t st = cbase + (CST_NSTRICT + t) * f32_tile;
            const uint32_t ut = cbase + (CST_UT + t) * f32_tile;
            const uint32_t ey = cbase + (CST_EYE + t) * f32_tile;
            const uint32_t bi = cbase + (CST_BIAS + t) * f32_tile;
            const uint32_t su = cbase + (CST_SUT + t) * f32_tile;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t gr = ti * 32 + r;
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t gc = si * 32 + c;
                    const uint32_t e = tile_elem_off(r, c);
                    if (gc <= gr) {
                        store_f32(lt, e, 1.0f);
                    } else {
                        store_f32(bi, e, -1.0e4f);
                    }
                    if (gc < gr) {
                        // NSTRICT carries -1 so both consumers of the strict mask (A and dA) come
                        // out already negated.
                        store_f32(st, e, -1.0f);
                    }
                    if (gc >= gr) {
                        store_f32(ut, e, 1.0f);
                    }
                    if (gc > gr) {
                        store_f32(su, e, 1.0f);
                    }
                    if (gc == gr) {
                        store_f32(ey, e, 1.0f);
                    }
                }
            }
        }
    }
    // Scale tile: column 0 only, consumed as a stride-0 COL-broadcast vector so the compute kernel
    // instantiates a single broadcast flavour.
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbase + CST_SCALE * f32_tile);
        for (uint32_t r = 0; r < 32; ++r) {
            p[tile_elem_off(r, 0)] = scale_bits;
        }
    }
    // Row-of-ones tiles: the [1,C] operand of the outer product that builds the column-broadcast
    // matrix for L.
    for (uint32_t t = 0; t < Ct; ++t) {
        const uint32_t base = cbase + (CST_ROWONES + t) * f32_tile;
        for (uint32_t c = 0; c < 32; ++c) {
            store_f32(base, tile_elem_off(0, c), 1.0f);
        }
    }
    cb_push_back(cb_const, nconst);

    // Column-of-ones tiles: the row-sum matmul operand (X @ colones == rowsum).
    constexpr uint32_t ncol = NCOL;
    cb_reserve_back(cb_colones, ncol);
    const uint32_t colbase = get_write_ptr(cb_colones);
    zero_l1(colbase, ncol * f32_tile);
    for (uint32_t t = 0; t < ncol; ++t) {
        const uint32_t base = colbase + t * f32_tile;
        for (uint32_t r = 0; r < 32; ++r) {
            store_f32(base, tile_elem_off(r, 0), 1.0f);
        }
    }
    cb_push_back(cb_colones, ncol);
}

// Column gather for the [B,1,T,HV] gate tensors: head h is COLUMN h of page (b, t/32).  One
// whole-page read per token tile, then 32 scalar extracts into column 0 of the destination.
template <typename ACC>
void gather_gate(
    const ACC& acc, uint32_t tile_bytes, uint32_t slot, uint32_t dst_base, uint32_t b, uint32_t t0, uint32_t h) {
    const uint32_t stage64 = (get_write_ptr(cb_stage) + 63u) & ~63u;
    for (uint32_t ct = 0; ct < Ct; ++ct) {
        const uint32_t tbase = t0 + ct * 32;
        if (tbase >= gT) {
            break;
        }
        const uint32_t tt = tbase >> 5;
        noc_async_read(acc.get_noc_addr(b * Tt + tt, 0), stage64, tile_bytes);
        noc_async_read_barrier();
        const uint32_t tile_addr = dst_base + (slot + ct) * tile_bytes;
        for (uint32_t r = 0; r < 32; ++r) {
            if (tbase + r >= gT) {
                break;
            }
            const uint32_t se = tile_elem_off(r, h);
            const uint32_t de = tile_elem_off(r, 0);
            if constexpr (ESZ == 4) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage64);
                volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr);
                d[de] = s[se];
            } else {
                volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage64);
                volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr);
                d[de] = s[se];
            }
        }
    }
}

// A [Ct, dn] block of head h out of a token-major [B,1,T,Hx*D] grid: Ct*dn whole pages, packed
// row-major (r*dn + j) into the CB.  `block_pages` >= Ct*dn is the CB's uniform push size; the tail
// is never read by the consumer.
template <typename ACC>
static __attribute__((noipa)) void read_flat_block(
    const ACC& acc,
    uint32_t cb,
    uint32_t block_pages,
    uint32_t b,
    uint32_t tt0,
    uint32_t hx_dt,
    uint32_t h,
    uint32_t dt,
    uint32_t d0,
    uint32_t dn) {
    constexpr uint32_t tile_bytes = 32 * 32 * ESZ;
    cb_reserve_back(cb, block_pages);
    const uint32_t dst = get_write_ptr(cb);
    for (uint32_t r = 0; r < Ct; ++r) {
        for (uint32_t j = 0; j < dn; ++j) {
            noc_async_read(
                acc.get_noc_addr(flat_page(b, tt0 + r, hx_dt, h, dt, d0 + j)),
                dst + (r * dn + j) * tile_bytes,
                tile_bytes);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb, block_pages);
}

// The [Ct, Vb] d_out block of value head h; head-major d_out is one head per outer index bh.
template <typename ACC>
static void read_do_block(const ACC& acc, uint32_t bh, uint32_t b, uint32_t tt0, uint32_t h, uint32_t vb) {
    if constexpr (DO_HEAD_MAJOR) {
        read_flat_block(acc, cb_doin, CtVb, bh, tt0, Vt, 0, Vt, vb * Vb, Vb);
    } else {
        read_flat_block(acc, cb_doin, CtVb, b, tt0, gH * Vt, h, Vt, vb * Vb, Vb);
    }
}

// Read a rows x cols sub-block out of the flat scratch tensor.
template <typename ACC>
static __attribute__((noipa)) void read_sub(
    const ACC& acc,
    uint32_t dst_addr,
    uint32_t base_tile,
    uint32_t rows,
    uint32_t row_stride,
    uint32_t col0,
    uint32_t cols,
    uint32_t tile_bytes) {
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_read(
                acc.get_noc_addr(base_tile + r * row_stride + col0 + c),
                dst_addr + (r * cols + c) * tile_bytes,
                tile_bytes);
        }
    }
}

// A [K, Vb] block of head (b, h) out of a [B,HV,K,V] state tensor (h0 or dht).
template <typename ACC>
static __attribute__((noipa)) void read_state_block(const ACC& acc, uint32_t b, uint32_t h, uint32_t vb) {
    constexpr uint32_t in_tile = 32 * 32 * ESZ;
    cb_reserve_back(cb_vin, MAXV);
    const uint32_t dst = get_write_ptr(cb_vin);
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        for (uint32_t j = 0; j < Vb; ++j) {
            noc_async_read(
                acc.get_noc_addr(((b * gH + h) * Kt + kt) * Vt + vb * Vb + j), dst + (kt * Vb + j) * in_tile, in_tile);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_vin, MAXV);
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t g_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t do_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t h0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dht_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t sc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t h_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t vn_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t item_start = get_arg_val<uint32_t>(arg++);
    const uint32_t num_items = get_arg_val<uint32_t>(arg++);
    const uint32_t num_release_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t scale_bits = get_arg_val<uint32_t>(arg++);
    const uint32_t num_owned = get_arg_val<uint32_t>(arg++);
    const uint32_t owned_base = arg;

    constexpr uint32_t in_tile = 32 * 32 * ESZ;
    constexpr uint32_t f32_tile = 32 * 32 * 4;

    constexpr auto qa = TensorAccessorArgs<CT_ACC_BASE>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto doa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto h0a = TensorAccessorArgs<doa.next_compile_time_args_offset()>();
    constexpr auto dhta = TensorAccessorArgs<h0a.next_compile_time_args_offset()>();
    constexpr auto sca = TensorAccessorArgs<dhta.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<sca.next_compile_time_args_offset()>();
    constexpr auto vna = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto aa = TensorAccessorArgs<vna.next_compile_time_args_offset()>();

    const auto q_acc = TensorAccessor(qa, q_addr, in_tile);
    const auto k_acc = TensorAccessor(ka, k_addr, in_tile);
    const auto v_acc = TensorAccessor(va, v_addr, in_tile);
    const auto g_acc = TensorAccessor(ga, g_addr, in_tile);
    const auto b_acc = TensorAccessor(ba, beta_addr, in_tile);
    const auto do_acc = TensorAccessor(doa, do_addr, in_tile);
    [[maybe_unused]] const auto h0_acc = TensorAccessor(h0a, h0_addr, in_tile);
    [[maybe_unused]] const auto dht_acc = TensorAccessor(dhta, dht_addr, in_tile);
    const auto sc_acc = TensorAccessor(sca, sc_addr, f32_tile);
    // Without the saved forward these alias the scratch at its S / v_new / Tinv bases.
    const auto h_acc = TensorAccessor(ha, h_addr, f32_tile);
    const auto vn_acc = TensorAccessor(vna, vn_addr, f32_tile);
    const auto a_acc = TensorAccessor(aa, a_addr, f32_tile);
    constexpr uint32_t S_BASE = HAS_SAVED ? 0 : SC_S;
    constexpr uint32_t VNEW_BASE = HAS_SAVED ? 0 : SC_VNEW;
    constexpr uint32_t TINV_BASE = HAS_SAVED ? 0 : SC_ATTN;

    build_constant_tiles(scale_bits);

    // ------------------------------------------------------------------
    // Stage P: one (bh, chunk) item per iteration.
    // ------------------------------------------------------------------
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;
        const uint32_t bh = wi / NC;
        const uint32_t ci = wi % NC;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t t0 = ci * CHUNK;
        const uint32_t tt0 = ci * Ct;
        const uint32_t hk = h / gG;

        read_flat_block(q_acc, cb_qin, CtKt, b, tt0, gHK * Kt, hk, Kt, 0, Kt);
        read_flat_block(k_acc, cb_kin, CtKt, b, tt0, gHK * Kt, hk, Kt, 0, Kt);

        cb_reserve_back(cb_gatein, 2 * Ct);
        {
            const uint32_t gbase = get_write_ptr(cb_gatein);
            zero_l1(gbase, 2 * Ct * in_tile);
            gather_gate(g_acc, in_tile, 0, gbase, b, t0, h);
            gather_gate(b_acc, in_tile, Ct, gbase, b, t0, h);
        }
        cb_push_back(cb_gatein, 2 * Ct);

        if constexpr (HAS_SAVED) {
            // Tinv rides in the first CtCt pages of a stage-G-sized cb_load_item block.
            cb_reserve_back(cb_load_item, LITEM);
            const uint32_t dst = get_write_ptr(cb_load_item);
            for (uint32_t i = 0; i < CtCt; ++i) {
                noc_async_read(a_acc.get_noc_addr(wi * ST_ATTN + i), dst + i * f32_tile, f32_tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_load_item, LITEM);
        }

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            if constexpr (!HAS_SAVED) {
                read_flat_block(v_acc, cb_vin, MAXV, b, tt0, gH * Vt, h, Vt, vb * Vb, Vb);
            }
            read_do_block(do_acc, bh, b, tt0, h, vb);
        }
    }

    // ------------------------------------------------------------------
    // Barrier 1: per-(bh) group fan-in.  Every core issues ALL of its prep before ANY wait, which
    // is what makes the rendezvous deadlock-free.
    // ------------------------------------------------------------------
    if (num_owned > 0) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PREP));
        noc_semaphore_wait_min(sem, NC * num_owned);

        // -------------------------------------------------------------
        // Stage S: the sequential scan, one core per (b, h).
        // -------------------------------------------------------------
        for (uint32_t oi = 0; oi < num_owned; ++oi) {
            const uint32_t bh = get_arg_val<uint32_t>(owned_base + oi);
            const uint32_t b = bh / gH;
            const uint32_t h = bh % gH;

            for (uint32_t vb = 0; vb < NVB; ++vb) {
                if constexpr (HAS_H0 && !HAS_SAVED) {
                    read_state_block(h0_acc, b, h, vb);
                }

                for (uint32_t i = 0; i < (HAS_SAVED ? 0 : NC); ++i) {
                    const uint32_t wi = bh * NC + i;
                    cb_reserve_back(cb_load_vb, LVB);
                    uint32_t dst = get_write_ptr(cb_load_vb);
                    read_sub(sc_acc, dst, SC_KCD + wi * ST_KCD, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_P + wi * ST_P, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_VCORR + wi * ST_VCORR, Ct, Vt, vb * Vb, Vb, f32_tile);
                    dst += CtVb * f32_tile;
                    noc_async_read(sc_acc.get_noc_addr(SC_VEC + wi * ST_VEC + 2 * Ct), dst, f32_tile);
                    noc_async_read_barrier();
                    cb_push_back(cb_load_vb, LVB);
                }

                if constexpr (HAS_DHT) {
                    read_state_block(dht_acc, b, h, vb);
                }

                for (int32_t i = (int32_t)NC - 1; i >= 0; --i) {
                    const uint32_t wi = bh * NC + (uint32_t)i;
                    cb_reserve_back(cb_load_vb, LVB);
                    uint32_t dst = get_write_ptr(cb_load_vb);
                    read_sub(sc_acc, dst, SC_P + wi * ST_P, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_U + wi * ST_U, Ct, Vt, vb * Vb, Vb, f32_tile);
                    dst += CtVb * f32_tile;
                    read_sub(sc_acc, dst, SC_KCD + wi * ST_KCD, Ct, Kt, 0, Kt, f32_tile);
                    dst += CtKt * f32_tile;
                    read_sub(sc_acc, dst, SC_C + wi * ST_C, Kt, Vt, vb * Vb, Vb, f32_tile);
                    dst += KtVb * f32_tile;
                    noc_async_read(sc_acc.get_noc_addr(SC_VEC + wi * ST_VEC + 2 * Ct), dst, f32_tile);
                    noc_async_read_barrier();
                    cb_push_back(cb_load_vb, LVB);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Barrier 2: per-group release.  Cores with no stage-G work wait for 0.
    // ------------------------------------------------------------------
    {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_SCAN));
        noc_semaphore_wait_min(sem, num_release_groups);
    }

    // ------------------------------------------------------------------
    // Stage G: gradient assembly.
    // ------------------------------------------------------------------
    for (uint32_t p = 0; p < num_items; ++p) {
        const uint32_t wi = item_start + p;
        const uint32_t bh = wi / NC;
        const uint32_t ci = wi % NC;
        const uint32_t b = bh / gH;
        const uint32_t h = bh % gH;
        const uint32_t tt0 = ci * Ct;

        read_flat_block(q_acc, cb_qin, CtKt, b, tt0, gHK * Kt, h / gG, Kt, 0, Kt);
        read_flat_block(k_acc, cb_kin, CtKt, b, tt0, gHK * Kt, h / gG, Kt, 0, Kt);
        cb_reserve_back(cb_load_item, LITEM);
        {
            uint32_t dst = get_write_ptr(cb_load_item);
            for (uint32_t i = 0; i < CtCt; ++i) {
                noc_async_read(a_acc.get_noc_addr(TINV_BASE + wi * ST_ATTN + i), dst + i * f32_tile, f32_tile);
            }
            dst += CtCt * f32_tile;
            for (uint32_t i = 0; i < 4 * Ct; ++i) {
                noc_async_read(sc_acc.get_noc_addr(SC_VEC + wi * ST_VEC + i), dst + i * f32_tile, f32_tile);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_load_item, LITEM);

        for (uint32_t vb = 0; vb < NVB; ++vb) {
            read_flat_block(v_acc, cb_vin, MAXV, b, tt0, gH * Vt, h, Vt, vb * Vb, Vb);
            read_do_block(do_acc, bh, b, tt0, h, vb);
            cb_reserve_back(cb_load_vb, LVB);
            {
                uint32_t dst = get_write_ptr(cb_load_vb);
                read_sub(vn_acc, dst, VNEW_BASE + wi * ST_VNEW, Ct, Vt, vb * Vb, Vb, f32_tile);
                dst += CtVb * f32_tile;
                read_sub(sc_acc, dst, SC_DVNEW + wi * ST_DVNEW, Ct, Vt, vb * Vb, Vb, f32_tile);
                dst += CtVb * f32_tile;
                read_sub(h_acc, dst, S_BASE + wi * ST_S, Kt, Vt, vb * Vb, Vb, f32_tile);
                dst += KtVb * f32_tile;
                read_sub(sc_acc, dst, SC_DS + wi * ST_DS, Kt, Vt, vb * Vb, Vb, f32_tile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_load_vb, LVB);
        }
    }
}
