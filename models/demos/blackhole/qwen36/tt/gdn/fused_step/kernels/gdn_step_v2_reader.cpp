// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode step v2, reader (fused_step/gdn_step_v2.py). Core (h, j): value head h, value-column tile j, key head kh.
// 1. dense conv inputs: row 0 of each channel tile the core needs (dense slots: q_t, k_t of kh, v_j; plus z_j in the
//    new-input tile) -> even rows of ONE tile per conv-history slot (st1, st2, st3) and one for the new input (proj),
//    64-byte reads of rows 0-1 of faces 0/1 (the odd rows hold the source's padding row 1, finite, multiplied by 0)
//    + the core's 4 dense tap tiles                                                         -> cb_h, cb_x, cb_tap
// 2. constants, gates (v1 layout), norm weight tile j, state tiles S[h][t][j]               -> cb_const .. cb_S
// 3. after every read of the shared conv history: +1 on the semaphore of each core of the key group (owners shift)
// 4. the compute core's normalised dense tile (cb_n, fp32) -> row-0 tiles q_t, k_t, v_j, silu(z_j) and the column
//    tiles k_t^T (exact fp32 copies; the target tiles were zero-filled in step 2)
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_h = 0, cb_x = 1, cb_tap = 2, cb_abraw = 3;
constexpr uint32_t cb_const = 6, cb_n = 10, cb_aq = 11, cb_ak = 12, cb_kcol = 13, cb_v = 14, cb_sz = 15;
constexpr uint32_t cb_a = 16, cb_b = 17, cb_na = 18, cb_dt = 19, cb_S = 22, cb_gather = 31, cb_w = 33;
constexpr uint32_t NCONST = 6;

constexpr uint32_t BF = 2048, F32 = 4096;

// byte offset of dense slot s (row r = 2s, or 16 + 2(s - 8)), face half f (0: columns 0-15, 1: 16-31)
template <uint32_t face_bytes, uint32_t row_bytes>
constexpr uint32_t slot_off(uint32_t s, uint32_t f) {
    const uint32_t r = s < 8 ? 2 * s : 16 + 2 * (s - 8);
    return ((r >= 16 ? 2 : 0) + f) * face_bytes + (r % 16) * row_bytes;
}

FORCE_INLINE void zero_fill(uint32_t dst, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < bytes; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, dst + z, (bytes - z) < MEM_ZEROS_SIZE ? (bytes - z) : MEM_ZEROS_SIZE);
    }
}

// zero the dense slots s0..15 of a bf16 tile (64 B per slot and face half)
FORCE_INLINE void zero_slots_bf16(uint32_t base, uint32_t s0) {
    for (uint32_t s = s0; s < 16; s++) {
        for (uint32_t f = 0; f < 2; f++) {
            zero_fill(base + slot_off<512, 32>(s, f), 64);
        }
    }
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t z_off = get_compile_time_arg_val(2);
    constexpr uint32_t ab_tile = get_compile_time_arg_val(3);
    constexpr uint32_t a_col = get_compile_time_arg_val(4);
    constexpr uint32_t Nv = get_compile_time_arg_val(5);
    constexpr uint32_t sem_rd = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);  // cores of the key group
    constexpr auto proj_args = TensorAccessorArgs<8>();
    constexpr auto st1_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();
    constexpr auto st2_args = TensorAccessorArgs<st1_args.next_compile_time_args_offset()>();
    constexpr auto st3_args = TensorAccessorArgs<st2_args.next_compile_time_args_offset()>();
    constexpr auto taps_args = TensorAccessorArgs<st3_args.next_compile_time_args_offset()>();
    constexpr auto gconst_args = TensorAccessorArgs<taps_args.next_compile_time_args_offset()>();
    constexpr auto normw_args = TensorAccessorArgs<gconst_args.next_compile_time_args_offset()>();
    constexpr auto consts_args = TensorAccessorArgs<normw_args.next_compile_time_args_offset()>();
    constexpr auto rec_args = TensorAccessorArgs<consts_args.next_compile_time_args_offset()>();
    constexpr uint32_t NS = 2 * Kt + 1;  // conv slots: q_t, k_t, v_j (z_j = slot NS, new input only)

    uint32_t i = 0;
    const uint32_t proj_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t taps_addr = get_arg_val<uint32_t>(i++);
    const uint32_t gconst_addr = get_arg_val<uint32_t>(i++);
    const uint32_t normw_addr = get_arg_val<uint32_t>(i++);
    const uint32_t consts_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);
    const uint32_t ci = get_arg_val<uint32_t>(i++);  // core index h * Vt + j
    uint32_t ch[NS];
    for (uint32_t s = 0; s < NS; s++) {
        ch[s] = get_arg_val<uint32_t>(i++);
    }
    const uint32_t group_arg = i;  // G (x, y) pairs

    const auto proj = TensorAccessor(proj_args, proj_addr, BF);
    const auto st1 = TensorAccessor(st1_args, st1_addr, BF);
    const auto st2 = TensorAccessor(st2_args, st2_addr, BF);
    const auto st3 = TensorAccessor(st3_args, st3_addr, BF);
    const auto taps = TensorAccessor(taps_args, taps_addr, BF);
    const auto gconst = TensorAccessor(gconst_args, gconst_addr, F32);
    const auto normw = TensorAccessor(normw_args, normw_addr, F32);
    const auto consts = TensorAccessor(consts_args, consts_addr, F32);
    const auto rec = TensorAccessor(rec_args, rec_addr, F32);

    // ---- 1. dense conv inputs ----
    {
        DeviceZoneScopedN("R2-dense");
        cb_reserve_back(cb_h, 3);
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_tap, 4);
        const uint32_t hb = get_write_ptr(cb_h);
        const uint32_t xb = get_write_ptr(cb_x);
        for (uint32_t s = 0; s < NS; s++) {
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t so = f * 512;  // row 0 of face f (+ row 1)
                const uint32_t d = slot_off<512, 32>(s, f);
                noc_async_read(st1.get_noc_addr(ch[s], so), hb + d, 64);
                noc_async_read(st2.get_noc_addr(ch[s], so), hb + BF + d, 64);
                noc_async_read(st3.get_noc_addr(ch[s], so), hb + 2 * BF + d, 64);
                noc_async_read(proj.get_noc_addr(ch[s], so), xb + d, 64);
            }
        }
        for (uint32_t f = 0; f < 2; f++) {  // z_j into slot NS of the new-input tile
            noc_async_read(proj.get_noc_addr(z_off + h * Vt + j, f * 512), xb + slot_off<512, 32>(NS, f), 64);
        }
        zero_slots_bf16(hb, NS);
        zero_slots_bf16(hb + BF, NS);
        zero_slots_bf16(hb + 2 * BF, NS);
        zero_slots_bf16(xb, NS + 1);
        uint32_t dst = get_write_ptr(cb_tap);
        for (uint32_t k = 0; k < 4; k++, dst += BF) {
            noc_async_read_page(ci * 4 + k, taps, dst);
        }
        noc_async_read_barrier();
        cb_push_back(cb_h, 3);
        cb_push_back(cb_x, 1);
        cb_push_back(cb_tap, 4);
    }

    // ---- 2. constants, gates, norm weight, state; zero the split targets and the gather tile ----
    {
        DeviceZoneScopedN("R2-consts");
        cb_reserve_back(cb_const, NCONST);
        uint32_t dst = get_write_ptr(cb_const);
        for (uint32_t t = 0; t < NCONST; t++, dst += F32) {
            noc_async_read_page(t, consts, dst);
        }
        cb_reserve_back(cb_abraw, 1);
        noc_async_read_page(ab_tile, proj, get_write_ptr(cb_abraw));
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        zero_fill(get_write_ptr(cb_a), F32);
        zero_fill(get_write_ptr(cb_b), F32);
        cb_reserve_back(cb_na, 1);
        cb_reserve_back(cb_dt, 1);
        noc_async_read_page(2 * h, gconst, get_write_ptr(cb_na));
        noc_async_read_page(2 * h + 1, gconst, get_write_ptr(cb_dt));
        cb_reserve_back(cb_w, 1);
        noc_async_read_page(j, normw, get_write_ptr(cb_w));
        cb_reserve_back(cb_S, Kt);
        dst = get_write_ptr(cb_S);
        for (uint32_t t = 0; t < Kt; t++, dst += F32) {
            noc_async_read_page((h * Kt + t) * Vt + j, rec, dst);
        }
        noc_async_read_barrier();
        cb_push_back(cb_const, NCONST);
        {
            volatile tt_l1_ptr uint16_t* raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_abraw));
            volatile tt_l1_ptr uint32_t* pa = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_a));
            volatile tt_l1_ptr uint32_t* pb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_b));
            pa[0] = static_cast<uint32_t>(raw[a_col + h]) << 16;
            pb[0] = static_cast<uint32_t>(raw[a_col + Nv + h]) << 16;
        }
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
        cb_push_back(cb_na, 1);
        cb_push_back(cb_dt, 1);
        cb_push_back(cb_S, Kt);
        cb_push_back(cb_w, 1);
        // split targets + gather tile: zero one tile, then copy it around (local NOC reads)
        cb_reserve_back(cb_aq, Kt);
        cb_reserve_back(cb_ak, Kt);
        cb_reserve_back(cb_kcol, Kt);
        cb_reserve_back(cb_v, 1);
        cb_reserve_back(cb_sz, 1);
        const uint32_t z0 = get_write_ptr(cb_v);
        zero_fill(z0, F32);
        noc_async_read_barrier();
        const uint64_t zsrc = get_noc_addr(z0);
        for (uint32_t t = 0; t < Kt; t++) {
            noc_async_read(zsrc, get_write_ptr(cb_aq) + t * F32, F32);
            noc_async_read(zsrc, get_write_ptr(cb_ak) + t * F32, F32);
            noc_async_read(zsrc, get_write_ptr(cb_kcol) + t * F32, F32);
        }
        noc_async_read(zsrc, get_write_ptr(cb_sz), F32);
        noc_async_read(zsrc, get_write_ptr(cb_gather), F32);  // the writer produces cb_gather; same address
        noc_async_read_barrier();
    }
    // ---- 3. every read of the shared conv history is done, and the gather tile is zero ----
    for (uint32_t g = 0; g < G; g++) {
        const uint32_t px = get_arg_val<uint32_t>(group_arg + 2 * g);
        const uint32_t py = get_arg_val<uint32_t>(group_arg + 2 * g + 1);
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_rd)), 1);
    }

    // ---- 4. split the normalised dense tile ----
    {
        DeviceZoneScopedN("R2-split");
        cb_wait_front(cb_n, 1);
        volatile tt_l1_ptr uint32_t* n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_n));
        auto row_copy = [&](uint32_t s, uint32_t dst_addr) {
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t so = slot_off<1024, 64>(s, f) / 4;
                for (uint32_t e = 0; e < 16; e++) {
                    d[f * 256 + e] = n[so + e];
                }
            }
        };
        for (uint32_t t = 0; t < Kt; t++) {
            row_copy(t, get_write_ptr(cb_aq) + t * F32);
            row_copy(Kt + t, get_write_ptr(cb_ak) + t * F32);
            // k_t as column 0: element r at face (r >= 16 ? 2 : 0), row r % 16, column 0
            volatile tt_l1_ptr uint32_t* d =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_kcol) + t * F32);
            for (uint32_t f = 0; f < 2; f++) {
                const uint32_t so = slot_off<1024, 64>(Kt + t, f) / 4;
                for (uint32_t e = 0; e < 16; e++) {
                    d[f * 512 + e * 16] = n[so + e];
                }
            }
        }
        row_copy(2 * Kt, get_write_ptr(cb_v));
        row_copy(2 * Kt + 1, get_write_ptr(cb_sz));
        cb_pop_front(cb_n, 1);
        cb_push_back(cb_aq, Kt);
        cb_push_back(cb_ak, Kt);
        cb_push_back(cb_kcol, Kt);
        cb_push_back(cb_v, 1);
        cb_push_back(cb_sz, 1);
    }
}
