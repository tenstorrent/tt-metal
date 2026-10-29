// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN gates over one prefill chunk, reader (NCRISC):
//   g = neg_a * softplus(a + dt) * mask,  beta = sigmoid(b) * mask
// a | b are the 96 columns at tile ab0 of the projection p (a: 48, b: 48); mask is 1 on the chunk's valid rows
// when the chunk updates the state (control page [0] valid rows, [1] update), else 0, so masked rows leave the
// recurrent state unchanged. Per row tile the reader regroups the three p tiles into a0 a1 b0 b1 by whole
// 16-column faces (48 = 3 faces).
//
// Compile-time args: 0 Pw (p width, tiles), 1 ab0 (tile column of a). Runtime args: 0 first row tile,
// 1 row tiles, 2 p, 3 dt, 4 neg_a (2 fp32 tiles, values in row 0), 5 control addresses.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_raw = 0, cb_a = 1, cb_b = 2, cb_dt = 3, cb_na = 4, cb_m = 5;
constexpr uint32_t kFace = 512;  // bf16 16 x 16

inline void copy_face(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < kFace / 4; i++) {
        d[i] = s[i];
    }
}

inline void zero_face(uint32_t dst) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < kFace / 4; i++) {
        d[i] = 0;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Pw = get_compile_time_arg_val(0);
    constexpr uint32_t ab0 = get_compile_time_arg_val(1);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    constexpr uint32_t bf_tile = get_tile_size(cb_raw), f32_tile = get_tile_size(cb_dt);
    const InterleavedAddrGenFast<true> p{
        .bank_base_address = get_arg_val<uint32_t>(2), .page_size = bf_tile, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> dt{
        .bank_base_address = get_arg_val<uint32_t>(3), .page_size = f32_tile, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> na{
        .bank_base_address = get_arg_val<uint32_t>(4), .page_size = f32_tile, .data_format = DataFormat::Float32};
    const InterleavedAddrGen<true> ctrl{.bank_base_address = get_arg_val<uint32_t>(5), .page_size = 32};

    cb_reserve_back(cb_dt, 2);
    cb_reserve_back(cb_na, 2);
    for (uint32_t c = 0; c < 2; c++) {
        noc_async_read_tile(c, dt, get_write_ptr(cb_dt) + c * f32_tile);
        noc_async_read_tile(c, na, get_write_ptr(cb_na) + c * f32_tile);
    }
    cb_reserve_back(cb_raw, 3);
    const uint32_t raw = get_write_ptr(cb_raw);
    noc_async_read(ctrl.get_noc_addr(0), raw, 32);
    noc_async_read_barrier();
    const uint32_t valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw)[0];
    const uint32_t update = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw)[1];
    cb_push_back(cb_dt, 2);
    cb_push_back(cb_na, 2);

    for (uint32_t r = first; r < first + rows; r++) {
        for (uint32_t i = 0; i < 3; i++) {
            noc_async_read_tile(r * Pw + ab0 + i, p, raw + i * bf_tile);
        }
        cb_reserve_back(cb_m, 1);
        {
            // fp32 tile, faces of 16 x 16: rows 16..31 are faces 2 and 3
            volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_m));
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t on = update && r * 32 + row < valid ? 0x3f800000 : 0;
                const uint32_t f0 = (row / 16) * 2;
                for (uint32_t f = f0; f < f0 + 2; f++) {
                    for (uint32_t col = 0; col < 16; col++) {
                        m[f * 256 + (row % 16) * 16 + col] = on;
                    }
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_m, 1);
        const uint32_t t0 = raw, t1 = raw + bf_tile, t2 = raw + 2 * bf_tile;
        cb_reserve_back(cb_a, 2);
        cb_reserve_back(cb_b, 2);
        const uint32_t a = get_write_ptr(cb_a), b = get_write_ptr(cb_b);
        // a0 = a cols 0..31; a1 = a cols 32..47; b0 = b cols 0..31; b1 = b cols 32..47
        for (uint32_t f = 0; f < 4; f++) {
            copy_face(a + f * kFace, t0 + f * kFace);
        }
        for (uint32_t h = 0; h < 2; h++) {  // face rows 0..15, 16..31
            const uint32_t L = 2 * h, R = 2 * h + 1;
            copy_face(a + bf_tile + L * kFace, t1 + L * kFace);
            zero_face(a + bf_tile + R * kFace);
            copy_face(b + L * kFace, t1 + R * kFace);
            copy_face(b + R * kFace, t2 + L * kFace);
            copy_face(b + bf_tile + L * kFace, t2 + R * kFace);
            zero_face(b + bf_tile + R * kFace);
        }
        cb_push_back(cb_a, 2);
        cb_push_back(cb_b, 2);
    }
}
