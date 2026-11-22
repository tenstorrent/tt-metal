// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Staged copy (M2, tt/mk/gate_stage.py) of tt/fused/kernels/gate_combine_{reader,writer}.cpp: one extra runtime arg
// (last), `stage`: when non-zero, this core's F (bytes 0 .. kz * nt * 4 KB) and G (next kz * nt * 4 KB) slices were
// copied into local L1 at `stage` earlier, in the order read here, and are taken from there instead of DRAM.
//
// Message-gate combine, reader (one core per nt column tiles): z (+ block mask), the core's forget / gain column
// slices, and its h / m row-form tiles.
#include "models/demos/inkling2_mini/tt/fused/kernels/rowio.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");          // z width in tiles (64 * gates / 32)
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");          // column tiles per core
    constexpr uint32_t nct = get_named_compile_time_arg_val("nct");        // C / 32
    constexpr uint32_t h_tile = get_named_compile_time_arg_val("h_tile");  // page bytes of h and m
    constexpr uint32_t cb_z = 0, cb_mask = 1, cb_f = 2, cb_g = 3, cb_h = 4, cb_m = 5;

    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t block = get_named_compile_time_arg_val("block");  // row i through gate i only (block mask)
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");

    const uint32_t z_addr = get_arg_val<uint32_t>(0);
    const uint32_t f_addr = get_arg_val<uint32_t>(2);
    const uint32_t h_addr = get_arg_val<uint32_t>(4);
    const uint32_t m_addr = get_arg_val<uint32_t>(5);
    const uint32_t col0 = get_arg_val<uint32_t>(6);
    const uint32_t h_dense = get_arg_val<uint32_t>(7);  // h / m given as dense twins (fp32): vector r = pages r*T ..
    const uint32_t m_dense = get_arg_val<uint32_t>(8);
    const uint32_t h_nvec = get_arg_val<uint32_t>(9);  // twin vectors (rows >= nvec repeat the last one)
    const uint32_t m_nvec = get_arg_val<uint32_t>(10);
    const uint32_t stage = get_arg_val<uint32_t>(12);

    DeviceZoneScopedN("R_all");
    // bulk weights first
    // G and F rows kt < f_w: read by the writer (the other NOC), in parallel
    constexpr uint32_t f_w = get_named_compile_time_arg_val("f_w");
    cb_reserve_back(cb_f, kz * nt);
    const uint32_t f = get_write_ptr(cb_f);
    for (uint32_t kt = f_w; kt < kz; ++kt) {
        if (stage) {
            noc_async_read(get_noc_addr(stage + kt * nt * 4096), f + kt * nt * 4096, nt * 4096);
        } else {
            rowio::read_pages(f_addr, 4096, kt * nct + col0, nt, f + kt * nt * 4096);
        }
    }
    // z: only rows 0..rows-1 matter (output rows >= rows are never written; matmul rows are independent). With
    // the block mask, row i keeps z tiles 2i, 2i+1 (gate i) and the rest of the row is zeroed.
    cb_reserve_back(cb_z, kz);
    const uint32_t zl = get_write_ptr(cb_z);
    // compact z (gatez): rows only, block (t, r) = 128 B at (t * rows + r) * 128 of copy z_copy; staged in cb 1
    const uint32_t zst = get_write_ptr(cb_mask);
    const InterleavedAddrGen<true> zs = {.bank_base_address = z_addr, .page_size = 4096};
    noc_async_read(zs.get_noc_addr(get_arg_val<uint32_t>(11)), zst, kz * rows * 128);
    noc_async_read_barrier();
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t off = rowio::row_offset(r, 4);
        for (uint32_t t = 0; t < kz; ++t) {
            const uint32_t d = zl + t * 4096 + off;
            if (!block || t / 2 == r) {
                const uint64_t a = get_noc_addr(zst + (t * rows + r) * 128);
                noc_async_read(a, d, 64);
                noc_async_read(a + 64, d + 1024, 64);
            } else {
                rowio::zero_words(d, 64);
                rowio::zero_words(d + 1024, 64);
            }
        }
    }
    // h, m: column tiles col0 .. of row form, or rows of the dense twins (elements 32 j .. 32 j + 31 of vector r:
    // 128 contiguous bytes -> the two face-row chunks of tile row r)
    cb_reserve_back(cb_h, nt);
    cb_reserve_back(cb_m, nt);
    const uint32_t srcs[2] = {h_addr, m_addr}, dense[2] = {h_dense, m_dense}, nvec[2] = {h_nvec, m_nvec};
    const uint32_t cbs[2] = {get_write_ptr(cb_h), get_write_ptr(cb_m)};
    for (uint32_t i = 0; i < 2; ++i) {
        if (!dense[i]) {
            rowio::read_pages(srcs[i], h_tile, col0, nt, cbs[i]);
            continue;
        }
        const InterleavedAddrGen<true> ds = {.bank_base_address = srcs[i], .page_size = 4096};
        for (uint32_t j = 0; j < nt; ++j) {
            const uint32_t e = 32 * (col0 + j);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t v = r < nvec[i] ? r : nvec[i] - 1;
                const uint64_t a = ds.get_noc_addr(v * T + e / 1024, (e % 1024) * 4);
                const uint32_t d = cbs[i] + j * 4096 + rowio::row_offset(r, 4);
                noc_async_read(a, d, 64);
                noc_async_read(a + 64, d + 1024, 64);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_z, kz);
    if constexpr (f_w > 0) {  // the writer's F rows have landed
        cb_wait_front(6, 1);
        cb_pop_front(6, 1);
    }
    cb_push_back(cb_f, kz * nt);
    cb_push_back(cb_h, nt);
    cb_push_back(cb_m, nt);
}
