// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, k / v head writer: this core's head row (one dense page, 128 elements = face 0 rows
// 0..7 of the packed tile) -> row `pos` of head `h` of up to two KV caches [1, 8, S, 32 * W] (TILE, interleaved
// DRAM), in place of paged_update_cache. The slot index is read on device (trace safe).
//   bf16:  each 16-element chunk is one 32 B face-row write.
//   bfp8:  the packed tile holds the chunk's shared exponent (byte = face row) and 16 mantissa bytes; the cache
//          tile's exponent block (64 B) is read, patched and written back, the mantissas written in place.
// mass: also write the far-cache mass chunk [1, 0 x 15] (packed in the same way, cb_mass) at tile t0_mass.
// chain (multi-token verify prep, tokens in one program): the bfp8 exponent blocks are read, patched and written
// back, so tokens that write rows of the same tile must not overlap. Runtime args after the destinations: wait,
// next_x, next_y, has_next. With wait, the core waits on semaphore 0 (the previous token's writer) before it
// reads the exponent blocks; after its writes land it increments the next token's semaphore.
#include "attn_io.hpp"

namespace {
// Row r of the cache tiles page0 .. (one per pair of 16-element chunks): chunk c of the packed source (bf16: 32 B
// at 32c; bfp8: exponent byte c and 16 mantissa bytes at 64 + 16c) -> face (r / 16) * 2 + c % 2, face row r % 16.
// bfp8: exps holds the tiles' prefetched 64 B exponent blocks (one per tile), patched here and written back.
FORCE_INLINE void write_row(
    uint32_t base,
    uint32_t tile_bytes,
    uint32_t page0,
    uint32_t r,
    uint32_t src,
    uint32_t nchunks,
    uint32_t exps,
    uint32_t bfp8) {
    const InterleavedAddrGen<true> dst = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t f0 = (r >> 4) * 2, fr = r & 15;
    for (uint32_t c = 0; c < nchunks; c += 2) {
        const uint32_t page = page0 + c / 2;
        if (bfp8) {
            const uint32_t e_l1 = exps + (c / 2) * 64;
            volatile tt_l1_ptr uint8_t* e = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(e_l1);
            volatile tt_l1_ptr uint8_t* s = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
            for (uint32_t h = 0; h < 2; ++h) {
                e[(f0 + h) * 16 + fr] = s[c + h];
                noc_async_write(src + 64 + (c + h) * 16, dst.get_noc_addr(page, 64 + (f0 + h) * 256 + fr * 16), 16);
            }
            noc_async_write(e_l1, dst.get_noc_addr(page, 0), 64);
        } else {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(src + (c + h) * 32, dst.get_noc_addr(page, ((f0 + h) * 256 + fr * 16) * 2), 32);
            }
        }
    }
}

FORCE_INLINE void read_exps(uint32_t base, uint32_t tile_bytes, uint32_t page0, uint32_t ntiles, uint32_t exps) {
    const InterleavedAddrGen<true> dst = {.bank_base_address = base, .page_size = tile_bytes};
    for (uint32_t t = 0; t < ntiles; ++t) {
        noc_async_read(dst.get_noc_addr(page0 + t, 0), exps + t * 64, 64);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t ndst = get_named_compile_time_arg_val("ndst");
    constexpr uint32_t bfp8 = get_named_compile_time_arg_val("bfp8");
    constexpr uint32_t mass = get_named_compile_time_arg_val("mass");
    constexpr uint32_t chain = get_named_compile_time_arg_val("chain");
    constexpr uint32_t cb_out = 19, cb_mass = 20, cb_stage = 25;
    constexpr uint32_t tile_bytes = bfp8 ? 1088 : 2048;

    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    const uint32_t h = get_arg_val<uint32_t>(0);
    // stage: [64 d) slot index of destination d, then per destination 5 exponent blocks (4 row tiles + mass)
    uint32_t base[ndst], page0[ndst], row0[ndst], r[ndst];  // row0: first tile of the cache row
    for (uint32_t d = 0; d < ndst; ++d) {
        const InterleavedAddrGen<true> pa = {.bank_base_address = get_arg_val<uint32_t>(1 + d * 5 + 4), .page_size = 4};
        noc_async_read(pa.get_noc_addr(0, 0), stage + 64 * d, 64);
    }
    noc_async_read_barrier();
    if constexpr (chain) {
        if (get_arg_val<uint32_t>(1 + ndst * 5)) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
            noc_semaphore_wait(sem, 1);
            noc_semaphore_set(sem, 0);
        }
    }
    for (uint32_t d = 0; d < ndst; ++d) {
        const uint32_t a = 1 + d * 5;
        base[d] = get_arg_val<uint32_t>(a);
        const uint32_t seq_tiles = get_arg_val<uint32_t>(a + 1);  // S / 32
        const uint32_t width = get_arg_val<uint32_t>(a + 2);      // tiles per row
        const uint32_t t0 = get_arg_val<uint32_t>(a + 3);
        const uint32_t pos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64 * d);
        row0[d] = (h * seq_tiles + (pos >> 5)) * width;
        page0[d] = row0[d] + t0;
        r[d] = pos & 31;
        if constexpr (bfp8) {  // prefetch the exponent blocks while compute runs
            const uint32_t exps = stage + 64 * ndst + d * 5 * 64;
            read_exps(base[d], tile_bytes, page0[d], 4, exps);
            if (mass && d == ndst - 1) {
                read_exps(base[d], tile_bytes, row0[d] + 4, 1, exps + 4 * 64);
            }
        }
    }
    noc_async_read_barrier();
    if constexpr (mass) {
        cb_wait_front(cb_mass, 1);
    }
    cb_wait_front(cb_out, 1);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t d = 0; d < ndst; ++d) {
        const uint32_t exps = stage + 64 * ndst + d * 5 * 64;
        write_row(base[d], tile_bytes, page0[d], r[d], src, 8, exps, bfp8);
        if (mass && d == ndst - 1) {  // the far cache is the last destination; mass chunk at tile 4
            write_row(base[d], tile_bytes, row0[d] + 4, r[d], get_read_ptr(cb_mass), 2, exps + 4 * 64, bfp8);
        }
    }
    noc_async_write_barrier();
    if constexpr (chain) {
        const uint32_t c = 1 + ndst * 5;
        if (get_arg_val<uint32_t>(c + 3)) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(c + 1), get_arg_val<uint32_t>(c + 2), get_semaphore(0)), 1);
            noc_async_atomic_barrier();
        }
    }
    cb_pop_front(cb_out, 1);
    if constexpr (mass) {
        cb_pop_front(cb_mass, 1);
    }
}
