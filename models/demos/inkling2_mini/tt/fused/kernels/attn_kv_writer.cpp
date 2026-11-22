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
#include "attn_io.hpp"

namespace {
FORCE_INLINE void write_row(
    uint32_t base,
    uint32_t tile_bytes,
    uint32_t page0,
    uint32_t r,
    uint32_t src,
    uint32_t first_chunk,
    uint32_t nchunks,
    uint32_t exp_stage,
    uint32_t bfp8) {
    const InterleavedAddrGen<true> dst = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t f0 = (r >> 4) * 2, fr = r & 15;
    for (uint32_t c = 0; c < nchunks; c += 2) {
        const uint32_t page = page0 + c / 2;
        if (bfp8) {
            noc_async_read(dst.get_noc_addr(page, 0), exp_stage, 64);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint8_t* e = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(exp_stage);
            volatile tt_l1_ptr uint8_t* s = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
            for (uint32_t h = 0; h < 2; ++h) {
                e[(f0 + h) * 16 + fr] = s[first_chunk + c + h];
                noc_async_write(
                    src + 64 + (first_chunk + c + h) * 16, dst.get_noc_addr(page, 64 + (f0 + h) * 256 + fr * 16), 16);
            }
            noc_async_write(exp_stage, dst.get_noc_addr(page, 0), 64);
            noc_async_write_barrier();  // exp_stage is reused for the next tile
        } else {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(
                    src + (first_chunk + c + h) * 32, dst.get_noc_addr(page, ((f0 + h) * 256 + fr * 16) * 2), 32);
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t ndst = get_named_compile_time_arg_val("ndst");
    constexpr uint32_t bfp8 = get_named_compile_time_arg_val("bfp8");
    constexpr uint32_t mass = get_named_compile_time_arg_val("mass");
    constexpr uint32_t cb_out = 19, cb_mass = 20, cb_stage = 25;
    constexpr uint32_t tile_bytes = bfp8 ? 1088 : 2048;

    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    const uint32_t h = get_arg_val<uint32_t>(0);
    uint32_t a = 1;
    if constexpr (mass) {
        cb_wait_front(cb_mass, 1);
    }
    cb_wait_front(cb_out, 1);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t d = 0; d < ndst; ++d) {
        const uint32_t base = get_arg_val<uint32_t>(a++);
        const uint32_t seq_tiles = get_arg_val<uint32_t>(a++);  // S / 32
        const uint32_t width = get_arg_val<uint32_t>(a++);      // tiles per row
        const uint32_t t0 = get_arg_val<uint32_t>(a++);
        const uint32_t pos_addr = get_arg_val<uint32_t>(a++);
        const uint32_t pos = attnio::read_u32(pos_addr, 4, stage);
        const uint32_t page0 = (h * seq_tiles + (pos >> 5)) * width;
        write_row(base, tile_bytes, page0 + t0, pos & 31, src, 0, 8, stage + 64, bfp8);
        if constexpr (mass) {
            if (d == ndst - 1) {  // the far cache is the last destination
                write_row(base, tile_bytes, page0 + 4, pos & 31, get_read_ptr(cb_mass), 0, 2, stage + 64, bfp8);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
    if constexpr (mass) {
        cb_pop_front(cb_mass, 1);
    }
}
