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
// chain (multi-token verify prep, tokens in one program, token t + 1's row = token t's row + 1 in every
// destination): the bfp8 exponent blocks are read, patched and written back, so tokens that write rows of the same
// tile must not overlap. Runtime args after the destinations: wait, next_x, next_y, has_next. Every token
// prefetches its blocks and writes its mantissas at once; the exponent blocks pass down the chain in L1: a token
// whose row is not a tile's last hands its patched blocks to the next token's core (same stage address, same CB
// set) and increments that core's semaphore 0 instead of writing them; a token with wait waits on its semaphore
// and patches the blocks it was handed (row != 0) or its own prefetch (row 0: the previous token wrote its tile).
// The last token of a tile writes the blocks.
// far4: the far cache (last destination) is bfp4 (source bfp8): same exponent byte, each mantissa rounded to 3 bits
//       (nearest, ties to even, as the host packer), two per byte (element 0 in the low nibble); the 16 B block
//       holding the face row and its neighbour is prefetched, patched and written back.
#include "attn_io.hpp"

namespace {
// bfp4 cache: offset of the 16 B block holding face f, row fr (8 B per face row) and of the row inside it
FORCE_INLINE uint32_t bfp4_blk(uint32_t f, uint32_t fr) { return 64 + f * 128 + (fr & ~1u) * 8; }

// Row r of a bfp4 cache's tiles page0 ..: like write_row's bfp8 case; mants holds, per chunk, the 64 B aligned block
// read from the tile that contains the chunk's 16 B block.
FORCE_INLINE void write_row4(
    uint32_t base, uint32_t page0, uint32_t r, uint32_t src, uint32_t nchunks, uint32_t exps, uint32_t mants) {
    const InterleavedAddrGen<true> dst = {.bank_base_address = base, .page_size = 576};
    const uint32_t f0 = (r >> 4) * 2, fr = r & 15;
    volatile tt_l1_ptr uint8_t* s = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
    for (uint32_t c = 0; c < nchunks; c += 2) {
        const uint32_t page = page0 + c / 2;
        const uint32_t e_l1 = exps + (c / 2) * 64;
        volatile tt_l1_ptr uint8_t* e = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(e_l1);
        for (uint32_t h = 0; h < 2; ++h) {
            e[(f0 + h) * 16 + fr] = s[c + h];
            const uint32_t blk = bfp4_blk(f0 + h, fr);
            const uint32_t l1 = mants + (c + h) * 64 + (blk & 63);  // the 16 B block inside the staged 64 B
            attnio::bfp8_to_bfp4(src + 64 + (c + h) * 16, l1 + (fr & 1) * 8, 8);
            noc_async_write(l1, dst.get_noc_addr(page, blk), 16);
        }
        noc_async_write(e_l1, dst.get_noc_addr(page, 0), 64);
    }
}

FORCE_INLINE void read_mants4(uint32_t base, uint32_t page0, uint32_t r, uint32_t nchunks, uint32_t mants) {
    const InterleavedAddrGen<true> dst = {.bank_base_address = base, .page_size = 576};
    const uint32_t f0 = (r >> 4) * 2, fr = r & 15;
    for (uint32_t c = 0; c < nchunks; ++c) {
        const uint32_t blk = bfp4_blk(f0 + c % 2, fr);
        noc_async_read(dst.get_noc_addr(page0 + c / 2, blk & ~63u), mants + c * 64, 64);
    }
}
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
    uint32_t bfp8,
    bool put_exps = true) {
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
            if (put_exps) {
                noc_async_write(e_l1, dst.get_noc_addr(page, 0), 64);
            }
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
    constexpr uint32_t far4 = get_named_compile_time_arg_val("far4");
    static_assert(!far4 || bfp8, "a bfp4 far cache needs bfp8 rows");
    constexpr uint32_t cb_out = 19, cb_mass = 20, cb_stage = 25;
    constexpr uint32_t tile_bytes = bfp8 ? 1088 : 2048;

    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    const uint32_t h = get_arg_val<uint32_t>(0);
    // stage: [64 d) slot index of destination d, then per destination 5 exponent blocks (4 row tiles + mass), then
    // (far4) 10 staged mantissa blocks of the far cache (8 row chunks + 2 mass chunks)
    uint32_t base[ndst], page0[ndst], row0[ndst], r[ndst];  // row0: first tile of the cache row
    for (uint32_t d = 0; d < ndst; ++d) {
        const InterleavedAddrGen<true> pa = {.bank_base_address = get_arg_val<uint32_t>(1 + d * 5 + 4), .page_size = 4};
        noc_async_read(pa.get_noc_addr(0, 0), stage + 64 * d, 64);
    }
    noc_async_read_barrier();
    static_assert(!(chain && far4), "chain: the handed blocks use the bfp4 staging area");
    // chain: the blocks the previous token hands over land here (the next token's at the same address)
    const uint32_t fwd = stage + 64 * ndst + ndst * 5 * 64;
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
            const bool d4 = far4 && d == ndst - 1;
            const uint32_t tb = d4 ? 576 : tile_bytes;
            read_exps(base[d], tb, page0[d], 4, exps);
            if (mass && d == ndst - 1) {
                read_exps(base[d], tb, row0[d] + 4, 1, exps + 4 * 64);
            }
            if (d4) {
                const uint32_t mants = stage + 64 * ndst + ndst * 5 * 64;
                read_mants4(base[d], page0[d], r[d], 8, mants);
                if (mass) {
                    read_mants4(base[d], row0[d] + 4, r[d], 2, mants + 8 * 64);
                }
            }
        }
    }
    noc_async_read_barrier();
    if constexpr (mass) {
        cb_wait_front(cb_mass, 1);
    }
    cb_wait_front(cb_out, 1);
    const uint32_t src = get_read_ptr(cb_out);
    bool has_next = false;
    if constexpr (chain) {
        const uint32_t c = 1 + ndst * 5;
        has_next = get_arg_val<uint32_t>(c + 3) != 0;
        if (bfp8 && get_arg_val<uint32_t>(c)) {
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
            noc_semaphore_wait(sem, 1);
            noc_semaphore_set(sem, 0);
            for (uint32_t d = 0; d < ndst; ++d) {
                if (r[d] != 0) {  // same tiles as the previous token: go on from its blocks
                    volatile tt_l1_ptr uint32_t* to =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64 * ndst + d * 5 * 64);
                    volatile tt_l1_ptr uint32_t* from =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd + d * 5 * 64);
                    for (uint32_t i = 0; i < 5 * 16; ++i) {
                        to[i] = from[i];
                    }
                }
            }
        }
    }
    for (uint32_t d = 0; d < ndst; ++d) {
        const uint32_t exps = stage + 64 * ndst + d * 5 * 64;
        if (far4 && d == ndst - 1) {
            const uint32_t mants = stage + 64 * ndst + ndst * 5 * 64;
            write_row4(base[d], page0[d], r[d], src, 8, exps, mants);
            if (mass) {
                write_row4(base[d], row0[d] + 4, r[d], get_read_ptr(cb_mass), 2, exps + 4 * 64, mants + 8 * 64);
            }
            continue;
        }
        const bool hand = chain && bfp8 && has_next && r[d] != 31;  // the next token goes on in these tiles
        write_row(base[d], tile_bytes, page0[d], r[d], src, 8, exps, bfp8, !hand);
        if (mass && d == ndst - 1) {  // the far cache is the last destination; mass chunk at tile 4
            write_row(base[d], tile_bytes, row0[d] + 4, r[d], get_read_ptr(cb_mass), 2, exps + 4 * 64, bfp8, !hand);
        }
        if constexpr (chain) {
            if (hand) {
                const uint32_t c = 1 + ndst * 5;
                noc_async_write(
                    exps,
                    get_noc_addr(get_arg_val<uint32_t>(c + 1), get_arg_val<uint32_t>(c + 2), fwd + d * 5 * 64),
                    5 * 64);
            }
        }
    }
    noc_async_write_barrier();
    if constexpr (chain) {
        const uint32_t c = 1 + ndst * 5;
        if (bfp8 && has_next) {
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
