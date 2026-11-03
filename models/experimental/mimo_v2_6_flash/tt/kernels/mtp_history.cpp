// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The target-hidden history and the MTP ring backfill's plan (tt/mtp_backfill.py; host mirror
// reference/mtp_backfill.py), one core, four modes. The history: a bf16 row-major buffer of S * H rows (one page per
// row, row_bytes each; entry e = slot * H + position % H) and one int32 page [2 * S * H]: each entry's position (-1
// none) at e, its token at S * H + e. The page is read whole, patched in L1 and written back whole; the hidden rows
// move one page at a time through L1. Every other buffer is a one-page interleaved DRAM tensor.
//
// Mode 0, a decode step (decode trace, after the final norm): slot s < S at cur_pos[s] >= 0 writes its source row s
// (the final norm's rows, row-major) with its input token (tokens, copied at the step's start).
// Mode 1, a verify pass (verify trace): verify slot j at P[j] >= 0 writes its source row j into the ring of its
// request's slot (the chain row's word chain_hslots + j, slot + 1; 0 none).
// Mode 2, a write list (prefill): per entry (source row, entry, position, token); source -1 invalidates the entry,
// -2 skips it.
// Mode 3, the backfill plan (backfill trace): from the upload (drafting: 0 idle, else the draft groups' width w; slot,
// A, draft group g) and the page, the fill
// of draft group g's whole
// ring from S0 = floor(max(0, A - W + 1) / block) * block (W the MTP window, H >= W + S - 1: the window before a
// verify pass at A and the pass's rows): row r at position q = S0 + r takes history entry slot * H + q % H when the
// history holds q in [A - W + 1, A - 1] (else entry 0), RoPE index min(q, A), and for layer i the token the history
// holds at q + i + 1 (else 0; past A a verify pass's, committed in every row the draft does not re-run); the fill's
// page table row (ring blocks from S0's, then the scratch blocks); with every layer the stash rows A - 2, A - 1
// (positions, layer-0 tokens, hidden rows copied into the stash's hidden buffer) when the history holds them and their
// tokens, in group g's last two stash slots g * w + w - 2, g * w + w - 1 (its other slots emptied; a group's ring and
// scratch blocks start at
// g * table_words). Idle: group 0's scratch blocks, the stash left alone.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE uint64_t dram_addr(uint32_t base, uint32_t page_size, uint32_t page = 0, uint32_t offset = 0) {
    const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = page_size};
    return get_noc_addr(page, g, offset);
}

constexpr uint32_t align64(uint32_t bytes) { return (bytes + 63) & ~63u; }

}  // namespace

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);          // history slots (SPEC_SLOTS)
    constexpr uint32_t H = get_compile_time_arg_val(3);          // history rows per slot
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);  // one hidden row (bf16)
    constexpr uint32_t N = get_compile_time_arg_val(5);          // backfill rows (the ring)
    constexpr uint32_t layers = get_compile_time_arg_val(6);
    constexpr uint32_t block = get_compile_time_arg_val(7);
    constexpr uint32_t chain_row = get_compile_time_arg_val(8);  // the verify upload's chain row
    constexpr uint32_t chain_hslots = get_compile_time_arg_val(9);
    constexpr uint32_t table_words = get_compile_time_arg_val(10);  // the fill table row's width
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(11);
    constexpr uint32_t W = get_compile_time_arg_val(12);   // the MTP layers' SWA window
    constexpr uint32_t GS = get_compile_time_arg_val(13);  // a draft group's slots (its stash slots)
    // The history's slots may spill past the chain row (adaptive draft groups: reference/spec_inputs.py VERIFY_ROWS).
    constexpr uint32_t verify_rows = chain_row + 1 + (chain_hslots + S + B - 1) / B - 1;
    static_assert(S <= B && chain_hslots + S <= (verify_rows - chain_row) * B && N % block == 0);
    static_assert(2 * (N / block) <= table_words);
    static_assert(GS >= 2 && S % GS == 0);
    static_assert(W + S - 1 <= H && W + block <= N);
    constexpr uint32_t page_bytes = 2 * S * H * 4;
    constexpr uint32_t ring_blocks = N / block;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t base = get_write_ptr(cb_scratch);
    const uint32_t page_l1 = base;
    const uint32_t row_l1 = page_l1 + align64(page_bytes);
    const uint32_t in_l1 = row_l1 + align64(row_bytes);
    volatile tt_l1_ptr int32_t* page = reinterpret_cast<volatile tt_l1_ptr int32_t*>(page_l1);

    // Copy hidden row `src` of the buffer at `src_addr` into history row `e`, and set the entry's position and token.
    auto write = [&](uint32_t src_addr, uint32_t src, uint32_t hist_addr, uint32_t e, int32_t pos, int32_t token) {
        noc_async_read(dram_addr(src_addr, row_bytes, src), row_l1, row_bytes);
        noc_async_read_barrier();
        noc_async_write(row_l1, dram_addr(hist_addr, row_bytes, e), row_bytes);
        noc_async_write_barrier();
        page[e] = pos;
        page[S * H + e] = token;
    };

    if constexpr (mode == 0 || mode == 1 || mode == 2) {
        // Runtime args: page, history rows, source rows, then the mode's inputs.
        const uint32_t page_addr = get_arg_val<uint32_t>(0);
        const uint32_t hist_addr = get_arg_val<uint32_t>(1);
        const uint32_t src_addr = get_arg_val<uint32_t>(2);
        noc_async_read(dram_addr(page_addr, page_bytes), page_l1, page_bytes);
        if constexpr (mode == 0) {
            noc_async_read(dram_addr(get_arg_val<uint32_t>(3), B * 4), in_l1, B * 4);                   // positions
            noc_async_read(dram_addr(get_arg_val<uint32_t>(4), B * 4), in_l1 + align64(B * 4), B * 4);  // tokens
            noc_async_read_barrier();
            const int32_t* pos = reinterpret_cast<const int32_t*>(in_l1);
            const int32_t* tokens = reinterpret_cast<const int32_t*>(in_l1 + align64(B * 4));
            for (uint32_t s = 0; s < S; ++s) {
                if (pos[s] >= 0) {
                    write(src_addr, s, hist_addr, s * H + static_cast<uint32_t>(pos[s]) % H, pos[s], tokens[s]);
                }
            }
        } else if constexpr (mode == 1) {
            constexpr uint32_t verify_bytes = verify_rows * B * 4;
            noc_async_read(dram_addr(get_arg_val<uint32_t>(3), verify_bytes), in_l1, verify_bytes);
            noc_async_read_barrier();
            const int32_t* tokens = reinterpret_cast<const int32_t*>(in_l1);
            const int32_t* pos = tokens + B;
            const int32_t* hslots = tokens + chain_row * B + chain_hslots;
            for (uint32_t j = 0; j < S; ++j) {
                const int32_t h = hslots[j];
                if (pos[j] >= 0 && h >= 1 && h <= static_cast<int32_t>(S)) {
                    const uint32_t e = static_cast<uint32_t>(h - 1) * H + static_cast<uint32_t>(pos[j]) % H;
                    write(src_addr, j, hist_addr, e, pos[j], tokens[j]);
                }
            }
        } else {
            const uint32_t list_addr = get_arg_val<uint32_t>(3);
            const uint32_t entries = get_arg_val<uint32_t>(4);
            noc_async_read(dram_addr(list_addr, entries * 16), in_l1, entries * 16);
            noc_async_read_barrier();
            const int32_t* list = reinterpret_cast<const int32_t*>(in_l1);
            for (uint32_t k = 0; k < entries; ++k) {
                const int32_t src = list[4 * k];
                const uint32_t e = static_cast<uint32_t>(list[4 * k + 1]);
                if (src >= 0) {
                    write(src_addr, static_cast<uint32_t>(src), hist_addr, e, list[4 * k + 2], list[4 * k + 3]);
                } else if (src == -1) {
                    page[e] = -1;
                    page[S * H + e] = 0;
                }
            }
        }
        noc_async_read_barrier();
        noc_async_write(page_l1, dram_addr(page_addr, page_bytes), page_bytes);
        noc_async_write_barrier();
    } else {
        // Runtime args: upload, page, history rows, hidden rows out, RoPE out, table out, stash page, stash hidden,
        // then the layers' tokens out.
        const uint32_t upload_addr = get_arg_val<uint32_t>(0);
        const uint32_t page_addr = get_arg_val<uint32_t>(1);
        const uint32_t hist_addr = get_arg_val<uint32_t>(2);
        const uint32_t stash_addr = get_arg_val<uint32_t>(6);
        const uint32_t stash_hidden_addr = get_arg_val<uint32_t>(7);
        constexpr uint32_t out_bytes = N * 4;
        const uint32_t upload_l1 = in_l1;
        const uint32_t stash_l1 = upload_l1 + align64(B * 4);
        const uint32_t out_l1 = stash_l1 + align64(B * 4);  // hidden rows, RoPE, layers x tokens, table
        noc_async_read(dram_addr(page_addr, page_bytes), page_l1, page_bytes);
        noc_async_read(dram_addr(upload_addr, B * 4), upload_l1, B * 4);
        if constexpr (layers > 1) {
            noc_async_read(dram_addr(stash_addr, B * 4), stash_l1, B * 4);
        }
        noc_async_read_barrier();
        const int32_t* upload = reinterpret_cast<const int32_t*>(upload_l1);
        volatile tt_l1_ptr int32_t* stash = reinterpret_cast<volatile tt_l1_ptr int32_t*>(stash_l1);
        volatile tt_l1_ptr int32_t* hidden = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1);
        volatile tt_l1_ptr int32_t* rope = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + out_bytes);
        auto tokens = [&](uint32_t i) {
            return reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + (2 + i) * out_bytes);
        };
        volatile tt_l1_ptr int32_t* table =
            reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + (2 + layers) * out_bytes);
        for (uint32_t r = 0; r < N; ++r) {
            hidden[r] = 0;
            rope[r] = static_cast<int32_t>(r);
            for (uint32_t i = 0; i < layers; ++i) {
                tokens(i)[r] = 0;
            }
        }
        for (uint32_t k = 0; k < table_words; ++k) {
            table[k] = static_cast<int32_t>(ring_blocks + k % ring_blocks);  // the scratch blocks
        }
        const bool drafting = upload[0] != 0;
        if (drafting) {
            const uint32_t slot = static_cast<uint32_t>(upload[1]);
            const int32_t a = upload[2];
            const uint32_t w = upload[0] <= static_cast<int32_t>(GS) ? static_cast<uint32_t>(upload[0]) : GS;
            const uint32_t first = static_cast<uint32_t>(upload[3]) * w;  // the group's stash slots
            const int32_t blocks = upload[3] * static_cast<int32_t>(table_words);
            const int32_t lo = a - static_cast<int32_t>(W) + 1 > 0 ? a - static_cast<int32_t>(W) + 1 : 0;
            const int32_t s0 = lo / static_cast<int32_t>(block) * static_cast<int32_t>(block);
            auto held = [&](int32_t q) { return q >= 0 && page[slot * H + static_cast<uint32_t>(q) % H] == q; };
            auto token = [&](int32_t q) { return page[S * H + slot * H + static_cast<uint32_t>(q) % H]; };
            for (uint32_t k = 0; k < table_words; ++k) {
                const uint32_t b = k < ring_blocks ? (static_cast<uint32_t>(s0) % N / block + k) % ring_blocks
                                                   : ring_blocks + k % ring_blocks;
                table[k] = blocks + static_cast<int32_t>(b);
            }
            for (uint32_t r = 0; r < N; ++r) {
                const int32_t q = s0 + static_cast<int32_t>(r);
                rope[r] = q < a ? q : a;
                if (q >= lo && q <= a - 1 && held(q)) {
                    hidden[r] = static_cast<int32_t>(slot * H + static_cast<uint32_t>(q) % H);
                    for (uint32_t i = 0; i < layers; ++i) {
                        const int32_t x = q + static_cast<int32_t>(i) + 1;
                        if (held(x)) {
                            tokens(i)[r] = token(x);
                        }
                    }
                }
            }
            if constexpr (layers > 1) {
                // The group's stash slots: its last two hold rows A - 2, A - 1 when the history holds them and their
                // layer-0 tokens; their hidden rows go to those slots' stash hidden rows.
                for (uint32_t k = first; k < first + w; ++k) {
                    stash[k] = -1;
                    stash[S + k] = 0;
                }
                for (uint32_t k = first + w - 2; k < first + w; ++k) {
                    const int32_t q = a - static_cast<int32_t>(first + w - k);
                    if (held(q) && held(q + 1)) {
                        stash[k] = q;
                        stash[S + k] = token(q + 1);
                        const uint32_t e = slot * H + static_cast<uint32_t>(q) % H;
                        noc_async_read(dram_addr(hist_addr, row_bytes, e), row_l1, row_bytes);
                        noc_async_read_barrier();
                        noc_async_write(row_l1, dram_addr(stash_hidden_addr, row_bytes, k), row_bytes);
                        noc_async_write_barrier();
                    }
                }
            }
        }
        noc_async_write(out_l1, dram_addr(get_arg_val<uint32_t>(3), out_bytes), out_bytes);
        noc_async_write(out_l1 + out_bytes, dram_addr(get_arg_val<uint32_t>(4), out_bytes), out_bytes);
        noc_async_write(
            out_l1 + (2 + layers) * out_bytes, dram_addr(get_arg_val<uint32_t>(5), table_words * 4), table_words * 4);
        for (uint32_t i = 0; i < layers; ++i) {
            noc_async_write(
                out_l1 + (2 + i) * out_bytes, dram_addr(get_arg_val<uint32_t>(8 + i), out_bytes), out_bytes);
        }
        if constexpr (layers > 1) {
            if (drafting) {
                noc_async_write(stash_l1, dram_addr(stash_addr, B * 4), B * 4);
            }
        }
        noc_async_write_barrier();
    }
}
