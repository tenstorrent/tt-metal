// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Routed-expert program, reader (NOC 0, one per core; 24 cores, 3 per DRAM bank): (1) the activation rows of the
// gate/up cores (80 x 576 bytes: face 0 and rows 0 and 1 of face 1 of every tile of the gathered hidden: the other
// rows of the x tile are never read: a row of a matmul tile depends on its own row only), (2) the routing: the drain
// core's L1 shard of top-10 indices and this die's owner row give the experts of this die the rows selected (published
// to compute and writer as a metadata page), (3) this core's cached BF4 weight blocks of every active expert, in the
// order compute consumes them: with a gate/up unit: G(0), then G(i + 1) before W(i); else W(i).
// Block = 40 BF4 tiles = two 11,520 B transactions of the bank's piece of the expert; the CB is a ring of `slots`
// blocks with slots - 1 reads in flight (one trid each), dm0.cpp's pipeline.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#include "moe_experts_common.h"
#include "../../kernels/zones.h"

using namespace moe_experts;

void kernel_main() {
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t cb_md = get_named_compile_time_arg_val("cb_md");
    constexpr uint32_t cb_mdw = get_named_compile_time_arg_val("cb_mdw");
    constexpr uint32_t cb_scratch = get_named_compile_time_arg_val("cb_scratch");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t owner_bytes = get_named_compile_time_arg_val("owner_bytes");
    constexpr uint32_t stage = get_named_compile_time_arg_val("stage");  // 2: gate/up only, 3: the whole program
    constexpr bool x_fragments = get_named_compile_time_arg_val("x_fragments") == 1;
    constexpr uint32_t x_read_bytes = 576;  // face 0 (512 B) + rows 0 and 1 of face 1 (64 B)
    constexpr uint32_t txn_bytes = TXN_TILES * BF4_TILE_BYTES;
    constexpr uint32_t block_bytes = BLOCK_TILES * BF4_TILE_BYTES;
    constexpr uint32_t blocks_in_flight = slots - 1;
    static_assert(rows >= 1 && rows <= 2, "two row slots in the metadata page");

    constexpr auto hidden_args = TensorAccessorArgs<0, 0>();
    constexpr auto indices_args = TensorAccessorArgs<
        hidden_args.next_compile_time_args_offset(),
        hidden_args.next_common_runtime_args_offset()>();
    constexpr auto owner_args = TensorAccessorArgs<
        indices_args.next_compile_time_args_offset(),
        indices_args.next_common_runtime_args_offset()>();

    uint32_t arg = 0;
    const uint32_t w01_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t hidden_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t indices_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t owner_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t vc = get_arg_val<uint32_t>(arg++);
    const uint32_t g_kind = get_arg_val<uint32_t>(arg++);  // 0 none, 1 pair unit (8 blocks), 2 half unit (4 blocks)
    const uint32_t g_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t w_blocks = get_arg_val<uint32_t>(arg++);
    uint32_t g_bank[MAX_G_BLOCKS], g_blk[MAX_G_BLOCKS], w_bank[MAX_W_BLOCKS], w_blk[MAX_W_BLOCKS];
    for (uint32_t i = 0; i < MAX_G_BLOCKS; ++i) {
        g_bank[i] = get_arg_val<uint32_t>(arg++);
        g_blk[i] = get_arg_val<uint32_t>(arg++);
    }
    for (uint32_t i = 0; i < MAX_W_BLOCKS; ++i) {
        w_bank[i] = get_arg_val<uint32_t>(arg++);
        w_blk[i] = get_arg_val<uint32_t>(arg++);
    }

    const auto hidden = TensorAccessor(hidden_args, hidden_addr);
    const auto indices = TensorAccessor(indices_args, indices_addr);
    const auto owner = TensorAccessor(owner_args, owner_addr);

    Noc noc;
    CircularBuffer x(cb_x);
    CircularBuffer wcb(cb_w);
    CircularBuffer md(cb_md);
    CircularBuffer mdw(cb_mdw);
    CircularBuffer scratch(cb_scratch);

    uint32_t n = 0;
    uint32_t entry_expert[MAX_ACTIVE];
    {
        FUSED_ZONE("fz_me_r_meta");
        if (g_kind != 0) {
            x.reserve_back(HT);
            const uint32_t x_base = x.get_write_ptr();
            for (uint32_t k = 0; k < HT; ++k) {
                if constexpr (x_fragments) {  // rows 0 and 1 of both faces: 2 x 64 B
                    noc.async_read(
                        hidden,
                        CoreLocalMem<uint32_t>(x_base + k * TILE_BYTES),
                        64,
                        {.page_id = k, .offset_bytes = 0},
                        {});
                    noc.async_read(
                        hidden,
                        CoreLocalMem<uint32_t>(x_base + k * TILE_BYTES + 512),
                        64,
                        {.page_id = k, .offset_bytes = 512},
                        {});
                } else {  // bytes [0, 576): face 0 whole and rows 0 and 1 of face 1
                    noc.async_read(
                        hidden,
                        CoreLocalMem<uint32_t>(x_base + k * TILE_BYTES),
                        x_read_bytes,
                        {.page_id = k, .offset_bytes = 0},
                        {});
                }
            }
        }
        scratch.reserve_back(1);
        const uint32_t owner_base = scratch.get_write_ptr();
        const uint32_t idx_base = owner_base + owner_bytes;
        const uint32_t route_words = indices.get_aligned_page_size() / 2;
        noc_async_read(indices.get_noc_addr(0), idx_base, rows * 2 * route_words);
        noc.async_read(owner, CoreLocalMem<uint32_t>(owner_base), owner_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        if (g_kind != 0) {
            x.push_back(HT);
        }
        volatile tt_l1_ptr uint16_t* owner_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(owner_base);
        volatile tt_l1_ptr uint16_t* idx_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_base);
        md.reserve_back(1);
        mdw.reserve_back(1);
        volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(md.get_write_ptr());
        volatile tt_l1_ptr uint32_t* pagew = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mdw.get_write_ptr());
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t k = 0; k < TOP_K; ++k) {
                const uint32_t id = idx_u16[r * route_words + k];
                if (owner_u16[id] == 0) {
                    continue;
                }
                const uint32_t e = id & (EXPERTS_PER_DEVICE - 1);
                uint32_t j = 0;
                while (j < n && entry_expert[j] != e) {
                    ++j;
                }
                if (j == n) {
                    entry_expert[n] = e;
                    page[1 + MD_ENTRY_WORDS * j] = e;
                    page[2 + MD_ENTRY_WORDS * j] = 0;
                    page[3 + MD_ENTRY_WORDS * j] = 0;
                    page[4 + MD_ENTRY_WORDS * j] = 0;
                    ++n;
                }
                page[2 + MD_ENTRY_WORDS * j] |= 1u << r;
                page[3 + MD_ENTRY_WORDS * j + r] = k;
            }
        }
        page[0] = n;
        for (uint32_t i = 0; i < 1 + MD_ENTRY_WORDS * n; ++i) {
            pagew[i] = page[i];
        }
        md.push_back(1);
        mdw.push_back(1);
    }

    {
        FUSED_ZONE("fz_me_r_weights");
        const uint32_t w_cb_base = wcb.get_write_ptr();
        uint32_t slot_addr[slots];
        for (uint32_t s = 0; s < slots; ++s) {
            slot_addr[s] = w_cb_base + s * block_bytes;
        }
        uint32_t trid_issue = 1, trid_wait = 1, slot = 0, pending = 0;
        uint32_t cur_bank = 0xFFFFFFFFu;
        wcb.reserve_back(BLOCK_TILES * (blocks_in_flight - 1));

        auto read_block = [&](uint32_t bank, uint32_t byte_offset) {
            noc_async_read_set_trid(trid_issue);
            if (bank != cur_bank) {
                noc_async_read_one_packet_set_state<true>(get_noc_addr_from_bank_id<true>(bank, 0), txn_bytes, vc);
                cur_bank = bank;
            }
            const uint64_t base = get_noc_addr_from_bank_id<true>(bank, 0);
            noc_async_read_one_packet_with_state_with_trid<false, false>(
                base, byte_offset, slot_addr[slot], trid_issue);
            noc_async_read_one_packet_with_state_with_trid<false, false>(
                base, byte_offset + txn_bytes, slot_addr[slot] + txn_bytes, trid_issue);
            if (++slot == slots) {
                slot = 0;
            }
            if (++trid_issue > slots) {
                trid_issue = 1;
            }
            if (++pending == blocks_in_flight) {
                noc_async_read_barrier_with_trid(trid_wait);
                wcb.push_back(BLOCK_TILES);
                if (++trid_wait > slots) {
                    trid_wait = 1;
                }
                --pending;
                wcb.reserve_back(BLOCK_TILES * blocks_in_flight);
            }
        };
        auto g_unit = [&](uint32_t e) {
            for (uint32_t b = 0; b < g_blocks; ++b) {
                read_block(g_bank[b], w01_addr + (e * BANK_BLOCKS_PER_EXPERT + g_blk[b]) * block_bytes);
            }
        };
        auto w_unit = [&](uint32_t e) {
            for (uint32_t b = 0; b < w_blocks; ++b) {
                read_block(w_bank[b], w2_addr + (e * W2_BLOCKS_PER_EXPERT + w_blk[b]) * block_bytes);
            }
        };
        if (n > 0 && g_kind != 0) {
            g_unit(entry_expert[0]);
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (g_kind != 0 && i + 1 < n) {
                g_unit(entry_expert[i + 1]);
            }
            if constexpr (stage >= 3) {
                w_unit(entry_expert[i]);
            }
        }
        for (; pending > 0; --pending) {
            noc_async_read_barrier_with_trid(trid_wait);
            wcb.push_back(BLOCK_TILES);
            if (++trid_wait > slots) {
                trid_wait = 1;
            }
        }
        wcb.push_back(BLOCK_TILES);  // the one reserved slot nobody uses, as dm0.cpp
    }
}
