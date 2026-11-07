// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Payload generator for a sender core's compute engine (one source for T0, T1 and T2). It produces every packet
// payload the core's sender kernel sends, so the sender does not store them word by word: T1's SFPU writes the
// packet's words (seed + i) into dest tiles 0, 1, ... (SFPU_TILE_WORDS words each) and T2 packs them as Int32, back to
// back, into the next page of the payload CB. A page holds the core's largest payload.
//
// Pages are produced in strict round-robin order over the configs that carry a payload, skipping configs that are
// done, with each config's seed stepped by prng_next per packet. The sender consumes them in the same order (it
// waits on credits instead of skipping a config), so page k is always the payload the sender expects next.
//
// No compile-time args: the payload CB is SFPU_PAYLOAD_CB_INDEX (tt_fabric_test_pattern.hpp).
// Runtime args: the number of configs (at most SFPU_PAYLOAD_MAX_CONFIGS), then per config {initial seed, number of
// payload packets (0 for configs whose payloads the sender fills itself), tiles per payload}.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "tt_fabric_test_pattern.hpp"

constexpr uint32_t PAYLOAD_CB = tt::tt_fabric::fabric_tests::SFPU_PAYLOAD_CB_INDEX;
constexpr uint32_t MAX_CONFIGS = tt::tt_fabric::fabric_tests::SFPU_PAYLOAD_MAX_CONFIGS;

// Per-config state in each thread's local memory rather than on its small stack.
uint32_t seeds[MAX_CONFIGS];
uint32_t remaining[MAX_CONFIGS];
uint32_t tiles[MAX_CONFIGS];

// LLK addresses tile t of a CB at t pages past its read/write pointer, but a payload's tiles are SFPU_TILE_BYTES apart
// within one page. While one is alive, the CB's page is one tile (in the unpacker's or packer's CB interface; the math
// thread has none); the real page size is back before the page is pushed or popped.
struct TileSizedPages {
#ifndef TRISC_MATH
    explicit TileSizedPages(uint32_t cb) : cb_(cb), page_size_(get_local_cb_interface(cb).fifo_page_size) {
        get_local_cb_interface(cb).fifo_page_size = tt::tt_fabric::fabric_tests::SFPU_TILE_BYTES >> cb_addr_shift;
    }
    ~TileSizedPages() { get_local_cb_interface(cb_).fifo_page_size = page_size_; }
    uint32_t cb_;
    uint32_t page_size_;
#else
    explicit TileSizedPages(uint32_t) {}
#endif
};

#ifdef TRISC_MATH
// Words seed + i of a dest tile in Int32. In 32-bit dest an SFPU row step covers 64 words of the packed tile:
// lane L holds word 64m + 2L at the even step of block m and word 64m + 2L + 1 at the odd step, and
// vConstTileId is 2L on lane L.
inline void generate_payload_page(uint32_t seed) {
    sfpi::vInt word = sfpi::vConstTileId;
    word += seed;
    for (int block = 0; block < 16; ++block) {
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = word;
        sfpi::dst_reg++;
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = word + 1;
        sfpi::dst_reg++;
        word += 64;
    }
}
#endif

void kernel_main() {
    const uint32_t num_configs = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < num_configs; i++) {
        seeds[i] = get_arg_val<uint32_t>(1 + 3 * i);
        remaining[i] = get_arg_val<uint32_t>(2 + 3 * i);
        tiles[i] = get_arg_val<uint32_t>(3 + 3 * i);
    }

    compute_kernel_hw_startup(PAYLOAD_CB, PAYLOAD_CB);
    fill_tile_init();

    bool pages_left = true;
    while (pages_left) {
        pages_left = false;
        for (uint32_t i = 0; i < num_configs; i++) {
            if (remaining[i] == 0) {
                continue;
            }
            pages_left = true;

            tile_regs_acquire();
            for (uint32_t t = 0; t < tiles[i]; t++) {
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    generate_payload_page,
                    t,
                    VectorMode::None,
                    seeds[i] + t * tt::tt_fabric::fabric_tests::SFPU_TILE_WORDS)));
            }
            tile_regs_commit();

            cb_reserve_back(PAYLOAD_CB, 1);
            tile_regs_wait();
            {
                TileSizedPages tile_pages(PAYLOAD_CB);
                for (uint32_t t = 0; t < tiles[i]; t++) {
                    pack_tile<true>(t, PAYLOAD_CB, t);
                }
            }
            tile_regs_release();
            cb_push_back(PAYLOAD_CB, 1);

            seeds[i] = tt::tt_fabric::fabric_tests::prng_next(seeds[i]);
            remaining[i]--;
        }
    }
}
