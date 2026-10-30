// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Payload verifier for a receiver core's compute engine (one source for T0, T1 and T2). For each packet the
// receiver kernel hands over, T0 unpacks the payload as Int32 tiles into dest tiles 0, 1, ... (SFPU_TILE_WORDS words
// each, so the payload's words are consecutive in dest), T1's SFPU ORs every word's XOR with its expected word into a
// flag, and T2 packs the flag for the receiver kernel. A flag is zero exactly when every word matches. Word w of the
// packet's slot is expected to be seed + w, shifted per segment for a scatter write, whose chunks land in the slot out
// of payload order.
//
// The receiver kernel sends one request per packet on REQ_CB, {op, config, seed, payload words, segments}, and
// pushes the packet's page on that config's data CB, which is laid over the config's receive ring so its read pointer
// follows the ring slot by slot. An EXIT request ends the kernel.
//
// No compile-time args: the CB indices are fixed (tt_fabric_test_pattern.hpp); config c's data CB is DATA_CB_BASE + c.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt_fabric_test_pattern.hpp"

constexpr uint32_t REQ_CB = tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_CB_INDEX;
constexpr uint32_t FLAG_CB = tt::tt_fabric::fabric_tests::SFPU_VERIFY_FLAG_CB_INDEX;
constexpr uint32_t DATA_CB_BASE = tt::tt_fabric::fabric_tests::SFPU_VERIFY_DATA_CB_BASE;

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

// A request's segments (SFPU_VERIFY_REQ_SEGMENTS): slot word w of segment s is expected to be seed + w + delta[s].
struct PayloadSegments {
    uint32_t count;
    uint32_t end[tt::tt_fabric::fabric_tests::SFPU_VERIFY_MAX_SEGMENTS];
    uint32_t delta[tt::tt_fabric::fabric_tests::SFPU_VERIFY_MAX_SEGMENTS];
};

#ifdef TRISC_MATH
// The flag of dest's first num_words words, from tile 0 on, into step 0 of tile 0. Lane L checks word 64m + 2L at the
// even step of block m and word 64m + 2L + 1 at the odd step; vConstTileId is 2L on lane L. A segment may end at either
// of a lane's words, and the last block's words past num_words (whatever the slot held before) are not checked.
inline void flag_payload_page(uint32_t seed, uint32_t num_words, const PayloadSegments* segments) {
    const sfpi::vInt lane_word = sfpi::vConstTileId;
    sfpi::vInt flag = 0;
    uint32_t s = 0;
    for (uint32_t block_start = 0; block_start < num_words; block_start += 64) {
        while (segments->end[s] <= block_start) {
            s++;
        }
        sfpi::vInt expected_even = lane_word;
        expected_even += seed + block_start + segments->delta[s];
        sfpi::vInt expected_odd = expected_even + 1;
        // Segments ending inside this block: the words past an end take the next segment's delta.
        for (uint32_t t = s; t + 1 < segments->count && segments->end[t] < block_start + 64; t++) {
            const int32_t end = static_cast<int32_t>(segments->end[t] - block_start);
            const uint32_t step = segments->delta[t + 1] - segments->delta[t];
            v_if (lane_word >= sfpi::vInt(end)) {
                expected_even += step;
            }
            v_endif;
            v_if (lane_word >= sfpi::vInt(end - 1)) {
                expected_odd += step;
            }
            v_endif;
        }
        const sfpi::vInt even = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg++;
        const sfpi::vInt odd = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg++;
        const int32_t words_left = static_cast<int32_t>(num_words - block_start);
        if (words_left >= 64) {
            flag |= even ^ expected_even;
            flag |= odd ^ expected_odd;
        } else {
            v_if (lane_word < sfpi::vInt(words_left)) {
                flag |= even ^ expected_even;
            }
            v_endif;
            v_if (lane_word < sfpi::vInt(words_left - 1)) {
                flag |= odd ^ expected_odd;
            }
            v_endif;
        }
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = flag;
}
#endif

void kernel_main() {
    compute_kernel_hw_startup(DATA_CB_BASE, FLAG_CB);
    copy_init(DATA_CB_BASE);
    fill_tile_init();
    while (true) {
        cb_wait_front(REQ_CB, 1);
        const uint32_t op = read_tile_value(REQ_CB, 0, tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_OP);
        const uint32_t config = read_tile_value(REQ_CB, 0, tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_CONFIG);
        const uint32_t seed = read_tile_value(REQ_CB, 0, tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_SEED);
        const uint32_t num_words = read_tile_value(REQ_CB, 0, tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_WORDS);
        // read_tile_value hands the unpacker's read to the other threads, so all three threads make every read.
        [[maybe_unused]] PayloadSegments segments;
        const uint32_t num_segments =
            op == tt::tt_fabric::fabric_tests::SFPU_VERIFY_OP_CHECK
                ? read_tile_value(REQ_CB, 0, tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_SEGMENTS)
                : 0;
        segments.count = num_segments;
        for (uint32_t s = 0; s < num_segments; s++) {
            segments.end[s] =
                read_tile_value(REQ_CB, 0, tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_SEGMENT_END + 2 * s);
            segments.delta[s] =
                read_tile_value(REQ_CB, 0, tt::tt_fabric::fabric_tests::SFPU_VERIFY_REQ_SEGMENT_DELTA + 2 * s);
        }
        cb_pop_front(REQ_CB, 1);
        if (op == tt::tt_fabric::fabric_tests::SFPU_VERIFY_OP_EXIT) {
            break;
        }

        const uint32_t data_cb = DATA_CB_BASE + config;
        cb_wait_front(data_cb, 1);
        tile_regs_acquire();
        {
            // A slot's tiles are back to back from its start; the last one runs past the payload into whatever follows.
            TileSizedPages tile_pages(data_cb);
            for (uint32_t t = 0; t * tt::tt_fabric::fabric_tests::SFPU_TILE_WORDS < num_words; t++) {
                copy_tile(data_cb, t, t);
            }
        }
        MATH((_llk_math_eltwise_unary_sfpu_params_(flag_payload_page, 0, VectorMode::None, seed, num_words, &segments)));
        tile_regs_commit();
        cb_pop_front(data_cb, 1);

        cb_reserve_back(FLAG_CB, 1);
        tile_regs_wait();
        pack_tile(0, FLAG_CB);
        tile_regs_release();
        cb_push_back(FLAG_CB, 1);
    }
}
