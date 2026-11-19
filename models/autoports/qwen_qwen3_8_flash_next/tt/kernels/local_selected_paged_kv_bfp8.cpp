// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

inline uint32_t qsa_face_element(uint32_t row, uint32_t column) {
    return (((row >> 4) * 2u + (column >> 4)) * 256u) + (row & 15u) * 16u + (column & 15u);
}
inline uint32_t qsa_cache_tile(
    uint32_t physical_page, uint32_t head, uint32_t page_size, uint32_t token_in_page, uint32_t channel_tile) {
    return (((physical_page + head) * (page_size / 32u) + token_in_page / 32u) * 8u) + channel_tile;
}

inline uint16_t bfp8_to_bf16(uint32_t data, uint32_t exponent) {
    const uint32_t sign = (data & 128u) << 8;
    uint32_t mantissa = data & 127u;
    if (mantissa == 0) {
        return sign;
    }
    uint32_t shift = 0;
    while ((mantissa & 64u) == 0) {
        mantissa <<= 1;
        ++shift;
    }
    ASSERT(exponent >= shift);
    return sign | ((exponent - shift) << 7) | ((mantissa << 1) & 127u);
}
void kernel_main() {
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t PHYSICAL_PAGES = get_compile_time_arg_val(1);
    constexpr uint32_t LOGICAL_PAGES = get_compile_time_arg_val(2);
    constexpr uint32_t QUERIES = get_compile_time_arg_val(3);
    constexpr uint32_t SLOT_TILES = 65;  // 2080 selected slots
    constexpr auto ka = TensorAccessorArgs<4>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto koa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto voa = TensorAccessorArgs<koa.next_compile_time_args_offset()>();
    const auto keys = TensorAccessor(ka, get_arg_val<uint32_t>(0), 1088);
    const auto values = TensorAccessor(va, get_arg_val<uint32_t>(1), 1088);
    const auto pages = TensorAccessor(pa, get_arg_val<uint32_t>(2));
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(3), 4096);
    const auto kout = TensorAccessor(koa, get_arg_val<uint32_t>(4), 2048);
    const auto vout = TensorAccessor(voa, get_arg_val<uint32_t>(5), 2048);
    const uint32_t begin = get_arg_val<uint32_t>(6);
    const uint32_t end = get_arg_val<uint32_t>(7);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t kp = scratch;
    const uint32_t vp = scratch + 16384;
    const uint32_t ip = scratch + 32768;
    const uint32_t pp = scratch + 36864;
    const uint32_t lp = scratch + 36928;
    volatile tt_l1_ptr uint32_t* kw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kp);
    volatile tt_l1_ptr uint32_t* vw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vp);
    volatile tt_l1_ptr uint16_t* kh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(kp);
    volatile tt_l1_ptr uint16_t* vh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(vp);
    volatile tt_l1_ptr uint32_t* selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ip);
    volatile tt_l1_ptr uint32_t* page_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp);
    volatile tt_l1_ptr uint8_t* lines = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(lp);
    const uint32_t ep = lp + 2048;
    volatile tt_l1_ptr uint8_t* exponents = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(ep);
    uint64_t previous_exponent[16];
    uint64_t previous_line[32];
    for (uint32_t group = begin; group < end; ++group) {
        const uint32_t query = group / SLOT_TILES;
        const uint32_t head = 0;
        const uint32_t slot_tile = group % SLOT_TILES;
        ASSERT(query < QUERIES);
        for (uint32_t i = 0; i < 4096; ++i) {
            kw[i] = 0;
            vw[i] = 0;
        }
        for (uint32_t i = 0; i < 32; ++i) {
            previous_line[i] = ~uint64_t(0);
        }
        for (uint32_t i = 0; i < 16; ++i) {
            previous_exponent[i] = ~uint64_t(0);
        }
        uint32_t previous_page_line = 0xffffffffu;
        // Q<=32, therefore selected_ids has exactly one physical tile row.
        noc_async_read(ids.get_noc_addr(slot_tile), ip, 4096);
        noc_async_read_barrier();
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t token = selected[qsa_face_element(query, row)];
            if (token == 0xffffffffu) {
                continue;
            }
            ASSERT(token < LOGICAL_PAGES * PAGE_SIZE);
            const uint32_t logical_page = token / PAGE_SIZE;
            const uint32_t page_line = (logical_page * 4u) & ~63u;
            if (page_line != previous_page_line) {
                noc_async_read(pages.get_noc_addr(0) + page_line, pp, 64);
                noc_async_read_barrier();
                previous_page_line = page_line;
            }
            const uint32_t physical_page = page_words[logical_page & 15u];
            ASSERT(physical_page < PHYSICAL_PAGES);
            const uint32_t local_token = token % PAGE_SIZE;
            const uint32_t source_row = local_token & 31u;
            // Eight channels and two faces are read in parallel for K and V.
            // Consecutive selected tokens often share their aligned source
            // lines; retain each channel/face line until its address changes.
            for (uint32_t channel = 0; channel < 8; ++channel) {
                const uint32_t tile = qsa_cache_tile(physical_page, head, PAGE_SIZE, local_token, channel);
                const uint64_t ke = keys.get_noc_addr(tile);
                const uint64_t ve = values.get_noc_addr(tile);
                if (ke != previous_exponent[channel]) {
                    noc_async_read(ke, ep + channel * 64u, 64);
                    previous_exponent[channel] = ke;
                }
                if (ve != previous_exponent[channel + 8]) {
                    noc_async_read(ve, ep + (channel + 8) * 64u, 64);
                    previous_exponent[channel + 8] = ve;
                }
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t offset = 64u + qsa_face_element(source_row, half * 16);
                    const uint32_t aligned = offset & ~63u;
                    const uint32_t line = channel * 2 + half;
                    const uint64_t kaddr = keys.get_noc_addr(tile) + aligned;
                    const uint64_t vaddr = values.get_noc_addr(tile) + aligned;
                    if (kaddr != previous_line[line]) {
                        noc_async_read(kaddr, lp + line * 64, 64);
                        previous_line[line] = kaddr;
                    }
                    if (vaddr != previous_line[line + 16]) {
                        noc_async_read(vaddr, lp + (line + 16) * 64, 64);
                        previous_line[line + 16] = vaddr;
                    }
                }
            }
            noc_async_read_barrier();
            for (uint32_t channel = 0; channel < 8; ++channel) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t line = channel * 2 + half;
                    const uint32_t source_offset = qsa_face_element(source_row, half * 16) & 63u;
                    const uint32_t exp_offset = qsa_face_element(source_row, half * 16) / 16u;
                    const uint32_t dest = channel * 1024 + qsa_face_element(row, half * 16);
                    for (uint32_t element = 0; element < 16; ++element) {
                        kh[dest + element] = bfp8_to_bf16(
                            lines[line * 64 + source_offset + element], exponents[channel * 64 + exp_offset]);
                        vh[dest + element] = bfp8_to_bf16(
                            lines[(line + 16) * 64 + source_offset + element],
                            exponents[(channel + 8) * 64 + exp_offset]);
                    }
                }
            }
        }
        const uint32_t first_output_tile = group * 8;
        for (uint32_t channel = 0; channel < 8; ++channel) {
            noc_async_write(kp + channel * 2048, kout.get_noc_addr(first_output_tile + channel), 2048);
            noc_async_write(vp + channel * 2048, vout.get_noc_addr(first_output_tile + channel), 2048);
        }
        noc_async_write_barrier();
    }
}
