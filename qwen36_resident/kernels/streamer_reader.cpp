// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident streamer reader (NCRISC): seeds x with x0 of the token state, then streams this core's columns of every
// layer's weights from its DRAM bank into the shared weight ring, running ahead of compute by up to the ring size
// (across the mixer phases and the all-reduces). A GDN layer also brings its conv taps and conv history (oldest first).
// A bank's cores interleave their blocks in the bank (slot r * cores_per_bank + j is block r of the
// bank's core j), and each block's pages alternate between NOC0 and NOC1: one NOC's reads cap at
// ~47 GB/s per bank, both reach the bank's ~64 GB/s.
//
// Runtime args: 0 bank_id, 1 vc, 2 slot j of this core in its bank, 3 cores_per_bank, then per entry e:
// 4 + 4e n_tiles, base address, per-copy stride (bytes per bank), copies; then 28 conv taps address
// (DRAM rows of 4 x kBlk 1x32 tiles), 29 conv history address (rows of kHistSlots x kBlk), 30 this core's first
// row (core index x GDN copies), 31 GDN copies, 32 token state address, 33 optional timeline buffer
// (0 = off): per layer, the wall clock at the end of each entry's stream and the cycles each entry
// spent waiting for ring room, 34 n_conv (q|k|v tiles of this core: only the 32x32 views holding them of
// the taps and history are read).

#include "api/dataflow/dataflow_api.h"
#include "streamer_common.hpp"

using namespace resident;

namespace {

constexpr uint32_t kInFlight = 2;

uint32_t ring_base;
volatile uint32_t* consumed;
RingCursor cursor;

volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
uint32_t stall_cycles;

FORCE_INLINE void wait_for_room() {
    constexpr uint32_t ring_units = ring_bytes / kUnit;
    if (((cursor.units - *consumed) & 0xffff) > ring_units) {
        const uint32_t t0 = *clk;
        while (((cursor.units - *consumed) & 0xffff) > ring_units) {
        }
        stall_cycles += *clk - t0;
    }
}

// copy k (of this entry's kind) of the weights
template <uint32_t E>
FORCE_INLINE void stream_entry(uint32_t bank_id, uint32_t vc, uint32_t slot, uint32_t slots, uint32_t k) {
    using En = Entry<E>;
    const uint32_t n_tiles = get_arg_val<uint32_t>(4 + 4 * E);
    const uint32_t copies = get_arg_val<uint32_t>(7 + 4 * E);
    const uint32_t weight_addr = get_arg_val<uint32_t>(5 + 4 * E) + (k % copies) * get_arg_val<uint32_t>(6 + 4 * E);
    const uint32_t num_blocks = n_tiles * En::nkb;
    const uint64_t base0 = get_noc_addr_from_bank_id<true>(bank_id, weight_addr, 0);
    const uint64_t base1 = get_noc_addr_from_bank_id<true>(bank_id, weight_addr, 1);
    noc_async_read_one_packet_set_state<true>(base0, En::page_size, vc, 0);
    noc_async_read_one_packet_set_state<true>(base1, En::page_size, vc, 1);

    const uint32_t stride = slots * En::block_bytes;
    uint32_t src = slot * En::block_bytes;
    uint32_t issued = 0;
    uint32_t next_trid = 1;
    uint32_t wait_trid = 1;
    auto retire = [&]() {
        noc_async_read_barrier_with_trid(wait_trid, 0);
        noc_async_read_barrier_with_trid(wait_trid, 1);
        cb_push_back(En::cb, En::sb);
        wait_trid = wait_trid == kInFlight + 1 ? 1 : wait_trid + 1;
        issued--;
    };
    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t l1 = ring_base + cursor.place<E>();
        wait_for_room();
        noc_async_read_set_trid(next_trid, 0);
        noc_async_read_set_trid(next_trid, 1);
        for (uint32_t p = 0; p < En::pages; p++) {
            const uint32_t off = p * En::page_size;
            const uint8_t noc = (b * En::pages + p) & 1;
            noc_async_read_one_packet_with_state_with_trid(noc ? base1 : base0, src + off, l1 + off, next_trid, noc);
        }
        src += stride;
        next_trid = next_trid == kInFlight + 1 ? 1 : next_trid + 1;
        if (++issued == kInFlight + 1) {
            retire();
        }
    }
    while (issued > 0) {
        retire();
    }
}

}  // namespace

void kernel_main() {
    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t slots = get_arg_val<uint32_t>(3);
    const uint32_t conv_w_addr = get_arg_val<uint32_t>(28);
    const uint32_t hist_addr = get_arg_val<uint32_t>(29);
    const uint32_t first_row = get_arg_val<uint32_t>(30);
    const uint32_t gdn_copies = get_arg_val<uint32_t>(31);
    const uint32_t tok_addr = get_arg_val<uint32_t>(32);
    const uint32_t ts_addr = get_arg_val<uint32_t>(33);
    const uint32_t conv_bytes = views_of(get_arg_val<uint32_t>(34)) * 2048;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 0);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 1);
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed = get_cb_tiles_acked_ptr(cb_consumed);
        *consumed = 0;
    }
    const InterleavedAddrGen<true> tok{.bank_base_address = tok_addr, .page_size = kTokBytes};
    const InterleavedAddrGen<true> taps{.bank_base_address = conv_w_addr, .page_size = 4 * kBlkBytes};
    const InterleavedAddrGen<true> hist{.bank_base_address = hist_addr, .page_size = kHistSlots * kBlkBytes};

    // x of round 0 = x0; the ring step (for the conv slot order) lands in cb_dout, unused until compute runs
    // (fc: this chip's half of the fc input instead, see streamer_common.hpp)
    cb_reserve_back(cb_x, Ht);
    noc_async_read(
        tok.get_noc_addr(0, kTokX0 + (fc_in && chip >= num_chips / 2 ? Ht * kTileBytes : 0)),
        get_write_ptr(cb_x),
        Ht * kTileBytes);
    noc_async_read(tok.get_noc_addr(0), get_write_ptr(cb_dout), 16);
    noc_async_read_barrier();
    const uint32_t ring = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_dout));
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ring)) = ring + 1;  // for the writer
    cb_push_back(cb_x, Ht);
    if constexpr (batch > 1) {
        // rmsnorm constants: ones, the fold matrix (1 where row = column mod batch), the identity, bf16 32x32
        cb_reserve_back(cb_ones_full, 1);
        cb_reserve_back(cb_fold, 1);
        cb_reserve_back(cb_eye, 1);
        volatile tt_l1_ptr uint16_t* ones = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_ones_full));
        volatile tt_l1_ptr uint16_t* fold = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_fold));
        volatile tt_l1_ptr uint16_t* eye = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_eye));
        for (uint32_t i = 0; i < 1024; i++) {
            const uint32_t face = i / 256, row = (face / 2) * 16 + (i % 256) / 16, col = (face % 2) * 16 + i % 16;
            ones[i] = 0x3f80;
            fold[i] = row % batch == col % batch ? 0x3f80 : 0;
            eye[i] = row == col ? 0x3f80 : 0;
        }
        cb_push_back(cb_ones_full, 1);
        cb_push_back(cb_fold, 1);
        cb_push_back(cb_eye, 1);
    }
    if constexpr (verify) {
        // the conv's row selections (see streamer_common.hpp), bf16 32x32: [0] keeps the even rows, [1] moves
        // row R - 1 to odd row R, [2] moves row R + 1 to even row R
        cb_reserve_back(cb_shift, 3);
        volatile tt_l1_ptr uint16_t* sel = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_shift));
        for (uint32_t i = 0; i < 1024; i++) {
            const uint32_t face = i / 256, row = (face / 2) * 16 + (i % 256) / 16, col = (face % 2) * 16 + i % 16;
            sel[i] = row % 2 == 0 && col == row ? 0x3f80 : 0;
            sel[1024 + i] = row % 2 == 1 && col + 1 == row ? 0x3f80 : 0;
            sel[2048 + i] = row % 2 == 0 && col == row + 1 ? 0x3f80 : 0;
        }
        cb_push_back(cb_shift, 3);
    }

    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    auto mark = [&](uint32_t l, uint32_t e) {
        if (ts_addr) {
            ts[l * kTsWords + e] = *clk;
            ts[l * kTsWords + 4 + e] = stall_cycles;
        }
        stall_cycles = 0;
    };
    stall_cycles = 0;
    if constexpr (fc_in) {
        stream_entry<kQkvz>(bank_id, vc, slot, slots, 0);
    }
    uint32_t g = 0, a = 0;  // GDN / attention layers so far
    for (uint32_t l = 0; l < layers; l++) {
        if (is_attn(l)) {
            stream_entry<kQkvg>(bank_id, vc, slot, slots, a++);
        } else {
            stream_entry<kQkvz>(bank_id, vc, slot, slots, g);
            // this layer's taps and history, oldest first (see conv_step); a copy used again within the step
            // waits for the writer's write-back of its previous use
            const uint32_t row = first_row + g % gdn_copies;
            const uint32_t step = conv_step(ring, g, gdn_copies);
            if (g >= gdn_copies) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local)), g - gdn_copies + 1);
            }
            cb_reserve_back(cb_conv_w, 4 * kBlk);
            cb_reserve_back(cb_conv_hist, 3 * kBlk);
            if (conv_bytes > 0) {
                for (uint32_t j = 0; j < 4; j++) {
                    noc_async_read(
                        taps.get_noc_addr(row, j * kBlkBytes), get_write_ptr(cb_conv_w) + j * kBlkBytes, conv_bytes);
                }
                for (uint32_t j = 0; j < 3; j++) {
                    noc_async_read(
                        hist.get_noc_addr(row, ((step + j) % kHistSlots) * kBlkBytes),
                        get_write_ptr(cb_conv_hist) + j * kBlkBytes,
                        conv_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_conv_w, 4 * kBlk);
            cb_push_back(cb_conv_hist, 3 * kBlk);
            g++;
        }
        mark(l, 0);
        stream_entry<kOut>(bank_id, vc, slot, slots, l);
        mark(l, 1);
        stream_entry<kGateUp>(bank_id, vc, slot, slots, l);
        mark(l, 2);
        stream_entry<kDown>(bank_id, vc, slot, slots, l);
        mark(l, 3);
    }
    if constexpr (lm_head) {
        stream_entry<kHead>(bank_id, vc, slot, slots, 0);
    }
}
